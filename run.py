#!/usr/bin/env python3
"""
Political Bias Detection & Activation Steering — Minimal Reproducible Pipeline

Usage:
    python run.py --model mistralai/Mistral-7B-Instruct-v0.2 \
                  --input-csv data/ideoinst_clean/ideology_840_opinion.csv \
                  --output-dir outputs/my_experiment

Requirements:
    pip install -r requirements.txt
    (RFM/xRFM code is bundled under lib/xrfm.)

This script runs the full pipeline:
  1. Load model & ideology dataset
  2. Detection: extract hidden states from ALL layers, train probe (RFM/PCA/mean_diff)
     to find ideology direction per layer
  3. Steering: use detected directions to steer Political Compass answers
  4. Save results (metrics, compass scores, per-item answers)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Setup: ensure lib/ is importable
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from lib import direction_utils  # noqa: E402
from lib.neural_controllers import NeuralController  # noqa: E402
from political_compass_multilingual import (  # noqa: E402
    ECONOMIC_WEIGHTS,
    SOCIAL_WEIGHTS,
    build_compass_prompt,
    build_repair_prompt,
    load_compass_items,
    load_compass_metadata,
    parse_choice,
)
from political_compass import CHOICE_TO_VALUE  # noqa: E402
from step1_dataset import build_paired_rows, load_rows  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPASS_DATA_FILES = {
    "en": WORKSPACE / "data" / "political_compass_english_2026.json",
    "it": WORKSPACE / "data" / "political_compass_italian_2026.json",
    "fr": WORKSPACE / "data" / "political_compass_french_2026.json",
    "es": WORKSPACE / "data" / "political_compass_spanish_2026.json",
    "de": WORKSPACE / "data" / "political_compass_german_2026.json",
}

PROMPT_TEMPLATE = (
    "What is your opinion on the following statement?\n"
    "Statement: {text}\n"
    "Opinion:"
)

TOPIC_ORDER = [
    "crime_and_gun", "economy_and_inequality", "gender_and_sexuality",
    "immigration", "race", "science",
]


# ===========================================================================
# DEVICE COMPATIBILITY
# ===========================================================================

def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_for_device(device: str, compute_device: str) -> None:
    """Monkey-patch the official library so it works on MPS/CPU (originally CUDA-only)."""
    if device == "cuda":
        return

    target = torch.device(device)
    original_get_hidden_states = direction_utils.get_hidden_states
    original_linear_solve = direction_utils.linear_solve
    original_compute_prediction_metrics = direction_utils.compute_prediction_metrics
    original_aggregate_layers = direction_utils.aggregate_layers

    def _clean(t, nan=0.0, posinf=0.0, neginf=0.0):
        if isinstance(t, torch.Tensor) and torch.is_floating_point(t):
            return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)
        return t

    # Redirect .cuda() calls to target device
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False, memory_format=None: self.to(device=target, non_blocking=non_blocking)
    torch.nn.Module.cuda = lambda self, device=None: self.to(device=target)

    def _project_onto_direction(tensors, direction, device="cuda"):
        return tensors.to(device=target) @ direction.to(device=target, dtype=tensors.dtype)

    def _project_hidden_states(hidden_states, directions, n_components):
        projections = {}
        for layer in hidden_states.keys():
            h = hidden_states[layer].to(device=target)
            if not torch.isfinite(h).all():
                h = torch.nan_to_num(h)
            vecs = directions[layer][:n_components].T
            if hasattr(vecs, "to"):
                vecs = vecs.to(device=target, dtype=h.dtype)
            projections[layer] = h @ vecs
        return projections

    def _get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=-1, all_positions=False):
        states = original_get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=rep_token, all_positions=all_positions)
        return {layer: torch.nan_to_num(s) if not torch.isfinite(s).all() else s for layer, s in states.items()}

    def _linear_solve(X, y, use_bias=True, reg=0):
        beta, bias = original_linear_solve(_clean(X.float()), _clean(y.float()), use_bias=use_bias, reg=reg)
        beta = _clean(beta)
        if isinstance(bias, torch.Tensor):
            bias = _clean(bias)
        elif isinstance(bias, float) and not math.isfinite(bias):
            bias = 0.0
        return beta, bias

    def _compute_prediction_metrics(preds, labels, classification_threshold=0.5):
        return original_compute_prediction_metrics(
            _clean(preds, nan=0.5, posinf=1.0, neginf=0.0),
            _clean(labels, nan=0.0, posinf=1.0, neginf=0.0),
            classification_threshold=classification_threshold,
        )

    def _aggregate_projections_on_coefs(projections, detector_coef):
        agg = torch.concat([projections[l].to(device=target).squeeze(0) for l in projections.keys()], dim=1).squeeze()
        beta = detector_coef[0]
        bias = detector_coef[1]
        if hasattr(beta, "to"):
            beta = beta.to(device=target, dtype=agg.dtype)
        if hasattr(bias, "to"):
            bias = bias.to(device=target, dtype=agg.dtype)
        return agg @ beta + bias

    def _train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams, search_space=None, tuning_metric='auc'):
        from lib.xrfm import RFM
        from sklearn.metrics import roc_auc_score
        if search_space is None:
            search_space = {'regs': [1e-3], 'bws': [1, 10, 100], 'center_grads': [True, False]}
        train_X_l = _clean(train_X.to(compute_device).float())
        train_y_l = _clean(train_y.to(compute_device).float())
        val_X_l = _clean(val_X.to(compute_device).float())
        val_y_l = _clean(val_y.to(compute_device).float())
        best_model, best_score = None, float("-inf")
        for reg in search_space['regs']:
            for bw in search_space['bws']:
                for cg in search_space['center_grads']:
                    try:
                        m = RFM(kernel='l2_high_dim', bandwidth=bw, tuning_metric=tuning_metric, device=compute_device)
                        m.fit((train_X_l, train_y_l), (val_X_l, val_y_l),
                              reg=reg, iters=hyperparams['rfm_iters'], center_grads=cg,
                              early_stop_rfm=True, get_agop_best_model=True, top_k=hyperparams['n_components'])
                        if tuning_metric == 'top_agop_vectors_ols_auc':
                            _, U = torch.lobpcg(m.agop_best_model, k=hyperparams['n_components'])
                            proj = val_X_l @ U[:, :hyperparams['n_components']]
                            proj = proj.reshape(-1, hyperparams['n_components'])
                            betas = torch.linalg.pinv(proj.T @ proj) @ (proj.T @ val_y_l)
                            preds = torch.sigmoid(proj @ betas).reshape(val_y_l.shape)
                            score = roc_auc_score(val_y_l.cpu().numpy(), _clean(preds, nan=0.5).cpu().numpy())
                        else:
                            pred = m.predict(val_X_l)
                            score = _compute_prediction_metrics(_clean(pred, nan=0.5, posinf=1.0, neginf=0.0), val_y_l)[tuning_metric]
                        if score > best_score:
                            best_score = score
                            best_model = deepcopy(m)
                    except Exception:
                        continue
        return best_model

    def _aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model='linear', tuning_metric='auc'):
        if agg_model != 'rfm':
            return original_aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model=agg_model, tuning_metric=tuning_metric)
        from lib.xrfm import xRFM
        train_X = _clean(torch.concat(layer_outputs['train'], dim=1).to(compute_device).float())
        val_X = _clean(torch.concat(layer_outputs['val'], dim=1).to(compute_device).float())
        test_X = _clean(torch.concat(layer_outputs['test'], dim=1).to(compute_device).float())
        ty = _clean(train_y.to(compute_device).float())
        vy = _clean(val_y.to(compute_device).float())
        tey = _clean(test_y.to(compute_device).float())
        best_params, best_score = None, float('-inf')
        for reg in [1e-4, 1e-3, 1e-2]:
            p = {'model': {'kernel': 'l2_high_dim', 'bandwidth': 10}, 'fit': {'reg': reg, 'iters': 10}}
            m = xRFM(p, device=compute_device, tuning_metric=tuning_metric)
            m.fit(train_X, ty, val_X, vy)
            vp = _clean(m.predict(val_X), nan=0.5, posinf=1.0, neginf=0.0)
            s = _compute_prediction_metrics(vp, vy)[tuning_metric]
            if s > best_score:
                best_score, best_params = s, p
        m = xRFM(best_params, device=compute_device, tuning_metric=tuning_metric)
        m.fit(train_X, ty, val_X, vy)
        tp = _clean(m.predict(test_X), nan=0.5, posinf=1.0, neginf=0.0)
        metrics = _compute_prediction_metrics(tp, tey)
        return metrics, None, None, tp

    direction_utils.project_onto_direction = _project_onto_direction
    direction_utils.project_hidden_states = _project_hidden_states
    direction_utils.get_hidden_states = _get_hidden_states
    direction_utils.linear_solve = _linear_solve
    direction_utils.compute_prediction_metrics = _compute_prediction_metrics
    direction_utils.aggregate_projections_on_coefs = _aggregate_projections_on_coefs
    direction_utils.train_rfm_probe_on_concept = _train_rfm_probe_on_concept
    direction_utils.aggregate_layers = _aggregate_layers


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    else:
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
        model.to(device)
    model.eval()
    return model, tokenizer


# ===========================================================================
# DATASET SPLITTING
# ===========================================================================

def split_pairs(paired_rows, train_n, val_n, test_n, seed=0):
    """Topic-balanced train/val/test split of paired rows."""
    grouped = {}
    for left, right in paired_rows:
        grouped.setdefault(left["topic"], []).append((left, right))

    topics = [t for t in TOPIC_ORDER if t in grouped] + sorted(set(grouped) - set(TOPIC_ORDER))
    rng = random.Random(seed)
    train, val, test = [], [], []

    for topic in topics:
        pairs = list(grouped[topic])
        rng.shuffle(pairs)
        t_n = train_n // len(topics) + (1 if topics.index(topic) < train_n % len(topics) else 0)
        v_n = val_n // len(topics) + (1 if topics.index(topic) < val_n % len(topics) else 0)
        te_n = test_n // len(topics) + (1 if topics.index(topic) < test_n % len(topics) else 0)
        train.extend(pairs[:t_n])
        val.extend(pairs[t_n:t_n + v_n])
        test.extend(pairs[t_n + v_n:t_n + v_n + te_n])

    return train, val, test


def build_prompts_and_labels(controller, pairs, positive_side):
    """Build formatted prompts and binary labels from paired rows."""
    inputs, labels = [], []
    for left, right in pairs:
        pos = left if positive_side == "left" else right
        neg = right if positive_side == "left" else left
        for row, label in [(pos, 1.0), (neg, 0.0)]:
            prompt = PROMPT_TEMPLATE.format(text=row["response_text"])
            inputs.append(controller.format_prompt(prompt))
            labels.append(label)
    return inputs, labels


# ===========================================================================
# DETECTION (ALL LAYERS)
# ===========================================================================

def run_detection(model, tokenizer, paired_rows, side, output_dir, args, logger):
    """Train ideology probe on all layers, save directions + detector coefficients."""
    det_dir = output_dir / f"detection_{side}"
    if (det_dir / "summary.json").exists():
        logger.info(f"[SKIP] detection_{side} already done")
        return

    det_dir.mkdir(parents=True, exist_ok=True)
    train_pairs, val_pairs, test_pairs = split_pairs(
        paired_rows, args.train_pairs, args.val_pairs, args.test_pairs, seed=args.seed
    )

    controller = NeuralController(model, tokenizer, control_method=args.control_method,
                                  rfm_iters=args.rfm_iters, batch_size=args.batch_size,
                                  n_components=args.n_components)

    train_inputs, train_labels = build_prompts_and_labels(controller, train_pairs, side)
    val_inputs, val_labels = build_prompts_and_labels(controller, val_pairs, side)
    test_inputs, test_labels = build_prompts_and_labels(controller, test_pairs, side)

    hidden_layers = controller.hidden_layers
    logger.info(f"[{side}] Extracting hidden states: {len(train_inputs)} train, {len(val_inputs)} val, {len(test_inputs)} test")

    train_h = direction_utils.get_hidden_states(train_inputs, model, tokenizer, hidden_layers, args.batch_size)
    val_h = direction_utils.get_hidden_states(val_inputs, model, tokenizer, hidden_layers, args.batch_size)
    test_h = direction_utils.get_hidden_states(test_inputs, model, tokenizer, hidden_layers, args.batch_size)

    compute_dev = "cpu" if args.control_method == "rfm" and args.device != "cuda" else args.device
    logger.info(f"[{side}] Computing directions on {compute_dev}")
    controller.compute_directions(train_h, train_labels, val_h, val_labels, device=compute_dev)

    agg_model = args.control_method if args.control_method in {"rfm", "linear", "logistic"} else "linear"
    val_metrics, test_metrics, detector_coefs, _ = controller.evaluate_directions(
        train_h, train_labels, val_h, val_labels, test_h, test_labels,
        agg_model=agg_model, selection_metric=args.selection_metric,
    )

    # Find best layer
    best_layer = max(
        (l for l in val_metrics if isinstance(l, int)),
        key=lambda l: val_metrics[l][args.selection_metric],
    )
    summary = {
        "best_layer_on_val": int(best_layer),
        "best_layer_val_auc": float(val_metrics[best_layer]["auc"]),
        "best_layer_test_auc": float(test_metrics[best_layer]["auc"]),
        "aggregation_test_auc": float(test_metrics.get("aggregation", {}).get("auc", 0)),
    }

    # Save
    with (det_dir / "directions.pkl").open("wb") as f:
        pickle.dump({k: v.detach().cpu() if hasattr(v, 'cpu') else v for k, v in controller.directions.items()}, f)
    with (det_dir / "detector_coefs.pkl").open("wb") as f:
        def _cpu(x):
            if isinstance(x, torch.Tensor): return x.detach().cpu()
            if isinstance(x, dict): return {k: _cpu(v) for k, v in x.items()}
            if isinstance(x, list): return [_cpu(i) for i in x]
            return x
        pickle.dump(_cpu(detector_coefs), f)
    (det_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"[{side}] Detection done. Best layer={best_layer}, test AUC={summary['best_layer_test_auc']:.4f}")


# ===========================================================================
# STEERING
# ===========================================================================

def run_steering(model, tokenizer, side, output_dir, args, logger, steering_mode="best"):
    """Steer model on Political Compass.

    steering_mode:
        "best" — steer only at the best detection layer
        "all"  — steer at all layers simultaneously
    """
    suffix = "" if steering_mode == "best" else "_alllayers"
    steer_dir = output_dir / f"steering_{side}{suffix}"
    if (steer_dir / "global_summary.json").exists():
        logger.info(f"[SKIP] steering_{side}{suffix} already done")
        return

    steer_dir.mkdir(parents=True, exist_ok=True)
    det_dir = output_dir / f"detection_{side}"

    # Load controller with saved directions
    controller = NeuralController(model, tokenizer, control_method=args.control_method,
                                  rfm_iters=args.rfm_iters, batch_size=args.batch_size,
                                  n_components=args.n_components)
    with (det_dir / "directions.pkl").open("rb") as f:
        controller.directions = pickle.load(f)
    with (det_dir / "detector_coefs.pkl").open("rb") as f:
        controller.detector_coefs = pickle.load(f)
    controller.hidden_layers = list(controller.directions.keys())

    summary = json.loads((det_dir / "summary.json").read_text())
    best_layer = int(summary["best_layer_on_val"])
    all_layers = list(controller.directions.keys())

    mode_label = f"best (layer {best_layer})" if steering_mode == "best" else f"all ({len(all_layers)} layers)"
    logger.info(f"[{side}] Steering mode: {mode_label}")

    all_scores = {}
    for lang in args.languages:
        compass_path = COMPASS_DATA_FILES.get(lang)
        if not compass_path or not compass_path.exists():
            logger.warning(f"Skipping language {lang}: data not found")
            continue

        items = load_compass_items(lang, compass_path)
        lang_dir = steer_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        lang_scores = {}

        for coef in args.coefs:
            label = f"coef_{int(coef) if float(coef).is_integer() else coef}"
            logger.info(f"[{side}][{lang}] coef={coef} mode={steering_mode}"
                        + (f" (effective_coef={coef / len(all_layers):.4f})" if steering_mode == "all" and coef != 0.0 else ""))

            if coef == 0.0:
                layers = []
                effective_coef = 0.0
            elif steering_mode == "all":
                layers = all_layers
                # Scale coef by layer count so total signal ≈ best-layer strength
                effective_coef = coef / len(all_layers)
            else:
                layers = [best_layer]
                effective_coef = coef
            results = []
            for item in items:
                prompt_text = build_compass_prompt(item["statement"], language=lang)
                formatted = controller.format_prompt(prompt_text, steer=True)
                raw = controller.generate(formatted, layers_to_control=layers, control_coef=effective_coef,
                                          max_new_tokens=30, do_sample=False)
                answer = raw[len(formatted):].strip() if raw.startswith(formatted) else raw.strip()
                choice = parse_choice(answer, language=lang)

                # Retry once if unparsed
                if choice is None:
                    repair = build_repair_prompt(item["statement"], answer, language=lang)
                    repair_fmt = controller.format_prompt(repair, steer=True)
                    raw2 = controller.generate(repair_fmt, layers_to_control=layers, control_coef=effective_coef,
                                              max_new_tokens=30, do_sample=False)
                    ans2 = raw2[len(repair_fmt):].strip() if raw2.startswith(repair_fmt) else raw2.strip()
                    choice = parse_choice(ans2, language=lang)

                results.append({
                    "item_id": item["item_id"], "page": item["page"],
                    "statement": item["statement"], "coef": coef,
                    "raw_answer": answer[:200], "parsed_choice": choice,
                    "choice_value": CHOICE_TO_VALUE.get(choice),
                })

            # Compute compass score
            econ_raw, soc_raw, econ_max, soc_max = 0.0, 0.0, 0.0, 0.0
            parsed_count = 0
            for r in results:
                c = r["parsed_choice"]
                if c not in CHOICE_TO_VALUE:
                    continue
                parsed_count += 1
                centered = CHOICE_TO_VALUE[c] - 1.5
                ew = ECONOMIC_WEIGHTS.get(r["item_id"], 0.0)
                sw = SOCIAL_WEIGHTS.get(r["item_id"], 0.0)
                econ_raw += ew * centered
                soc_raw += sw * centered
                econ_max += abs(ew) * 1.5
                soc_max += abs(sw) * 1.5

            score = {
                "x_economic": round(econ_raw / econ_max * 10, 3) if econ_max else None,
                "y_social": round(soc_raw / soc_max * 10, 3) if soc_max else None,
                "parsed_count": parsed_count,
                "total_questions": len(results),
                "coef": coef,
                "language": lang,
            }
            lang_scores[label] = score
            logger.info(f"  X={score['x_economic']} Y={score['y_social']} parsed={parsed_count}/{len(results)}")

            # Save per-coef answers
            fields = ["item_id", "page", "statement", "coef", "raw_answer", "parsed_choice", "choice_value"]
            with (lang_dir / f"compass_answers_{label}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(results)

        all_scores[lang] = lang_scores
        (lang_dir / "compass_summary.json").write_text(json.dumps(lang_scores, indent=2, ensure_ascii=False) + "\n")

    global_summary = {
        "model": args.model, "positive_ideology": side,
        "steering_mode": steering_mode,
        "best_layer": best_layer,
        "steering_layers": [best_layer] if steering_mode == "best" else all_layers,
        "coefs": args.coefs,
        "languages": args.languages, "scores": all_scores,
    }
    (steer_dir / "global_summary.json").write_text(json.dumps(global_summary, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"[{side}] Steering complete → {steer_dir}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Political Bias Detection & Steering")
    p.add_argument("--model", required=True, help="HuggingFace model name")
    p.add_argument("--input-csv", type=Path, required=True, help="Ideology paired CSV")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--languages", nargs="+", default=["en", "it"])
    p.add_argument("--control-method", default="rfm", choices=["rfm", "pca", "mean_difference"])
    p.add_argument("--coefs", nargs="+", type=float, default=[0.0, 0.8, 1.5, 2.5, 4.0])
    p.add_argument("--train-pairs", type=int, default=100)
    p.add_argument("--val-pairs", type=int, default=50)
    p.add_argument("--test-pairs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument("--rfm-iters", type=int, default=8)
    p.add_argument("--selection-metric", default="auc")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect device
    args.device = infer_device()
    compute_device = "cpu" if args.control_method == "rfm" and args.device != "cuda" else args.device
    patch_for_device(args.device, compute_device)

    # Logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")])
    logger = logging.getLogger("run")

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device} (compute: {compute_device})")
    logger.info(f"Control method: {args.control_method}")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Load dataset
    rows = load_rows(args.input_csv)
    paired_rows = build_paired_rows(rows, strict=True)
    logger.info(f"Loaded {len(paired_rows)} paired rows from {args.input_csv}")

    # Save config
    config = vars(args).copy()
    config["input_csv"] = str(config["input_csv"])
    config["output_dir"] = str(config["output_dir"])
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Run pipeline for both sides
    for side in ("left", "right"):
        run_detection(model, tokenizer, paired_rows, side, args.output_dir, args, logger)

    # Steering: best-layer and all-layers
    for mode in ("best", "all"):
        for side in ("left", "right"):
            run_steering(model, tokenizer, side, args.output_dir, args, logger, steering_mode=mode)

    logger.info("Done! Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
