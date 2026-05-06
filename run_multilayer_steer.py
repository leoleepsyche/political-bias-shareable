#!/usr/bin/env python3
# Multi-layer Steering Experiment for Political Bias
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.detect import (
    infer_compute_device,
    infer_device,
    load_detection_controller,
    load_model_and_tokenizer,
    patch_runtime_for_device,
)
from pipeline.steer import compute_compass_score, format_coef_label, COMPASS_DATA_FILES
from political_compass_multilingual import (
    build_compass_prompt,
    build_repair_prompt,
    load_compass_items,
    parse_choice,
)
from political_compass import CHOICE_TO_VALUE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_multilayer_compass_with_steering(
    controller,
    compass_items: list[dict],
    language: str,
    steering_layers: list[int],
    coef: float,
    max_new_tokens: int = 30,
    max_retries: int = 1,
) -> list[dict]:
    results = []
    effective_coef = coef
    if len(steering_layers) > 0:
        effective_coef = coef / (len(steering_layers) ** 0.5)

    active_layers = steering_layers if coef != 0.0 else []

    for item in compass_items:
        prompt_text = build_compass_prompt(item["statement"], language=language)
        formatted_prompt = controller.format_prompt(prompt_text, steer=True) 

        raw_response = controller.generate(
            formatted_prompt,
            layers_to_control=active_layers,
            control_coef=effective_coef,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if raw_response.startswith(formatted_prompt):
            answer_text = raw_response[len(formatted_prompt):].strip()
        else:
            answer_text = raw_response.strip()

        choice = parse_choice(answer_text, language=language)

        if choice is None and max_retries > 0:
            repair_text = build_repair_prompt(item["statement"], answer_text, language=language)
            repair_prompt = controller.format_prompt(repair_text, steer=True) 
            repair_response = controller.generate(
                repair_prompt,
                layers_to_control=active_layers,
                control_coef=effective_coef,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            if repair_response.startswith(repair_prompt):
                repair_answer = repair_response[len(repair_prompt):].strip()
            else:
                repair_answer = repair_response.strip()
            choice = parse_choice(repair_answer, language=language)

        results.append({
            "item_id": item["item_id"],
            "page": item["page"],
            "statement": item["statement"],
            "coef": coef,
            "effective_coef": effective_coef,
            "raw_answer": answer_text[:200],
            "parsed_choice": choice,
            "choice_value": CHOICE_TO_VALUE.get(choice, None),
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--detection-dir", type=Path, required=True)
    parser.add_argument("--output-base-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to steer. Omit or use --all-layers to steer all available layers.")
    parser.add_argument("--all-layers", action="store_true",
                        help="Steer all layers available in directions.pkl")
    parser.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = infer_device(args.device) 
    compute_device = infer_compute_device("rfm", device)
    patch_runtime_for_device(device, compute_device)

    logger.info("Loading model %s...", args.model)
    model_obj, tokenizer_obj = load_model_and_tokenizer(args.model, device)

    logger.info("Loading controller...")
    controller, _, _ = load_detection_controller(
        model=model_obj,
        tokenizer=tokenizer_obj,
        detection_dir=args.detection_dir,
        control_method="rfm",
        batch_size=1,
        n_components=1,
        rfm_iters=8,
    )

    available_layers = sorted(controller.directions.keys())

    if args.all_layers or args.layers is None:
        steering_layers = available_layers
    else:
        num_hidden_layers = model_obj.config.num_hidden_layers
        # Normalise to negative indices (directions.pkl uses negative keys)
        steering_layers = sorted({
            (idx - num_hidden_layers) if idx >= 0 else idx
            for idx in args.layers
        })
        missing = set(steering_layers) - set(available_layers)
        if missing:
            logger.error(
                "Layers %s not in directions.pkl (available: %s)",
                sorted(missing), available_layers,
            )
            sys.exit(1)

    # Use absolute values for directory naming (e.g. L2_L3_L4_L5)
    layer_spec = "_".join(f"L{abs(l)}" for l in steering_layers)
    output_dir = args.output_base_dir / f"multilayer_steer_{layer_spec}" / args.language
    output_dir.mkdir(parents=True, exist_ok=True)

    compass_path = COMPASS_DATA_FILES.get(args.language)
    if compass_path is None or not compass_path.exists():
        logger.error("Compass data not found")
        sys.exit(1)
        
    compass_items = load_compass_items(args.language, compass_path)
    logger.info("Starting Multi-Layer Steering on layers: %s", steering_layers)

    all_scores = {}
    for coef in args.coefs:
        logger.info("--- Steer Multi-Layer (layers=%s) | Coef=%s ---", steering_layers, coef)
        
        results = run_multilayer_compass_with_steering(
            controller=controller,
            compass_items=compass_items,
            language=args.language,
            steering_layers=steering_layers,
            coef=coef,
        )
        
        score = compute_compass_score(results)
        label = format_coef_label(coef)
        all_scores[label] = score
        
        detail_path = output_dir / f"compass_answers_{label}.csv"
        fieldnames = ["item_id", "page", "statement", "coef", "effective_coef", "raw_answer", "parsed_choice", "choice_value"]
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "model": args.model,
        "device": device,
        "compute_device": compute_device,
        "steering_layers": steering_layers,
        "coefs": args.coefs,
        "language": args.language,
        "scores": all_scores,
    }
    (output_dir / "multilayer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    logger.info("Multi-layer steering completed! Results saved to %s", output_dir)

if __name__ == "__main__":
    main()
