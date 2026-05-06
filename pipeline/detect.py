from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repo_paths import resolve_repo_path
from step1_dataset import allocate_topic_counts, ordered_topics

WORKSPACE = Path(__file__).resolve().parent.parent
OFFICIAL_REPO = resolve_repo_path(WORKSPACE, "neural_controllers_official")

if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from neural_controllers import NeuralController  # noqa: E402
import direction_utils as official_direction_utils  # noqa: E402

PROMPT_FORMATS = {
    "party": (
        "Which political party supports the following position?\n"
        "Position: {text}"
    ),
    "opinion": (
        "What is your opinion on the following statement?\n"
        "Statement: {text}\n"
        "Opinion:"
    ),
}

DIRECTIONS_FILENAME = "directions.pkl"
DETECTOR_COEFS_FILENAME = "detector_coefs.pkl"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_control_method_for_device(control_method: str, device: str) -> None:
    if device != "cuda" and control_method == "linear":
        raise ValueError(
            "control_method='linear' is not supported on local MPS/CPU in this repo. "
            "Use control_method='rfm', 'pca', or 'mean_difference' instead."
        )


def infer_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_compute_device(control_method: str, device: str) -> str:
    return "cpu" if control_method == "rfm" and device != "cuda" else device


_PATCHED_RUNTIME: tuple[str, str] | None = None


def patch_runtime_for_device(default_device: str, rfm_device: str | None = None) -> None:
    """Make the official CUDA-first code path usable on MPS/CPU without upstream edits."""
    global _PATCHED_RUNTIME
    rfm_target = rfm_device or default_device
    if _PATCHED_RUNTIME == (default_device, rfm_target):
        return
    if default_device == "cuda":
        _PATCHED_RUNTIME = (default_device, rfm_target)
        return

    target = torch.device(default_device)
    original_aggregate_layers = official_direction_utils.aggregate_layers
    original_get_hidden_states = official_direction_utils.get_hidden_states
    original_linear_solve = official_direction_utils.linear_solve
    original_compute_prediction_metrics = official_direction_utils.compute_prediction_metrics

    def _clean_tensor(values, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
        if isinstance(values, torch.Tensor) and torch.is_floating_point(values):
            return torch.nan_to_num(values, nan=nan, posinf=posinf, neginf=neginf)
        return values

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        del device, memory_format
        return self.to(device=target, non_blocking=non_blocking)

    def _module_cuda(self, device=None):
        del device
        return self.to(device=target)

    def _project_onto_direction(tensors, direction, device="cuda"):
        del device
        assert len(tensors.shape) == 2
        assert tensors.shape[1] == direction.shape[0]
        return tensors.to(device=target) @ direction.to(device=target, dtype=tensors.dtype)

    def _project_hidden_states(hidden_states, directions, n_components):
        assert hidden_states.keys() == directions.keys()
        projections = {}
        for layer in hidden_states.keys():
            layer_hidden = hidden_states[layer].to(device=target)
            if not torch.isfinite(layer_hidden).all():
                layer_hidden = torch.nan_to_num(layer_hidden, nan=0.0, posinf=0.0, neginf=0.0)
            vecs = directions[layer][:n_components].T
            if hasattr(vecs, "to"):
                vecs = vecs.to(device=target, dtype=layer_hidden.dtype)
            projections[layer] = layer_hidden @ vecs
        return projections

    def _get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=-1, all_positions=False):
        hidden_states = original_get_hidden_states(
            prompts,
            model,
            tokenizer,
            hidden_layers,
            forward_batch_size,
            rep_token=rep_token,
            all_positions=all_positions,
        )
        cleaned_hidden_states = {}
        for layer, states in hidden_states.items():
            if torch.isfinite(states).all():
                cleaned_hidden_states[layer] = states
            else:
                cleaned_hidden_states[layer] = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned_hidden_states

    def _linear_solve(X, y, use_bias=True, reg=0):
        X = _clean_tensor(X.float())
        y = _clean_tensor(y.float())
        beta, bias = original_linear_solve(X, y, use_bias=use_bias, reg=reg)
        beta = _clean_tensor(beta)
        if isinstance(bias, torch.Tensor):
            bias = _clean_tensor(bias)
        elif isinstance(bias, float) and not math.isfinite(bias):
            bias = 0.0
        return beta, bias

    def _compute_prediction_metrics(preds, labels, classification_threshold=0.5):
        preds = _clean_tensor(preds, nan=0.5, posinf=1.0, neginf=0.0)
        labels = _clean_tensor(labels, nan=0.0, posinf=1.0, neginf=0.0)
        return original_compute_prediction_metrics(
            preds,
            labels,
            classification_threshold=classification_threshold,
        )

    def _aggregate_projections_on_coefs(projections, detector_coef):
        agg_projections = []
        for layer in projections.keys():
            X = projections[layer].to(device=target)
            agg_projections.append(X.squeeze(0))

        agg_projections = torch.concat(agg_projections, dim=1).squeeze()
        agg_beta = detector_coef[0]
        agg_bias = detector_coef[1]
        if hasattr(agg_beta, "to"):
            agg_beta = agg_beta.to(device=target, dtype=agg_projections.dtype)
        if hasattr(agg_bias, "to"):
            agg_bias = agg_bias.to(device=target, dtype=agg_projections.dtype)
        agg_preds = agg_projections @ agg_beta + agg_bias
        return agg_preds

    def _train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams, search_space=None, tuning_metric="auc"):
        if search_space is None:
            search_space = {
                "regs": [1e-3],
                "bws": [1, 10, 100],
                "center_grads": [True, False],
            }

        train_X_local = _clean_tensor(train_X.to(rfm_target).float())
        train_y_local = _clean_tensor(train_y.to(rfm_target).float())
        val_X_local = _clean_tensor(val_X.to(rfm_target).float())
        val_y_local = _clean_tensor(val_y.to(rfm_target).float())

        best_model = None
        maximize_metric = tuning_metric in ["f1", "auc", "acc", "top_agop_vectors_ols_auc"]
        best_score = float("-inf") if maximize_metric else float("inf")
        best_reg = None
        best_bw = None
        best_center_grads = None

        for reg in search_space["regs"]:
            for bw in search_space["bws"]:
                for center_grads in search_space["center_grads"]:
                    try:
                        rfm_params = {
                            "model": {
                                "kernel": "l2_high_dim",
                                "bandwidth": bw,
                                "tuning_metric": tuning_metric,
                            },
                            "fit": {
                                "reg": reg,
                                "iters": hyperparams["rfm_iters"],
                                "center_grads": center_grads,
                                "early_stop_rfm": True,
                                "get_agop_best_model": True,
                                "top_k": hyperparams["n_components"],
                            },
                        }
                        model = official_direction_utils.RFM(**rfm_params["model"], device=rfm_target)
                        model.fit((train_X_local, train_y_local), (val_X_local, val_y_local), **rfm_params["fit"])

                        if tuning_metric == "top_agop_vectors_ols_auc":
                            top_k = hyperparams["n_components"]
                            targets = val_y_local
                            _, U = torch.lobpcg(model.agop_best_model, k=top_k)
                            top_eigenvectors = U[:, :top_k]
                            projections = val_X_local @ top_eigenvectors
                            projections = projections.reshape(-1, top_k)
                            xtx = projections.T @ projections
                            xty = projections.T @ targets
                            betas = torch.linalg.pinv(xtx) @ xty
                            preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
                            preds = _clean_tensor(preds, nan=0.5, posinf=1.0, neginf=0.0)
                            val_score = official_direction_utils.roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
                        else:
                            pred_proba = model.predict(val_X_local)
                            pred_proba = _clean_tensor(pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                            val_score = official_direction_utils.compute_prediction_metrics(pred_proba, val_y_local)[tuning_metric]

                        if (maximize_metric and val_score > best_score) or (not maximize_metric and val_score < best_score):
                            best_score = val_score
                            best_reg = reg
                            best_bw = bw
                            best_center_grads = center_grads
                            best_model = official_direction_utils.deepcopy(model)
                    except Exception as exc:  # pragma: no cover
                        logger = logging.getLogger(__name__)
                        logger.exception("Error fitting RFM on device=%s: %s", rfm_target, exc)
                        continue

        logging.getLogger(__name__).info(
            "Best RFM %s=%s reg=%s bw=%s center_grads=%s device=%s",
            tuning_metric,
            best_score,
            best_reg,
            best_bw,
            best_center_grads,
            rfm_target,
        )
        return best_model

    def _aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model="linear", tuning_metric="auc"):
        if agg_model != "rfm":
            return original_aggregate_layers(
                layer_outputs,
                train_y,
                val_y,
                test_y,
                agg_model=agg_model,
                tuning_metric=tuning_metric,
            )

        train_X = _clean_tensor(torch.concat(layer_outputs["train"], dim=1).to(rfm_target).float())
        val_X = _clean_tensor(torch.concat(layer_outputs["val"], dim=1).to(rfm_target).float())
        test_X = _clean_tensor(torch.concat(layer_outputs["test"], dim=1).to(rfm_target).float())
        train_y = _clean_tensor(train_y.to(rfm_target).float())
        val_y = _clean_tensor(val_y.to(rfm_target).float())
        test_y = _clean_tensor(test_y.to(rfm_target).float())

        bw_search_space = [10]
        reg_search_space = [1e-4, 1e-3, 1e-2]
        kernel_search_space = ["l2_high_dim"]
        maximize_metric = tuning_metric in ["f1", "auc", "acc"]

        best_rfm_params = None
        best_rfm_score = float("-inf") if maximize_metric else float("inf")
        for bw in bw_search_space:
            for reg in reg_search_space:
                for kernel in kernel_search_space:
                    rfm_params = {
                        "model": {
                            "kernel": kernel,
                            "bandwidth": bw,
                        },
                        "fit": {
                            "reg": reg,
                            "iters": 10,
                        },
                    }
                    model = official_direction_utils.xRFM(rfm_params, device=rfm_target, tuning_metric=tuning_metric)
                    model.fit(train_X, train_y, val_X, val_y)
                    val_preds = _clean_tensor(model.predict(val_X), nan=0.5, posinf=1.0, neginf=0.0)
                    metrics = official_direction_utils.compute_prediction_metrics(val_preds, val_y)

                    if (maximize_metric and metrics[tuning_metric] > best_rfm_score) or (
                        not maximize_metric and metrics[tuning_metric] < best_rfm_score
                    ):
                        best_rfm_score = metrics[tuning_metric]
                        best_rfm_params = rfm_params

        model = official_direction_utils.xRFM(best_rfm_params, device=rfm_target, tuning_metric=tuning_metric)
        model.fit(train_X, train_y, val_X, val_y)
        test_preds = _clean_tensor(model.predict(test_X), nan=0.5, posinf=1.0, neginf=0.0)
        metrics = official_direction_utils.compute_prediction_metrics(test_preds, test_y)
        return metrics, None, None, test_preds

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[assignment]
    torch.nn.Module.cuda = _module_cuda  # type: ignore[assignment]
    official_direction_utils.project_onto_direction = _project_onto_direction
    official_direction_utils.project_hidden_states = _project_hidden_states
    official_direction_utils.get_hidden_states = _get_hidden_states
    official_direction_utils.aggregate_projections_on_coefs = _aggregate_projections_on_coefs
    official_direction_utils.linear_solve = _linear_solve
    official_direction_utils.compute_prediction_metrics = _compute_prediction_metrics
    official_direction_utils.train_rfm_probe_on_concept = _train_rfm_probe_on_concept
    official_direction_utils.aggregate_layers = _aggregate_layers
    _PATCHED_RUNTIME = (default_device, rfm_target)


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    model.eval()
    return model, tokenizer


def split_pairs_by_topic(
    paired_rows: list[tuple[dict, dict]],
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    seed: int,
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    paired_left_rows = [left_row for left_row, _ in paired_rows]
    train_counts = allocate_topic_counts(paired_left_rows, train_pairs)
    val_counts = allocate_topic_counts(paired_left_rows, val_pairs)
    test_counts = allocate_topic_counts(paired_left_rows, test_pairs)

    grouped_pairs: dict[str, list[tuple[dict, dict]]] = {}
    for left_row, right_row in paired_rows:
        grouped_pairs.setdefault(left_row["topic"], []).append((left_row, right_row))

    train_split: list[tuple[dict, dict]] = []
    val_split: list[tuple[dict, dict]] = []
    test_split: list[tuple[dict, dict]] = []

    rng = random.Random(seed)
    for topic in ordered_topics(paired_left_rows):
        topic_pairs = list(grouped_pairs[topic])
        rng.shuffle(topic_pairs)

        train_n = train_counts.get(topic, 0)
        val_n = val_counts.get(topic, 0)
        test_n = test_counts.get(topic, 0)
        total_needed = train_n + val_n + test_n
        if len(topic_pairs) < total_needed:
            raise ValueError(
                f"Topic {topic} has {len(topic_pairs)} matched pairs, "
                f"cannot allocate train={train_n}, val={val_n}, test={test_n}."
            )

        train_split.extend(topic_pairs[:train_n])
        val_split.extend(topic_pairs[train_n : train_n + val_n])
        test_split.extend(topic_pairs[train_n + val_n : total_needed])

    return train_split, val_split, test_split


def build_detection_split(
    controller: NeuralController,
    paired_rows: list[tuple[dict, dict]],
    positive_ideology: str,
    prompt_format: str = "opinion",
) -> tuple[list[str], list[float]]:
    if positive_ideology not in {"left", "right"}:
        raise ValueError("positive_ideology must be 'left' or 'right'.")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unknown prompt_format {prompt_format!r}; expected one of {sorted(PROMPT_FORMATS)}")

    template = PROMPT_FORMATS[prompt_format]
    inputs: list[str] = []
    labels: list[float] = []

    for left_row, right_row in paired_rows:
        ideology_to_row = {"left": left_row, "right": right_row}
        positive_row = ideology_to_row[positive_ideology]
        negative_row = ideology_to_row["right" if positive_ideology == "left" else "left"]

        for row, label in ((positive_row, 1.0), (negative_row, 0.0)):
            prompt = template.format(text=row["response_text"])
            inputs.append(controller.format_prompt(prompt))
            labels.append(label)

    return inputs, labels


def summarize_metrics(val_metrics: dict, test_metrics: dict, selection_metric: str) -> dict:
    best_layer = max(
        (layer for layer in val_metrics.keys() if isinstance(layer, int)),
        key=lambda layer: val_metrics[layer][selection_metric],
    )

    return {
        "selection_metric": selection_metric,
        "best_layer_on_val": int(best_layer),
        "best_layer_val_metrics": val_metrics[best_layer],
        "best_layer_test_metrics": test_metrics[best_layer],
        "official_best_layer_test_metrics": test_metrics.get("best_layer"),
        "aggregation_test_metrics": test_metrics.get("aggregation"),
    }


def to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(key): to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _to_cpu_builtin(value: Any):
    if isinstance(value, dict):
        return {key: _to_cpu_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_cpu_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_builtin(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def write_metrics_csv(output_path: Path, split_name: str, metrics: dict) -> None:
    rows: list[dict] = []
    for layer_key, layer_metrics in metrics.items():
        layer_name = str(layer_key)
        for metric_name, metric_value in layer_metrics.items():
            rows.append(
                {
                    "split": split_name,
                    "layer": layer_name,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "layer", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def detection_complete(output_dir: Path) -> bool:
    return (
        (output_dir / "summary.json").exists()
        and (output_dir / DIRECTIONS_FILENAME).exists()
        and (output_dir / DETECTOR_COEFS_FILENAME).exists()
    )


def run_detection(
    *,
    model,
    tokenizer,
    paired_rows: list[tuple[dict, dict]],
    output_dir: Path,
    model_name: str,
    input_csv: Path,
    device: str,
    compute_device: str,
    control_method: str,
    prompt_format: str,
    positive_ideology: str,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    batch_size: int,
    n_components: int,
    rfm_iters: int,
    selection_metric: str,
    seed: int,
    logger: logging.Logger,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split, val_split, test_split = split_pairs_by_topic(
        paired_rows,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        seed=seed,
    )

    controller = NeuralController(
        model,
        tokenizer,
        rfm_iters=rfm_iters,
        batch_size=batch_size,
        n_components=n_components,
        control_method=control_method,
    )

    train_inputs, train_labels = build_detection_split(
        controller, train_split, positive_ideology=positive_ideology, prompt_format=prompt_format
    )
    val_inputs, val_labels = build_detection_split(
        controller, val_split, positive_ideology=positive_ideology, prompt_format=prompt_format
    )
    test_inputs, test_labels = build_detection_split(
        controller, test_split, positive_ideology=positive_ideology, prompt_format=prompt_format
    )

    hidden_layers = controller.hidden_layers
    logger.info(
        "[%s] Extracting hidden states for %d train, %d val, %d test inputs",
        positive_ideology,
        len(train_inputs),
        len(val_inputs),
        len(test_inputs),
    )
    train_hidden = official_direction_utils.get_hidden_states(train_inputs, model, tokenizer, hidden_layers, batch_size)
    val_hidden = official_direction_utils.get_hidden_states(val_inputs, model, tokenizer, hidden_layers, batch_size)
    test_hidden = official_direction_utils.get_hidden_states(test_inputs, model, tokenizer, hidden_layers, batch_size)

    logger.info("[%s] Computing concept directions", positive_ideology)
    controller.compute_directions(train_hidden, train_labels, val_hidden, val_labels, device=compute_device)
    agg_model = control_method if control_method in {"rfm", "linear", "logistic"} else "linear"
    val_metrics, test_metrics, detector_coefs, _ = controller.evaluate_directions(
        train_hidden,
        train_labels,
        val_hidden,
        val_labels,
        test_hidden,
        test_labels,
        agg_model=agg_model,
        selection_metric=selection_metric,
    )

    summary = to_builtin(summarize_metrics(val_metrics, test_metrics, selection_metric))
    metadata = to_builtin(
        {
            "model": model_name,
            "device": device,
            "compute_device": compute_device,
            "control_method": control_method,
            "prompt_format": prompt_format,
            "positive_ideology": positive_ideology,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "train_inputs": len(train_inputs),
            "val_inputs": len(val_inputs),
            "test_inputs": len(test_inputs),
            "selection_metric": selection_metric,
            "input_csv": str(input_csv),
            "batch_size": batch_size,
            "n_components": n_components,
            "rfm_iters": rfm_iters,
            "seed": seed,
        }
    )

    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "detector_layers.json").write_text(
        json.dumps(sorted(int(layer) for layer in detector_coefs.keys() if isinstance(layer, int)), indent=2) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "val_metrics_long.csv", "val", val_metrics)
    write_metrics_csv(output_dir / "test_metrics_long.csv", "test", test_metrics)

    with (output_dir / DIRECTIONS_FILENAME).open("wb") as handle:
        pickle.dump(_to_cpu_builtin(controller.directions), handle)
    with (output_dir / DETECTOR_COEFS_FILENAME).open("wb") as handle:
        pickle.dump(_to_cpu_builtin(controller.detector_coefs), handle)

    logger.info("[%s] Saved detection artifacts to %s", positive_ideology, output_dir)
    return summary


def load_detection_controller(
    *,
    model,
    tokenizer,
    detection_dir: Path,
    control_method: str,
    batch_size: int,
    n_components: int,
    rfm_iters: int,
) -> tuple[NeuralController, int, dict]:
    controller = NeuralController(
        model,
        tokenizer,
        rfm_iters=rfm_iters,
        batch_size=batch_size,
        n_components=n_components,
        control_method=control_method,
    )
    with (detection_dir / DIRECTIONS_FILENAME).open("rb") as handle:
        controller.directions = pickle.load(handle)
    with (detection_dir / DETECTOR_COEFS_FILENAME).open("rb") as handle:
        controller.detector_coefs = pickle.load(handle)
    controller.hidden_layers = list(controller.directions.keys())

    summary = load_json(detection_dir / "summary.json")
    best_layer = int(summary["best_layer_on_val"])
    return controller, best_layer, summary
