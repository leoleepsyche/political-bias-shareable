#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pipeline.detect import (
    detection_complete,
    infer_compute_device,
    infer_device,
    load_detection_controller,
    load_model_and_tokenizer,
    patch_runtime_for_device,
    run_detection,
    validate_control_method_for_device,
)
from pipeline.report import build_report
from pipeline.steer import run_steering_stage, steering_complete
from step1_dataset import build_paired_rows, load_rows

WORKSPACE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end Political Compass steering experiment")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--languages", nargs="+", default=["en", "it"])
    parser.add_argument("--control-method", default="rfm")
    parser.add_argument("--prompt-format", default="opinion", choices=["party", "opinion"])
    parser.add_argument("--coefs", nargs="+", type=float, default=[0.0, 0.8, 1.5, 2.5, 4.0])
    parser.add_argument("--train-pairs", type=int, default=100)
    parser.add_argument("--val-pairs", type=int, default=50)
    parser.add_argument("--test-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--rfm-iters", type=int, default=8)
    parser.add_argument("--selection-metric", default="auc")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "logs" / "experiment.log")

    device = infer_device(args.device)
    validate_control_method_for_device(args.control_method, device)
    compute_device = infer_compute_device(args.control_method, device)
    patch_runtime_for_device(device, rfm_device=compute_device)

    config = {
        "model": args.model,
        "input_csv": str(args.input_csv.resolve()),
        "device": device,
        "requested_device": args.device,
        "compute_device": compute_device,
        "languages": args.languages,
        "control_method": args.control_method,
        "prompt_format": args.prompt_format,
        "coefs": args.coefs,
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "test_pairs": args.test_pairs,
        "batch_size": args.batch_size,
        "n_components": args.n_components,
        "rfm_iters": args.rfm_iters,
        "selection_metric": args.selection_metric,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    (output_dir / "config.json").write_text(json.dumps(to_builtin(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("Starting experiment in %s", output_dir)
    logger.info("Model=%s device=%s compute_device=%s", args.model, device, compute_device)

    model = None
    tokenizer = None
    paired_rows = None

    def ensure_model():
        nonlocal model, tokenizer
        if model is None or tokenizer is None:
            logger.info("Loading model %s on %s", args.model, device)
            model, tokenizer = load_model_and_tokenizer(args.model, device)
        return model, tokenizer

    def ensure_paired_rows():
        nonlocal paired_rows
        if paired_rows is None:
            logger.info("Loading ideology pairs from %s", args.input_csv)
            rows = load_rows(args.input_csv)
            paired_rows = build_paired_rows(rows, strict=True)
            logger.info("Loaded %d paired rows", len(paired_rows))
        return paired_rows

    for side in ("left", "right"):
        detection_dir = output_dir / f"detection_{side}"
        if detection_complete(detection_dir):
            logger.info("[SKIP] detection_%s already exists, skipping", side)
            continue
        model_obj, tokenizer_obj = ensure_model()
        paired = ensure_paired_rows()
        run_detection(
            model=model_obj,
            tokenizer=tokenizer_obj,
            paired_rows=paired,
            output_dir=detection_dir,
            model_name=args.model,
            input_csv=args.input_csv.resolve(),
            device=device,
            compute_device=compute_device,
            control_method=args.control_method,
            prompt_format=args.prompt_format,
            positive_ideology=side,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            test_pairs=args.test_pairs,
            batch_size=args.batch_size,
            n_components=args.n_components,
            rfm_iters=args.rfm_iters,
            selection_metric=args.selection_metric,
            seed=args.seed,
            logger=logger,
        )

    for side in ("left", "right"):
        steering_dir = output_dir / f"steering_{side}"
        if steering_complete(steering_dir):
            logger.info("[SKIP] steering_%s already exists, skipping", side)
            continue
        model_obj, tokenizer_obj = ensure_model()
        detection_dir = output_dir / f"detection_{side}"
        controller, best_layer, _summary = load_detection_controller(
            model=model_obj,
            tokenizer=tokenizer_obj,
            detection_dir=detection_dir,
            control_method=args.control_method,
            batch_size=args.batch_size,
            n_components=args.n_components,
            rfm_iters=args.rfm_iters,
        )
        run_steering_stage(
            controller=controller,
            best_layer=best_layer,
            positive_ideology=side,
            model_name=args.model,
            device=device,
            compute_device=compute_device,
            control_method=args.control_method,
            prompt_format=args.prompt_format,
            languages=args.languages,
            coefs=args.coefs,
            max_new_tokens=args.max_new_tokens,
            output_dir=steering_dir,
            logger=logger,
        )

    report_path = build_report(output_dir, logger=logger)
    logger.info("Experiment complete. Report: %s", report_path)


if __name__ == "__main__":
    main()
