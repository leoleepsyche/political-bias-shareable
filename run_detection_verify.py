#!/usr/bin/env python3
"""Re-run detection only (no steering) with a different seed to verify AUC."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.detect import (
    infer_compute_device,
    infer_device,
    load_model_and_tokenizer,
    patch_runtime_for_device,
    run_detection,
    validate_control_method_for_device,
)
from step1_dataset import build_paired_rows, load_rows

WORKSPACE = Path(__file__).resolve().parent
INPUT_CSV = WORKSPACE / "data" / "ideoinst_clean" / "ideology_840_opinion.csv"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

SEED = 42  # Different from original seed=0
TRAIN_PAIRS = 600
VAL_PAIRS = 120
TEST_PAIRS = 120


def main() -> None:
    output_root = WORKSPACE / "outputs" / "detection_verify_seed42"
    output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("detection_verify")

    device = infer_device("auto")
    validate_control_method_for_device("rfm", device)
    compute_device = infer_compute_device("rfm", device)
    patch_runtime_for_device(device, rfm_device=compute_device)

    rows = load_rows(INPUT_CSV)
    paired_rows = build_paired_rows(rows, strict=True)
    logger.info("Loaded %d paired rows", len(paired_rows))

    for model_name in MODELS:
        slug = model_name.replace("/", "_").replace("-", "_")
        model_dir = output_root / slug

        logger.info("=" * 60)
        logger.info("Loading model %s on %s", model_name, device)
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        for side in ("left", "right"):
            detection_dir = model_dir / f"detection_{side}"
            if (detection_dir / "summary.json").exists():
                logger.info("[SKIP] %s %s already done", model_name, side)
                continue

            logger.info("Running detection: %s %s (seed=%d)", model_name, side, SEED)
            summary = run_detection(
                model=model,
                tokenizer=tokenizer,
                paired_rows=paired_rows,
                output_dir=detection_dir,
                model_name=model_name,
                input_csv=INPUT_CSV,
                device=device,
                compute_device=compute_device,
                control_method="rfm",
                prompt_format="opinion",
                positive_ideology=side,
                train_pairs=TRAIN_PAIRS,
                val_pairs=VAL_PAIRS,
                test_pairs=TEST_PAIRS,
                batch_size=1,
                n_components=1,
                rfm_iters=8,
                selection_metric="auc",
                seed=SEED,
                logger=logger,
            )
            logger.info(
                "RESULT: %s %s | best_layer=%s | test_auc=%s | test_acc=%s",
                model_name,
                side,
                summary.get("best_layer_on_val"),
                summary.get("best_layer_test_metrics", {}).get("auc"),
                summary.get("best_layer_test_metrics", {}).get("acc"),
            )

        # Free model memory before loading the next one
        del model, tokenizer
        import gc
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print final comparison
    logger.info("=" * 60)
    logger.info("FINAL RESULTS COMPARISON")
    logger.info("=" * 60)
    for model_name in MODELS:
        slug = model_name.replace("/", "_").replace("-", "_")
        model_dir = output_root / slug
        for side in ("left", "right"):
            summary_path = model_dir / f"detection_{side}" / "summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                test = s.get("best_layer_test_metrics", {})
                logger.info(
                    "%s %s: layer=%s test_auc=%.6f test_acc=%.2f%%",
                    model_name, side,
                    s.get("best_layer_on_val"),
                    test.get("auc", -1),
                    test.get("acc", -1),
                )


if __name__ == "__main__":
    main()
