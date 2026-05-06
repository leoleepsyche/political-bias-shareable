#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import ROOT, SIDE_COLORS, savefig as _savefig

OUTPUT_DIR = ROOT / "outputs" / "figures_instruct_detection"

DETECTION_SUMMARIES = {
    "Qwen2.5-7B-Instruct": ROOT
    / "outputs"
    / "ideology_840_mps"
    / "run_840_mps_promptfixed_20260423_141925"
    / "comparison"
    / "detection_summary.csv",
    "Mistral-7B-Instruct-v0.2": ROOT
    / "outputs"
    / "multimodel_840_batch_20260424_103114_pty"
    / "mistralai_Mistral_7B_Instruct_v0_2"
    / "comparison"
    / "detection_summary.csv",
    "Llama-3-8B-Instruct": ROOT
    / "outputs"
    / "multimodel_840_batch_20260424_103114_pty"
    / "meta_llama_Meta_Llama_3_8B_Instruct"
    / "comparison"
    / "detection_summary.csv",
}

MODEL_LABELS = {
    "Qwen2.5-7B-Instruct": "Qwen",
    "Mistral-7B-Instruct-v0.2": "Mistral",
    "Llama-3-8B-Instruct": "Llama-3",
}

def load_detection_metrics() -> pd.DataFrame:
    rows = []
    for model, path in DETECTION_SUMMARIES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "side": row["ideology"],
                        "best_layer": int(row["best_layer"]),
                        "best_test_auc": float(row["best_test_auc"]),
                        "aggregation_test_auc": float(row["aggregation_test_auc"]),
                        "best_test_acc": float(row["best_test_acc"]),
                        "source_csv": str(path),
                    }
                )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, stem: str) -> None:
    _savefig(fig, stem, OUTPUT_DIR, dpi=240)


def plot_detection_auc(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    model_order = ["Qwen", "Mistral", "Llama-3"]
    side_order = ["left", "right"]
    bar_width = 0.34
    x_positions = list(range(len(model_order)))

    for idx, side in enumerate(side_order):
        offset = (idx - 0.5) * bar_width
        values = [
            float(
                df.loc[
                    (df["model_label"] == model_label) & (df["side"] == side),
                    "best_test_auc",
                ].iloc[0]
            )
            for model_label in model_order
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width,
            label=side,
            color=SIDE_COLORS[side],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)

    ax.set_ylim(0.82, 1.015)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_order)
    ax.set_xlabel("")
    ax.set_ylabel("Best-layer test AUC")
    ax.set_title("Best-layer test AUC by model and detector", pad=12, fontsize=15, weight="bold")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.legend(title="Detector", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    fig.subplots_adjust(top=0.84, bottom=0.14, right=0.82)
    savefig(fig, "detection_best_layer_auc")


def plot_detection_summary_table(df: pd.DataFrame) -> None:
    table_df = df.copy()
    table_df["Detector"] = table_df["side"].str.title()
    table_df["Best AUC"] = table_df["best_test_auc"].map(lambda x: f"{x:.4f}")
    table_df["Agg. AUC"] = table_df["aggregation_test_auc"].map(lambda x: f"{x:.4f}")
    table_df["Best ACC"] = table_df["best_test_acc"].map(lambda x: f"{x:.2f}%")
    display = table_df[["model", "Detector", "best_layer", "Best AUC", "Agg. AUC", "Best ACC"]].rename(
        columns={"model": "Model", "best_layer": "Best layer"}
    )

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")
    ax.set_title("Detection metric summary", fontsize=14, weight="bold", pad=12)
    savefig(fig, "detection_metric_table")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_detection_metrics()
    df.to_csv(OUTPUT_DIR / "detection_metrics_instruct.csv", index=False)
    plot_detection_auc(df)
    plot_detection_summary_table(df)
    print(f"Wrote detection figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
