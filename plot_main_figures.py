#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import (
    ROOT,
    COEF_ORDER,
    LANG_ORDER,
    MODEL_COLORS,
    MODEL_ORDER,
    SIDE_ORDER,
    savefig as _savefig,
    setup_style,
)

BATCH_ROOT = ROOT / "outputs" / "multimodel_840_batch_20260424_103114_pty"
OUTPUT_DIR = ROOT / "outputs" / "figures_main"

MODEL_RUNS = {
    "Qwen": BATCH_ROOT / "Qwen_Qwen2_5_7B",
    "Mistral": BATCH_ROOT / "mistralai_Mistral_7B_Instruct_v0_2",
    "Llama-3": BATCH_ROOT / "meta_llama_Meta_Llama_3_8B_Instruct",
}

SIDE_COLORS = {"left": "#0072B2", "right": "#E69F00"}
LANG_MARKERS = {"en": "o", "it": "s", "fr": "^", "es": "D", "de": "P"}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def savefig(fig: plt.Figure, stem: str) -> None:
    _savefig(fig, stem, OUTPUT_DIR, dpi=240)


def load_shift_data() -> pd.DataFrame:
    frames = []
    for model, run_dir in MODEL_RUNS.items():
        path = require_file(run_dir / "comparison" / "branch_shift_summary.csv")
        df = pd.read_csv(path)
        df["model"] = model
        df["source_csv"] = str(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["coef"] = data["coef"].astype(float)
    data["shift"] = data["shift"].astype(float)
    return data


def load_detection_data() -> pd.DataFrame:
    rows = []
    for model, run_dir in MODEL_RUNS.items():
        path = require_file(run_dir / "comparison" / "detection_summary.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "model": model,
                        "ideology": row["ideology"],
                        "best_layer": int(row["best_layer"]),
                        "best_test_auc": float(row["best_test_auc"]),
                        "aggregation_test_auc": float(row["aggregation_test_auc"]),
                        "best_test_acc": float(row["best_test_acc"]),
                        "source_csv": str(path),
                    }
                )
    return pd.DataFrame(rows)


def write_plot_data(shift_df: pd.DataFrame, detection_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shift_df.to_csv(OUTPUT_DIR / "main_shift_data.csv", index=False)
    detection_df.to_csv(OUTPUT_DIR / "main_detection_data.csv", index=False)


def plot_dose_response_english(shift_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = shift_df[(shift_df["model"] == model) & (shift_df["language"] == "en")]
        for side in SIDE_ORDER:
            side_df = subset[subset["ideology"] == side].sort_values("coef")
            ax.plot(
                side_df["coef"],
                side_df["shift"],
                marker="o",
                linewidth=2.2,
                color=SIDE_COLORS[side],
                label=f"{side} steering",
            )
        ax.set_title(model)
        ax.set_xlabel("Steering coefficient alpha")
        ax.set_xticks(COEF_ORDER)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Compass shift from alpha=0")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Steering Dose-Response on English Political Compass Items", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.22)
    savefig(fig, "fig2_dose_response_english")


def plot_compass_trajectories_english(shift_df: pd.DataFrame) -> None:
    en_df = shift_df[shift_df["language"] == "en"]
    x_min = min(-10.0, float(en_df["x"].min()) - 0.8)
    x_max = max(10.0, float(en_df["x"].max()) + 0.8)
    y_min = min(-10.0, float(en_df["y"].min()) - 0.8)
    y_max = max(10.0, float(en_df["y"].max()) + 0.8)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = en_df[en_df["model"] == model]
        for side in SIDE_ORDER:
            side_df = subset[subset["ideology"] == side].sort_values("coef")
            ax.plot(
                side_df["x"],
                side_df["y"],
                marker="o",
                linewidth=2.0,
                color=SIDE_COLORS[side],
                label=f"{side} steering",
            )
            for _, row in side_df.iterrows():
                ax.annotate(
                    f"{row['coef']:g}",
                    (row["x"], row["y"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=SIDE_COLORS[side],
                )
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_title(model)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Economic coordinate")
    axes[0].set_ylabel("Social coordinate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Compass Plane Trajectories Under Steering (English)", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.80, wspace=0.18)
    savefig(fig, "fig3_compass_trajectories_english")


def plot_crosslingual_heatmap(shift_df: pd.DataFrame) -> None:
    vmax = float(shift_df["shift"].max())
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.0), sharex=True, sharey=True)
    image = None
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, side in enumerate(SIDE_ORDER):
            ax = axes[row_idx, col_idx]
            subset = shift_df[(shift_df["model"] == model) & (shift_df["ideology"] == side)]
            pivot = (
                subset.pivot_table(index="language", columns="coef", values="shift", aggfunc="mean")
                .reindex(index=LANG_ORDER, columns=COEF_ORDER)
            )
            masked_values = np.ma.masked_invalid(pivot.values)
            image = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{model}: {side} steering")
            ax.set_xticks(range(len(COEF_ORDER)))
            ax.set_xticklabels([f"{coef:g}" for coef in COEF_ORDER])
            ax.set_yticks(range(len(LANG_ORDER)))
            ax.set_yticklabels(LANG_ORDER)
            for i in range(len(LANG_ORDER)):
                for j in range(len(COEF_ORDER)):
                    value = pivot.values[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > vmax * 0.45 else "#222222")
                    else:
                        ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="#555555")
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Steering coefficient alpha")
            if col_idx == 0:
                ax.set_ylabel("Language")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("Compass shift from alpha=0")
    fig.suptitle("Cross-Lingual Steering Transfer", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.92, right=0.86, hspace=0.36, wspace=0.14)
    savefig(fig, "fig4_crosslingual_transfer_heatmap")


def plot_detection_vs_steering(shift_df: pd.DataFrame, detection_df: pd.DataFrame) -> None:
    auc = (
        detection_df.groupby("model", as_index=False)["best_test_auc"]
        .mean()
        .rename(columns={"best_test_auc": "mean_best_auc"})
    )
    steer = (
        shift_df[shift_df["coef"] == 4.0]
        .groupby(["model", "language"], as_index=False)["shift"]
        .max()
        .rename(columns={"shift": "max_shift_at_alpha4"})
    )
    plot_df = steer.merge(auc, on="model", how="left")
    plot_df.to_csv(OUTPUT_DIR / "detection_vs_steering_data.csv", index=False)
    plot_df_valid = plot_df.dropna(subset=["mean_best_auc", "max_shift_at_alpha4"])

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    for model in MODEL_ORDER:
        model_df = plot_df_valid[plot_df_valid["model"] == model]
        for _, row in model_df.iterrows():
            ax.scatter(
                row["mean_best_auc"],
                row["max_shift_at_alpha4"],
                s=90,
                color=MODEL_COLORS[model],
                marker=LANG_MARKERS[row["language"]],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
            )
        if not model_df.empty:
            center = model_df[["mean_best_auc", "max_shift_at_alpha4"]].mean()
            ax.text(
                center["mean_best_auc"] + 0.0007,
                center["max_shift_at_alpha4"],
                model,
                fontsize=10,
                color=MODEL_COLORS[model],
                fontweight="bold",
            )

    marker_handles = [
        plt.Line2D([0], [0], marker=LANG_MARKERS[lang], color="none", markerfacecolor="#555555", markersize=8, label=lang)
        for lang in LANG_ORDER
    ]
    ax.legend(handles=marker_handles, title="Language", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel("Mean best-layer detection AUC")
    ax.set_ylabel("Max steering shift at alpha=4")
    ax.set_title("Detection Strength vs Behavioral Steerability", fontweight="bold")
    ax.grid(alpha=0.25)
    x_min = max(0.5, float(plot_df_valid["mean_best_auc"].min()) - 0.01)
    ax.set_xlim(x_min, 1.005)
    savefig(fig, "fig5_detection_vs_steerability")


def main() -> None:
    setup_style()
    shift_df = load_shift_data()
    detection_df = load_detection_data()
    write_plot_data(shift_df, detection_df)
    plot_dose_response_english(shift_df)
    plot_compass_trajectories_english(shift_df)
    plot_crosslingual_heatmap(shift_df)
    plot_detection_vs_steering(shift_df, detection_df)
    print(f"Wrote main figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
