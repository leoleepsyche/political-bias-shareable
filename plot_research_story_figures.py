#!/usr/bin/env python3
"""Research-story figures: detection AUC, steering displacement, item-level analysis.

This script produces the full set of figures for the research narrative,
combining data from the Qwen Instruct promptfixed run and the multimodel batch.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

from plot_utils import (
    ROOT,
    COEF_ORDER,
    LANG_LABELS,
    LANG_ORDER,
    MODEL_COLORS,
    MODEL_ORDER,
    SIDE_COLORS,
    SIDE_MARKERS,
    SIDE_ORDER,
    STANCE_COLORS,
    STANCE_ORDER,
    coef_label,
    read_answer_file,
    savefig as _savefig,
    setup_style,
)

MAIN_DIR = ROOT / "outputs" / "figures_main"
OUTPUT_DIR = ROOT / "outputs" / "figures_research_story"
BATCH_ROOT = ROOT / "outputs" / "multimodel_840_batch_20260424_103114_pty"
QWEN_INSTRUCT_840_ROOT = (
    ROOT
    / "outputs"
    / "ideology_840_mps"
    / "run_840_mps_promptfixed_20260423_141925"
)
QWEN_INSTRUCT_840_COMPARISON = QWEN_INSTRUCT_840_ROOT / "comparison"
QWEN_INSTRUCT_840_LOGS = {
    "left": QWEN_INSTRUCT_840_ROOT / "logs" / "steer_left.log",
    "right": QWEN_INSTRUCT_840_ROOT / "logs" / "steer_right.log",
}

SHIFT_CSV = MAIN_DIR / "main_shift_data.csv"
DETECTION_CSV = MAIN_DIR / "main_detection_data.csv"

COEF_COLORS = {
    0.0: "#4D4D4D",
    0.8: "#56B4E9",
    1.5: "#009E73",
    2.5: "#E69F00",
    4.0: "#CC79A7",
}

MODEL_STANCE_ROOTS = {
    "Qwen": QWEN_INSTRUCT_840_ROOT,
    "Mistral": BATCH_ROOT / "mistralai_Mistral_7B_Instruct_v0_2",
    "Llama-3": BATCH_ROOT / "meta_llama_Meta_Llama_3_8B_Instruct",
}
MODEL_COMPARISON_ROOTS = {
    "Qwen": QWEN_INSTRUCT_840_ROOT,
    "Mistral": BATCH_ROOT / "mistralai_Mistral_7B_Instruct_v0_2",
    "Llama-3": BATCH_ROOT / "meta_llama_Meta_Llama_3_8B_Instruct",
}
SOURCE_NOTE = "Qwen = Qwen2.5-7B-Instruct 840-pair promptfixed run; Mistral/Llama-3 = 840-pair multimodel batch."


def savefig(fig: plt.Figure, stem: str) -> None:
    _savefig(fig, stem, OUTPUT_DIR)


def load_shift_data() -> pd.DataFrame:
    df = pd.read_csv(SHIFT_CSV)
    df["coef"] = df["coef"].astype(float)
    df = df[df["model"] != "Qwen"].copy()
    qwen_df = load_qwen_instruct_shift_data()
    story_df = pd.concat([qwen_df, df], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    story_df.to_csv(OUTPUT_DIR / "story_shift_data.csv", index=False)
    return story_df


def load_detection_data() -> pd.DataFrame:
    df = pd.read_csv(DETECTION_CSV)
    df = df[df["model"] != "Qwen"].copy()
    qwen_df = load_qwen_instruct_detection_data()
    story_df = pd.concat([qwen_df, df], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    story_df.to_csv(OUTPUT_DIR / "story_detection_data.csv", index=False)
    return story_df


def load_qwen_instruct_shift_data() -> pd.DataFrame:
    source_csv = QWEN_INSTRUCT_840_COMPARISON / "branch_shift_summary.csv"
    df = pd.read_csv(source_csv)
    df["coef"] = df["coef"].astype(float)
    df["model"] = "Qwen"
    df["source_csv"] = str(source_csv)
    df["source_note"] = "Qwen/Qwen2.5-7B-Instruct 840-pair promptfixed run"
    return df


def load_qwen_instruct_detection_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"Best layer:\s+(-?\d+)\s+\(val AUC=([0-9.]+), test AUC=([0-9.]+)\)")
    for side, log_path in QWEN_INSTRUCT_840_LOGS.items():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Could not parse Qwen-Instruct best layer from {log_path}")
        rows.append(
            {
                "ideology": side,
                "best_layer": int(match.group(1)),
                "best_test_auc": float(match.group(3)),
                "aggregation_test_auc": np.nan,
                "best_test_acc": np.nan,
                "model": "Qwen",
                "source_csv": str(log_path),
            }
        )
    return pd.DataFrame(rows)


def alpha_color_mapping(cmap_name: str = "plasma", *, start: float = 0.05, end: float = 0.95) -> dict[float, object]:
    cmap = plt.get_cmap(cmap_name)
    nonzero_coefs = [coef for coef in COEF_ORDER if float(coef) != 0.0]
    nonzero_colors = cmap(np.linspace(start, end, len(nonzero_coefs)))
    coef_colors: dict[float, object] = {0.0: COEF_COLORS[0.0]}
    coef_colors.update({float(coef): nonzero_colors[i] for i, coef in enumerate(nonzero_coefs)})
    return coef_colors


def compute_symmetric_limit(df: pd.DataFrame, pad: float = 0.6, minimum: float = 5.0) -> float:
    xy = df.dropna(subset=["x", "y"])
    if xy.empty:
        return 6.0
    x_abs = np.abs(pd.to_numeric(xy["x"], errors="coerce").to_numpy(dtype=float))
    y_abs = np.abs(pd.to_numeric(xy["y"], errors="coerce").to_numpy(dtype=float))
    max_abs = float(np.nanmax(np.concatenate([x_abs, y_abs])))
    return max(minimum, float(np.ceil((max_abs + pad) / 1.0) * 1.0))


def jitter_xy(
    x: float,
    y: float,
    counts: dict[tuple[int, int], int],
    overlap_eps: float,
    jitter: float,
    jitter_offsets: np.ndarray,
) -> tuple[float, float]:
    key = (int(np.round(x / overlap_eps)), int(np.round(y / overlap_eps)))
    used = counts.get(key, 0)
    counts[key] = used + 1
    offset = jitter_offsets[min(used, len(jitter_offsets) - 1)] * jitter
    return x + float(offset[0]), y + float(offset[1])


def load_item_change_data() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, root in MODEL_COMPARISON_ROOTS.items():
        source_csv = root / "comparison" / "branch_changed_items_summary.csv"
        df = pd.read_csv(source_csv)
        df["model"] = model
        df["coef"] = df["coef"].astype(float)
        df["changed_items"] = pd.to_numeric(df["changed_items"], errors="coerce")
        df["change_rate"] = df["changed_items"] / 62.0
        df["source_csv"] = str(source_csv)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def build_item_coordinate_change_table(shift_df: pd.DataFrame) -> pd.DataFrame:
    changes = load_item_change_data()
    coord_cols = ["model", "ideology", "language", "coef", "delta_x", "delta_y", "shift"]
    table = changes.merge(shift_df[coord_cols], on=["model", "ideology", "language", "coef"], how="left")
    table = table[
        [
            "model",
            "language",
            "ideology",
            "coef",
            "changed_items",
            "change_rate",
            "delta_x",
            "delta_y",
            "shift",
            "examples",
            "source_csv",
        ]
    ].copy()
    table = table.sort_values(
        by=["model", "language", "ideology", "coef"],
        key=lambda col: col.map({value: idx for idx, value in enumerate(MODEL_ORDER + LANG_ORDER + SIDE_ORDER)})
        if col.name in {"model", "language", "ideology"}
        else col,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "story_item_coordinate_change_table.csv", index=False)
    return table


def plot_detection_auc(detection_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    auc_values = pd.to_numeric(detection_df["best_test_auc"], errors="coerce").to_numpy(dtype=float)
    auc_values = auc_values[np.isfinite(auc_values)]
    y_min = 0.84 if len(auc_values) == 0 else min(0.84, float(np.min(auc_values)) - 0.03)
    ax.set_ylim(y_min, 1.0)
    with plt.style.context("tableau-colorblind10"):
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, side in enumerate(SIDE_ORDER):
        offsets = x + (idx - 0.5) * width
        sub = (
            detection_df[detection_df["ideology"] == side]
            .set_index("model")
            .reindex(MODEL_ORDER)
            .reset_index()
        )
        bars = ax.bar(
            offsets,
            sub["best_test_auc"],
            width=width,
            color=palette[idx % len(palette)],
            label=f"{side.capitalize()} steering",
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Best-layer test AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))

    fig.text(0.5, 0.012, SOURCE_NOTE, ha="center", fontsize=8.3, color="#555555")
    fig.subplots_adjust(bottom=0.30, top=0.90)
    savefig(fig, "fig1_detection_auc")


def add_trajectory_arrow(ax: plt.Axes, xs: pd.Series, ys: pd.Series, color: str) -> None:
    clean = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if len(clean) < 2:
        return
    for start, end in zip(clean.iloc[:-1].itertuples(), clean.iloc[1:].itertuples(), strict=False):
        dx = end.x - start.x
        dy = end.y - start.y
        if abs(dx) + abs(dy) < 1e-8:
            continue
        ax.annotate(
            "",
            xy=(end.x, end.y),
            xytext=(start.x, start.y),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.6,
                "shrinkA": 4,
                "shrinkB": 4,
                "alpha": 0.85,
            },
        )


def plot_english_displacement(shift_df: pd.DataFrame) -> None:
    en = shift_df[shift_df["language"] == "en"].copy()
    pad = 0.6
    lim = compute_symmetric_limit(en, pad=pad, minimum=5.0)
    major_step = 2.0 if lim <= 10.0 else 4.0
    minor_step = major_step / 2.0

    fig, axes = plt.subplots(
        2,
        len(MODEL_ORDER),
        figsize=(11.8, 6.0),
        sharex=False,
        sharey=False,
        gridspec_kw={"wspace": 0.24, "hspace": 0.22},
    )

    overlap_eps = 0.12
    jitter = 0.28
    coef_colors = alpha_color_mapping("plasma")
    jitter_offsets = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
    )

    for col_idx, model in enumerate(MODEL_ORDER):
        model_df = en[en["model"] == model].copy()
        for row_idx, side in enumerate(SIDE_ORDER):
            ax = axes[row_idx, col_idx]
            side_df = model_df[model_df["ideology"] == side]
            side_df = side_df.dropna(subset=["x", "y", "coef"]).sort_values("coef")

            counts: dict[tuple[int, int], int] = {}
            for _, point in side_df.iterrows():
                coef = float(point["coef"])
                coef_key = float(np.round(coef, 3))

                x = float(point["x"])
                y = float(point["y"])
                x, y = jitter_xy(x, y, counts, overlap_eps, jitter, jitter_offsets)

                ax.scatter(
                    x,
                    y,
                    s=72,
                    marker=SIDE_MARKERS[side],
                    color=coef_colors.get(coef_key, "#777777"),
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.92,
                    zorder=3,
                )

            ax.axhline(0, color="#999999", linewidth=0.8)
            ax.axvline(0, color="#999999", linewidth=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MultipleLocator(major_step))
            ax.yaxis.set_major_locator(MultipleLocator(major_step))
            ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
            ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
            ax.grid(which="major", alpha=0.18)
            ax.grid(which="minor", alpha=0.05)

            if row_idx == 1:
                ax.set_xlabel("Economic")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Social")
            else:
                ax.set_ylabel("")

    fig.text(0.02, 0.67, "Left steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")
    fig.text(0.02, 0.29, "Right steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")

    alpha_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=coef_colors[float(coef)],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=9,
            label=f"α={coef:g}",
        )
        for coef in COEF_ORDER
    ]

    fig.suptitle(
        "English Compass Positions by Steering Coefficient",
        y=0.985,
        fontweight="bold",
    )
    fig.legend(
        handles=alpha_handles,
        loc="upper center",
        ncol=len(COEF_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    for col_idx, model in enumerate(MODEL_ORDER):
        top_ax = axes[0, col_idx]
        pos = top_ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.01,
            model,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    fig.text(
        0.5,
        0.012,
        "Each point is one alpha; color encodes steering coefficient and marker shape encodes steering side.\n"
        "Qwen = Qwen2.5-7B-Instruct promptfixed run; Mistral/Llama-3 = multimodel batch.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.16, top=0.78, left=0.07)
    savefig(fig, "fig2_english_steering_displacement")


def plot_multilingual_positions_by_model(shift_df: pd.DataFrame) -> None:
    df = shift_df.copy()
    coef_colors = alpha_color_mapping("plasma")
    overlap_eps = 0.12
    jitter = 0.28
    jitter_offsets = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
    )

    alpha_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=coef_colors[float(coef)],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=9,
            label=f"α={coef:g}",
        )
        for coef in COEF_ORDER
    ]

    for model in MODEL_ORDER:
        model_df = df[df["model"] == model].copy()
        lim = compute_symmetric_limit(model_df, pad=0.6, minimum=5.0)
        major_step = 2.0 if lim <= 10.0 else 4.0
        minor_step = major_step / 2.0

        fig, axes = plt.subplots(
            2,
            len(LANG_ORDER),
            figsize=(16.0, 5.9),
            sharex=False,
            sharey=False,
            gridspec_kw={"wspace": 0.22, "hspace": 0.22},
        )

        for col_idx, lang in enumerate(LANG_ORDER):
            lang_df = model_df[model_df["language"] == lang]
            for row_idx, side in enumerate(SIDE_ORDER):
                ax = axes[row_idx, col_idx]
                side_df = lang_df[lang_df["ideology"] == side]
                side_df = side_df.dropna(subset=["x", "y", "coef"]).sort_values("coef")

                counts: dict[tuple[int, int], int] = {}
                for _, point in side_df.iterrows():
                    coef = float(point["coef"])
                    coef_key = float(np.round(coef, 3))

                    x = float(point["x"])
                    y = float(point["y"])
                    x, y = jitter_xy(x, y, counts, overlap_eps, jitter, jitter_offsets)

                    ax.scatter(
                        x,
                        y,
                        s=70,
                        marker=SIDE_MARKERS[side],
                        color=coef_colors.get(coef_key, "#777777"),
                        edgecolor="white",
                        linewidth=1.0,
                        alpha=0.92,
                        zorder=3,
                    )

                ax.axhline(0, color="#999999", linewidth=0.8)
                ax.axvline(0, color="#999999", linewidth=0.8)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal", adjustable="box")
                ax.xaxis.set_major_locator(MultipleLocator(major_step))
                ax.yaxis.set_major_locator(MultipleLocator(major_step))
                ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
                ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
                ax.grid(which="major", alpha=0.18)
                ax.grid(which="minor", alpha=0.05)

                if row_idx == 1:
                    ax.set_xlabel("Economic")
                else:
                    ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel("Social")
                else:
                    ax.set_ylabel("")

        for col_idx, lang in enumerate(LANG_ORDER):
            top_ax = axes[0, col_idx]
            pos = top_ax.get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                pos.y1 + 0.01,
                LANG_LABELS.get(lang, lang),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.text(0.02, 0.67, "Left steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")
        fig.text(0.02, 0.29, "Right steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")

        fig.suptitle(f"{model}: Compass Positions by Steering Coefficient Across Languages", y=0.985, fontweight="bold")
        fig.legend(
            handles=alpha_handles,
            loc="upper center",
            ncol=len(COEF_ORDER),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5,
            0.012,
            "Each point is one alpha; color encodes steering coefficient and marker shape encodes steering side.",
            ha="center",
            fontsize=8.2,
            color="#555555",
        )
        fig.subplots_adjust(bottom=0.16, top=0.78, left=0.06)
        savefig(fig, f"fig2_multilingual_positions_{model.lower().replace('-', '')}")


def plot_multilingual_positions_by_language(shift_df: pd.DataFrame) -> None:
    df = shift_df.copy()
    coef_colors = alpha_color_mapping("plasma")
    overlap_eps = 0.12
    jitter = 0.28
    jitter_offsets = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
    )

    alpha_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=coef_colors[float(coef)],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=9,
            label=f"α={coef:g}",
        )
        for coef in COEF_ORDER
    ]

    for lang in LANG_ORDER:
        lang_df = df[df["language"] == lang].copy()
        lim = compute_symmetric_limit(lang_df, pad=0.6, minimum=5.0)
        major_step = 2.0 if lim <= 10.0 else 4.0
        minor_step = major_step / 2.0

        fig, axes = plt.subplots(
            2,
            len(MODEL_ORDER),
            figsize=(11.8, 6.0),
            sharex=False,
            sharey=False,
            gridspec_kw={"wspace": 0.24, "hspace": 0.22},
        )

        for col_idx, model in enumerate(MODEL_ORDER):
            model_df = lang_df[lang_df["model"] == model]
            for row_idx, side in enumerate(SIDE_ORDER):
                ax = axes[row_idx, col_idx]
                side_df = model_df[model_df["ideology"] == side]
                side_df = side_df.dropna(subset=["x", "y", "coef"]).sort_values("coef")

                counts: dict[tuple[int, int], int] = {}
                for _, point in side_df.iterrows():
                    coef = float(point["coef"])
                    coef_key = float(np.round(coef, 3))

                    x = float(point["x"])
                    y = float(point["y"])
                    x, y = jitter_xy(x, y, counts, overlap_eps, jitter, jitter_offsets)

                    ax.scatter(
                        x,
                        y,
                        s=70,
                        marker=SIDE_MARKERS[side],
                        color=coef_colors.get(coef_key, "#777777"),
                        edgecolor="white",
                        linewidth=1.0,
                        alpha=0.92,
                        zorder=3,
                    )

                ax.axhline(0, color="#999999", linewidth=0.8)
                ax.axvline(0, color="#999999", linewidth=0.8)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal", adjustable="box")
                ax.xaxis.set_major_locator(MultipleLocator(major_step))
                ax.yaxis.set_major_locator(MultipleLocator(major_step))
                ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
                ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
                ax.grid(which="major", alpha=0.18)
                ax.grid(which="minor", alpha=0.05)

                if row_idx == 1:
                    ax.set_xlabel("Economic")
                else:
                    ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel("Social")
                else:
                    ax.set_ylabel("")

        for col_idx, model in enumerate(MODEL_ORDER):
            top_ax = axes[0, col_idx]
            pos = top_ax.get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                pos.y1 + 0.01,
                model,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.text(0.02, 0.67, "Left steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")
        fig.text(0.02, 0.29, "Right steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")

        lang_label = LANG_LABELS.get(lang, lang)
        fig.suptitle(f"{lang_label}: Compass Positions by Steering Coefficient Across Models", y=0.985, fontweight="bold")
        fig.legend(
            handles=alpha_handles,
            loc="upper center",
            ncol=len(COEF_ORDER),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5,
            0.012,
            "Each point is one alpha; color encodes steering coefficient and marker shape encodes steering side.",
            ha="center",
            fontsize=8.2,
            color="#555555",
        )
        fig.subplots_adjust(bottom=0.16, top=0.78, left=0.07)
        savefig(fig, f"fig2_multilingual_positions_{lang}")


def format_signed(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:+.2f}"


def format_distance(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.2f}"


def add_table_figure(
    df: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    stem: str,
    title: str,
    footnote: str,
    figsize: tuple[float, float],
    font_size: float = 8.5,
    header_size: float = 9.0,
    col_widths: list[float] | None = None,
    highlight_rows: list[bool] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    display_df = df[columns].copy()
    table = ax.table(
        cellText=display_df.values,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DEE8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(header_size)
        elif highlight_rows is not None and highlight_rows[row - 1]:
            cell.set_facecolor("#FFF2CC")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F8FA")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")

    fig.suptitle(title, y=0.985, fontweight="bold")
    fig.text(0.5, 0.015, footnote, ha="center", fontsize=8.0, color="#555555")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.91, bottom=0.07)
    savefig(fig, stem)


def plot_item_coordinate_change_tables(item_coord_df: pd.DataFrame) -> None:
    for lang_code in LANG_ORDER:
        lang_label = LANG_LABELS[lang_code]
        lang_df = item_coord_df[(item_coord_df["language"] == lang_code) & (item_coord_df["coef"] > 0)].copy()
        lang_df["Model"] = lang_df["model"]
        lang_df["Steering"] = lang_df["ideology"].str.capitalize()
        lang_df["Alpha"] = lang_df["coef"].map(lambda value: f"{value:g}")
        lang_df["Changed / 62"] = lang_df["changed_items"].map(lambda value: f"{int(value)}/62")
        lang_df["Change %"] = lang_df["change_rate"].map(lambda value: f"{value:.0%}")
        lang_df["Delta Econ"] = lang_df["delta_x"].map(format_signed)
        lang_df["Delta Social"] = lang_df["delta_y"].map(format_signed)
        lang_df["Distance"] = lang_df["shift"].map(format_distance)
        lang_df = lang_df.sort_values(
            by=["model", "ideology", "coef"],
            key=lambda col: col.map({value: idx for idx, value in enumerate(MODEL_ORDER + SIDE_ORDER)})
            if col.name in {"model", "ideology"}
            else col,
        )
        highlight_rows = lang_df["coef"].eq(4.0).tolist()
        add_table_figure(
            lang_df,
            ["Model", "Steering", "Alpha", "Changed / 62", "Change %", "Delta Econ", "Delta Social", "Distance"],
            ["Model", "Steer", "alpha", "Changed\nitems", "Change\nrate", "Delta\nEcon", "Delta\nSocial", "Compass\ndistance"],
            f"fig2a_{lang_code}_item_coordinate_change_table",
            f"{lang_label} Steering Effects Across Alpha",
            f"Yellow rows are alpha=4 and should match the {lang_label} rows in the cross-lingual alpha=4 table. Changed items compare parsed stance labels against alpha=0.",
            figsize=(11.4, 7.2),
            font_size=8.2,
            header_size=8.4,
            col_widths=[0.14, 0.10, 0.07, 0.12, 0.10, 0.11, 0.12, 0.12],
            highlight_rows=highlight_rows,
        )

    alpha4_rows: list[dict[str, str]] = []
    alpha4 = item_coord_df[item_coord_df["coef"] == 4.0].copy()
    for model in MODEL_ORDER:
        for language in LANG_ORDER:
            row = {"Model": model, "Language": LANG_LABELS[language]}
            for side, prefix in [("left", "L"), ("right", "R")]:
                side_row = alpha4[
                    (alpha4["model"] == model)
                    & (alpha4["language"] == language)
                    & (alpha4["ideology"] == side)
                ]
                if side_row.empty:
                    row[f"{prefix} changed"] = "NA"
                    row[f"{prefix} dE"] = "NA"
                    row[f"{prefix} dS"] = "NA"
                    row[f"{prefix} dist"] = "NA"
                    continue
                values = side_row.iloc[0]
                row[f"{prefix} changed"] = f"{int(values['changed_items'])}/62"
                row[f"{prefix} dE"] = format_signed(values["delta_x"])
                row[f"{prefix} dS"] = format_signed(values["delta_y"])
                row[f"{prefix} dist"] = format_distance(values["shift"])
            alpha4_rows.append(row)
    alpha4 = pd.DataFrame(alpha4_rows)
    add_table_figure(
        alpha4,
        ["Model", "Language", "L changed", "L dE", "L dS", "L dist", "R changed", "R dE", "R dS", "R dist"],
        ["Model", "Language", "Left\nchanged", "Left\nDelta E", "Left\nDelta S", "Left\ndist.", "Right\nchanged", "Right\nDelta E", "Right\nDelta S", "Right\ndist."],
        "fig2b_crosslingual_alpha4_item_coordinate_change_table",
        "Cross-Lingual Steering Effects at alpha=4",
        "Each row compares alpha=4 against that language's alpha=0 baseline. This table separates item-level changes from coordinate movement.",
        figsize=(11.4, 6.2),
        font_size=7.8,
        header_size=7.6,
        col_widths=[0.10, 0.12, 0.105, 0.09, 0.09, 0.08, 0.105, 0.09, 0.09, 0.08],
    )


def plot_detection_vs_behavior_summary(shift_df: pd.DataFrame, detection_df: pd.DataFrame) -> None:
    mean_auc = detection_df.groupby("model", as_index=False)["best_test_auc"].mean()
    english_alpha4 = (
        shift_df[(shift_df["language"] == "en") & (shift_df["coef"] == 4.0)]
        .groupby("model", as_index=False)["shift"]
        .max()
        .rename(columns={"shift": "max_english_shift_alpha4"})
    )
    transfer_alpha4 = (
        shift_df[shift_df["coef"] == 4.0]
        .groupby("model", as_index=False)["shift"]
        .max()
        .rename(columns={"shift": "max_any_language_shift_alpha4"})
    )
    summary = (
        mean_auc.merge(english_alpha4, on="model", how="left")
        .merge(transfer_alpha4, on="model", how="left")
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "story_detection_behavior_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    metrics = [
        ("best_test_auc", "Mean detection AUC", (0.965, 1.006), "{:.3f}"),
        ("max_english_shift_alpha4", "Max English shift at alpha=4", None, "{:.1f}"),
        ("max_any_language_shift_alpha4", "Max any-language shift at alpha=4", None, "{:.1f}"),
    ]

    for ax, (col, title, ylim, fmt) in zip(axes, metrics, strict=True):
        values = summary[col].astype(float)
        bars = ax.bar(
            summary["model"],
            values,
            color=[MODEL_COLORS[m] for m in summary["model"]],
            edgecolor="white",
            linewidth=0.8,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.18))
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025,
                    fmt.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Detection Strength and Behavioral Steerability Are Separable", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"All models have high hidden-state detection AUC, but their compass-coordinate movement differs sharply. {SOURCE_NOTE}",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    savefig(fig, "fig3_detection_vs_behavior_summary")


def plot_crosslingual_heatmap_clean(shift_df: pd.DataFrame) -> None:
    vmax = float(np.nanmax(shift_df["shift"].to_numpy()))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#EFEFEF")

    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(SIDE_ORDER),
        figsize=(11.4, 8.6),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.34, "wspace": 0.12},
    )
    image = None
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, side in enumerate(SIDE_ORDER):
            ax = axes[row_idx, col_idx]
            subset = shift_df[(shift_df["model"] == model) & (shift_df["ideology"] == side)]
            pivot = (
                subset.pivot_table(index="language", columns="coef", values="shift", aggfunc="mean")
                .reindex(index=LANG_ORDER, columns=COEF_ORDER)
            )
            masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
            image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{model}: {side} steering", fontweight="bold")
            ax.set_xticks(range(len(COEF_ORDER)))
            ax.set_xticklabels([f"{coef:g}" for coef in COEF_ORDER])
            ax.set_yticks(range(len(LANG_ORDER)))
            ax.set_yticklabels([LANG_LABELS[lang] for lang in LANG_ORDER])

            for i, lang in enumerate(LANG_ORDER):
                for j, coef in enumerate(COEF_ORDER):
                    value = pivot.loc[lang, coef]
                    if np.isfinite(value):
                        text_color = "white" if value > vmax * 0.42 else "#1F1F1F"
                        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
                    else:
                        ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="#555555")
                        ax.add_patch(
                            plt.Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                fill=False,
                                hatch="///",
                                edgecolor="#C9C9C9",
                                linewidth=0.0,
                            )
                        )

            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Steering coefficient alpha")
            if col_idx == 0:
                ax.set_ylabel("Language")

    if image is not None:
        cbar_ax = fig.add_axes([0.89, 0.18, 0.025, 0.64])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Compass displacement from alpha=0")

    fig.suptitle("Cross-Lingual Steering Transfer", y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.03,
        f"Cell value = within-language coordinate displacement from alpha=0. {SOURCE_NOTE}",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.10, right=0.84, top=0.92)
    savefig(fig, "fig4_crosslingual_transfer_clean")


def stance_counts_for_file(path: Path) -> dict[str, float]:
    answers = read_answer_file(path)
    total = len(answers)
    counts = {stance: 0 for stance in STANCE_ORDER}
    for answer in answers:
        choice = (answer.get("parsed_choice") or "").strip().lower()
        if choice in counts and choice != "unparsed":
            counts[choice] += 1
        else:
            counts["unparsed"] += 1
    return {stance: counts[stance] / total if total else np.nan for stance in STANCE_ORDER}


def normalize_stance(choice: str | None) -> str:
    value = (choice or "").strip().lower()
    return value if value in STANCE_ORDER else "unparsed"


def build_focused_stance_data(model: str = "Llama-3", language: str = "en") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    root = MODEL_STANCE_ROOTS[model]
    for side in SIDE_ORDER:
        for coef in COEF_ORDER:
            path = root / f"steering_{side}" / language / f"compass_answers_{coef_label(coef)}.csv"
            proportions = stance_counts_for_file(path)
            for stance, value in proportions.items():
                rows.append(
                    {
                        "model": model,
                        "language": language,
                        "side": side,
                        "coef": coef,
                        "stance": stance,
                        "proportion": value,
                        "source_csv": str(path),
                    }
                )
    return pd.DataFrame(rows)


def plot_focused_stance_distribution() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        for lang_code in LANG_ORDER:
            lang_label = LANG_LABELS[lang_code]
            model_slug = model.lower().replace("-", "")
            df = build_focused_stance_data(model=model, language=lang_code)
            df.to_csv(OUTPUT_DIR / f"focused_stance_distribution_{model_slug}_{lang_code}.csv", index=False)

            fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
            x = np.arange(len(COEF_ORDER))
            for ax, side in zip(axes, SIDE_ORDER, strict=True):
                side_df = df[df["side"] == side]
                bottom = np.zeros(len(COEF_ORDER))
                for stance in STANCE_ORDER:
                    values = [
                        float(
                            side_df[(side_df["coef"] == coef) & (side_df["stance"] == stance)]["proportion"].iloc[0]
                        )
                        for coef in COEF_ORDER
                    ]
                    ax.bar(
                        x,
                        values,
                        bottom=bottom,
                        color=STANCE_COLORS[stance],
                        edgecolor="white",
                        linewidth=0.5,
                        label=stance,
                    )
                    bottom += np.array(values)

                ax.set_title(f"{model} {lang_label}: {side} steering", fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{coef:g}" for coef in COEF_ORDER])
                ax.set_xlabel("Steering coefficient alpha")
                ax.grid(axis="y", alpha=0.2)
            axes[0].set_ylabel("Share of 62 Political Compass items")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(STANCE_ORDER), frameon=False, bbox_to_anchor=(0.5, 0.98))
            fig.suptitle(f"Item-Level Stance Distribution: {model} {lang_label}", y=1.08, fontweight="bold")
            fig.text(
                0.5,
                0.01,
                "This supplementary view shows answer-category proportions; it is not a coordinate trajectory.",
                ha="center",
                fontsize=9,
                color="#555555",
            )
            fig.subplots_adjust(top=0.77, bottom=0.20, wspace=0.12)
            savefig(fig, f"figS1_focused_stance_distribution_{model_slug}_{lang_code}")


def plot_item_level_alpha_stance_example(
    model: str = "Mistral",
    language: str = "en",
    target_stance: str = "strongly agree",
) -> None:
    if target_stance not in STANCE_ORDER:
        raise ValueError(f"Unknown target stance: {target_stance}")

    root = MODEL_STANCE_ROOTS[model]
    lang_label = LANG_LABELS.get(language, language)
    model_slug = model.lower().replace("-", "")
    coef_colors = alpha_color_mapping("plasma")
    stance_slug = re.sub(r"[^a-z0-9]+", "_", target_stance.lower()).strip("_")

    order_path = (
        root
        / f"steering_{SIDE_ORDER[0]}"
        / language
        / f"compass_answers_{coef_label(float(COEF_ORDER[0]))}.csv"
    )
    ordered_answers = read_answer_file(order_path)
    item_order = [row.get("item_id", "").strip() for row in ordered_answers if row.get("item_id", "").strip()]
    if not item_order:
        raise ValueError(f"Could not determine item order from {order_path}")

    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_order)}

    alpha_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=coef_colors[float(coef)],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            label=f"α={coef:g}",
        )
        for coef in COEF_ORDER
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 4.6), sharey=True)
    for ax, side in zip(axes, SIDE_ORDER, strict=True):
        for coef_idx, coef in enumerate(COEF_ORDER):
            path = root / f"steering_{side}" / language / f"compass_answers_{coef_label(float(coef))}.csv"
            answers = read_answer_file(path)
            stance_items = {
                row.get("item_id", "").strip()
                for row in answers
                if row.get("item_id", "").strip() and normalize_stance(row.get("parsed_choice")) == target_stance
            }
            xs = [item_to_idx[item_id] for item_id in stance_items if item_id in item_to_idx]
            ys = [coef_idx] * len(xs)
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=54,
                    marker="s",
                    color=coef_colors[float(coef)],
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.95,
                )

        ax.set_title(f"{model} {lang_label}: {side} steering", fontweight="bold")
        ax.set_xlim(-0.5, len(item_order) - 0.5)
        ax.set_ylim(-0.6, len(COEF_ORDER) - 0.4)
        ax.invert_yaxis()

        ax.set_xticks(np.arange(len(item_order)))
        ax.set_xticklabels([str(i + 1) for i in range(len(item_order))], fontsize=6, rotation=90)
        ax.set_xlabel("Item index (1-62)")
        ax.set_yticks(np.arange(len(COEF_ORDER)))
        ax.set_yticklabels([f"α={coef:g}" for coef in COEF_ORDER])
        ax.grid(axis="y", alpha=0.18)
        ax.grid(axis="x", alpha=0.06)

    axes[0].set_ylabel("Steering coefficient alpha")
    fig.legend(
        handles=alpha_handles,
        loc="upper center",
        ncol=len(COEF_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        f"Where the model answers '{target_stance}' across 62 questions ({model} {lang_label})",
        y=1.08,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.78, bottom=0.30, wspace=0.10)
    savefig(fig, f"figS2_item_level_{stance_slug}_{model_slug}_{language}")


def item_answer_path(root: Path, side: str, language: str, coef: float) -> Path:
    return root / f"steering_{side}" / language / f"compass_answers_{coef_label(float(coef))}.csv"


def load_item_rows(
    root: Path,
    side: str,
    language: str,
    coef: float,
    *,
    required: bool = True,
) -> list[dict[str, str]]:
    path = item_answer_path(root, side, language, coef)
    if not path.exists():
        message = f"Missing answer CSV: {path}"
        if required:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}; skipping")
        return []
    rows = read_answer_file(path)
    missing = sum(1 for row in rows if not (row.get("item_id") or "").strip())
    if missing:
        print(f"WARNING: {path} has {missing} rows with missing item_id; skipping those rows")
    return rows


def item_order_from_baseline(root: Path, model: str, language: str, side: str = "left") -> list[str]:
    rows = load_item_rows(root, side, language, COEF_ORDER[0])
    item_order = [row.get("item_id", "").strip() for row in rows if row.get("item_id", "").strip()]
    if not item_order:
        raise ValueError(f"Could not determine item order for {model} {language} from {item_answer_path(root, side, language, COEF_ORDER[0])}")
    return item_order


def model_slug(model: str) -> str:
    return model.lower().replace("-", "")


def plot_item_level_stance_matrix(
    model: str = "Mistral",
    language: str = "en",
    sides: list[str] | tuple[str, ...] = tuple(SIDE_ORDER),
    color_by: str = "stance",
    stance_cmap_name: str | None = None,
    stance_cmap_levels: list[float] | None = None,
    unparsed_color: object | None = None,
    stance_hatches: dict[str, str] | None = None,
    hatch_color: object = (0.0, 0.0, 0.0, 0.35),
    coef_cmap_name: str = "YlGn",
    coef_cmap_range: tuple[float, float] = (0.20, 0.85),
    filename_suffix: str = "",
) -> Path:
    root = MODEL_STANCE_ROOTS[model]
    lang_label = LANG_LABELS.get(language, language)
    slug = model_slug(model)

    item_order = item_order_from_baseline(root, model, language, side=sides[0])
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_order)}
    stance_to_idx = {stance: idx for idx, stance in enumerate(STANCE_ORDER)}
    if color_by not in {"stance", "coef"}:
        raise ValueError(f"color_by must be 'stance' or 'coef', got {color_by!r}")

    if color_by == "stance":
        if stance_cmap_name is None:
            stance_colors = [STANCE_COLORS[stance] for stance in STANCE_ORDER]
        else:
            cmap = plt.get_cmap(stance_cmap_name)
            default_levels = [0.85, 0.70, 0.55, 0.40]
            levels = stance_cmap_levels or default_levels
            if len(levels) not in {len(STANCE_ORDER), len(STANCE_ORDER) - 1}:
                raise ValueError(
                    f"stance_cmap_levels must have {len(STANCE_ORDER)} or {len(STANCE_ORDER) - 1} values, got {len(levels)}"
                )
            if len(levels) == len(STANCE_ORDER):
                stance_colors = list(cmap(np.array(levels, dtype=float)))
            else:
                stance_colors = [None] * len(STANCE_ORDER)
                for stance, lvl in zip(STANCE_ORDER[:-1], levels, strict=True):
                    stance_colors[stance_to_idx[stance]] = cmap(float(lvl))
                stance_colors[stance_to_idx["unparsed"]] = unparsed_color or STANCE_COLORS["unparsed"]
        cmap = ListedColormap(stance_colors)
        coef_colors: dict[float, object] | None = None
    else:
        coef_map = plt.get_cmap(coef_cmap_name)
        levels = np.linspace(coef_cmap_range[0], coef_cmap_range[1], len(COEF_ORDER))
        coef_colors = {float(coef): coef_map(float(levels[idx])) for idx, coef in enumerate(COEF_ORDER)}
        stance_colors = [unparsed_color or STANCE_COLORS["unparsed"]] * len(STANCE_ORDER)
        cmap = None

    fig, axes = plt.subplots(1, len(sides), figsize=(7.8 * len(sides), 5.0), sharey=True)
    if len(sides) == 1:
        axes = [axes]
    for ax, side in zip(axes, sides, strict=True):
        matrix = np.full((len(COEF_ORDER), len(item_order)), stance_to_idx["unparsed"], dtype=int)
        for coef_idx, coef in enumerate(COEF_ORDER):
            answers = load_item_rows(root, side, language, float(coef), required=False)
            seen_item_ids: set[str] = set()
            for row in answers:
                item_id = row.get("item_id", "").strip()
                if not item_id:
                    continue
                seen_item_ids.add(item_id)
                if item_id not in item_to_idx:
                    print(f"WARNING: {model} {language} {side} alpha={coef:g} has item_id not in baseline order: {item_id}; skipping")
                    continue
                stance = normalize_stance(row.get("parsed_choice"))
                matrix[coef_idx, item_to_idx[item_id]] = stance_to_idx[stance]
            missing_from_file = [item_id for item_id in item_order if item_id not in seen_item_ids]
            if missing_from_file:
                print(
                    f"WARNING: {model} {language} {side} alpha={coef:g} missing "
                    f"{len(missing_from_file)} baseline item_id values; leaving as unparsed"
                )

        baseline = matrix[0, :]
        change_counts = [0]
        for coef_idx in range(1, len(COEF_ORDER)):
            changed_cols = np.where(matrix[coef_idx, :] != baseline)[0]
            change_counts.append(int(len(changed_cols)))
            for col in changed_cols:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, coef_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.0,
                        zorder=3,
                    )
                )

        if color_by == "stance":
            ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-0.5,
                vmax=len(STANCE_ORDER) - 0.5,
            )
            if stance_hatches:
                plt.rcParams["hatch.linewidth"] = 0.7
                for r in range(matrix.shape[0]):
                    for c in range(matrix.shape[1]):
                        stance = STANCE_ORDER[int(matrix[r, c])]
                        hatch = stance_hatches.get(stance, "")
                        if not hatch:
                            continue
                        ax.add_patch(
                            plt.Rectangle(
                                (c - 0.5, r - 0.5),
                                1.0,
                                1.0,
                                fill=False,
                                hatch=hatch,
                                edgecolor=hatch_color,
                                linewidth=0.0,
                                zorder=2,
                            )
                        )
        else:
            plt.rcParams["hatch.linewidth"] = 0.7
            for r in range(matrix.shape[0]):
                coef = float(COEF_ORDER[r])
                row_color = coef_colors[coef]
                for c in range(matrix.shape[1]):
                    stance = STANCE_ORDER[int(matrix[r, c])]
                    face = (unparsed_color or STANCE_COLORS["unparsed"]) if stance == "unparsed" else row_color
                    hatch = stance_hatches.get(stance, "") if stance_hatches else ""
                    ax.add_patch(
                        plt.Rectangle(
                            (c - 0.5, r - 0.5),
                            1.0,
                            1.0,
                            facecolor=face,
                            edgecolor=hatch_color if hatch else "none",
                            linewidth=0.0 if hatch else 0.0,
                            hatch=hatch,
                            zorder=1,
                        )
                    )
        ax.set_title(f"{model} {lang_label}: {side} steering", fontweight="bold")
        ax.set_yticks(np.arange(len(COEF_ORDER)))
        ax.set_yticklabels([f"α={coef:g}" for coef in COEF_ORDER])
        ax.set_xticks(np.arange(len(item_order)))
        ax.set_xticklabels([str(i + 1) for i in range(len(item_order))], fontsize=6, rotation=90)
        ax.set_xlabel("Item index (1-62)")

    if stance_hatches:
        legend_face = (
            plt.get_cmap(stance_cmap_name)(0.65) if (color_by == "stance" and stance_cmap_name is not None) else "#ffffff"
        )
        legend_handles = [
            plt.Rectangle(
                (0, 0),
                1.0,
                1.0,
                facecolor=legend_face if color_by == "coef" else stance_colors[stance_to_idx[stance]],
                edgecolor=hatch_color,
                linewidth=0.8,
                hatch=stance_hatches.get(stance, ""),
                label=stance,
            )
            for stance in STANCE_ORDER
        ]
    else:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=stance_colors[stance_to_idx[stance]],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=10,
                label=stance,
            )
            for stance in STANCE_ORDER
        ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(STANCE_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(f"Item-level stance categories across alpha ({model} {lang_label})", y=1.08, fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.30, wspace=0.08)
    savefig(fig, f"figS2_item_level_stance_matrix_{slug}_{language}{filename_suffix}")
    return OUTPUT_DIR / f"figS2_item_level_stance_matrix_{slug}_{language}{filename_suffix}.png"


def plot_changes_only(
    model: str = "Mistral",
    language: str = "en",
    sides: list[str] | tuple[str, ...] = tuple(SIDE_ORDER),
    coef_cmap_name: str = "plasma",
    coef_cmap_range: tuple[float, float] = (0.05, 0.95),
    filename_suffix: str = "",
) -> Path:
    root = MODEL_STANCE_ROOTS[model]
    lang_label = LANG_LABELS.get(language, language)
    slug = model_slug(model)

    item_order = item_order_from_baseline(root, model, language, side=sides[0])
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_order)}
    coef_colors = alpha_color_mapping(coef_cmap_name, start=coef_cmap_range[0], end=coef_cmap_range[1])

    fig, axes = plt.subplots(1, len(sides), figsize=(7.8 * len(sides), 4.6), sharey=True)
    if len(sides) == 1:
        axes = [axes]
    for ax, side in zip(axes, sides, strict=True):
        baseline_rows = load_item_rows(root, side, language, float(COEF_ORDER[0]))
        baseline = {
            row.get("item_id", "").strip(): normalize_stance(row.get("parsed_choice"))
            for row in baseline_rows
            if row.get("item_id", "").strip()
        }

        total_changed = 0
        for coef_idx, coef in enumerate(COEF_ORDER[1:], start=1):
            answers = load_item_rows(root, side, language, float(coef), required=False)
            changed_items = []
            for row in answers:
                item_id = row.get("item_id", "").strip()
                if not item_id:
                    continue
                if item_id not in baseline or item_id not in item_to_idx:
                    print(f"WARNING: {model} {language} {side} alpha={coef:g} has item_id not in baseline order: {item_id}; skipping")
                    continue
                stance = normalize_stance(row.get("parsed_choice"))
                if stance != baseline[item_id]:
                    changed_items.append(item_id)
            total_changed += len(changed_items)
            if changed_items:
                xs = [item_to_idx[item_id] for item_id in changed_items]
                ys = [coef_idx] * len(xs)
                ax.scatter(
                    xs,
                    ys,
                    s=70,
                    marker="s",
                    color=coef_colors[float(coef)],
                    edgecolor="#111111",
                    linewidth=0.9,
                    alpha=0.95,
                )

        ax.set_title(f"{model} {lang_label}: {side} steering (total Δ={total_changed})", fontweight="bold")
        ax.set_xlim(-0.5, len(item_order) - 0.5)
        ax.set_ylim(-0.6, len(COEF_ORDER) - 0.4)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(item_order)))
        ax.set_xticklabels([str(i + 1) for i in range(len(item_order))], fontsize=6, rotation=90)
        ax.set_xlabel("Item index (1-62)")
        ax.set_yticks(np.arange(len(COEF_ORDER)))
        ax.set_yticklabels([f"α={coef:g}" for coef in COEF_ORDER])
        ax.grid(axis="x", alpha=0.05)
        ax.grid(axis="y", alpha=0.15)

    alpha_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=coef_colors[float(coef)],
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=9,
            label=f"α={coef:g} (changed)",
        )
        for coef in COEF_ORDER[1:]
    ]
    fig.legend(
        handles=alpha_handles,
        loc="upper center",
        ncol=len(COEF_ORDER) - 1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(f"Only the items that change vs α=0 ({model} {lang_label})", y=1.08, fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.30, wspace=0.08)
    savefig(fig, f"figS2_item_level_changes_only_{slug}_{language}{filename_suffix}")
    return OUTPUT_DIR / f"figS2_item_level_changes_only_{slug}_{language}{filename_suffix}.png"


def write_item_answer_compare(
    model: str = "Mistral",
    language: str = "en",
    side: str = "left",
    coef_a: float = 0.0,
    coef_b: float = 0.8,
) -> Path:
    root = MODEL_STANCE_ROOTS[model]
    rows_a = load_item_rows(root, side, language, float(coef_a))
    rows_b = load_item_rows(root, side, language, float(coef_b))

    by_id_a = {row.get("item_id", "").strip(): row for row in rows_a if row.get("item_id", "").strip()}
    by_id_b = {row.get("item_id", "").strip(): row for row in rows_b if row.get("item_id", "").strip()}

    order = [row.get("item_id", "").strip() for row in rows_a if row.get("item_id", "").strip()]
    missing_in_b = [item_id for item_id in order if item_id not in by_id_b]
    if missing_in_b:
        print(f"WARNING: {model} {language} {side} alpha={coef_b:g} missing {len(missing_in_b)} item_id values from alpha={coef_a:g}")

    out_rows: list[dict[str, object]] = []
    for idx, item_id in enumerate(order, start=1):
        ra = by_id_a.get(item_id, {})
        rb = by_id_b.get(item_id, {})
        out_rows.append(
            {
                "item_index": idx,
                "item_id": item_id,
                "statement": (ra.get("statement") or rb.get("statement") or "").strip(),
                "parsed_choice_a": (ra.get("parsed_choice") or "").strip(),
                "raw_answer_a": (ra.get("raw_answer") or "").strip(),
                "parsed_choice_b": (rb.get("parsed_choice") or "").strip(),
                "raw_answer_b": (rb.get("raw_answer") or "").strip(),
                "changed": normalize_stance(ra.get("parsed_choice")) != normalize_stance(rb.get("parsed_choice")),
            }
        )

    out_df = pd.DataFrame(out_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"item_answer_compare_{model_slug(model)}_{language}_{side}_a{coef_a:g}_b{coef_b:g}.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def plot_item_level_stance_matrix_example(model: str = "Mistral", language: str = "en") -> Path:
    return plot_item_level_stance_matrix(model=model, language=language)


def plot_item_level_changes_only_example(model: str = "Mistral", language: str = "en") -> Path:
    return plot_changes_only(model=model, language=language)


def write_item_answer_comparison(
    model: str,
    language: str,
    side: str,
    coef_a: float,
    coef_b: float,
) -> Path:
    return write_item_answer_compare(model=model, language=language, side=side, coef_a=coef_a, coef_b=coef_b)


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shift_df = load_shift_data()
    detection_df = load_detection_data()
    item_coord_df = build_item_coordinate_change_table(shift_df)

    plot_detection_auc(detection_df)
    plot_english_displacement(shift_df)
    plot_multilingual_positions_by_language(shift_df)
    plot_item_coordinate_change_tables(item_coord_df)
    plot_detection_vs_behavior_summary(shift_df, detection_df)
    plot_crosslingual_heatmap_clean(shift_df)
    plot_focused_stance_distribution()
    for lang in LANG_ORDER:
        plot_item_level_stance_matrix(model="Llama-3", language=lang)
        plot_changes_only(model="Llama-3", language=lang)
        write_item_answer_compare(model="Llama-3", language=lang, side="left", coef_a=0.0, coef_b=0.8)
        write_item_answer_compare(model="Llama-3", language=lang, side="right", coef_a=0.0, coef_b=0.8)
    print(f"Wrote research-story figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
