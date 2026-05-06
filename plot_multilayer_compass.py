#!/usr/bin/env python3
"""Plot compass positions for all-layer steering experiments, one figure per language."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from plot_utils import ROOT, LANG_LABELS, LANG_ORDER, SIDE_MARKERS, savefig as _savefig, setup_style

OUTPUT_DIR = ROOT / "outputs" / "figures_research_story"

MULTILAYER_LEFT = ROOT / "outputs" / "multilayer_experiments"
MULTILAYER_RIGHT = ROOT / "outputs" / "multilayer_experiments_right"

# Multilayer experiments use a different coefficient sweep than the main pipeline.
COEF_ORDER = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def coef_color_mapping() -> dict[float, str]:
    cmap = plt.get_cmap("plasma")
    nonzero = [c for c in COEF_ORDER if c != 0.0]
    colors = cmap(np.linspace(0.05, 0.95, len(nonzero)))
    mapping: dict[float, object] = {0.0: "#4D4D4D"}
    for i, c in enumerate(nonzero):
        mapping[c] = colors[i]
    return mapping


def load_multilayer_data() -> list[dict]:
    """Load all multilayer summary JSONs into a flat list of records."""
    rows = []
    for side, base in [("left", MULTILAYER_LEFT), ("right", MULTILAYER_RIGHT)]:
        for summary_path in sorted(base.rglob("multilayer_summary.json")):
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            lang = data.get("language", summary_path.parent.name)
            for label, sc in data["scores"].items():
                x = sc.get("x_economic")
                y = sc.get("y_social")
                coef_raw = sc.get("coef")
                if coef_raw is not None:
                    coef = coef_raw
                else:
                    coef = float(label.replace("coef_", ""))
                parsed = sc.get("parsed_count", 0)
                total = sc.get("total_questions", 62)
                if x is None or y is None:
                    continue
                rows.append({
                    "language": lang,
                    "ideology": side,
                    "coef": float(coef),
                    "x": float(x),
                    "y": float(y),
                    "parsed": parsed,
                    "total": total,
                    "model": data.get("model", "Llama-3"),
                })
    return rows


def jitter_xy(x, y, counts, eps=0.12, jitter=0.28):
    """Apply deterministic jitter to overlapping points."""
    offsets = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    ]
    gx, gy = int(round(x / eps)), int(round(y / eps))
    key = (gx, gy)
    idx = counts.get(key, 0)
    counts[key] = idx + 1
    if idx > 0 and idx < len(offsets):
        dx, dy = offsets[idx]
        x += dx * jitter
        y += dy * jitter
    return x, y


def plot_multilayer_by_language(rows: list[dict]) -> None:
    """One figure per language: 1 column (Llama-3 only), 2 rows (left/right)."""
    coef_colors = coef_color_mapping()

    alpha_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=coef_colors[c],
            markeredgecolor="white", markeredgewidth=0.9,
            markersize=9, label=f"\u03b1={c:g}",
        )
        for c in COEF_ORDER
    ]

    for lang in LANG_ORDER:
        lang_rows = [r for r in rows if r["language"] == lang]
        if not lang_rows:
            continue

        xs = [r["x"] for r in lang_rows]
        ys = [r["y"] for r in lang_rows]
        max_abs = max(max(abs(v) for v in xs), max(abs(v) for v in ys))
        lim = max(5.0, float(np.ceil((max_abs + 0.6) / 1.0)))
        major_step = 2.0 if lim <= 10.0 else 4.0
        minor_step = major_step / 2.0

        fig, axes = plt.subplots(
            2, 1, figsize=(5.0, 8.0),
            sharex=False, sharey=False,
            gridspec_kw={"hspace": 0.25},
        )

        for row_idx, side in enumerate(["left", "right"]):
            ax = axes[row_idx]
            side_rows = [r for r in lang_rows if r["ideology"] == side]
            side_rows.sort(key=lambda r: r["coef"])

            counts = {}
            for r in side_rows:
                coef = float(r["coef"])
                x, y = r["x"], r["y"]
                x, y = jitter_xy(x, y, counts)

                ax.scatter(
                    x, y, s=70,
                    marker=SIDE_MARKERS[side],
                    color=coef_colors.get(coef, "#777777"),
                    edgecolor="white", linewidth=1.0,
                    alpha=0.92, zorder=3,
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
            ax.set_xlabel("Economic")
            ax.set_ylabel("Social")
            side_label = "Left steering" if side == "left" else "Right steering"
            ax.set_title(side_label, fontsize=11, fontweight="bold")

        lang_label = LANG_LABELS.get(lang, lang)
        fig.suptitle(
            f"{lang_label}: All-Layer Compass Positions\n(Llama-3-8B-Instruct, 31 layers)",
            y=0.98, fontweight="bold",
        )
        fig.legend(
            handles=alpha_handles, loc="lower center",
            ncol=len(COEF_ORDER), frameon=False,
            fontsize=8.5, bbox_to_anchor=(0.5, -0.01),
        )
        fig.text(
            0.5, -0.04,
            "Each point is one \u03b1; color encodes steering coefficient and marker shape encodes steering side.",
            ha="center", fontsize=7.5, color="#666666",
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"fig_multilayer_compass_{lang}"
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=260, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem}")


def plot_multilayer_all_languages(rows: list[dict]) -> None:
    """Single figure: 2 rows (left/right) x 5 columns (languages), matching the original style."""
    coef_colors = coef_color_mapping()

    alpha_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=coef_colors[c],
            markeredgecolor="white", markeredgewidth=0.9,
            markersize=9, label=f"\u03b1={c:g}",
        )
        for c in COEF_ORDER
    ]

    fig, axes = plt.subplots(
        2, len(LANG_ORDER),
        figsize=(20.0, 7.5),
        sharex=False, sharey=False,
        gridspec_kw={"wspace": 0.24, "hspace": 0.22},
    )

    for col_idx, lang in enumerate(LANG_ORDER):
        lang_rows = [r for r in rows if r["language"] == lang]
        xs = [r["x"] for r in lang_rows] if lang_rows else [0]
        ys = [r["y"] for r in lang_rows] if lang_rows else [0]
        max_abs = max(max(abs(v) for v in xs), max(abs(v) for v in ys))
        lim = max(5.0, float(np.ceil((max_abs + 0.6))))
        major_step = 2.0 if lim <= 10.0 else 4.0
        minor_step = major_step / 2.0

        for row_idx, side in enumerate(["left", "right"]):
            ax = axes[row_idx, col_idx]
            side_rows = sorted(
                [r for r in lang_rows if r["ideology"] == side],
                key=lambda r: r["coef"],
            )

            counts = {}
            for r in side_rows:
                coef = float(r["coef"])
                x, y = r["x"], r["y"]
                x, y = jitter_xy(x, y, counts)

                ax.scatter(
                    x, y, s=70,
                    marker=SIDE_MARKERS[side],
                    color=coef_colors.get(coef, "#777777"),
                    edgecolor="white", linewidth=1.0,
                    alpha=0.92, zorder=3,
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
            if col_idx == 0:
                ax.set_ylabel("Social")

    # Column titles = language names
    for col_idx, lang in enumerate(LANG_ORDER):
        top_ax = axes[0, col_idx]
        pos = top_ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2, pos.y1 + 0.01,
            LANG_LABELS.get(lang, lang),
            ha="center", va="bottom", fontweight="bold",
        )

    # Row labels
    fig.text(0.02, 0.67, "Left steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")
    fig.text(0.02, 0.29, "Right steering", rotation=90, va="center", ha="center", fontsize=10, color="#333333")

    fig.suptitle(
        "All-Layer Steering: Compass Positions Across Languages (Llama-3-8B-Instruct, 31 layers)",
        y=0.985, fontweight="bold",
    )
    fig.legend(
        handles=alpha_handles, loc="upper center",
        ncol=len(COEF_ORDER), frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.text(
        0.5, 0.005,
        "Each point is one \u03b1; color encodes steering coefficient and marker shape encodes steering side.",
        ha="center", fontsize=8, color="#666666",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "fig_multilayer_compass_all_languages"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


if __name__ == "__main__":
    setup_style()
    rows = load_multilayer_data()
    print(f"Loaded {len(rows)} data points")
    plot_multilayer_by_language(rows)
    plot_multilayer_all_languages(rows)
