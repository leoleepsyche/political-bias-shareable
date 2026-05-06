#!/usr/bin/env python3
"""Shared constants and utilities for all plot scripts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

# Canonical ordering used across all figures.
MODEL_ORDER = ["Qwen", "Mistral", "Llama-3"]
LANG_ORDER = ["en", "it", "fr", "es", "de"]
COEF_ORDER = [0.0, 0.8, 1.5, 2.5, 4.0]
SIDE_ORDER = ["left", "right"]

LANG_LABELS = {
    "en": "English",
    "it": "Italian",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}

# Okabe-Ito colorblind-safe palette.
SIDE_COLORS = {"left": "#0072B2", "right": "#D55E00"}
MODEL_COLORS = {"Qwen": "#009E73", "Mistral": "#0072B2", "Llama-3": "#D55E00"}
SIDE_MARKERS = {"left": "o", "right": "^"}

STANCE_ORDER = ["strongly disagree", "disagree", "agree", "strongly agree", "unparsed"]
STANCE_COLORS = {
    "strongly disagree": "#3B6FB6",
    "disagree": "#F28E2B",
    "agree": "#59A14F",
    "strongly agree": "#E15759",
    "unparsed": "#B8B8B8",
}


def setup_style() -> None:
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
        }
    )


def savefig(fig: plt.Figure, stem: str, output_dir: Path, *, dpi: int = 260) -> None:
    """Save figure as both PNG and PDF, then close."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def coef_label(coef: float) -> str:
    """Format a steering coefficient for use in filenames (e.g. 'coef_0', 'coef_0.8')."""
    return f"coef_{int(coef) if float(coef).is_integer() else coef}"


def read_answer_file(path: Path) -> list[dict[str, str]]:
    """Read a compass_answers CSV into a list of row dicts."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
