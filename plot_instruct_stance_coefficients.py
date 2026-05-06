#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import (
    ROOT,
    COEF_ORDER,
    LANG_ORDER,
    SIDE_ORDER,
    STANCE_COLORS,
    STANCE_ORDER,
    coef_label,
    read_answer_file,
    savefig as _savefig,
)

OUTPUT_DIR = ROOT / "outputs" / "figures_instruct_coefficients"

MODELS = {
    "Qwen2.5-7B-Instruct": ROOT
    / "outputs"
    / "ideology_840_mps"
    / "run_840_mps_promptfixed_20260423_141925",
    "Mistral-7B-Instruct-v0.2": ROOT
    / "outputs"
    / "multimodel_840_batch_20260424_103114_pty"
    / "mistralai_Mistral_7B_Instruct_v0_2",
    "Llama-3-8B-Instruct": ROOT
    / "outputs"
    / "multimodel_840_batch_20260424_103114_pty"
    / "meta_llama_Meta_Llama_3_8B_Instruct",
}

LANGUAGES = LANG_ORDER
SIDES = SIDE_ORDER
COEFS = COEF_ORDER


def build_long_table() -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, root in MODELS.items():
        for side in SIDES:
            for language in LANGUAGES:
                for coef in COEFS:
                    path = root / f"steering_{side}" / language / f"compass_answers_{coef_label(coef)}.csv"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    answers = read_answer_file(path)
                    total = len(answers)
                    parsed = 0
                    choice_values: list[float] = []
                    counts = {stance: 0 for stance in STANCE_ORDER}

                    for answer in answers:
                        choice = (answer.get("parsed_choice") or "").strip().lower()
                        if choice in counts and choice != "unparsed":
                            parsed += 1
                            counts[choice] += 1
                            value = answer.get("choice_value")
                            if value not in (None, ""):
                                choice_values.append(float(value))
                        else:
                            counts["unparsed"] += 1

                    mean_choice = float(np.mean(choice_values)) if choice_values else np.nan
                    for stance in STANCE_ORDER:
                        rows.append(
                            {
                                "model": model_name,
                                "root": str(root),
                                "side": side,
                                "language": language,
                                "coef": coef,
                                "stance": stance,
                                "count": counts[stance],
                                "total": total,
                                "parsed": parsed,
                                "parse_rate": parsed / total if total else np.nan,
                                "proportion_total": counts[stance] / total if total else np.nan,
                                "proportion_parsed": (
                                    counts[stance] / parsed
                                    if parsed and stance != "unparsed"
                                    else (0.0 if stance != "unparsed" else np.nan)
                                ),
                                "mean_choice": mean_choice,
                            }
                        )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, stem: str) -> None:
    _savefig(fig, stem, OUTPUT_DIR, dpi=220)


def plot_parse_rate(df: pd.DataFrame) -> None:
    meta = df.drop_duplicates(["model", "side", "language", "coef"])
    fig, axes = plt.subplots(len(MODELS), len(SIDES), figsize=(12, 10), sharex=True, sharey=True)
    for row_idx, model in enumerate(MODELS):
        for col_idx, side in enumerate(SIDES):
            ax = axes[row_idx, col_idx]
            sub = meta[(meta["model"] == model) & (meta["side"] == side)]
            table = sub.pivot(index="language", columns="coef", values="parse_rate").reindex(LANGUAGES)
            sns.heatmap(
                table,
                ax=ax,
                vmin=0,
                vmax=1,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                cbar=col_idx == len(SIDES) - 1,
            )
            ax.set_title(f"{model}\n{side} steering", fontsize=10, pad=8)
            ax.set_xlabel("Coefficient")
            ax.set_ylabel("Language" if col_idx == 0 else "")
    fig.suptitle("Parse reliability by model, language, steering side, and coefficient", y=0.98, fontsize=14)
    fig.subplots_adjust(top=0.88, hspace=0.55, wspace=0.2)
    savefig(fig, "parse_rate_heatmap")


def plot_delta_mean_choice(df: pd.DataFrame) -> None:
    meta = df.drop_duplicates(["model", "side", "language", "coef"])
    baseline = meta[meta["coef"] == 0.0][["model", "side", "language", "mean_choice"]].rename(
        columns={"mean_choice": "baseline_mean_choice"}
    )
    merged = meta.merge(baseline, on=["model", "side", "language"], how="left")
    merged["delta_mean_choice"] = merged["mean_choice"] - merged["baseline_mean_choice"]

    fig, axes = plt.subplots(len(MODELS), len(SIDES), figsize=(13, 12), sharex=True)
    palette = dict(zip(LANGUAGES, sns.color_palette("tab10", n_colors=len(LANGUAGES))))
    for row_idx, model in enumerate(MODELS):
        for col_idx, side in enumerate(SIDES):
            ax = axes[row_idx, col_idx]
            sub = merged[(merged["model"] == model) & (merged["side"] == side)]
            for language in LANGUAGES:
                lang = sub[sub["language"] == language].sort_values("coef")
                ax.plot(
                    lang["coef"],
                    lang["delta_mean_choice"],
                    marker="o",
                    linewidth=1.8,
                    label=language,
                    color=palette[language],
                )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set_title(f"{model}\n{side} steering", fontsize=10, pad=8)
            ax.set_xlabel("Coefficient")
            ax.set_ylabel("Delta mean stance score" if col_idx == 0 else "")
            ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(LANGUAGES), frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Dose-response: change in mean stance score from coef=0", y=0.985, fontsize=15)
    fig.subplots_adjust(top=0.88, hspace=0.62, wspace=0.2)
    savefig(fig, "delta_mean_choice_by_coef")


def plot_stacked_for_model(df: pd.DataFrame, model: str) -> None:
    sub = df[df["model"] == model]
    fig, axes = plt.subplots(len(SIDES), 1, figsize=(14, 6.8), sharey=True)
    x_labels = [f"{language}\n{coef:g}" for language in LANGUAGES for coef in COEFS]
    x = np.arange(len(x_labels))

    for ax, side in zip(axes, SIDES):
        side_df = sub[sub["side"] == side]
        bottom = np.zeros(len(x_labels))
        for stance in STANCE_ORDER:
            values = []
            for language in LANGUAGES:
                for coef in COEFS:
                    row = side_df[
                        (side_df["language"] == language)
                        & (side_df["coef"] == coef)
                        & (side_df["stance"] == stance)
                    ]
                    values.append(float(row["proportion_total"].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=STANCE_COLORS[stance],
                edgecolor="white",
                linewidth=0.35,
                label=stance,
            )
            bottom += np.array(values)

        for boundary in range(len(COEFS), len(x_labels), len(COEFS)):
            ax.axvline(boundary - 0.5, color="#cccccc", linewidth=0.8)
        ax.set_title(f"{side.capitalize()} steering")
        ax.set_ylabel("Share of all 62 questions")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=len(STANCE_ORDER), frameon=False)
    fig.suptitle(f"Stance distribution across coefficients: {model}", y=1.03)
    safe_model = model.lower().replace("/", "_").replace(" ", "_").replace(".", "_")
    savefig(fig, f"stance_stacked_{safe_model}")


def plot_radar_coef4(df: pd.DataFrame) -> None:
    # Radar is intentionally restricted to coef=4 to keep the figure readable.
    radar_stances = STANCE_ORDER[:-1]
    radar_labels = ["Strongly\nDisagree", "Disagree", "Agree", "Strongly\nAgree"]
    angles = np.linspace(0, 2 * np.pi, len(radar_stances), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        len(MODELS),
        len(SIDES),
        figsize=(11, 14),
        subplot_kw={"projection": "polar"},
    )
    palette = dict(zip(LANGUAGES, sns.color_palette("tab10", n_colors=len(LANGUAGES))))
    for row_idx, model in enumerate(MODELS):
        for col_idx, side in enumerate(SIDES):
            ax = axes[row_idx, col_idx]
            sub = df[(df["model"] == model) & (df["side"] == side) & (df["coef"] == 4.0)]
            for language in LANGUAGES:
                vals = []
                for stance in radar_stances:
                    row = sub[(sub["language"] == language) & (sub["stance"] == stance)]
                    vals.append(float(row["proportion_parsed"].iloc[0]) if not row.empty else 0.0)
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=1.5, label=language, color=palette[language])
                ax.fill(angles, vals, color=palette[language], alpha=0.04)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_labels, fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", pad=8)
            ax.set_title(f"{model}\n{side} steering, coef=4", va="bottom", fontsize=10, pad=22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(LANGUAGES), frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Radar view of parsed stance proportions at coefficient 4", y=0.995, fontsize=15)
    fig.subplots_adjust(top=0.9, hspace=0.75, wspace=0.35)
    savefig(fig, "radar_coef4_stance_proportions")


def plot_radar_coef4_by_model(df: pd.DataFrame) -> None:
    radar_stances = STANCE_ORDER[:-1]
    radar_labels = ["Strongly\nDisagree", "Disagree", "Agree", "Strongly\nAgree"]
    angles = np.linspace(0, 2 * np.pi, len(radar_stances), endpoint=False).tolist()
    angles += angles[:1]
    palette = dict(zip(LANGUAGES, sns.color_palette("tab10", n_colors=len(LANGUAGES))))

    for model in MODELS:
        fig, axes = plt.subplots(1, len(SIDES), figsize=(10.5, 4.9), subplot_kw={"projection": "polar"})
        for ax, side in zip(axes, SIDES):
            sub = df[(df["model"] == model) & (df["side"] == side) & (df["coef"] == 4.0)]
            for language in LANGUAGES:
                vals = []
                for stance in radar_stances:
                    row = sub[(sub["language"] == language) & (sub["stance"] == stance)]
                    vals.append(float(row["proportion_parsed"].iloc[0]) if not row.empty else 0.0)
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=1.7, label=language, color=palette[language])
                ax.fill(angles, vals, color=palette[language], alpha=0.04)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_labels, fontsize=8)
            ax.tick_params(axis="x", pad=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"{side.capitalize()} steering, coef=4", fontsize=11, pad=20)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(LANGUAGES), frameon=False, bbox_to_anchor=(0.5, 0.86))
        fig.suptitle(f"Radar stance proportions at strongest steering: {model}", y=0.98, fontsize=14)
        fig.subplots_adjust(top=0.68, wspace=0.35)
        safe_model = model.lower().replace("/", "_").replace(" ", "_").replace(".", "_")
        savefig(fig, f"radar_coef4_{safe_model}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_long_table()
    df.to_csv(OUTPUT_DIR / "stance_distribution_long.csv", index=False)
    plot_parse_rate(df)
    plot_delta_mean_choice(df)
    plot_radar_coef4(df)
    plot_radar_coef4_by_model(df)
    for model in MODELS:
        plot_stacked_for_model(df, model)

    print(f"Wrote figures and aggregate CSV to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
