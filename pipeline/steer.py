from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from repo_paths import resolve_repo_path

WORKSPACE = Path(__file__).resolve().parent.parent
OFFICIAL_REPO = resolve_repo_path(WORKSPACE, "neural_controllers_official")

if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from neural_controllers import NeuralController  # noqa: E402
from political_compass import CHOICE_TO_VALUE  # noqa: E402
from political_compass_multilingual import (  # noqa: E402
    ECONOMIC_WEIGHTS,
    SOCIAL_WEIGHTS,
    build_compass_prompt,
    build_repair_prompt,
    load_compass_items,
    load_compass_metadata,
    parse_choice,
)

COMPASS_DATA_FILES = {
    "en": WORKSPACE / "data" / "political_compass_english_2026.json",
    "it": WORKSPACE / "data" / "political_compass_italian_2026.json",
    "fr": WORKSPACE / "data" / "political_compass_french_2026.json",
    "es": WORKSPACE / "data" / "political_compass_spanish_2026.json",
    "de": WORKSPACE / "data" / "political_compass_german_2026.json",
}


def format_coef_label(coef: float) -> str:
    return f"coef_{int(coef) if float(coef).is_integer() else coef}"


def steering_complete(output_dir: Path) -> bool:
    return (output_dir / "global_summary.json").exists()


def load_complete_language_summary(lang_dir: Path, coefs: list[float]) -> dict[str, dict] | None:
    summary_path = lang_dir / "compass_summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for coef in coefs:
        label = format_coef_label(coef)
        if label not in summary:
            return None
        if not (lang_dir / f"compass_answers_{label}.csv").exists():
            return None
    return summary


def run_compass_with_steering(
    controller: NeuralController,
    compass_items: list[dict],
    language: str,
    steering_layer: int,
    coef: float,
    max_new_tokens: int = 30,
    max_retries: int = 1,
) -> list[dict]:
    """Answer all Political Compass questions with optional steering."""
    results = []
    layers = [steering_layer] if coef != 0.0 else []

    for item in compass_items:
        prompt_text = build_compass_prompt(item["statement"], language=language)
        formatted_prompt = controller.format_prompt(prompt_text, steer=True)

        raw_response = controller.generate(
            formatted_prompt,
            layers_to_control=layers,
            control_coef=coef,
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
                layers_to_control=layers,
                control_coef=coef,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            if repair_response.startswith(repair_prompt):
                repair_answer = repair_response[len(repair_prompt):].strip()
            else:
                repair_answer = repair_response.strip()
            choice = parse_choice(repair_answer, language=language)

        results.append(
            {
                "item_id": item["item_id"],
                "page": item["page"],
                "statement": item["statement"],
                "coef": coef,
                "raw_answer": answer_text[:200],
                "parsed_choice": choice,
                "choice_value": CHOICE_TO_VALUE.get(choice, None),
            }
        )

    return results


def compute_compass_score(results: list[dict]) -> dict:
    """Compute approximate Political Compass coordinates using shared item weights."""
    econ_raw = 0.0
    soc_raw = 0.0
    econ_max = 0.0
    soc_max = 0.0
    econ_items = 0
    social_items = 0

    for row in results:
        choice = row["parsed_choice"]
        if choice not in CHOICE_TO_VALUE:
            continue
        centered = CHOICE_TO_VALUE[choice] - 1.5
        item_id = row["item_id"]
        ew = ECONOMIC_WEIGHTS.get(item_id, 0.0)
        sw = SOCIAL_WEIGHTS.get(item_id, 0.0)
        econ_raw += ew * centered
        soc_raw += sw * centered
        econ_max += abs(ew) * 1.5
        soc_max += abs(sw) * 1.5
        if ew != 0.0:
            econ_items += 1
        if sw != 0.0:
            social_items += 1

    x_score = econ_raw / econ_max * 10 if econ_max else None
    y_score = soc_raw / soc_max * 10 if soc_max else None
    parsed = sum(1 for row in results if row["parsed_choice"] is not None)

    return {
        "x_economic": round(x_score, 3) if x_score is not None else None,
        "y_social": round(y_score, 3) if y_score is not None else None,
        "parsed_count": parsed,
        "total_questions": len(results),
        "econ_items": econ_items,
        "social_items": social_items,
    }


def run_steering_stage(
    *,
    controller: NeuralController,
    best_layer: int,
    positive_ideology: str,
    model_name: str,
    device: str,
    compute_device: str,
    control_method: str,
    prompt_format: str,
    languages: list[str],
    coefs: list[float],
    max_new_tokens: int,
    output_dir: Path,
    logger,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores: dict[str, dict] = {}
    actual_languages: list[str] = []

    for lang in languages:
        compass_path = COMPASS_DATA_FILES.get(lang)
        if compass_path is None or not compass_path.exists():
            logger.warning("Skipping language '%s': compass data not found at %s", lang, compass_path)
            continue

        compass_items = load_compass_items(lang, compass_path)
        compass_meta = load_compass_metadata(lang, compass_path)
        logger.info("[%s] Language %s (%s), %d questions", positive_ideology, lang, compass_meta["language"], len(compass_items))

        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        lang_scores: dict[str, dict] = {}

        existing_scores = load_complete_language_summary(lang_dir, coefs)
        if existing_scores is not None:
            logger.info("[%s][%s] Existing complete steering outputs found, skipping language", positive_ideology, lang)
            actual_languages.append(lang)
            all_scores[lang] = existing_scores
            continue

        for coef in coefs:
            label = format_coef_label(coef)
            logger.info("[%s][%s] Steering coef=%s (layer=%s)", positive_ideology, lang, coef, best_layer)
            results = run_compass_with_steering(
                controller,
                compass_items,
                language=lang,
                steering_layer=best_layer,
                coef=coef,
                max_new_tokens=max_new_tokens,
            )

            score = compute_compass_score(results)
            score["coef"] = coef
            score["language"] = lang
            lang_scores[label] = score

            logger.info(
                "[%s][%s][%s] X=%s Y=%s parsed=%s/%s",
                positive_ideology,
                lang,
                coef,
                score["x_economic"],
                score["y_social"],
                score["parsed_count"],
                score["total_questions"],
            )

            detail_path = lang_dir / f"compass_answers_{label}.csv"
            fieldnames = ["item_id", "page", "statement", "coef", "raw_answer", "parsed_choice", "choice_value"]
            with detail_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        actual_languages.append(lang)
        all_scores[lang] = lang_scores
        (lang_dir / "compass_summary.json").write_text(
            json.dumps(lang_scores, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    global_summary = {
        "model": model_name,
        "device": device,
        "compute_device": compute_device,
        "control_method": control_method,
        "prompt_format": prompt_format,
        "positive_ideology": positive_ideology,
        "best_layer": best_layer,
        "coefs": coefs,
        "languages": actual_languages,
        "scores": all_scores,
    }
    (output_dir / "global_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("[%s] Saved steering artifacts to %s", positive_ideology, output_dir)
    return global_summary
