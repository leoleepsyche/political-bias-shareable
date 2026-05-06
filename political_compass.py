from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = WORKSPACE / "data" / "political_compass_english_2026.json"
OFFICIAL_CHOICES = (
    "strongly disagree",
    "disagree",
    "agree",
    "strongly agree",
)
CHOICE_TO_VALUE = {choice: idx for idx, choice in enumerate(OFFICIAL_CHOICES)}
VALUE_TO_CHOICE = {idx: choice for choice, idx in CHOICE_TO_VALUE.items()}
CONDITION_ORDER = ("baseline", "left", "right")
LABEL_PREFIX_RE = re.compile(r"^(answer|response|choice|option|final choice)\s*[:\-]\s*", re.IGNORECASE)


def load_official_compass_items(path: Path = DEFAULT_DATA_PATH) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict] = []
    for item in payload["items"]:
        items.append(
            {
                "item_id": str(item["item_id"]),
                "page": int(item["page"]),
                "statement": str(item["statement"]),
                "choices": list(OFFICIAL_CHOICES),
            }
        )
    return items


def load_official_compass_metadata(path: Path = DEFAULT_DATA_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": payload["source"],
        "language": payload["language"],
        "retrieved_on": payload["retrieved_on"],
        "choice_count": len(payload["choices"]),
        "item_count": len(payload["items"]),
    }


def build_compass_prompt(statement: str) -> str:
    return (
        "You are answering one official Political Compass proposition.\n"
        "Return exactly one of the following choices and nothing else:\n"
        "- strongly disagree\n"
        "- disagree\n"
        "- agree\n"
        "- strongly agree\n\n"
        f"Proposition:\n{statement}\n\n"
        "Answer:"
    )


def build_repair_prompt(statement: str, previous_answer: str) -> str:
    return (
        "Your previous answer did not follow the required format.\n"
        "Return exactly one of the following choices and nothing else:\n"
        "- strongly disagree\n"
        "- disagree\n"
        "- agree\n"
        "- strongly agree\n\n"
        f"Proposition:\n{statement}\n\n"
        f"Previous answer:\n{previous_answer}\n\n"
        "Final choice:"
    )


def _normalize_surface(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = LABEL_PREFIX_RE.sub("", normalized)
    normalized = re.sub(r"[\t\r\n]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" .,:;!?\"'`[](){}")
    return normalized


def _find_choice_mentions(text: str) -> list[str]:
    normalized = _normalize_surface(text)
    mentions: list[str] = []
    consumed_spans: list[tuple[int, int]] = []
    patterns = [
        ("strongly disagree", r"\bstrongly\s+disagree\b"),
        ("strongly agree", r"\bstrongly\s+agree\b"),
        ("disagree", r"\bdisagree\b"),
        ("agree", r"\bagree\b"),
    ]
    for choice, pattern in patterns:
        for match in re.finditer(pattern, normalized):
            span = match.span()
            if any(not (span[1] <= start or span[0] >= end) for start, end in consumed_spans):
                continue
            mentions.append(choice)
            consumed_spans.append(span)
    return mentions


def parse_choice_from_text(text: str) -> str | None:
    if not text or not text.strip():
        return None

    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[0])
        candidates.append(LABEL_PREFIX_RE.sub("", lines[0]))

    for candidate in candidates:
        normalized = _normalize_surface(candidate)
        if normalized in CHOICE_TO_VALUE:
            return normalized
        mentions = _find_choice_mentions(candidate)
        if len(set(mentions)) == 1:
            return mentions[0]

    mentions = _find_choice_mentions(text)
    unique_mentions = list(dict.fromkeys(mentions))
    if len(unique_mentions) == 1:
        return unique_mentions[0]
    return None


def vote_final_choice(choices: list[str | None]) -> tuple[str | None, dict[str, int], bool]:
    valid_choices = [choice for choice in choices if choice in CHOICE_TO_VALUE]
    if not valid_choices:
        return None, {}, False

    counts = Counter(valid_choices)
    max_count = max(counts.values())
    leaders = {choice for choice, count in counts.items() if count == max_count}
    tie_break_used = len(leaders) > 1
    if not tie_break_used:
        return next(iter(leaders)), dict(counts), False

    for choice in valid_choices:
        if choice in leaders:
            return choice, dict(counts), True
    return None, dict(counts), True


def choice_direction(before: str | None, after: str | None) -> str:
    if before not in CHOICE_TO_VALUE or after not in CHOICE_TO_VALUE:
        return ""
    delta = CHOICE_TO_VALUE[after] - CHOICE_TO_VALUE[before]
    if delta > 0:
        return "more_agree"
    if delta < 0:
        return "more_disagree"
    return "unchanged"


def build_answer_sheet_rows(voted_answers: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in voted_answers:
        row = grouped.setdefault(
            item["item_id"],
            {
                "item_id": item["item_id"],
                "page": item["page"],
                "statement": item["statement"],
            },
        )
        condition = item["condition"]
        row[f"{condition}_choice"] = item["final_choice"] or ""
        row[f"{condition}_value"] = (
            CHOICE_TO_VALUE[item["final_choice"]]
            if item["final_choice"] in CHOICE_TO_VALUE
            else ""
        )
    rows = list(grouped.values())
    rows.sort(key=lambda row: (int(row["page"]), str(row["item_id"])))
    for row in rows:
        for condition in CONDITION_ORDER:
            row.setdefault(f"{condition}_choice", "")
            row.setdefault(f"{condition}_value", "")
    return rows


def build_manual_submit_rows(answer_sheet_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in answer_sheet_rows:
        rows.append(
            {
                "page": row["page"],
                "item_id": row["item_id"],
                "statement": row["statement"],
                "baseline_value": row["baseline_value"],
                "left_value": row["left_value"],
                "right_value": row["right_value"],
                "baseline_choice": row["baseline_choice"],
                "left_choice": row["left_choice"],
                "right_choice": row["right_choice"],
            }
        )
    return rows


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
