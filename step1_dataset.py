"""
Step 1: dataset utilities for ideology cosine and generation experiments.

This module defines a single canonical row schema:
- instruction_id
- topic
- ideology
- instruction
- response_text

Additional source columns are preserved, but downstream code should rely on the
canonical fields above.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

IDEOINST_TOPIC_ORDER = [
    "crime_and_gun",
    "economy_and_inequality",
    "gender_and_sexuality",
    "immigration",
    "race",
    "science",
]
VALID_IDEOLOGIES = {"left", "right"}


def _canonicalize_row(row: dict, source: str) -> dict:
    normalized = dict(row)

    instruction_id = str(row.get("instruction_id") or row.get("pair_id") or "").strip()
    if not instruction_id:
        raise ValueError(f"{source}: missing instruction_id/pair_id")

    topic = str(row.get("topic") or row.get("category") or "").strip()
    if not topic:
        raise ValueError(f"{source}: missing topic/category")

    ideology = str(row.get("ideology") or "").strip()
    if ideology not in VALID_IDEOLOGIES:
        raise ValueError(
            f"{source}: invalid ideology {ideology!r}; expected one of {sorted(VALID_IDEOLOGIES)}"
        )

    response_text = str(row.get("response_text") or row.get("text") or "").strip()
    if not response_text:
        raise ValueError(f"{source}: missing response_text/text")

    normalized["instruction_id"] = instruction_id
    normalized["topic"] = topic
    normalized["ideology"] = ideology
    normalized["instruction"] = str(row.get("instruction") or "").strip()
    normalized["response_text"] = response_text
    return normalized


def normalize_rows(rows: list[dict], source: str = "rows") -> list[dict]:
    """Normalize arbitrary IdeoINST-style rows into the canonical schema."""
    return [
        _canonicalize_row(row, source=f"{source} row {idx}")
        for idx, row in enumerate(rows, start=1)
    ]


def load_rows(path: Path) -> list[dict]:
    """Load a canonicalized CSV for ideology experiments."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return normalize_rows(rows, source=str(csv_path))


def ordered_topics(rows: list[dict], topic_order: list[str] | None = None) -> list[str]:
    """Return present topics in preferred order, followed by alphabetical fallback."""
    order = topic_order or IDEOINST_TOPIC_ORDER
    present_topics = {row["topic"] for row in rows}
    preferred = [topic for topic in order if topic in present_topics]
    fallback = sorted(present_topics - set(order))
    return preferred + fallback


def topic_sort_key(topic: str, topic_order: list[str] | None = None) -> tuple[int, str]:
    """Stable topic ordering helper for matched-pair outputs and summaries."""
    order = topic_order or IDEOINST_TOPIC_ORDER
    if topic in order:
        return (order.index(topic), topic)
    return (len(order), topic)


def allocate_topic_counts(rows: list[dict], target_total: int) -> dict[str, int]:
    """
    Allocate per-topic sample counts as evenly as possible.

    Topic-balanced sampling requires at least one row per present topic when
    `target_total` is positive. Smaller requests would silently bias toward the
    earliest topics in `IDEOINST_TOPIC_ORDER`, so they are rejected.
    """
    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    available_by_topic: dict[str, int] = {}
    for row in rows:
        available_by_topic[row["topic"]] = available_by_topic.get(row["topic"], 0) + 1

    topics = ordered_topics(rows)
    if not topics:
        raise ValueError("No topics found in input rows.")
    if 0 < target_total < len(topics):
        raise ValueError(
            f"Topic-balanced sampling requires at least one row per topic. "
            f"Requested {target_total} rows across {len(topics)} topics."
        )

    base = target_total // len(topics)
    remainder = target_total % len(topics)
    counts: dict[str, int] = {}
    for idx, topic in enumerate(topics):
        counts[topic] = base + (1 if idx < remainder else 0)
        if available_by_topic.get(topic, 0) < counts[topic]:
            raise ValueError(
                f"Topic {topic} only has {available_by_topic.get(topic, 0)} rows, "
                f"cannot allocate {counts[topic]}."
            )
    return counts


def select_rows(rows: list[dict], ideology: str, target_total: int) -> list[dict]:
    """Select topic-balanced rows for a single ideology."""
    if ideology not in VALID_IDEOLOGIES:
        raise ValueError(f"Unknown ideology {ideology!r}; expected one of {sorted(VALID_IDEOLOGIES)}")

    ideology_rows = [row for row in rows if row["ideology"] == ideology]
    ideology_rows.sort(key=lambda row: (topic_sort_key(row["topic"]), row["instruction_id"]))
    if target_total == 0:
        return ideology_rows

    counts = allocate_topic_counts(ideology_rows, target_total)
    grouped: dict[str, list[dict]] = {}
    for row in ideology_rows:
        grouped.setdefault(row["topic"], []).append(row)

    selected: list[dict] = []
    for topic in ordered_topics(ideology_rows):
        take = counts.get(topic, 0)
        if take:
            selected.extend(grouped[topic][:take])
    return selected


def _index_rows_by_instruction_id(rows: list[dict], ideology: str) -> dict[str, dict]:
    """
    Index rows by instruction_id and fail fast on duplicates.

    Silent overwrites make matched-pair analysis unreliable, so duplicates are
    treated as a data error instead of "last row wins".
    """
    if ideology not in VALID_IDEOLOGIES:
        raise ValueError(f"Unknown ideology {ideology!r}; expected one of {sorted(VALID_IDEOLOGIES)}")

    rows_by_id: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        if row.get("ideology") != ideology:
            continue
        instruction_id = str(row.get("instruction_id") or "").strip()
        if not instruction_id:
            raise ValueError(f"Found {ideology} row without instruction_id during pairing.")
        if instruction_id in rows_by_id:
            duplicate_ids.add(instruction_id)
            continue
        rows_by_id[instruction_id] = row

    if duplicate_ids:
        duplicate_list = ", ".join(sorted(duplicate_ids)[:5])
        raise ValueError(
            f"Duplicate {ideology} rows found for instruction_id(s): {duplicate_list}"
        )
    return rows_by_id


def build_paired_rows(rows: list[dict], strict: bool = True) -> list[tuple[dict, dict]]:
    """Build matched left/right row pairs keyed by `instruction_id`."""
    left_by_id = _index_rows_by_instruction_id(rows, "left")
    right_by_id = _index_rows_by_instruction_id(rows, "right")

    if strict:
        left_only = sorted(set(left_by_id) - set(right_by_id))
        right_only = sorted(set(right_by_id) - set(left_by_id))
        if left_only or right_only:
            details: list[str] = []
            if left_only:
                details.append(f"left-only={', '.join(left_only[:5])}")
            if right_only:
                details.append(f"right-only={', '.join(right_only[:5])}")
            raise ValueError(
                "Unpaired instruction_id(s) found while building matched rows: "
                + "; ".join(details)
            )

    common_ids = sorted(
        set(left_by_id) & set(right_by_id),
        key=lambda instruction_id: (
            topic_sort_key(left_by_id[instruction_id]["topic"]),
            instruction_id,
        ),
    )

    paired_rows: list[tuple[dict, dict]] = []
    for instruction_id in common_ids:
        left_row = left_by_id[instruction_id]
        right_row = right_by_id[instruction_id]

        if left_row["topic"] != right_row["topic"]:
            raise ValueError(
                f"Mismatched topics for instruction_id {instruction_id}: "
                f"left={left_row['topic']} right={right_row['topic']}"
            )

        left_instruction = str(left_row.get("instruction") or "").strip()
        right_instruction = str(right_row.get("instruction") or "").strip()
        if left_instruction and right_instruction and left_instruction != right_instruction:
            raise ValueError(
                f"Mismatched instructions for instruction_id {instruction_id}."
            )

        paired_rows.append((left_row, right_row))

    return paired_rows


def select_paired_rows(rows: list[dict], target_total: int, strict: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Select a topic-balanced matched subset of left/right rows.

    `target_total` is the desired number of rows per ideology after balancing
    across topics, not the total number of rows across both ideologies.
    """
    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    paired_rows = build_paired_rows(rows, strict=strict)
    if not paired_rows:
        raise ValueError("No matched left/right instruction pairs found in input rows.")
    if target_total == 0:
        left_rows = [left_row for left_row, _ in paired_rows]
        right_rows = [right_row for _, right_row in paired_rows]
        return left_rows, right_rows

    paired_left_rows = [left_row for left_row, _ in paired_rows]
    counts = allocate_topic_counts(paired_left_rows, target_total)

    grouped_pairs: dict[str, list[tuple[dict, dict]]] = {}
    for left_row, right_row in paired_rows:
        grouped_pairs.setdefault(left_row["topic"], []).append((left_row, right_row))

    selected_pairs: list[tuple[dict, dict]] = []
    for topic in ordered_topics(paired_left_rows):
        take = counts.get(topic, 0)
        if take:
            selected_pairs.extend(grouped_pairs[topic][:take])

    left_rows = [left_row for left_row, _ in selected_pairs]
    right_rows = [right_row for _, right_row in selected_pairs]
    return left_rows, right_rows


def prepare_rows(rows: list[dict], per_ideology: int, strict: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Prepare matched left/right rows for cosine experiments.

    If `per_ideology` is zero, all matched rows are returned. Otherwise a
    topic-balanced matched subset is selected.

    strict=True makes the pipeline fail fast if either side is missing for any
    instruction_id.
    """
    return select_paired_rows(rows, per_ideology, strict=strict)


def load_ideoinst(data_dir: Path) -> list[dict]:
    """
    Load IdeoINST data from local JSON/JSONL files into the canonical schema.
    """
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON/JSONL files found in {data_dir}. "
            f"Please download IdeoINST from: https://github.com/kaichen23/llm_ideo_manipulate"
        )

    rows: list[dict] = []
    fallback_pair_id = 0
    for json_file in sorted(json_files):
        with json_file.open("r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if content.startswith("["):
            entries = json.loads(content)
        else:
            entries = [json.loads(line) for line in content.splitlines() if line.strip()]

        for entry in entries:
            instruction_id = str(entry.get("instruction_id") or entry.get("pair_id") or fallback_pair_id)
            instruction = str(entry.get("instruction") or entry.get("question") or "").strip()
            topic = str(entry.get("topic") or entry.get("category") or "unknown").strip()

            left_text = str(entry.get("left_response") or entry.get("left") or "").strip()
            if left_text:
                rows.append(
                    {
                        "pair_id": instruction_id,
                        "instruction_id": instruction_id,
                        "topic": topic,
                        "group": topic,
                        "ideology": "left",
                        "instruction": instruction,
                        "response_text": left_text,
                    }
                )

            right_text = str(entry.get("right_response") or entry.get("right") or "").strip()
            if right_text:
                rows.append(
                    {
                        "pair_id": instruction_id,
                        "instruction_id": instruction_id,
                        "topic": topic,
                        "group": topic,
                        "ideology": "right",
                        "instruction": instruction,
                        "response_text": right_text,
                    }
                )

            fallback_pair_id += 1

    return normalize_rows(rows, source=str(data_dir))
