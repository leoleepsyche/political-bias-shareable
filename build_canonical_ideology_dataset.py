#!/usr/bin/env python3
"""
Build a canonical ideology dataset for detection/steering runners from the
clean paired master table.

Input schema:
  topic,instruction_id,source_index,instruction,left_output,right_output,...

Output schema:
  instruction_id,topic,group,ideology,instruction,response_text

This output is directly compatible with:
  - run_compass_with_steering.py
  - run_official_neural_controller_detection.py
  - run_multilingual_compass_eval.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


TOPIC_TO_GROUP = {
    "crime_and_gun": "crime_and_guns",
    "economy_and_inequality": "economy_and_inequality",
    "gender_and_sexuality": "gender_and_sexuality",
    "immigration": "immigration",
    "race": "race_and_justice",
    "science": "science_and_environment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical ideology CSV from ideoinst_clean_master.csv"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/ideoinst_clean/ideoinst_clean_master.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pairs-per-topic",
        type=int,
        default=0,
        help="If > 0, keep at most this many paired examples per topic.",
    )
    return parser.parse_args()


def load_master_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def select_rows(rows: list[dict], pairs_per_topic: int) -> list[dict]:
    if pairs_per_topic <= 0:
        return rows

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["topic"]].append(row)

    selected: list[dict] = []
    for topic in sorted(grouped):
        topic_rows = sorted(grouped[topic], key=lambda row: row["instruction_id"])
        if len(topic_rows) < pairs_per_topic:
            raise ValueError(
                f"Topic {topic} has only {len(topic_rows)} clean pairs; "
                f"cannot take {pairs_per_topic}."
            )
        selected.extend(topic_rows[:pairs_per_topic])
    return selected


def to_canonical_rows(master_rows: list[dict]) -> list[dict]:
    canonical_rows: list[dict] = []
    for row in master_rows:
        instruction_id = row["instruction_id"].strip()
        topic = row["topic"].strip()
        instruction = row["instruction"].strip()
        group = TOPIC_TO_GROUP.get(topic, topic)

        left_text = row["left_output"].strip()
        right_text = row["right_output"].strip()
        if not instruction_id or not topic or not instruction:
            raise ValueError(f"Missing required fields for instruction_id={instruction_id!r}")
        if not left_text or not right_text:
            raise ValueError(f"Missing left/right text for instruction_id={instruction_id}")

        canonical_rows.append(
            {
                "instruction_id": instruction_id,
                "topic": topic,
                "group": group,
                "ideology": "left",
                "instruction": instruction,
                "response_text": left_text,
            }
        )
        canonical_rows.append(
            {
                "instruction_id": instruction_id,
                "topic": topic,
                "group": group,
                "ideology": "right",
                "instruction": instruction,
                "response_text": right_text,
            }
        )
    return canonical_rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instruction_id",
        "topic",
        "group",
        "ideology",
        "instruction",
        "response_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(master_rows: list[dict], canonical_rows: list[dict], output_csv: Path) -> None:
    topic_counts = Counter(row["topic"] for row in master_rows)
    ideology_counts = Counter(row["ideology"] for row in canonical_rows)
    print(f"Written: {output_csv}")
    print(f"Pairs: {len(master_rows)}")
    print(f"Rows:  {len(canonical_rows)}")
    print(f"Ideologies: {dict(sorted(ideology_counts.items()))}")
    print("Pairs by topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")


def main() -> None:
    args = parse_args()
    master_rows = load_master_rows(args.input_csv)
    selected_rows = select_rows(master_rows, args.pairs_per_topic)
    canonical_rows = to_canonical_rows(selected_rows)
    write_rows(canonical_rows, args.output_csv)
    print_stats(selected_rows, canonical_rows, args.output_csv)


if __name__ == "__main__":
    main()
