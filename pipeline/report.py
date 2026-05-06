from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path

LEFT_COLOR = "#2563eb"
RIGHT_COLOR = "#dc2626"
GRID_COLOR = "#d1d5db"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#6b7280"
BG_COLOR = "#ffffff"


def coef_key(coef: float) -> str:
    return f"coef_{int(coef) if float(coef).is_integer() else coef}"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_answers(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def color_scale(value: float, vmax: float) -> str:
    if vmax <= 0:
        return "#f9fafb"
    ratio = max(0.0, min(1.0, value / vmax))
    light = 98 - ratio * 30
    sat = 25 + ratio * 45
    return f"hsl(210 {sat:.0f}% {light:.0f}%)"


def axis_scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    t = (value - src_min) / (src_max - src_min)
    return dst_min + t * (dst_max - dst_min)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe_sub(lhs, rhs) -> float | None:
    if not (is_number(lhs) and is_number(rhs)):
        return None
    return float(lhs) - float(rhs)


def safe_distance(x1, y1, x2, y2) -> float | None:
    if not all(is_number(v) for v in (x1, y1, x2, y2)):
        return None
    return math.hypot(float(x1) - float(x2), float(y1) - float(y2))


def format_optional(value, value_fmt: str) -> str:
    if not is_number(value):
        return "NA"
    return format(value, value_fmt)


def build_trajectory_svg(language: str, left_points: list[tuple[float, float | None, float | None]], right_points: list[tuple[float, float | None, float | None]]) -> str:
    width = 280
    height = 240
    margin = 28
    x_min, x_max = -8.0, 0.0
    y_min, y_max = -8.0, 0.0

    def project(pt: tuple[float, float | None, float | None]) -> tuple[float, float]:
        _, x, y = pt
        if not (is_number(x) and is_number(y)):
            raise ValueError("Cannot project point with missing coordinates")
        px = axis_scale(x, x_min, x_max, margin, width - margin)
        py = axis_scale(y, y_min, y_max, height - margin, margin)
        return px, py

    def valid_points(points: list[tuple[float, float | None, float | None]]) -> list[tuple[float, float | None, float | None]]:
        return [pt for pt in points if is_number(pt[1]) and is_number(pt[2])]

    def path(points: list[tuple[float, float | None, float | None]]) -> str:
        coords = [project(pt) for pt in valid_points(points)]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)

    def labels(points: list[tuple[float, float | None, float | None]], color: str) -> str:
        chunks: list[str] = []
        for coef, x, y in valid_points(points):
            px, py = project((coef, x, y))
            label = str(int(coef) if float(coef).is_integer() else coef)
            chunks.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" />'
                f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" font-size="10" fill="{color}">{escape(label)}</text>'
            )
        return "".join(chunks)

    grid_lines = []
    for tick in range(-8, 1, 2):
        gx = axis_scale(tick, x_min, x_max, margin, width - margin)
        gy = axis_scale(tick, y_min, y_max, height - margin, margin)
        grid_lines.append(f'<line x1="{gx:.1f}" y1="{margin}" x2="{gx:.1f}" y2="{height-margin}" stroke="{GRID_COLOR}" stroke-width="1" />')
        grid_lines.append(f'<line x1="{margin}" y1="{gy:.1f}" x2="{width-margin}" y2="{gy:.1f}" stroke="{GRID_COLOR}" stroke-width="1" />')
        grid_lines.append(f'<text x="{gx:.1f}" y="{height-8}" text-anchor="middle" font-size="10" fill="{MUTED_COLOR}">{tick}</text>')
        grid_lines.append(f'<text x="8" y="{gy+3:.1f}" text-anchor="start" font-size="10" fill="{MUTED_COLOR}">{tick}</text>')

    left_polyline = ""
    right_polyline = ""
    if len(valid_points(left_points)) >= 2:
        left_polyline = f'<polyline fill="none" stroke="{LEFT_COLOR}" stroke-width="2.5" points="{path(left_points)}" />'
    if len(valid_points(right_points)) >= 2:
        right_polyline = f'<polyline fill="none" stroke="{RIGHT_COLOR}" stroke-width="2.5" points="{path(right_points)}" />'

    no_data_note = ""
    if not valid_points(left_points) and not valid_points(right_points):
        no_data_note = f'<text x="{width/2:.1f}" y="{height/2:.1f}" text-anchor="middle" font-size="12" fill="{MUTED_COLOR}">No valid coordinates</text>'

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="{BG_COLOR}" stroke="#e5e7eb" />
      <text x="{margin}" y="18" font-size="14" font-weight="700" fill="{TEXT_COLOR}">{escape(language.upper())}</text>
      {''.join(grid_lines)}
      <rect x="{margin}" y="{margin}" width="{width - 2*margin}" height="{height - 2*margin}" fill="none" stroke="#9ca3af" />
      {left_polyline}
      {right_polyline}
      {labels(left_points, LEFT_COLOR)}
      {labels(right_points, RIGHT_COLOR)}
      {no_data_note}
      <text x="{width/2:.1f}" y="{height-2}" text-anchor="middle" font-size="10" fill="{MUTED_COLOR}">x: economic</text>
      <text x="14" y="{height/2:.1f}" text-anchor="middle" font-size="10" fill="{MUTED_COLOR}" transform="rotate(-90 14 {height/2:.1f})">y: social</text>
    </svg>
    """


def build_branch_cards(title: str, color: str, strongest_shift: dict, strongest_changed: dict) -> str:
    if strongest_shift is None:
        strongest_shift = {"language": "NA", "coef": "NA", "shift": "NA"}
    if strongest_changed is None:
        strongest_changed = {"language": "NA", "coef": "NA", "changed_items": "NA"}
    return f"""
    <section>
      <h2>{escape(title)}</h2>
      <div class="cards">
        <div class="card">
          <h3 style="color:{color}">Largest Coordinate Shift</h3>
          <div class="metric"><span>Language</span><strong>{escape(str(strongest_shift['language']).upper())}</strong></div>
          <div class="metric"><span>Coef</span><strong>{strongest_shift['coef']}</strong></div>
          <div class="metric"><span>Shift magnitude</span><strong>{strongest_shift['shift']}</strong></div>
        </div>
        <div class="card">
          <h3 style="color:{color}">Largest Answer Change</h3>
          <div class="metric"><span>Language</span><strong>{escape(str(strongest_changed['language']).upper())}</strong></div>
          <div class="metric"><span>Coef</span><strong>{strongest_changed['coef']}</strong></div>
          <div class="metric"><span>Changed items</span><strong>{strongest_changed['changed_items']}</strong></div>
        </div>
      </div>
    </section>
    """


def build_report(run_root: Path, output_dir: Path | None = None, logger=None) -> Path:
    run_root = run_root.resolve()
    output_dir = (output_dir or (run_root / "comparison")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_left = load_json(run_root / "detection_left" / "summary.json")
    detection_right = load_json(run_root / "detection_right" / "summary.json")
    steering_left = load_json(run_root / "steering_left" / "global_summary.json")
    steering_right = load_json(run_root / "steering_right" / "global_summary.json")

    languages = steering_left["languages"]
    coefs = steering_left["coefs"]

    coord_rows: list[dict] = []
    distance_rows: list[dict] = []
    changed_summary_rows: list[dict] = []
    changed_detail_rows: list[dict] = []
    branch_shift_rows: list[dict] = []
    branch_changed_rows: list[dict] = []

    max_distance = 0.0
    max_changed = 0
    strongest_distance = None
    strongest_changed = None
    branch_shift_max = {"left": 0.0, "right": 0.0}
    branch_changed_max = {"left": 0, "right": 0}
    strongest_branch_shift = {"left": None, "right": None}
    strongest_branch_changed = {"left": None, "right": None}

    steering_by_side = {"left": steering_left, "right": steering_right}

    for language in languages:
        baseline_answers = {
            side: load_answers(run_root / f"steering_{side}" / language / "compass_answers_coef_0.csv")
            for side in ("left", "right")
        }
        for coef in coefs:
            key = coef_key(coef)
            lscore = steering_left["scores"][language][key]
            rscore = steering_right["scores"][language][key]
            distance = safe_distance(
                lscore["x_economic"],
                lscore["y_social"],
                rscore["x_economic"],
                rscore["y_social"],
            )
            if distance is not None:
                max_distance = max(max_distance, distance)
            coord_rows.append(
                {
                    "language": language,
                    "coef": coef,
                    "left_x": lscore["x_economic"],
                    "left_y": lscore["y_social"],
                    "right_x": rscore["x_economic"],
                    "right_y": rscore["y_social"],
                    "distance": round(distance, 3) if distance is not None else None,
                    "left_parsed": lscore["parsed_count"],
                    "right_parsed": rscore["parsed_count"],
                }
            )

            left_answers = load_answers(run_root / "steering_left" / language / f"compass_answers_{key}.csv")
            right_answers = load_answers(run_root / "steering_right" / language / f"compass_answers_{key}.csv")
            changed = 0
            changed_examples = []
            for item_id in sorted(set(left_answers) & set(right_answers)):
                lrow = left_answers[item_id]
                rrow = right_answers[item_id]
                if lrow["parsed_choice"] != rrow["parsed_choice"]:
                    changed += 1
                    if len(changed_examples) < 6:
                        changed_examples.append(f"{item_id}: {lrow['parsed_choice'] or 'None'} -> {rrow['parsed_choice'] or 'None'}")
                    changed_detail_rows.append(
                        {
                            "language": language,
                            "coef": coef,
                            "item_id": item_id,
                            "left_choice": lrow["parsed_choice"],
                            "right_choice": rrow["parsed_choice"],
                            "left_raw": lrow["raw_answer"],
                            "right_raw": rrow["raw_answer"],
                        }
                    )
            max_changed = max(max_changed, changed)
            distance_rows.append(
                {
                    "language": language,
                    "coef": coef,
                    "distance": round(distance, 3) if distance is not None else None,
                }
            )
            changed_summary_rows.append(
                {
                    "language": language,
                    "coef": coef,
                    "changed_items": changed,
                    "examples": " | ".join(changed_examples),
                }
            )
            if distance is not None and (strongest_distance is None or distance > strongest_distance["distance"]):
                strongest_distance = {"language": language, "coef": coef, "distance": round(distance, 3)}
            if strongest_changed is None or changed > strongest_changed["changed_items"]:
                strongest_changed = {"language": language, "coef": coef, "changed_items": changed}

            for side, side_data in steering_by_side.items():
                score = side_data["scores"][language][key]
                base = side_data["scores"][language]["coef_0"]
                shift_dx = safe_sub(score["x_economic"], base["x_economic"])
                shift_dy = safe_sub(score["y_social"], base["y_social"])
                shift_mag = safe_distance(score["x_economic"], score["y_social"], base["x_economic"], base["y_social"])
                if shift_mag is not None:
                    branch_shift_max[side] = max(branch_shift_max[side], shift_mag)
                branch_shift_rows.append(
                    {
                        "ideology": side,
                        "language": language,
                        "coef": coef,
                        "baseline_x": base["x_economic"],
                        "baseline_y": base["y_social"],
                        "x": score["x_economic"],
                        "y": score["y_social"],
                        "delta_x": round(shift_dx, 3) if shift_dx is not None else None,
                        "delta_y": round(shift_dy, 3) if shift_dy is not None else None,
                        "shift": round(shift_mag, 3) if shift_mag is not None else None,
                    }
                )
                if shift_mag is not None and (strongest_branch_shift[side] is None or shift_mag > strongest_branch_shift[side]["shift"]):
                    strongest_branch_shift[side] = {"language": language, "coef": coef, "shift": round(shift_mag, 3)}

                side_answers = load_answers(run_root / f"steering_{side}" / language / f"compass_answers_{key}.csv")
                base_answers = baseline_answers[side]
                side_changed = 0
                side_examples = []
                for item_id in sorted(set(base_answers) & set(side_answers)):
                    brow = base_answers[item_id]
                    srow = side_answers[item_id]
                    if brow["parsed_choice"] != srow["parsed_choice"]:
                        side_changed += 1
                        if len(side_examples) < 6:
                            side_examples.append(f"{item_id}: {brow['parsed_choice'] or 'None'} -> {srow['parsed_choice'] or 'None'}")
                branch_changed_max[side] = max(branch_changed_max[side], side_changed)
                branch_changed_rows.append(
                    {
                        "ideology": side,
                        "language": language,
                        "coef": coef,
                        "changed_items": side_changed,
                        "examples": " | ".join(side_examples),
                    }
                )
                if strongest_branch_changed[side] is None or side_changed > strongest_branch_changed[side]["changed_items"]:
                    strongest_branch_changed[side] = {
                        "language": language,
                        "coef": coef,
                        "changed_items": side_changed,
                    }

    detection_rows = [
        {
            "ideology": "left",
            "best_layer": detection_left["best_layer_on_val"],
            "best_test_auc": round(detection_left["best_layer_test_metrics"]["auc"], 4),
            "aggregation_test_auc": round(detection_left["aggregation_test_metrics"]["auc"], 4),
            "best_test_acc": round(detection_left["best_layer_test_metrics"]["acc"], 2),
        },
        {
            "ideology": "right",
            "best_layer": detection_right["best_layer_on_val"],
            "best_test_auc": round(detection_right["best_layer_test_metrics"]["auc"], 4),
            "aggregation_test_auc": round(detection_right["aggregation_test_metrics"]["auc"], 4),
            "best_test_acc": round(detection_right["best_layer_test_metrics"]["acc"], 2),
        },
    ]

    write_csv(output_dir / "detection_summary.csv", detection_rows, ["ideology", "best_layer", "best_test_auc", "aggregation_test_auc", "best_test_acc"])
    write_csv(output_dir / "coordinate_comparison.csv", coord_rows, ["language", "coef", "left_x", "left_y", "right_x", "right_y", "distance", "left_parsed", "right_parsed"])
    write_csv(output_dir / "changed_items_summary.csv", changed_summary_rows, ["language", "coef", "changed_items", "examples"])
    write_csv(output_dir / "changed_items_detailed.csv", changed_detail_rows, ["language", "coef", "item_id", "left_choice", "right_choice", "left_raw", "right_raw"])
    write_csv(output_dir / "branch_shift_summary.csv", branch_shift_rows, ["ideology", "language", "coef", "baseline_x", "baseline_y", "x", "y", "delta_x", "delta_y", "shift"])
    write_csv(output_dir / "branch_changed_items_summary.csv", branch_changed_rows, ["ideology", "language", "coef", "changed_items", "examples"])

    distance_map = {(row["language"], row["coef"]): row["distance"] for row in distance_rows}
    changed_map = {(row["language"], row["coef"]): row["changed_items"] for row in changed_summary_rows}
    coord_map = {(row["language"], row["coef"]): row for row in coord_rows}
    branch_shift_map = {
        side: {(row["language"], row["coef"]): row["shift"] for row in branch_shift_rows if row["ideology"] == side}
        for side in ("left", "right")
    }
    branch_changed_map = {
        side: {(row["language"], row["coef"]): row["changed_items"] for row in branch_changed_rows if row["ideology"] == side}
        for side in ("left", "right")
    }

    trajectory_svgs = []
    for language in languages:
        left_points = [(coef, coord_map[(language, coef)]["left_x"], coord_map[(language, coef)]["left_y"]) for coef in coefs]
        right_points = [(coef, coord_map[(language, coef)]["right_x"], coord_map[(language, coef)]["right_y"]) for coef in coefs]
        trajectory_svgs.append(build_trajectory_svg(language, left_points, right_points))

    def build_heatmap(cell_map: dict[tuple[str, float], float], vmax: float, title: str, value_fmt: str) -> str:
        headers = "".join(f"<th>{escape(str(coef))}</th>" for coef in coefs)
        rows_html = []
        for language in languages:
            cells = []
            for coef in coefs:
                value = cell_map[(language, coef)]
                bg = color_scale(value, vmax) if is_number(value) else "#f9fafb"
                cells.append(f'<td style="background:{bg}">{format_optional(value, value_fmt)}</td>')
            rows_html.append(f"<tr><th>{escape(language.upper())}</th>{''.join(cells)}</tr>")
        return f"""
        <section>
          <h2>{escape(title)}</h2>
          <table class="heatmap">
            <thead><tr><th>Language</th>{headers}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </section>
        """

    def detection_card(row: dict) -> str:
        return f"""
        <div class="card">
          <h3>{escape(row['ideology'].upper())}</h3>
          <div class="metric"><span>Best layer</span><strong>{row['best_layer']}</strong></div>
          <div class="metric"><span>Best-layer test AUC</span><strong>{row['best_test_auc']}</strong></div>
          <div class="metric"><span>Aggregation test AUC</span><strong>{row['aggregation_test_auc']}</strong></div>
          <div class="metric"><span>Best-layer test ACC</span><strong>{row['best_test_acc']}</strong></div>
        </div>
        """

    top_change_rows = sorted(changed_summary_rows, key=lambda row: (-row["changed_items"], row["language"], row["coef"]))[:10]
    top_change_html = "".join(
        f"<tr><td>{escape(row['language'].upper())}</td><td>{row['coef']}</td><td>{row['changed_items']}</td><td>{escape(row['examples'])}</td></tr>"
        for row in top_change_rows
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>840 Left vs Right Comparison</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 24px;
      color: {TEXT_COLOR};
      background: #f8fafc;
    }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    section {{ margin: 22px 0 28px 0; }}
    .subtle {{ color: {MUTED_COLOR}; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 320px));
      gap: 16px;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      margin: 8px 0;
      gap: 12px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      align-items: center;
      margin: 8px 0 14px 0;
    }}
    .legend span::before {{
      content: '';
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 8px;
      border-radius: 999px;
    }}
    .legend .left::before {{ background: {LEFT_COLOR}; }}
    .legend .right::before {{ background: {RIGHT_COLOR}; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      border: 1px solid #e5e7eb;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: center;
      font-size: 13px;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 700;
    }}
    td {{
      min-width: 52px;
    }}
    .note {{
      background: white;
      border-left: 4px solid #94a3b8;
      padding: 12px 14px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>840-Pair Steering Report</h1>
  <p class="subtle">Run root: {escape(str(run_root))}</p>
  <p class="subtle">Detection root: {escape(str(run_root))}</p>
  <p class="subtle">Generated: {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>

  <section class="note">
    <div><strong>Summary</strong></div>
    <div>Detection best layers diverged: left = {detection_left['best_layer_on_val']}, right = {detection_right['best_layer_on_val']}.</div>
    <div>Left branch strongest baseline-relative shift: {strongest_branch_shift['left']['language'].upper()} at coef {strongest_branch_shift['left']['coef']} with shift {strongest_branch_shift['left']['shift']}.</div>
    <div>Right branch strongest baseline-relative shift: {strongest_branch_shift['right']['language'].upper()} at coef {strongest_branch_shift['right']['coef']} with shift {strongest_branch_shift['right']['shift']}.</div>
  </section>

  <section>
    <h2>Detection Overview</h2>
    <div class="cards">
      {''.join(detection_card(row) for row in detection_rows)}
    </div>
  </section>

  <section>
    <h2>Compass Plane Trajectories</h2>
    <div class="legend"><span class="left">Left steering</span><span class="right">Right steering</span></div>
    <div class="grid">
      {''.join(trajectory_svgs)}
    </div>
  </section>

  {build_branch_cards('Left Steering vs Left Baseline', LEFT_COLOR, strongest_branch_shift['left'], strongest_branch_changed['left'])}
  {build_heatmap(branch_shift_map['left'], branch_shift_max['left'], 'Left Coordinate Shift From Baseline', '.3f')}
  {build_heatmap(branch_changed_map['left'], float(branch_changed_max['left']), 'Left Changed Item Count vs Baseline', '.0f')}

  {build_branch_cards('Right Steering vs Right Baseline', RIGHT_COLOR, strongest_branch_shift['right'], strongest_branch_changed['right'])}
  {build_heatmap(branch_shift_map['right'], branch_shift_max['right'], 'Right Coordinate Shift From Baseline', '.3f')}
  {build_heatmap(branch_changed_map['right'], float(branch_changed_max['right']), 'Right Changed Item Count vs Baseline', '.0f')}

  <section>
    <h2>Top Changed Contexts</h2>
    <table>
      <thead><tr><th>Language</th><th>Coef</th><th>Changed items</th><th>Examples</th></tr></thead>
      <tbody>{top_change_html}</tbody>
    </table>
  </section>
</body>
</html>
"""

    report_path = output_dir / "left_right_report.html"
    report_path.write_text(html, encoding="utf-8")
    metadata = {
        "run_root": str(run_root),
        "detection_root": str(run_root),
        "output_dir": str(output_dir),
        "strongest_distance": strongest_distance,
        "strongest_changed": strongest_changed,
        "strongest_branch_shift": strongest_branch_shift,
        "strongest_branch_changed": strongest_branch_changed,
    }
    (output_dir / "report_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if logger is not None:
        logger.info("Report written to %s", report_path)
    return report_path
