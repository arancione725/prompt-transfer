#!/usr/bin/env python3
"""Offline validation of RFA ranking against held-out target accuracy.

This module is intentionally separate from rfa.py.  Accuracy is loaded only
after an immutable selection report exists and can never affect selection.
"""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Mapping


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(first, second):
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in first_centered)
        * sum(value * value for value in second_centered)
    )
    if denominator == 0:
        return float("nan")
    return sum(
        first_value * second_value
        for first_value, second_value in zip(first_centered, second_centered)
    ) / denominator


def spearman(first, second):
    return _pearson(_rankdata(first), _rankdata(second))


def _normalize_record(value, fallback_key=None):
    if isinstance(value, (int, float)) and fallback_key is not None:
        key, accuracy = str(fallback_key), float(value)
    elif isinstance(value, Mapping):
        key = next(
            (
                str(value[field])
                for field in ("candidate", "id", "name")
                if value.get(field) not in (None, "")
            ),
            str(fallback_key) if fallback_key is not None else None,
        )
        if key is None or "accuracy" not in value:
            raise ValueError("Each record needs candidate and accuracy")
        accuracy = float(value["accuracy"])
    else:
        raise ValueError("Accuracy records must be mappings or name-to-number pairs")
    if not math.isfinite(accuracy):
        raise ValueError(f"Non-finite accuracy for {key}")
    return {"candidate": key, "accuracy": accuracy}


def load_accuracy(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Accuracy file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [_normalize_record(row) for row in csv.DictReader(stream)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_normalize_record(row) for row in payload]
    if isinstance(payload, Mapping):
        return [_normalize_record(value, key) for key, value in payload.items()]
    raise ValueError("Accuracy JSON must be a mapping or list")


def analyze(selection, accuracy_records):
    ranking = selection.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("Selection report contains no ranking")
    lookup = {}
    for record in accuracy_records:
        key = record["candidate"]
        if key in lookup and lookup[key] != record["accuracy"]:
            raise ValueError(f"Conflicting accuracy values for {key}")
        lookup[key] = record["accuracy"]
    joined = []
    for row in ranking:
        candidate = row["candidate"]
        if candidate not in lookup:
            raise ValueError(f"No accuracy found for {candidate!r}")
        combined = dict(row)
        combined["accuracy"] = lookup[candidate]
        joined.append(combined)
    correlation = spearman(
        [-float(row["rfa_score"]) for row in joined],
        [float(row["accuracy"]) for row in joined],
    )
    selected = next((row for row in joined if row.get("selected")), None)
    if selected is None:
        raise ValueError("Selection report does not mark a selected candidate")
    oracle = max(float(row["accuracy"]) for row in joined)
    selected_accuracy = float(selected["accuracy"])
    return joined, {
        "num_candidates": len(joined),
        "spearman_negative_rfa_accuracy": correlation,
        "selected_candidate": selected["candidate"],
        "selected_accuracy": selected_accuracy,
        "oracle_best_accuracy": oracle,
        "top1_selection_regret": oracle - selected_accuracy,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(output_dir, selection_path, accuracy_path, joined, summary):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"rfa_analysis_{stamp}.json"
    csv_path = output_dir / f"rfa_analysis_{stamp}.csv"
    columns = [
        "rank", "selected", "candidate", "checkpoint", "seed", "bridge_k",
        "rfa_score", "accuracy",
    ]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in joined])
    payload = {
        "selection_report": str(Path(selection_path).expanduser().resolve()),
        "accuracy_file": str(Path(accuracy_path).expanduser().resolve()),
        "accuracy_role": "offline_validation_only",
        "summary": _json_safe(summary),
        "candidates": joined,
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return json_path, csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--accuracy-file", required=True)
    parser.add_argument("--output-dir", default="outputs/rfa_analysis")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selection_path = Path(args.selection_json).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    joined, summary = analyze(selection, load_accuracy(args.accuracy_file))
    json_path, csv_path = save(
        args.output_dir,
        selection_path,
        args.accuracy_file,
        joined,
        summary,
    )
    rho = summary["spearman_negative_rfa_accuracy"]
    print(f"Spearman(-RFA, accuracy): {rho:.6f}" if math.isfinite(rho) else "Spearman: N/A")
    print(f"Top-1 selection regret:   {summary['top1_selection_regret']:.6f}")
    print(f"JSON: {json_path}\nCSV:  {csv_path}")


if __name__ == "__main__":
    main()
