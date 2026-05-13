#!/usr/bin/env python3
"""
Merge detection_output_*.csv files from experiment folders with metadata columns.

Edit EXPERIMENT_CONFIG below then run:
  python merge_detection_csvs.py
  python merge_detection_csvs.py --output path/to/merged.csv

Each row is prefixed with experiment (folder name) and fly_id (from detection_output_N.csv).
Gender is inferred from the numeric suffix N in detection_output_N.csv using
male_id_range and female_id_range per folder.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_CONFIG: dict[str, dict] = {
    "sd_09_12": {
        "age": "young",
        "male_id_range": (1, 6),
        "female_id_range": (7, 12),
    },
    "sd_09_17": {
        "age": "young",
        "male_id_range": (1, 6),
        "female_id_range": (7, 12),
    },
    "sd_09_19": {
        "age": "young",
        "male_id_range": (1, 6),
        "female_id_range": (7, 12),
    },
    "sd_10_28": {
        "age": "old",
        "male_id_range": (1, 8),
        "female_id_range": (9, 16),
    },
    "sd_10_30": {
        "age": "old",
        "male_id_range": (1, 8),
        "female_id_range": (9, 16),
    },
}

DEFAULT_GENOTYPE = "WIG"
FILENAME_RE = re.compile(r"detection_output_(\d+)\.csv$", re.IGNORECASE)


def infer_gender(fly_id: int, male_rng: tuple[int, int], female_rng: tuple[int, int]) -> str:
    lo_m, hi_m = male_rng
    lo_f, hi_f = female_rng
    if lo_m <= fly_id <= hi_m:
        return "male"
    if lo_f <= fly_id <= hi_f:
        return "female"
    return "unknown"


def parse_fly_id(path: Path) -> int | None:
    m = FILENAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def merge_experiments(
    base_dir: Path,
    config: dict[str, dict],
    genotype: str = DEFAULT_GENOTYPE,
) -> tuple[list[str], list[list[str]], int]:
    """Returns (header_row, data_rows, unknown_gender_row_count)."""
    rows_out: list[list[str]] = []
    header: list[str] | None = None
    unknown_rows = 0

    for folder_name, meta in sorted(config.items()):
        exp_dir = base_dir / folder_name
        if not exp_dir.is_dir():
            continue
        male_rng = tuple(meta["male_id_range"])
        female_rng = tuple(meta["female_id_range"])
        age = meta["age"]
        for csv_path in sorted(exp_dir.glob("detection_output_*.csv")):
            fly_id = parse_fly_id(csv_path)
            if fly_id is None:
                continue
            gender = infer_gender(fly_id, male_rng, female_rng)
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = ["experiment", "fly_id", *file_header, "age", "gender", "genotype"]
                for row in reader:
                    if len(row) != len(file_header):
                        row = row + [""] * (len(file_header) - len(row))
                        row = row[: len(file_header)]
                    out = [folder_name, str(fly_id), *row, age, gender, genotype]
                    rows_out.append(out)
                    if gender == "unknown":
                        unknown_rows += 1

    if header is None:
        raise SystemExit("No CSV files found; check base_dir and folder names in config.")
    return header, rows_out, unknown_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Merge detection CSVs with age/gender/genotype.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing sd_* experiment folders (default: script directory).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=SCRIPT_DIR / "merged_detection_all.csv",
        help="Output merged CSV path.",
    )
    p.add_argument(
        "--genotype",
        default=DEFAULT_GENOTYPE,
        help=f"Value for genotype column (default: {DEFAULT_GENOTYPE}).",
    )
    args = p.parse_args()

    header, rows, unknown = merge_experiments(
        args.base_dir,
        EXPERIMENT_CONFIG,
        genotype=args.genotype,
    )
    if unknown:
        print(f"Warning: {unknown} rows have gender 'unknown' (fly_id outside configured ranges).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
