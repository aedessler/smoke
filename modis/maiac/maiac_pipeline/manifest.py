"""Per-month run records, and the CSV assembled from them.

One JSON file per month rather than appended rows in a shared CSV: several
month workers run as separate processes, and concurrent appends to one file
interleave and corrupt each other. Each worker owns exactly one path, writes it
atomically, and the main process collates at the end.
"""
from __future__ import annotations

import csv
import glob
import json
import os

FIELDS = (
    "month",
    "status",
    "n_granules",
    "download_gb",
    "n_days",
    "n_days_partial",
    "coverage_pct",
    "mean_aod_median",
    "smoke_fraction_mean",
    "total_valid_pixel_days",
    "elapsed_s",
    "output_file",
    "note",
)


def record_path(outdir: str, month: str) -> str:
    return os.path.join(outdir, "manifest", f"{month}.json")


def write_record(outdir: str, month: str, record: dict) -> None:
    path = record_path(outdir, month)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"month": month, **record}, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_record(outdir: str, month: str) -> dict | None:
    path = record_path(outdir, month)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def collate(outdir: str) -> str:
    """Assemble manifest.csv from every per-month JSON record."""
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, "manifest", "*.json"))):
        try:
            with open(path) as fh:
                rows.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    out = os.path.join(outdir, "manifest.csv")
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.get("month", "")):
            writer.writerow(row)
    os.replace(tmp, out)
    return out
