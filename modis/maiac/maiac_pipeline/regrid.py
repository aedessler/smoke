"""Accumulate native-resolution sufficient statistics onto the 25 km grid.

Every field goes through the identical `np.bincount` weighting, which is what
makes the downstream ratios coverage-weighted means rather than means of means
(plan sections 15 and 17). float64 accumulators throughout: a month of CONUS
pixel-days is ~10^8 additions into 31,460 cells, and float32 would visibly
drift.
"""
from __future__ import annotations

import numpy as np

from . import config

FIELDS = ("A", "B", "C", "D")


def empty_accumulator() -> dict[str, np.ndarray]:
    return {k: np.zeros(config.TARGET_NCELLS, dtype="float64") for k in FIELDS}


def accumulate_tile(
    acc: dict[str, np.ndarray],
    cell_index: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    """Add one tile's native-resolution fields into a flat 25 km accumulator.

    `cell_index` is the cached (1200, 1200) int32 map from `conus_masks`, with
    -1 for pixels outside CONUS or off the target grid. Because MODIS
    sinusoidal tiles partition the globe without overlap, summing tile by tile
    never double-counts a source pixel.
    """
    keep = cell_index >= 0
    if not keep.any():
        return
    flat = cell_index[keep]
    for name in FIELDS:
        acc[name] += np.bincount(
            flat,
            weights=fields[name][keep].astype("float64", copy=False),
            minlength=config.TARGET_NCELLS,
        )


def add(acc: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> None:
    for name in FIELDS:
        acc[name] += other[name]


def as_2d(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.reshape(config.TARGET_NY, config.TARGET_NX) for k, v in acc.items()}


def monthly_ratios(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Turn summed sufficient statistics into the published monthly fields.

    Cells with a zero denominator become NaN rather than 0 -- "no valid
    retrieval anywhere this month" and "retrievals that averaged zero" are
    different statements and must not collapse together.
    """
    g = as_2d(acc)
    A, B, C, D = g["A"], g["B"], g["C"], g["D"]

    with np.errstate(invalid="ignore", divide="ignore"):
        valid_ok = B > 0
        smoke_ok = D > 0
        out = {
            "mean_aod_055": np.where(valid_ok, A / np.where(valid_ok, B, 1), np.nan),
            "mean_smoke_aod_055": np.where(
                smoke_ok, C / np.where(smoke_ok, D, 1), np.nan
            ),
            "smoke_aod_index": np.where(valid_ok, C / np.where(valid_ok, B, 1), np.nan),
            "smoke_pixel_day_fraction": np.where(
                valid_ok, D / np.where(valid_ok, B, 1), np.nan
            ),
            "valid_pixel_day_weight": B.copy(),
            "smoke_pixel_day_weight": D.copy(),
        }
    return out
