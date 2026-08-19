"""AOD_QA bit decoding and the observation -> pixel-day collapse.

Pure numpy, no GDAL: this is the scientific core and it is unit-tested
directly (plan sections 12 and 13).
"""
from __future__ import annotations

import numpy as np

from . import config


def decode_qa(qa: np.ndarray) -> dict[str, np.ndarray]:
    """Split AOD_QA into the three fields the pipeline uses."""
    qa = qa.astype("uint16", copy=False)
    return {
        "surface": (qa >> config.QA_SURFACE_SHIFT) & config.QA_SURFACE_MASK,
        "quality": (qa >> config.QA_QUALITY_SHIFT) & config.QA_QUALITY_MASK,
        "model": (qa >> config.QA_MODEL_SHIFT) & config.QA_MODEL_MASK,
    }


def valid_masks(
    aod_raw: np.ndarray,
    qa: np.ndarray,
    quality_set: tuple[int, ...] = config.QUALITY_PRIMARY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid, smoke) boolean masks with the same shape as `aod_raw`.

    `valid` is the strict land / best-quality / non-negative-AOD mask.
    `smoke` is the subset of `valid` where MAIAC selected the smoke aerosol
    model. Note that this identifies the *conditions* under which the smoke
    model was chosen; Optical_Depth_055 remains total-column AOD, not a
    smoke-only retrieval.

    `quality_set` is a tuple so the section 25 sensitivity run ({0, 11}) shares
    this code path rather than forking it.
    """
    fields = decode_qa(qa)
    quality_ok = np.isin(fields["quality"], np.asarray(quality_set, dtype="uint16"))
    valid = (
        (fields["surface"] == config.SURFACE_LAND)
        & quality_ok
        & (aod_raw >= 0)
    )
    smoke = valid & (fields["model"] == config.AEROSOL_MODEL_SMOKE)
    return valid, smoke


def collapse_to_pixel_day(
    aod_raw: np.ndarray,
    qa: np.ndarray,
    quality_set: tuple[int, ...] = config.QUALITY_PRIMARY,
) -> dict[str, np.ndarray]:
    """Reduce (n_obs, ny, nx) observations to one value per native pixel-day.

    A daily MCD19A2 tile carries several Terra/Aqua overpasses stacked on the
    leading axis. Collapsing them here, before any spatial aggregation, is what
    keeps a pixel observed four times from outweighing one observed once.

    Returns the four sufficient statistics of plan section 15, already zeroed
    where the corresponding indicator is false:

        A  daily_aod        where valid_day, else 0
        B  1                where valid_day, else 0
        C  daily_smoke_aod  where smoke_day, else 0
        D  1                where smoke_day, else 0
    """
    aod_raw = _as_3d(aod_raw)
    qa = _as_3d(qa)
    if aod_raw.shape != qa.shape:
        raise ValueError(f"AOD {aod_raw.shape} and QA {qa.shape} shapes differ")

    valid, smoke = valid_masks(aod_raw, qa, quality_set)
    aod = aod_raw.astype("float32") * config.AOD_SCALE

    valid_count = valid.sum(axis=0)
    smoke_count = smoke.sum(axis=0)

    valid_day = valid_count > 0
    smoke_day = smoke_count > 0

    daily_aod = np.where(valid, aod, 0.0).sum(axis=0) / np.maximum(valid_count, 1)
    daily_smoke = np.where(smoke, aod, 0.0).sum(axis=0) / np.maximum(smoke_count, 1)

    return {
        "A": np.where(valid_day, daily_aod, 0.0).astype("float64"),
        "B": valid_day.astype("float64"),
        "C": np.where(smoke_day, daily_smoke, 0.0).astype("float64"),
        "D": smoke_day.astype("float64"),
    }


def _as_3d(a: np.ndarray) -> np.ndarray:
    """Normalise a tile to (n_obs, ny, nx).

    GDAL hands back a 2-D array when a day happens to have a single
    observation, which would otherwise silently reduce along the wrong axis.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a[None, :, :]
    if a.ndim == 3:
        return a
    raise ValueError(f"expected a 2-D or 3-D tile array, got shape {a.shape}")
