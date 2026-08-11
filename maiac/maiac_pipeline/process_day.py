"""One day of CONUS tiles -> four 25 km sufficient-statistic arrays.

This is the resumable UNIT. Everything it needs is derivable from its
arguments, it writes exactly one small cache file atomically, and it deletes
its own raw HDF scratch on the way out -- so losing one to a spot preemption
costs at most a few hundred MB of re-transfer, never a month of work.
"""
from __future__ import annotations

import logging
import os
import shutil

import numpy as np

from . import config, conus_masks, download, granules as gran, hdf_reader, qa, regrid

log = logging.getLogger("maiac.day")


def unit_cache_path(outdir: str, month: str, day: str) -> str:
    return os.path.join(outdir, "units", month, f"{day}.npz")


def load_unit(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in regrid.FIELDS}


def process_day(
    month: str,
    day: str,
    day_granules: list,
    outdir: str,
    scratch: str,
    cache_dir: str,
    threads: int = 8,
    quality_set: tuple[int, ...] = config.QUALITY_PRIMARY,
    keep_raw: bool = False,
) -> dict:
    """Download, process and cache one acquisition day.

    Returns a small summary dict for the run log; the arrays themselves go to
    the unit cache so a restart can pick them up without recomputing.
    """
    cache_path = unit_cache_path(outdir, month, day)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    day_scratch = os.path.join(scratch, month, day)
    n_expected = len(day_granules)

    paths = download.download_granules(day_granules, day_scratch, threads=threads)

    acc = regrid.empty_accumulator()
    n_read = 0
    failures = []
    for path in paths:
        try:
            gid = gran.parse_granule_name(path)
            aod_raw, qa_raw = hdf_reader.read_granule(path)
            fields = qa.collapse_to_pixel_day(aod_raw, qa_raw, quality_set)
            cell_index = conus_masks.tile_cell_index(gid.h, gid.v, cache_dir)
            regrid.accumulate_tile(acc, cell_index, fields)
            n_read += 1
        except Exception as exc:
            failures.append(f"{os.path.basename(path)}: {exc}")
            log.warning("skipping %s: %s", os.path.basename(path), exc)

    _write_unit_atomic(cache_path, acc, n_expected, n_read)

    if not keep_raw:
        shutil.rmtree(day_scratch, ignore_errors=True)

    summary = {
        "day": day,
        "n_expected": n_expected,
        "n_downloaded": len(paths),
        "n_read": n_read,
        "failures": failures,
    }
    if n_read < n_expected:
        log.warning(
            "[%s] %s: only %d/%d tiles contributed", month, day, n_read, n_expected
        )
    return summary


def _write_unit_atomic(path: str, acc: dict, n_expected: int, n_read: int) -> None:
    """Write the unit cache via a .tmp + os.replace.

    Without the rename a preemption mid-write leaves a truncated .npz that
    looks finished on the next boot and gets silently trusted -- the one
    failure mode that would quietly corrupt a month.
    """
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        n_expected=np.int32(n_expected),
        n_read=np.int32(n_read),
        **{k: acc[k] for k in regrid.FIELDS},
    )
    os.replace(tmp, path)


def unit_tile_coverage(path: str) -> tuple[int, int]:
    """(n_read, n_expected) recorded in a cached unit."""
    with np.load(path) as z:
        return int(z["n_read"]), int(z["n_expected"])
