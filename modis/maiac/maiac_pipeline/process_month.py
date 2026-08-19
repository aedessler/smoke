"""One month -> one validated, uploaded NetCDF. This is the resumable MEMBER.

Sequence follows plan section 21 exactly, and the order matters:

    search -> dedupe -> per-day download+process -> accumulate -> ratios
    -> validate -> write -> upload -> confirm object exists -> delete raw

Raw HDFs are never deleted on the strength of a successful `cp` alone; the
object is confirmed present in Cloud Storage first.
"""
from __future__ import annotations

import logging
import os
import shutil
import time

from . import (
    config,
    gcs,
    granules as gran,
    manifest,
    process_day,
    regrid,
    write_netcdf,
)

log = logging.getLogger("maiac.month")


def output_filename(month: str) -> str:
    return f"maiac_smoke_25km_{month.replace('-', '_')}.nc"


def month_is_done(month: str, outdir: str, bucket_uri: str | None, gcs_prefix: str) -> bool:
    """Has this month already produced a usable NetCDF?

    Checks the local disk first (free), then Cloud Storage (survives losing the
    VM entirely). A local file that fails to open is treated as absent and
    reprocessed rather than trusted.
    """
    local = os.path.join(outdir, "monthly", output_filename(month))
    if os.path.exists(local) and _opens_cleanly(local):
        return True
    if gcs.enabled(bucket_uri):
        uri = gcs.month_uri(bucket_uri, gcs_prefix, output_filename(month))
        if gcs.exists(uri):
            return True
    return False


def _opens_cleanly(path: str) -> bool:
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            return "mean_aod_055" in ds and ds.sizes["x"] == config.TARGET_NX
    except Exception:
        return False


def process_month(
    month: str,
    outdir: str,
    scratch: str,
    cache_dir: str,
    bucket_uri: str | None = None,
    gcs_prefix: str = "maiac/monthly",
    threads: int = 8,
    quality_set: tuple[int, ...] = config.QUALITY_PRIMARY,
    keep_raw: bool = False,
    overwrite: bool = False,
    tiles: tuple[str, ...] | None = None,
) -> str:
    t0 = time.time()
    out_name = output_filename(month)
    local_out = os.path.join(outdir, "monthly", out_name)

    if not overwrite and month_is_done(month, outdir, bucket_uri, gcs_prefix):
        return f"[{month}] already complete, skipped"

    try:
        month_granules = gran.search_month(month)
    except Exception as exc:
        manifest.write_record(outdir, month, {"status": "failed", "note": f"CMR search: {exc}"})
        return f"[{month}] FAILED: CMR search: {exc}"

    if not month_granules:
        manifest.write_record(
            outdir, month, {"status": "empty", "n_granules": 0, "note": "no granules"}
        )
        return f"[{month}] no granules returned, skipped"

    # Drop tiles that hold no CONUS land before a single byte is transferred.
    if tiles:
        wanted = set(tiles)
        n_before = len(month_granules)
        month_granules = [
            g for g in month_granules
            if gran.parse_granule_name(gran.granule_name(g)).tile in wanted
        ]
        if n_before != len(month_granules):
            log.info(
                "[%s] %d/%d granules are on CONUS tiles", month, len(month_granules), n_before
            )

    by_day = gran.group_by_day(month_granules)
    download_gb = sum(gran.granule_size_mb(g) for g in month_granules) / 1024.0
    log.info(
        "[%s] %d granules over %d days (~%.1f GB)",
        month, len(month_granules), len(by_day), download_gb,
    )

    acc = regrid.empty_accumulator()
    n_partial = 0
    cached_units = 0

    for day, day_granules in by_day.items():
        cache_path = process_day.unit_cache_path(outdir, month, day)
        if os.path.exists(cache_path) and not overwrite:
            try:
                unit = process_day.load_unit(cache_path)
                n_read, n_exp = process_day.unit_tile_coverage(cache_path)
                regrid.add(acc, unit)
                cached_units += 1
                if n_read < n_exp:
                    n_partial += 1
                continue
            except Exception as exc:
                log.warning("[%s] cached unit %s unreadable (%s); redoing", month, day, exc)
                _safe_unlink(cache_path)

        summary = process_day.process_day(
            month, day, day_granules, outdir, scratch, cache_dir,
            threads=threads, quality_set=quality_set, keep_raw=keep_raw,
        )
        if summary["n_read"] < summary["n_expected"]:
            n_partial += 1
        regrid.add(acc, process_day.load_unit(cache_path))

    if cached_units:
        log.info("[%s] %d/%d days came from cache", month, cached_units, len(by_day))

    fields = regrid.monthly_ratios(acc)
    ds = write_netcdf.build_dataset(
        fields,
        month,
        {
            "n_granules": len(month_granules),
            "n_days": len(by_day),
            "n_days_partial_coverage": n_partial,
            "aod_quality_codes_accepted": ",".join(str(q) for q in quality_set),
        },
    )

    problems = write_netcdf.validate(ds)
    if problems:
        note = "; ".join(problems)
        manifest.write_record(
            outdir, month,
            {"status": "failed", "n_granules": len(month_granules),
             "download_gb": round(download_gb, 2), "note": f"validation: {note}"},
        )
        return f"[{month}] FAILED validation: {note}"

    diag = write_netcdf.diagnostics(ds)
    write_netcdf.write(ds, local_out)
    ds.close()

    uploaded = True
    if gcs.enabled(bucket_uri):
        uri = gcs.month_uri(bucket_uri, gcs_prefix, out_name)
        uploaded = gcs.upload(local_out, uri)
        if not uploaded:
            manifest.write_record(
                outdir, month,
                {"status": "failed", "output_file": local_out,
                 "note": "GCS upload/verify failed; raw scratch retained"},
            )
            return f"[{month}] FAILED: NetCDF written but GCS upload unverified"

    # Only now, with the object confirmed in place, is the raw month expendable.
    if not keep_raw:
        shutil.rmtree(os.path.join(scratch, month), ignore_errors=True)

    elapsed = time.time() - t0
    manifest.write_record(
        outdir, month,
        {
            "status": "complete",
            "n_granules": len(month_granules),
            "download_gb": round(download_gb, 2),
            "n_days": len(by_day),
            "n_days_partial": n_partial,
            "elapsed_s": round(elapsed, 1),
            "output_file": local_out,
            "note": "" if n_partial == 0 else f"{n_partial} days short of full tile coverage",
            **diag,
        },
    )
    return (
        f"[{month}] wrote {out_name} "
        f"({len(month_granules)} granules, {download_gb:.1f} GB, "
        f"{diag['coverage_pct']}% cells, {elapsed / 60:.1f} min)"
    )


def _safe_unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
