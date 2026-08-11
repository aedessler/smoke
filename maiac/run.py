#!/usr/bin/env python3
"""CLI entry point for the MAIAC 25 km pipeline.

Resumable by construction (gcp-spot-batch-job skill, step 2):

  MEMBER  one calendar month -> one NetCDF in output/monthly/
  UNIT    one acquisition day -> one .npz in output/units/<month>/

Every unit is written with an atomic rename and skipped on restart if already
present, so a spot preemption costs at most the one day in flight. Months whose
NetCDF already exists locally or in Cloud Storage are skipped outright.

Writes .complete (strict: nothing failed) and .finished (terminal either way)
into the job directory for the self-stop watcher.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maiac_pipeline import config, conus_masks, granules as gran, manifest  # noqa: E402
from maiac_pipeline.process_month import process_month  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("maiac.run")

# The 22 MODIS sinusoidal tiles CMR returns for the CONUS bounding box,
# confirmed against a real query rather than assumed. Only 14 of them turn out
# to carry any CONUS land -- `conus_masks.useful_tiles` works out which at
# startup and the rest are never downloaded.
CONUS_TILES = [
    "h07v05", "h07v06", "h08v04", "h08v05", "h08v06",
    "h09v03", "h09v04", "h09v05", "h09v06",
    "h10v03", "h10v04", "h10v05", "h10v06",
    "h11v03", "h11v04", "h11v05", "h11v06",
    "h12v03", "h12v04", "h12v05", "h13v03", "h13v04",
]


def _months(args) -> list[str]:
    if args.months:
        return [m.strip() for m in args.months.split(",") if m.strip()]
    return gran.months_in_range(args.start, args.end)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=config.ARCHIVE_START, help="first month, YYYY-MM")
    p.add_argument("--end", default="2025-07", help="last month, YYYY-MM (inclusive)")
    p.add_argument("--months", default="", help="explicit comma-separated YYYY-MM list, overrides --start/--end")
    p.add_argument("--jobdir", default="/opt/maiac-25km")
    p.add_argument("--outdir", default="", help="defaults to <jobdir>/output")
    p.add_argument("--scratch", default="", help="raw HDF staging; defaults to <jobdir>/raw")
    p.add_argument("--cache", default="", help="tile masks + CONUS polygon; defaults to <jobdir>/cache")
    p.add_argument("--bucket", default="", help="gs://bucket for monthly checkpoints; empty disables GCS")
    p.add_argument("--gcs-prefix", default="maiac/monthly")
    p.add_argument("--workers", type=int, default=6, help="parallelism across MONTHS")
    p.add_argument("--threads", type=int, default=8, help="HTTP download threads per month worker")
    p.add_argument("--permissive-quality", action="store_true",
                   help="section 25 sensitivity run: accept AOD quality 0 and 11. "
                        "Never merge into the primary record -- use a separate --gcs-prefix.")
    p.add_argument("--keep-raw", action="store_true", help="do not delete raw HDFs (debugging)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list the work without touching the network")
    p.add_argument("--list-members", action="store_true")
    p.add_argument("--describe", default="", help="Phase A: print what is inside one HDF file and exit")
    args = p.parse_args(argv)

    outdir = args.outdir or os.path.join(args.jobdir, "output")
    scratch = args.scratch or os.path.join(args.jobdir, "raw")
    cache = args.cache or os.path.join(args.jobdir, "cache")
    quality = config.QUALITY_PERMISSIVE if args.permissive_quality else config.QUALITY_PRIMARY

    if args.describe:
        from maiac_pipeline import hdf_reader

        for k, v in hdf_reader.describe_granule(args.describe).items():
            print(f"{k}: {v}")
        return 0

    months = _months(args)
    if args.list_members:
        print("\n".join(months))
        return 0
    if args.dry_run:
        print(f"{len(months)} months: {months[0]} .. {months[-1]}")
        print(f"outdir={outdir} scratch={scratch} cache={cache}")
        print(f"bucket={args.bucket or '(none)'} workers={args.workers} threads={args.threads}")
        print(f"quality codes accepted: {quality}")
        return 0

    for d in (outdir, scratch, cache, os.path.join(outdir, "monthly")):
        os.makedirs(d, exist_ok=True)

    # Build every tile mask up front, single-threaded. Left to the pool, N
    # workers would race to create the same file on the first day of the run.
    # This also tells us which tiles are worth downloading at all.
    tiles = conus_masks.useful_tiles(CONUS_TILES, cache)
    log.info("%d/%d tiles carry CONUS land: %s", len(tiles), len(CONUS_TILES), " ".join(tiles))

    _clear_stale_tmp(outdir)

    log.info("processing %d months (%s .. %s) with %d workers",
             len(months), months[0], months[-1], args.workers)
    t0 = time.time()
    kwargs = dict(
        outdir=outdir, scratch=scratch, cache_dir=cache,
        bucket_uri=args.bucket or None, gcs_prefix=args.gcs_prefix,
        threads=args.threads, quality_set=quality,
        keep_raw=args.keep_raw, overwrite=args.overwrite,
        tiles=tuple(tiles),
    )

    results = []
    if args.workers <= 1:
        for m in months:
            results.append(_guard(m, kwargs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_guard, m, kwargs): m for m in months}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                log.info("%s   [%d/%d months done]", results[-1], done, len(months))

    manifest_csv = manifest.collate(outdir)
    failed = [r for r in results if "FAILED" in r]

    log.info("===== Summary =====")
    for r in sorted(results):
        log.info(r)
    log.info("manifest: %s", manifest_csv)
    log.info("%d/%d months OK, %d failed, %.1f min total",
             len(results) - len(failed), len(results), len(failed), (time.time() - t0) / 60)

    if args.bucket:
        from maiac_pipeline import gcs

        gcs.upload(manifest_csv, f"{args.bucket.rstrip('/')}/maiac/manifest.csv")

    # Markers for the self-stop watcher. .complete is strict so a partial run
    # never looks done; .finished is terminal either way so a failed run still
    # stops the VM instead of billing overnight.
    if not failed:
        open(os.path.join(args.jobdir, ".complete"), "w").close()
    open(os.path.join(args.jobdir, ".finished"), "w").close()
    return 1 if failed else 0


def _guard(month: str, kwargs: dict) -> str:
    """Never let one month's exception take down the pool."""
    try:
        return process_month(month, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("maiac.run").exception("month %s blew up", month)
        return f"[{month}] FAILED: {exc!r}"


def _clear_stale_tmp(outdir: str) -> None:
    """Drop half-written caches left by an interrupted run.

    A .tmp is by definition never trusted as finished, so removing one is
    always safe -- and leaving it costs disk for the rest of the run.
    """
    for pattern in ("units/*/*.tmp.npz", "monthly/*.tmp", "manifest/*.tmp"):
        for path in glob.glob(os.path.join(outdir, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
