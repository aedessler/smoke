"""Authenticated HTTPS transfer of granules from NASA Earthdata Cloud.

Deliberately *not* direct S3. NASA's protected buckets only honour the
temporary S3 credentials from inside AWS us-west-2; from a Google VM the
external HTTPS links are the supported path, and `earthaccess.download`
already handles the Earthdata Login redirect dance that plain `requests`
gets wrong (auth headers are stripped across the URS redirect).
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("maiac.download")

_AUTH = None


def ensure_auth():
    """Log in to Earthdata once per process, from ~/.netrc.

    The password never appears in this repo, in the job code, or in a CLI
    argument -- it is read from a 0600 .netrc the startup script materialises
    from Secret Manager at boot.
    """
    global _AUTH
    if _AUTH is None:
        import earthaccess

        _AUTH = earthaccess.login(strategy="netrc", persist=False)
        if not getattr(_AUTH, "authenticated", False):
            raise RuntimeError(
                "Earthdata authentication failed -- check ~/.netrc has a "
                "'machine urs.earthdata.nasa.gov' entry"
            )
    return _AUTH


def download_granules(granules, dest: str, threads: int = 8, attempts: int = 3) -> list[str]:
    """Fetch `granules` into `dest`, returning the local .hdf paths present.

    Retries the whole short-fall rather than individual files: earthaccess
    skips anything already on disk, so a retry only re-fetches what is actually
    missing.
    """
    import earthaccess

    ensure_auth()
    os.makedirs(dest, exist_ok=True)
    want = len(granules)

    for attempt in range(1, attempts + 1):
        try:
            earthaccess.download(granules, local_path=dest, threads=threads)
        except Exception as exc:
            log.warning("download attempt %d/%d raised: %s", attempt, attempts, exc)
        got = local_hdfs(dest)
        if len(got) >= want:
            return got
        if attempt < attempts:
            delay = 5 * 2 ** (attempt - 1)
            log.warning(
                "have %d/%d granules in %s; retrying in %ds",
                len(got), want, dest, delay,
            )
            time.sleep(delay)

    got = local_hdfs(dest)
    log.error("gave up with %d/%d granules in %s", len(got), want, dest)
    return got


def local_hdfs(dest: str) -> list[str]:
    """Non-empty .hdf files sitting in `dest`.

    Size-zero files are treated as absent: a transfer killed mid-write leaves
    one behind, and trusting it would silently drop a tile from the day.
    """
    if not os.path.isdir(dest):
        return []
    out = []
    for name in sorted(os.listdir(dest)):
        if not name.endswith(".hdf"):
            continue
        path = os.path.join(dest, name)
        try:
            if os.path.getsize(path) > 0:
                out.append(path)
        except OSError:
            pass
    return out
