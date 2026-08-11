"""Cloud Storage checkpointing via the gcloud CLI.

Shelling out to `gcloud storage` rather than adding google-cloud-storage to the
conda environment: the VM already has an authenticated gcloud through its
attached service account, so this needs no key file and no extra dependency.
"""
from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger("maiac.gcs")


def _run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def enabled(bucket_uri: str | None) -> bool:
    return bool(bucket_uri)


def exists(uri: str) -> bool:
    return _run(["gcloud", "storage", "ls", uri], timeout=120).returncode == 0


def upload(local: str, uri: str, attempts: int = 3) -> bool:
    """Copy one file up, then confirm the object is actually there.

    A successful `cp` exit code is not on its own proof the object landed, so
    the existence check is a separate call -- the raw month is only deleted
    after this returns True.
    """
    for attempt in range(1, attempts + 1):
        proc = _run(["gcloud", "storage", "cp", local, uri])
        if proc.returncode == 0 and exists(uri):
            return True
        log.warning(
            "upload attempt %d/%d for %s failed: %s",
            attempt, attempts, uri, (proc.stderr or "").strip()[:300],
        )
    return False


def download(uri: str, local: str) -> bool:
    os.makedirs(os.path.dirname(local), exist_ok=True)
    return _run(["gcloud", "storage", "cp", uri, local]).returncode == 0


def month_uri(bucket_uri: str, prefix: str, filename: str) -> str:
    return f"{bucket_uri.rstrip('/')}/{prefix.strip('/')}/{filename}"
