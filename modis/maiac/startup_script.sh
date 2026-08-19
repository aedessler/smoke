#!/bin/bash
# GCP startup script for the MAIAC 25 km spot-VM job.
#
# Runs as root on EVERY boot, including the boot after a spot preemption --
# that is what makes preemption recovery automatic rather than something a
# human has to notice. Adapted from the gcp-spot-batch-job skill template.
#
# Metadata keys it reads:
#   job-code        base64(tar.gz) of the pipeline package        (required)
#   job-args        CLI arguments for run.py                      (required)
#   job-max-runtime absolute backstop in seconds  (default 4 days)
#   retrieval-hold  "1" suppresses watcher+worker for one boot    (optional)
#
# The Earthdata credential is NOT a metadata key -- it lives in Secret Manager
# and is read here through the VM's attached service account, so the password
# never appears in instance metadata (visible to anyone with
# compute.instances.get) or in this repo.

set -u

JOBDIR=/opt/maiac-25km
OUT="$JOBDIR/output"
ENVDIR="$JOBDIR/env"
CODE_METADATA_KEY="job-code"
EARTHDATA_SECRET="earthdata-netrc"

mkdir -p "$JOBDIR" "$OUT"
exec >>"$JOBDIR/startup.log" 2>&1
echo "===== startup $(date -u) ====="

meta() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null
}

# A fatal setup problem must still stop the VM. Without this the worker never
# runs, no marker is ever written, and the machine bills until the backstop.
die() {
  echo "FATAL: $*"
  touch "$JOBDIR/.finished"
  exit 1
}

RUN_ARGS="$(meta job-args)"
[ -n "$RUN_ARGS" ] || die "job-args metadata key is empty"
MAX_RUNTIME_S="$(meta job-max-runtime)"
MAX_RUNTIME_S="${MAX_RUNTIME_S:-345600}"

# --- one-time environment setup (idempotent across restarts) ---------------
# conda-forge rather than apt: GDAL needs the HDF4 driver to read MCD19A2 at
# all, and since GDAL 3.8 conda-forge ships that as a separate libgdal-hdf4
# plugin. Debian/Ubuntu packaging of it is not dependable.
if [ ! -f "$JOBDIR/.setup_done" ]; then
  echo "--- installing micromamba + geospatial stack ---"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y curl bzip2 ca-certificates

  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local bin/micromamba || die "micromamba download failed"

  export MAMBA_ROOT_PREFIX="$JOBDIR/mamba"
  /usr/local/bin/micromamba create -y -p "$ENVDIR" -c conda-forge \
      python=3.12 \
      numpy pandas xarray netcdf4 h5netcdf \
      gdal libgdal-hdf4 \
      pyproj geopandas shapely pyogrio \
      requests tqdm pip \
    || die "micromamba env creation failed"

  "$ENVDIR/bin/pip" install --no-cache-dir earthaccess || die "earthaccess install failed"

  touch "$JOBDIR/.setup_done"
  echo "--- environment ready ---"
fi

# Hard gate: without an HDF4 driver every granule read fails, one file at a
# time, and the run would grind through the whole archive producing nothing.
if ! "$ENVDIR/bin/gdalinfo" --formats 2>/dev/null | grep -qi 'HDF4'; then
  die "GDAL in $ENVDIR has no HDF4 driver -- cannot read MCD19A2"
fi

# --- Earthdata credential from Secret Manager ------------------------------
if [ ! -s /root/.netrc ]; then
  gcloud secrets versions access latest --secret="$EARTHDATA_SECRET" \
    >/root/.netrc 2>"$JOBDIR/secret.err" \
    || die "could not read secret '$EARTHDATA_SECRET': $(cat "$JOBDIR/secret.err")"
  chmod 600 /root/.netrc
fi
grep -q 'urs.earthdata.nasa.gov' /root/.netrc \
  || die "/root/.netrc has no urs.earthdata.nasa.gov entry"

# --- fetch the pipeline code fresh from metadata on every boot -------------
# A single `gcloud compute instances add-metadata --metadata-from-file
# job-code=...` pushes new code with no image rebuild; it takes effect on the
# next boot, preemption restarts included.
echo "--- unpacking job code ---"
rm -rf "$JOBDIR/code"
mkdir -p "$JOBDIR/code"
meta "$CODE_METADATA_KEY" | base64 -d | tar -xz -C "$JOBDIR/code" \
  || die "could not unpack $CODE_METADATA_KEY"
[ -f "$JOBDIR/code/run.py" ] || die "job-code archive has no run.py at its root"

# --- write the run wrapper --------------------------------------------------
cat >"$JOBDIR/run.sh" <<RUNEOF
#!/bin/bash
export HOME=/root
cd "$JOBDIR/code"
echo "===== job start \$(date -u) ====="
"$ENVDIR/bin/python" "$JOBDIR/code/run.py" $RUN_ARGS --jobdir "$JOBDIR" --outdir "$OUT"
rc=\$?
echo "===== job exit code \$rc at \$(date -u) ====="
if [ \$rc -eq 0 ]; then touch "$JOBDIR/.complete"; fi
# .finished marks a TERMINAL state regardless of outcome. Without it a failed
# run writes no marker, the watcher waits forever, and the VM bills
# indefinitely -- the exact failure this whole pattern exists to prevent.
touch "$JOBDIR/.finished"
RUNEOF
chmod +x "$JOBDIR/run.sh"

# --- retrieval hold ---------------------------------------------------------
# Once .complete exists the watcher stops the VM within seconds of every boot.
# Right for cost control, fatal for retrieval: restarting a finished VM to
# collect results gets a machine that shuts down before sshd is reliably up.
# `retrieval-hold=1` opts out of the cost guard for one boot -- the VM then
# runs until stopped by hand, so clear it as soon as the download is verified.
HOLD="$(meta retrieval-hold)"
if [ "${HOLD:-0}" = "1" ]; then
  echo "retrieval-hold=1 -- watcher and worker suppressed for this boot."
  echo "VM stays up until stopped or deleted manually."
  exit 0
fi

# --- deliberate re-run of an already-complete job ---------------------------
# .complete is never cleared automatically -- that is what makes a finished VM
# stop again instead of redoing hours of work. Starting a NEW run on the same
# disk (e.g. one month for validation, then the full archive) therefore needs an
# explicit signal.
#
# The key holds an arbitrary token, not a flag, and the script records the token
# it consumed. A preemption reboot re-reads the SAME token, sees it already
# consumed, and does nothing -- so this cannot silently wipe .complete and
# restart a finished archive. A plain `job-reset=1` flag would do exactly that.
#
#   gcloud compute instances add-metadata NAME --zone=ZONE \
#     --metadata job-reset=$(date +%s)
RESET_TOKEN="$(meta job-reset)"
if [ -n "${RESET_TOKEN:-}" ] \
   && [ "$RESET_TOKEN" != "$(cat "$JOBDIR/.reset_token" 2>/dev/null)" ]; then
  echo "new job-reset token ($RESET_TOKEN) -- clearing markers for a fresh run"
  rm -f "$JOBDIR/.complete" "$JOBDIR/.finished"
  echo "$RESET_TOKEN" >"$JOBDIR/.reset_token"
fi

# --- clear a stale terminal marker BEFORE arming the watcher ---------------
# Ordering is silently fatal to get wrong: the watcher tests its markers before
# its first sleep, so arming it while a stale .finished from a previous failed
# run is on disk shuts the VM down seconds into this boot's retry. .complete is
# deliberately NOT cleared -- a genuinely finished job should stop again rather
# than redo work it already did.
if [ ! -f "$JOBDIR/.complete" ]; then
  rm -f "$JOBDIR/.finished"
fi

# --- (re-)arm the self-stop watcher, unconditionally, on every boot --------
# Must happen every boot, not just the first: a transient systemd unit does not
# survive a reboot, and a preemption IS a reboot. The watcher lives on the
# persistent disk ($JOBDIR), not /tmp, for the same reason.
cat >"$JOBDIR/selfstop.sh" <<'SELFSTOPEOF'
#!/bin/bash
set -u
JOBDIR="__JOBDIR__"
MAX_RUNTIME_S=__MAX_RUNTIME_S__
exec >>"$JOBDIR/selfstop.log" 2>&1
echo "=== selfstop watcher armed $(date -u) (max ${MAX_RUNTIME_S}s) ==="
elapsed=0
while true; do
  if [ -f "$JOBDIR/.complete" ]; then
    echo "$(date -u) .complete detected; stopping VM now."
    break
  fi
  if [ -f "$JOBDIR/.finished" ]; then
    echo "$(date -u) .finished detected (terminal, not fully successful);"
    echo "stopping VM to avoid idle billing. Inspect run.log before restarting."
    break
  fi
  if [ "$elapsed" -ge "$MAX_RUNTIME_S" ]; then
    echo "$(date -u) max runtime ${MAX_RUNTIME_S}s reached; stopping VM now."
    break
  fi
  sleep 120
  elapsed=$((elapsed + 120))
done
sync
sleep 5
/sbin/shutdown -h now
SELFSTOPEOF
sed -i "s|__JOBDIR__|$JOBDIR|; s|__MAX_RUNTIME_S__|$MAX_RUNTIME_S|" "$JOBDIR/selfstop.sh"
chmod +x "$JOBDIR/selfstop.sh"

SELFSTOP_UNIT="maiac-selfstop"
if ! systemctl is-active --quiet "$SELFSTOP_UNIT" 2>/dev/null; then
  systemctl reset-failed "$SELFSTOP_UNIT" 2>/dev/null || true
  systemd-run --unit="$SELFSTOP_UNIT" --collect bash "$JOBDIR/selfstop.sh"
  echo "self-stop watcher ($SELFSTOP_UNIT) (re-)armed."
else
  echo "self-stop watcher ($SELFSTOP_UNIT) already active; leaving it alone."
fi

# --- launch the worker (skip if already finished or already running) -------
if [ -f "$JOBDIR/.complete" ]; then
  echo "job already complete; nothing to do (watcher above will stop the VM)."
  exit 0
fi
if pgrep -f "$JOBDIR/code/run.py" >/dev/null; then
  echo "job already running; leaving it alone."
  exit 0
fi
echo "launching job in background..."
setsid bash "$JOBDIR/run.sh" >>"$JOBDIR/run.log" 2>&1 </dev/null &
echo "launched."
