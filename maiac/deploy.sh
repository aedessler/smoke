#!/bin/bash
# Deploy the MAIAC 25 km job to a GCP spot VM.
#
# Records the exact gcloud calls, so the deployment is reproducible and
# reviewable rather than living only in a shell history. Safe to read; running
# it creates a VM.
#
# Prerequisites (see provision.sh, which creates them):
#   - bucket gs://bullet-climate-analysis-maiac-25km in us-west1
#   - service account maiac-runner@ with objectAdmin on that bucket and
#     secretAccessor on the earthdata-netrc secret
#   - Secret Manager secret `earthdata-netrc` holding a .netrc with a
#     urs.earthdata.nasa.gov entry  <- YOU create this, see README
#
# No service-account key file is created anywhere: the VM authenticates via its
# attached service account through the metadata server.
#
# Phase C (default) processes June 2023 only. For the full archive:
#   JOB_ARGS='--start 2000-02 --end 2025-07 --workers 8 --threads 8' \
#   MAX_RUNTIME=345600 NAME=maiac-25km-full bash deploy.sh

set -euo pipefail

PROJECT="${PROJECT:-bullet-climate-analysis}"
# us-west1 (Oregon) is the closest GCP region to AWS us-west-2, where NASA's
# Earthdata Cloud data physically sits. Cross-cloud HTTPS is this job's
# bottleneck, so the hop length is worth minimising.
ZONE="${ZONE:-us-west1-b}"
NAME="${NAME:-maiac-25km}"
MACHINE="${MACHINE:-n2-standard-16}"
PROVISIONING="${PROVISIONING:-SPOT}"
BUCKET="${BUCKET:-gs://bullet-climate-analysis-maiac-25km}"
SA="maiac-runner@${PROJECT}.iam.gserviceaccount.com"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Phase C of the plan: one month first, validated by eye, before committing to
# 25 years of transfer.
JOB_ARGS="${JOB_ARGS:---months 2023-06 --workers 1 --threads 12}"
MAX_RUNTIME="${MAX_RUNTIME:-21600}"   # 6 h backstop for the single-month run

# Disk is sized for THROUGHPUT, not capacity. Only ~4 GB of raw HDF is ever on
# disk at once (one day per month-worker), but pd-balanced scales at
# 0.28 MB/s per GB, so 500 GB buys ~140 MB/s -- and this job is a download.
DISK_GB="${DISK_GB:-500}"

if [ "$PROVISIONING" = "SPOT" ]; then
  # STOP, never the default DELETE: a preempted VM must keep its disk, and with
  # it every cached unit the worker has already paid for.
  SCHED_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
else
  SCHED_FLAGS=(--provisioning-model=STANDARD --maintenance-policy=MIGRATE)
fi

echo "packing pipeline code..."
CODE_B64="$(mktemp -t maiac-code)"
trap 'rm -f "$CODE_B64"' EXIT
tar -czf - --exclude='__pycache__' --exclude='*.pyc' \
    -C "$HERE" maiac_pipeline run.py | base64 >"$CODE_B64"
echo "  job-code payload: $(wc -c <"$CODE_B64") bytes"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  "${SCHED_FLAGS[@]}" \
  --service-account="$SA" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata="job-args=${JOB_ARGS} --bucket ${BUCKET},job-max-runtime=${MAX_RUNTIME}" \
  --metadata-from-file \
      startup-script="$HERE/startup_script.sh",job-code="$CODE_B64" \
  --labels=job=maiac-25km

cat <<EOF

Deployed $NAME ($MACHINE, $PROVISIONING) to $ZONE.
  args:    $JOB_ARGS --bucket $BUCKET
  backstop: ${MAX_RUNTIME}s

Watch the boot (environment build takes ~5-8 min the first time):
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT --command='sudo tail -f /opt/maiac-25km/startup.log'

Watch the job:
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT --command='sudo tail -f /opt/maiac-25km/run.log'

Progress at a glance:
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT --command='ls /opt/maiac-25km/output/monthly 2>/dev/null | wc -l; sudo find /opt/maiac-25km/output/units -name "*.npz" | wc -l; sudo tail -3 /opt/maiac-25km/run.log'

Push a code fix (takes effect on next boot, no image rebuild):
  bash $HERE/push_code.sh $NAME $ZONE

The VM stops itself when the job reaches a terminal state (.complete or
.finished) or hits the backstop. Restarting it, downloading results, and
deleting it are three separate actions that each need an explicit go-ahead.
EOF
