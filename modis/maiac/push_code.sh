#!/bin/bash
# Push updated pipeline code to an existing VM.
#
# Shipping the code as instance metadata rather than a baked image is what
# makes this a single call with no rebuild: the startup script re-reads
# `job-code` on every boot, so the update lands on the next one.
#
# Works on a STOPPED or a RUNNING VM. On a running one it does not disturb the
# job in flight -- the new code takes effect at the next boot. Add --reset to
# force that boot now, which is safe precisely because the unit cache makes the
# worker resumable: it picks up from the last cached day.
#
# Usage:
#   ./push_code.sh [VM_NAME] [ZONE] [--reset]
#   ./push_code.sh maiac-25km us-west1-b --reset
#
# To change the months/workers instead of the code, push job-args:
#   gcloud compute instances add-metadata VM --zone=ZONE \
#     --metadata "job-args=--start 2000-02 --end 2025-07 --workers 8 --threads 8 --bucket gs://BUCKET"

set -euo pipefail

VM="${1:-maiac-25km}"
ZONE="${2:-us-west1-b}"
RESET="${3:-}"
PROJECT="${PROJECT:-bullet-climate-analysis}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODE_B64="$(mktemp -t maiac-code)"
trap 'rm -f "$CODE_B64"' EXIT
tar -czf - --exclude='__pycache__' --exclude='*.pyc' \
    -C "$HERE" maiac_pipeline run.py | base64 >"$CODE_B64"

echo "pushing $(wc -c <"$CODE_B64") bytes of job-code to $VM ($ZONE)..."
gcloud compute instances add-metadata "$VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --metadata-from-file \
      job-code="$CODE_B64",startup-script="$HERE/startup_script.sh"

if [ "$RESET" = "--reset" ]; then
  echo "resetting $VM so the new code takes effect now..."
  gcloud compute instances reset "$VM" --project="$PROJECT" --zone="$ZONE"
else
  echo "Done. Takes effect on the next boot; pass --reset to force one."
fi
