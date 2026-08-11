#!/bin/bash
# One-time GCP setup for the MAIAC 25 km job. Idempotent -- re-running it is
# harmless and each step reports whether it created anything.
#
# What it deliberately does NOT do: put the Earthdata password anywhere. It
# creates an EMPTY Secret Manager secret and prints the one command you run
# yourself to load your ~/.netrc into it. The credential goes straight from
# your machine into Secret Manager.

set -euo pipefail

PROJECT="${PROJECT:-bullet-climate-analysis}"
REGION="${REGION:-us-west1}"
BUCKET="${BUCKET:-gs://bullet-climate-analysis-maiac-25km}"
SA_NAME=maiac-runner
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
SECRET=earthdata-netrc

echo "=== enabling APIs ==="
gcloud services enable compute.googleapis.com storage.googleapis.com \
  secretmanager.googleapis.com --project="$PROJECT"

echo "=== bucket $BUCKET ($REGION) ==="
if gcloud storage buckets describe "$BUCKET" --project="$PROJECT" >/dev/null 2>&1; then
  echo "  exists"
else
  # Same region as the VM: the monthly checkpoints are written from the VM on
  # every completed month, and same-region traffic is free.
  gcloud storage buckets create "$BUCKET" \
    --project="$PROJECT" --location="$REGION" --uniform-bucket-level-access
fi

echo "=== service account $SA ==="
if gcloud iam service-accounts describe "$SA" --project="$PROJECT" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT" \
    --display-name="MAIAC 25 km spot-VM runner"
fi

echo "=== secret $SECRET (container only, no value) ==="
if gcloud secrets describe "$SECRET" --project="$PROJECT" >/dev/null 2>&1; then
  echo "  exists"
else
  # User-managed replication pinned to $REGION, not automatic: this org's
  # constraints/gcp.resourceLocations policy forbids creating a secret in
  # "global", which is what --replication-policy=automatic asks for.
  gcloud secrets create "$SECRET" --project="$PROJECT" \
    --replication-policy=user-managed --locations="$REGION"
fi

echo "=== IAM ==="
# Scoped deliberately: write to this one bucket, read this one secret. Nothing
# else in the project is reachable from the VM.
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --project="$PROJECT" \
  --member="serviceAccount:$SA" \
  --role=roles/storage.objectAdmin >/dev/null
echo "  storage.objectAdmin on $BUCKET"

gcloud secrets add-iam-policy-binding "$SECRET" \
  --project="$PROJECT" \
  --member="serviceAccount:$SA" \
  --role=roles/secretmanager.secretAccessor >/dev/null
echo "  secretmanager.secretAccessor on $SECRET"

HAS_VERSION=$(gcloud secrets versions list "$SECRET" --project="$PROJECT" \
  --filter='state:ENABLED' --format='value(name)' 2>/dev/null | head -1)

cat <<EOF

=== provisioning complete ===
  project  $PROJECT
  bucket   $BUCKET
  runner   $SA
  secret   $SECRET
EOF

if [ -z "$HAS_VERSION" ]; then
  cat <<EOF

  !! The secret has no value yet. Load your Earthdata credential yourself --
     this is the one step that must not pass through anyone else's hands:

       gcloud secrets versions add $SECRET --project=$PROJECT --data-file=\$HOME/.netrc

     Your ~/.netrc already has a urs.earthdata.nasa.gov entry. If you would
     rather not upload the whole file, write a one-machine copy first:

       grep -A2 'urs.earthdata.nasa.gov' ~/.netrc > /tmp/earthdata.netrc
       gcloud secrets versions add $SECRET --project=$PROJECT --data-file=/tmp/earthdata.netrc
       rm /tmp/earthdata.netrc

     Then: bash deploy.sh
EOF
else
  echo
  echo "  secret has an enabled version -- ready to deploy."
fi
