#!/bin/bash
# Download a finished job's outputs from the VM and verify every file
# byte-for-byte before trusting them. Run this FROM the local machine.
# Exits non-zero on ANY verification failure -- treat that as "do not delete
# the VM," full stop.
#
# Usage:
#   ./retrieve_and_verify.sh VM_NAME ZONE [LOCAL_DEST_DIR]
#
# Note that the monthly NetCDFs are ALSO checkpointed to Cloud Storage as each
# month completes, so the quickest safe retrieval is usually:
#
#   gcloud storage cp -r gs://bullet-climate-analysis-maiac-25km/maiac ./data/
#
# This script exists for what the bucket does not hold: the run log, the
# per-month JSON records, and the cached daily units.
#
# Why tar+sha256 rather than `scp --recurse` or a file count:
#   - scp --recurse over many small files is slow and historically the flakiest
#     step in this pipeline; one tarball is one transfer.
#   - a file count can match while a file is silently truncated mid-transfer --
#     only a checksum catches that.
#
# Reaching a state where this can run at all requires the retrieval hold, or
# the self-stop watcher stops the VM within seconds of boot:
#   gcloud compute instances add-metadata VM --zone=ZONE --metadata retrieval-hold=1
#   gcloud compute instances start VM --zone=ZONE

set -u
VM="${1:?usage: retrieve_and_verify.sh VM_NAME ZONE [LOCAL_DEST_DIR]}"
ZONE="${2:?missing ZONE}"
LOCAL_DIR="${3:-./data/maiac_25km}"
REMOTE_DIR=/opt/maiac-25km

mkdir -p "$LOCAL_DIR"

echo "=== tar outputs + sha256 manifest on the VM ==="
# The file list is built with `find`, not a flat glob: the outputs live in
# monthly/, manifest/ and units/, and a top-level "*.nc *.csv" glob would
# quietly leave every one of them behind.
gcloud compute ssh "$VM" --zone="$ZONE" --ssh-flag="-o ConnectTimeout=30" --command="
  sudo bash -c '
    cd $REMOTE_DIR &&
    find output/monthly -name \"*.nc\" -type f > /tmp/retrieve.list 2>/dev/null;
    find output/manifest -name \"*.json\" -type f >> /tmp/retrieve.list 2>/dev/null;
    [ -f output/manifest.csv ] && echo output/manifest.csv >> /tmp/retrieve.list;
    for f in run.log startup.log selfstop.log; do
      [ -f \"\$f\" ] && echo \"\$f\" >> /tmp/retrieve.list;
    done;
    sort -u /tmp/retrieve.list -o /tmp/retrieve.list;
    xargs -a /tmp/retrieve.list sha256sum > /tmp/retrieve.sha256 &&
    tar cf /tmp/retrieve.tar -T /tmp/retrieve.list &&
    chmod 644 /tmp/retrieve.tar /tmp/retrieve.sha256 &&
    echo TARRED files=\$(wc -l < /tmp/retrieve.sha256)
  '
" || { echo "FAIL: could not tar/checksum on the VM"; exit 1; }

echo "=== download tarball + manifest ==="
gcloud compute scp "$VM:/tmp/retrieve.tar"    "$LOCAL_DIR/" --zone="$ZONE" \
  || { echo "FAIL: scp tarball"; exit 1; }
gcloud compute scp "$VM:/tmp/retrieve.sha256" "$LOCAL_DIR/" --zone="$ZONE" \
  || { echo "FAIL: scp manifest"; exit 1; }

cd "$LOCAL_DIR" || exit 1
if [ ! -f retrieve.tar ] || ! tar tf retrieve.tar >/dev/null 2>&1; then
  echo "FAIL: tarball missing or corrupt after download -- VM left untouched."
  exit 1
fi
tar xf retrieve.tar

echo "=== verify sha256 for every file ==="
fail=0
total=0
while read -r expected name; do
  [ -z "${expected:-}" ] && continue
  total=$((total + 1))
  if [ ! -f "$name" ]; then
    echo "  MISSING: $name"
    fail=$((fail + 1))
    continue
  fi
  actual=$(shasum -a 256 "$name" 2>/dev/null | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "  MISMATCH: $name"
    fail=$((fail + 1))
  fi
done < retrieve.sha256

if [ "$fail" -ne 0 ]; then
  echo "VERIFICATION FAILED: $fail of $total file(s) bad. Do NOT delete the VM."
  echo "Partial/possibly-corrupt files are in $LOCAL_DIR -- treat with suspicion."
  exit 1
fi

rm -f retrieve.tar retrieve.sha256
echo "VERIFIED: $total/$total files, all sha256 match."
echo
echo "Checksums prove the transfer, not the science. Before trusting the data,"
echo "open a month and look at it:"
echo "  python3 -c \"import xarray as xr; print(xr.open_dataset('$LOCAL_DIR/output/monthly/maiac_smoke_25km_2023_06.nc'))\""
echo
echo "A verified download does NOT by itself authorize deleting the VM --"
echo "that is a separate, explicit decision."
exit 0
