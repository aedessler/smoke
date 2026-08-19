# MAIAC 25 km CONUS AOD and smoke statistics

Monthly 25 km gridded aerosol optical depth and smoke-condition statistics over
the contiguous United States, built from raw NASA MCD19A2.061 granules on a
Google Compute Engine spot VM. No Earth Engine.

Implements `../maiac_aws_to_google_vm_25km_plan.md` (a working note kept outside
this repository), deployed with the `gcp-spot-batch-job` pattern.

```
NASA CMR  ->  authenticated HTTPS  ->  GCE spot VM  ->  monthly NetCDF  ->  GCS
                                            |
                                     raw HDF deleted per day
```

---

## Quick start

```bash
bash provision.sh          # bucket, service account, IAM, empty secret (idempotent)
```

```bash
gcloud secrets versions add earthdata-netrc --project=bullet-climate-analysis --data-file=$HOME/.netrc
```

```bash
bash deploy.sh             # Phase C: June 2023 only
```

Then, once June 2023 looks right:

```bash
JOB_ARGS='--start 2000-02 --end 2025-07 --workers 8 --threads 8' MAX_RUNTIME=345600 bash deploy.sh
```

---

## What the numbers actually are

Six variables per month on a frozen 130 × 242 grid (`EPSG:5070`, 25 km):

| Variable | Meaning |
|---|---|
| `mean_aod_055` | Coverage-weighted mean AOD at 550 nm |
| `mean_smoke_aod_055` | Mean AOD **conditional on** the smoke aerosol model being selected |
| `smoke_aod_index` | Smoke-model AOD summed over **all** valid pixel-days — combines how often and how thick |
| `smoke_pixel_day_fraction` | Fraction of valid pixel-days flagged smoke, in [0, 1] |
| `valid_pixel_day_weight` | Denominator of every ratio above — the sample size |
| `smoke_pixel_day_weight` | Smoke-model pixel-day count |

Two caveats that matter for interpretation:

- **`mean_smoke_aod_055` is not smoke-only AOD.** MAIAC's aerosol-model bits say
  which model the retrieval *selected*; `Optical_Depth_055` remains total-column
  AOD under that model. A cell can have high smoke-model AOD partly from
  background aerosol.
- **`mean_smoke_aod_055` and `smoke_aod_index` answer different questions.**
  The conditional mean is NaN where smoke never occurred; the index is 0 there.
  For "how smoky was this month," the index is usually what you want — the
  conditional mean is conditioned on an event whose frequency is itself the
  signal.

`valid_pixel_day_weight` is not decoration. Read every ratio next to it: a cell
with 3 valid pixel-days and one with 30,000 both produce a `mean_aod_055`, and
only one of them means anything.

---

## Aggregation chain

Four sufficient statistics carry everything (plan §15), which is what lets a
month be summarised without ever materialising a 1 km monthly raster:

```
per observation   ->  QA mask: land, best quality, AOD >= 0
per native pixel-day  ->  A = mean AOD over valid obs      B = 1 if any valid obs
                          C = mean AOD over smoke obs      D = 1 if any smoke obs
per 25 km cell    ->  sum A, B, C, D over pixels and days
per month         ->  mean_aod = sum(A)/sum(B),  index = sum(C)/sum(B),  ...
```

Ratio of sums, never mean of means: a cell whose western half was cloudy for
three weeks must not have its clear half's mean count equally. `tests/test_pipeline.py`
pins this with an explicit counterexample.

`float64` accumulators throughout — a month is ~10⁸ additions into 31,460 cells,
and `float32` drifts visibly at that scale.

### Two deliberate deviations from the plan

**1. Binning instead of `gdalwarp -r sum` (plan §17).** Each native pixel is
assigned to the 25 km cell containing its centre and summed with `np.bincount`,
using a per-tile index map cached once. The property §17 actually requires —
that A, B, C and D undergo *identical* spatial weighting, so their ratios stay
coverage-weighted means — holds exactly. Binning is also exact rather than
approximate (every observation counted once, no resampling kernel) and removes
~660 `gdalwarp` invocations per month. At a 27:1 linear downsample the
difference from partial-pixel area weighting is confined to cell edges.

**2. Eight of the 22 tiles are never downloaded.** CMR's bounding-box query
returns 22 MODIS tiles for CONUS, but the whole `v03` row is 50–60 °N (north of
the 49th parallel), `h07v05`/`h07v06` are Pacific, and `h11v06` is Atlantic.
They contain zero CONUS land, verified by rasterising the Census state polygons
onto each tile. Skipping them is ~36 % less transfer for a bit-identical result.
The 14 that matter:

```
h08v04 h08v05 h08v06 h09v04 h09v05 h09v06 h10v04 h10v05 h10v06
h11v04 h11v05 h12v04 h12v05 h13v04
```

### The CONUS mask is applied at 1 km, not 25 km

A 25 km cell straddling the Canadian or Mexican border would otherwise let
foreign pixels into a U.S. cell. The mask is rasterised onto each tile's native
1200 × 1200 sinusoidal grid and applied before aggregation (plan §14).

Point-in-polygon runs in **lon/lat, not Albers**. Reprojecting the outline into
EPSG:5070 first would chord its long straight segments — the 49th-parallel
border is one straight line in lon/lat and a curve in Albers — misplacing the
northern boundary by tens of km.

Sanity check on the resulting mask: total retained area **7.83 M km²** against a
true CONUS area of ~7.7–8.1 M km², covering 13,076 of the 31,460 cells in the
bounding rectangle.

---

## Resumability

Following the `gcp-spot-batch-job` skill:

| | |
|---|---|
| **MEMBER** | one calendar month → one NetCDF, one worker process |
| **UNIT** | one acquisition day → one `.npz` of the four 25 km arrays (~300 KB) |

- Every unit is written `name.tmp.npz` → `os.replace(...)`. A crash mid-write
  can never leave a truncated file that looks finished on the next boot.
- On startup, cached units are loaded and skipped. **A preemption costs at most
  the one day in flight** — a few hundred MB of re-transfer, never a month.
- A month whose NetCDF exists locally *or in Cloud Storage* is skipped outright,
  so the whole run is idempotent.
- `tests/test_integration.py` proves the resume path: it deletes a finished
  month's NetCDF, reruns, and asserts **zero** downloads were attempted.

Raw HDFs are deleted per day. A month's scratch is only removed after its
NetCDF is confirmed *present in the bucket* — never on the strength of a
successful `cp` exit code alone.

### Cost guard

The VM stops itself the moment the job reaches a terminal state. Two markers,
because one is a trap:

- `.complete` — strict, written only if nothing failed.
- `.finished` — terminal either way.

The watcher waits on **either**, plus a wall-clock backstop. Keying only off
`.complete` means a *failed* run writes no marker and the VM bills forever —
exactly the failure the guard exists to prevent.

The watcher is written to the persistent disk and re-armed by the startup
script on **every** boot, because a preemption is a reboot and a transient
systemd unit does not survive one.

---

## Layout

```
maiac_pipeline/
    config.py         frozen grid, QA bit positions, tile geometry
    granules.py       CMR search, filename parsing, duplicate resolution
    download.py       authenticated HTTPS via earthaccess
    hdf_reader.py     the only module that needs GDAL
    qa.py             QA bit decode + observation -> pixel-day collapse
    conus_masks.py    per-tile native CONUS mask fused with the 25 km index
    regrid.py         bincount accumulation + monthly ratios
    process_day.py    the resumable UNIT
    process_month.py  the resumable MEMBER
    write_netcdf.py   dataset assembly, validation, diagnostics
    manifest.py       per-month JSON records -> manifest.csv
    gcs.py            Cloud Storage checkpointing
run.py                CLI entry point
tests/                44 tests, all network- and GDAL-free
provision.sh          one-time GCP setup
deploy.sh             create the VM
push_code.sh          update code on a live VM (no image rebuild)
retrieve_and_verify.sh  tar + sha256 download, verified
```

```bash
python3 tests/test_pipeline.py && python3 tests/test_integration.py
```

Everything with a scientific consequence is tested without credentials, network
or an HDF4-capable GDAL, so it runs on a laptop as well as on the VM.

---

## Deployment facts

| | |
|---|---|
| Project | `bullet-climate-analysis` |
| Zone | `us-west1-b` — closest GCP region to AWS `us-west-2`, where the data sits |
| Machine | `n2-standard-16` SPOT, `--instance-termination-action=STOP` |
| Disk | 500 GB pd-balanced |
| Bucket | `gs://bullet-climate-analysis-maiac-25km` |
| Credential | Secret Manager `earthdata-netrc`, read via the attached SA |

**The disk is sized for throughput, not capacity.** Only ~4 GB of raw HDF is on
disk at any moment (one day per month-worker), but pd-balanced delivers
0.28 MB/s per GB, so 500 GB buys ~140 MB/s — and this job is a download.

**Measured from CMR before provisioning:** 22 granules/day, 660/month,
**6.4 GB per month** before tile filtering, ~4.1 GB after. Full archive
(2000-02 → 2025-07, 306 months) ≈ **1.2 TB**.

The Earthdata password is never in this repo, in instance metadata, or in a CLI
argument. It goes from your `~/.netrc` into Secret Manager by your own hand, and
the startup script materialises a 0600 `/root/.netrc` from it at boot.

---

## Operating a running job

```bash
gcloud compute ssh maiac-25km --zone=us-west1-b --command='sudo tail -f /opt/maiac-25km/run.log'
```

```bash
gcloud compute ssh maiac-25km --zone=us-west1-b --command='ls /opt/maiac-25km/output/monthly | wc -l; sudo find /opt/maiac-25km/output/units -name "*.npz" | wc -l'
```

Completed months land in the bucket as they finish, so the quickest safe
retrieval needs no VM at all:

```bash
gcloud storage cp -r gs://bullet-climate-analysis-maiac-25km/maiac ./data/
```

`retrieve_and_verify.sh` is for what the bucket does not hold — run logs,
per-month JSON records, cached units. It needs the retrieval hold first, or the
watcher stops the VM within seconds of boot:

```bash
gcloud compute instances add-metadata maiac-25km --zone=us-west1-b --metadata retrieval-hold=1
```

Clear it (`retrieval-hold=0`) once the download is verified.

**Restarting a stopped VM, downloading results, and deleting the VM are three
separate actions, each needing an explicit go-ahead.** A verified download does
not by itself authorise deletion.

---

## Inspecting and concatenating

Four-panel map plus diagnostics for one month:

```bash
python3 plot_month.py data/maiac/monthly/maiac_smoke_25km_2023_06.nc
```

Combine the monthly checkpoints into one archive file:

```bash
python3 concat_archive.py data/maiac/monthly -o data/maiac_smoke_25km_monthly.nc
```

Every monthly file shares one CRS, transform and coordinate vector by
construction, and `write_netcdf.validate` refuses to write one that does not.
`concat_archive.py` nonetheless **re-checks the grid before combining** and
refuses rather than letting xarray quietly outer-join two grids that disagree.
It also enumerates missing months, because a gap you know about and a gap you
discover in a plot two months later are very different things.

### Real gaps exist upstream

MCD19A2 has genuine outages. **2002-08-01 to 2002-08-07 is a contiguous 7-day
hole in CMR itself** — verified by direct query, not inferred — so August 2002
is a 24-day month.

The ratio-of-sums design absorbs this correctly: absent days contribute to
neither numerator nor denominator, so that month's `mean_aod_055` is the mean
over the days that exist. A pipeline that averaged daily rasters would have
biased the month low with seven implicit zeros. `n_days` in every file and in
`manifest.csv` is the audit trail.

---

## Run of record — 2026-08-07

Full archive **2000-02 → 2025-07, 306/306 months complete, 0 failures, 0
preemptions.** ~2.5 h wall clock on one `n2-standard-16` spot VM at 8 workers,
**1,068 GB** transferred at a sustained ~127 MB/s. Median 234 s per month.

- **0 months had a partial-coverage day** — every acquisition day present
  carried all 14 CONUS tiles.
- 6 months have fewer than 28 acquisition days, all upstream gaps:
  `2000-02` (6 — product starts 2000-02-24), `2000-08` (19), `2001-06` (14),
  `2002-03` (23), `2002-08` (24), `2022-01` (26).
- Coverage 28.6–41.5 % of the grid rectangle. The low end is winter: snow sets
  the land/water/snow/ice QA bits away from "land", so those pixels are
  correctly excluded rather than retrieved through snow.
- Combined archive: `data/maiac_smoke_25km_monthly.nc`, 39.9 MB, 306 × 130 × 242.

### Independent validation

CONUS-mean smoke pixel-day fraction by year reproduces the western U.S. fire
record without ever being told about it:

| Year | Fraction | | Year | Fraction |
|---|---|---|---|---|
| **2021** | **0.0507** | | 2010 | 0.0024 |
| **2020** | **0.0353** | | 2016 | 0.0028 |
| **2023** | **0.0267** | | 2009 | 0.0033 |
| 2024 | 0.0219 | | 2004 | 0.0047 |
| 2018 | 0.0199 | | 2001 | 0.0049 |

Top individual months: `2020-09` (0.257), `2021-08` (0.248), `2021-07` (0.217),
`2018-08` (0.179). Those are the September 2020 West Coast event, the Dixie and
Bootleg fires, and the Mendocino Complex / Carr fires respectively — recovered
from raw retrievals, with no fire data as input.

## Known limitations

- **`aod_raw >= 0` discards genuinely small negative retrievals** (MAIAC's valid
  range starts at −0.1), which biases `mean_aod_055` slightly high in very clean
  conditions. This follows plan §12; it is paired with the strict
  `aod_quality == 0` filter, which already removes most such cases.
- **Pre-2002 months are Terra-only.** Aqua joins in mid-2002, roughly doubling
  observations per pixel-day. That is a real feature of the record, not a
  defect, but it means `valid_pixel_day_weight` steps up in 2002 and trends
  spanning that boundary need care.
- **The permissive-quality sensitivity run (plan §25) is implemented but not
  run.** `--permissive-quality` accepts AOD quality 0 and 11. Send it to a
  separate `--gcs-prefix`; it must never merge into the primary record.
- **Validated against NOAA HMS, not against AERONET or surface PM2.5** (plan
  §24). `../../paper/` compares this product to HMS heavy-density smoke days
  over 2003–2024 — r = 0.960 monthly, 0.946 annual, regression slope 0.96 — in
  `paper/scripts/fig6_validation.py`. AERONET and surface PM2.5 remain open.
