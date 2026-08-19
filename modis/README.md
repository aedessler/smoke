# MODIS MAIAC smoke AOD over CONUS

Monthly, 25 km, CONUS-wide aerosol optical depth and smoke-condition statistics from MODIS
MAIAC (`MCD19A2.061`), 2000-02 → 2025-07.

---

## Layout

```
maiac/        the pipeline — raw NASA granules on a GCE spot VM
              (see maiac/README.md for everything operational)
figures/      example month maps
```

Commands throughout assume **this folder** as the working directory, so `cd modis`
first (`python3 maiac/plot_month.py data/maiac/monthly/…`).

This is one of two smoke products in the repository; [`../noaa/`](../noaa/) derives the
same quantity from NOAA HMS analyst polygons, and [`../paper/`](../paper/) is the
manuscript that analyses this record and checks it against HMS.

**Not in this repository.** The paths below are referenced throughout because that is where the
code writes and reads them, but they are outputs and working notes rather than source, so they
are not archived here. Running the pipeline recreates `data/`; the rest is local:

```
data/maiac_smoke_25km_monthly.nc    the combined archive — the file you want
data/maiac/monthly/                 306 per-month NetCDF checkpoints
data/maiac/manifest.csv             per-month run records
dashboard/                          click-a-cell index browser, plus its own README
maiac_aws_to_google_vm_25km_plan.md the plan maiac/ implements
```

## The pipeline

The product is `maiac/`: it downloads raw `MCD19A2.061` HDF granules from NASA over
authenticated HTTPS and aggregates them on a Google Compute Engine spot VM. No Earth Engine.
Full archive run of record — 306/306 months, 0 failures, 1,068 GB transferred in ~2.5 h — is
documented in [`maiac/README.md`](maiac/README.md).

## The archive file

`data/maiac_smoke_25km_monthly.nc` — dims `time × y × x` = 306 × 130 × 242, EPSG:5070 CONUS
Albers, 25 km, monthly 2000-02 → 2025-07. Built by `maiac/concat_archive.py` from the monthly
checkpoints, which re-checks that every month shares one grid before combining.

All six variables are built from **four accumulators** per 25 km cell per month, which is why
they are so tightly related:

```
one observation      →  QA screen: land, best quality (AOD_QA bits 8-11 == 0), AOD >= 0
one native pixel-day →  A = mean AOD over that day's valid obs (Terra+Aqua averaged)
   (1 km, 1 day)        B = 1 if any valid obs
                        C = mean AOD over obs where the smoke model was selected
                        D = 1 if any smoke obs
one 25 km cell/month →  sum A, B, C, D over all pixels and all days
```

The file stores `ΣB` and `ΣD` directly plus three ratios. Medians and NaN fractions below are
measured over the archive as shipped:

| Variable | Formula | Answers | Median | NaN |
|---|---|---|---|---|
| `valid_pixel_day_weight` | `ΣB` | **Sample size** — 1 km pixel-days that survived QA (0 → 23,142) | — | 0 % |
| `smoke_pixel_day_weight` | `ΣD` | How many of those had the smoke model selected | — | 0 % |
| `smoke_pixel_day_fraction` | `ΣD/ΣB` | **How often** — smoke frequency, in [0, 1] | 0.00 | 60.2 % |
| `mean_aod_055` | `ΣA/ΣB` | **How hazy overall** — coverage-weighted mean 550 nm AOD | 0.10 | 60.2 % |
| `mean_smoke_aod_055` | `ΣC/ΣD` | **How thick when smoky** — AOD conditional on smoke | 0.32 | 89.9 % |
| `smoke_aod_index` | `ΣC/ΣB` | **How smoky, full stop** — frequency × intensity | 0.00 | 60.2 % |

`crs` is not data. It is a 4-byte CF grid-mapping stub carrying the projection (Albers
equal-area, standard parallels 29.5/45.5, central meridian −96°, EPSG:5070); every other
variable points at it via `grid_mapping`. `units = 1` throughout means dimensionless — AOD
genuinely has no units.

Two identities hold in the file (verified, not just intended):

```
smoke_aod_index          = smoke_pixel_day_fraction × mean_smoke_aod_055   (to 2e-7, float32 noise)
smoke_pixel_day_fraction = smoke_pixel_day_weight / valid_pixel_day_weight (exact)
```

The index factorizes cleanly into occurrence × intensity. That is the whole reason all three
smoke variables exist rather than one.

### Three things to know before using it

**1. `mean_smoke_aod_055` is not smoke-only AOD.** MAIAC does not retrieve separate smoke and
background optical depths. The `AOD_QA` aerosol-model bits say which aerosol model the
retrieval *selected*, nothing more, and `Optical_Depth_055` remains total-column AOD under that
model. Reporting it as the smoke contribution to total AOD would be wrong.

The two companion fields split the signal: `smoke_pixel_day_fraction` isolates **occurrence**;
`smoke_aod_index` combines occurrence and magnitude, since non-smoke valid days contribute zero
to the numerator but still count in the denominator. For "how smoky was this month," the index
is usually what you want — the conditional mean is conditioned on an event whose frequency is
itself the signal.

It is also **NaN exactly where the answer is "no smoke."** That is the 89.9 % NaN in the table
against 60.2 % for the other ratios: the extra ~30 % is cells where `ΣD = 0`, leaving the
conditional mean undefined. Average it over time or space and you silently drop every clean
case, biasing the result high — its median of 0.32 is three times `mean_aod_055`'s partly for
this reason. `smoke_aod_index` is **0** in those same cells, which is the truthful value.

(The 60.2 % NaN floor shared by the other ratios is the CONUS mask plus zero-coverage cells:
only 13,076 of the 31,460 cells in the bounding rectangle fall inside the U.S. boundary.)

**2. Aggregation is a ratio of sums, not a mean of means.** Screen at native 1 km → collapse to
pixel-days → sum through the month → bin to 25 km → *then* form ratios. The same logic applies
across **time**, which is why the weights are in the file:

```python
annual = ds["mean_aod_055"] * ds["valid_pixel_day_weight"]   # weight before averaging
annual = annual.sum("time") / ds["valid_pixel_day_weight"].sum("time")   # right
annual = ds["mean_aod_055"].mean("time")                                 # biased
```

The bias is worst exactly where it matters most — winter and the cloudy Pacific Northwest,
where valid-day counts swing hardest month to month.

**3. Read every ratio next to its weight.** A cell with 3 valid pixel-days and one with 23,000
both produce a `mean_aod_055`, and only one of them means anything:

```python
ds = ds.where(ds["valid_pixel_day_weight"] >= 5)   # reasonable first pass; vary it
```

Coverage runs 28.6–41.5 % of the grid rectangle; the low end is winter, where snow sets the QA
bits away from "land" and those pixels are correctly excluded rather than retrieved through
snow. `valid_pixel_day_weight` also **steps up in mid-2002** when Aqua joins Terra, roughly
doubling observations per pixel-day — a real feature of the record, but trends spanning that
boundary need care.


## Requirements

Reading the archive locally needs only `xarray`, `netCDF4`, `numpy`, `matplotlib` (and
`rioxarray` + `requests` for `maiac/plot_month.py`). The pipeline itself needs `earthaccess`,
`google-cloud-storage` and an HDF4-capable GDAL, but only on the VM —
`maiac/startup_script.sh` installs them there. Nothing in `maiac/` needs to run locally, and
its 44 tests run without credentials, network or GDAL.
