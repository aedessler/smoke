# CONUS smoke coverage from NOAA HMS smoke polygons

Annual maps of smoke coverage over the contiguous United States, and a monthly time series
of the area covered, built from NOAA's Hazard Mapping System (HMS) smoke polygon archive.

Two scripts do the work, run in order:

```bash
python3 process_smoke.py                 # download + compute -> data/processed/  (~35 min)
python3 plot_smoke.py                    # data/processed/ -> figures/            (~1 min)
python3 plot_smoke.py --class all        # any density, 2006-2025 (time series only)
python3 plot_smoke.py --class heavy      # thick smoke only
```

`process_smoke.py` is the only script that touches the network. `plot_smoke.py` reads
nothing but `data/processed/`, so figures can be restyled and regenerated cheaply.

Everything in this folder runs from this folder — the scripts resolve `data/` and `figures/`
relative to themselves, so `cd noaa` first.

Three more scripts sit alongside those two, each self-contained and none of them on the path
above:

```bash
python3 grid_all_smoke.py                # the all-density gridded field process_smoke.py
                                         # does not write — covers the full 2006-2025 record
python3 plot_annual_bars.py              # the annual area figure as bars, from the annual table
python3 plot_period_change.py            # 2006-2009 vs 2021-2024 maps, and their difference
```

**Not in this repository.** `data/` and `figures/` are outputs rather than source and are not
archived here — `data/` because it is 113 MB, most of it the raw HMS annual bundles, and
`figures/` because `plot_smoke.py` redraws it from `data/processed/` in about a minute. The
catalogues under [`data/processed/`](#dataprocessed) and [`figures/`](#figures) below describe
what the scripts produce, not what ships. Run the two commands above, in order, to get both.

There is a **second, independent product** in [`../modis/`](../modis/): monthly 25 km
smoke-classified aerosol optical depth from MODIS MAIAC. It answers the same question from
satellite retrievals rather than analyst judgment, and exists mainly as a cross-check on the
HMS record. [`../paper/`](../paper/) is the manuscript that reports the MAIAC record and uses
the HMS heavy-density product here as its independent check.

`--class` takes `all`, `light`, `medium` or `heavy` (default `light`). A density class
requires that `process_smoke.py` was run with the matching `SMOKE_CLASS`; if it wasn't,
the script says which files are missing and exits rather than failing partway through.
`--class all` needs only the daily table.

---

## Data source

NOAA NESDIS Hazard Mapping System, Smoke Polygons:
<https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/>

HMS smoke polygons are **hand-drawn by NOAA analysts** tracing visible smoke in
geostationary and polar-orbiter imagery. They describe smoke *seen from above* — a plume
aloft over a city counts, whether or not it affects surface air quality. They are an
analyst product, not a measurement, and the analysis effort has grown over the record
(see "Caveats").

The archive offers the same data three ways:

| Path | Contents |
|---|---|
| `Shapefile/YYYY/MM/hms_smokeYYYYMMDD.zip` | daily shapefiles, 2005-08-05 → present |
| `Shapefile/Annual_Bundles/hms_smokeYYYY.zip` | **one shapefile per year** — what this project uses |
| `Shapefile/ArcGIS_WFS_source/hms_smoke.zip` | live snapshot only |
| `KML/…` | same data as KML, for Google Earth |
| `Geotiff/smoke_days_{CONUS,AK}_YYYY.tif` | NOAA's own annual "smoke days" rasters |

We use the annual bundles: 22 files, ~97 MB total, one `hms_smokeYYYY.shp` each.

**Why not NOAA's GeoTIFFs?** They count days with smoke of *any* density and carry no
density breakdown, so they cannot answer a density-specific question. They are still a
useful independent check on the all-density gridding here.

### Attributes

Each polygon carries `Satellite`, `Start`, `End`, `Density`.

- `Start` / `End` are strings like `"2020001 1546"` — year + day-of-year + HHMM, UTC.
  Parsed with `format="%Y%j %H%M"`.
- `Density` is `"Light"` / `"Medium"` / `"Heavy"`, or `"NA"` when the analyst did not
  classify the plume.

---

## Three properties of this dataset that drive the method

All three were verified against the archive rather than assumed. They are the reason the
code looks the way it does.

### 1. `Density` is unusable before 2011

It is ~100% `"NA"` in 2005–2007 and 2009, and 22–27% `"NA"` in 2008 and 2010.

Density-based products therefore **start in 2011**. All-density area is still computed from
2006 onward and written to the daily CSV, so the longer record is available for context.
(2005 is excluded entirely — the record starts 2005-08-05, so it is not a full year.)

### 2. The density classes are nested, not disjoint

On any given day, Heavy ⊂ Medium ⊂ Light (verified to within 0.1%). They are nested
contours, like elevation bands on a topographic map — not three separate regions.

So each class means **"at least this dense"**:

- `heavy_km2` = area under at least heavy smoke.
- `light_km2` = area under *any* smoke. Light is the **outer envelope**.

This matters for interpretation. The repo is currently configured for `SMOKE_CLASS =
"Light"`, so the figures show **total smoke extent at any density**, not a light-only band.
The check is in the data: `light_km2 / all_km2` has a median of 1.0000 across all 5,400
days. If you want thick smoke specifically, set `SMOKE_CLASS = "Heavy"`.

### 3. Polygon area is extremely long-tailed

The median Heavy polygon is ~1,000–5,000 km². But analysts also draw single
continental-scale plumes reaching ~11,000,000 km² — larger than CONUS itself.

These giants are **genuine analyses** (100–350 vertices, valid geometry), not artifacts, so
they are **not filtered out**. But a single polygon can dominate a day's total, so the daily
CSV records `n_<cls>_polys` and `<cls>_max_poly_km2` alongside the area. Check those columns
before trusting any individual monthly spike.

---

## Method

### Configuration

`process_smoke.py` computes one density class per run, selected by a constant:

```python
SMOKE_CLASS = "Light"   # "Light" | "Medium" | "Heavy"
```

[`process_smoke.py:81`](process_smoke.py). Output filenames follow the class, so switching
does not overwrite the previous class's products — several classes can coexist in
`data/processed/`, and `plot_smoke.py --class <name>` picks between whichever have been
processed. (`plot_smoke.py` has a `DEFAULT_CLASS` constant used only when `--class` is
omitted; it does not need to match `process_smoke.py`.)

### The `all` class

`--class all` plots the all-density series, which is **not** density-filtered and so covers
**2006–2025** — five years longer than any density class. It reads `all_km2` from the daily
table and aggregates to months, also writing `monthly_all_smoke_area.csv`.

`all` produces **time series only**. `process_smoke.py` grids just the one class named by its
own `SMOKE_CLASS`, so there is no `all_smoke_days.nc` and the maps are skipped with a message.

Over 2011–2025 the `all` and `light` series are near-identical (median `light_km2/all_km2` =
1.0000), because Light is the outer envelope. The extra years are the reason to use `all`;
"any density" is also the more honest label for what the polygons actually measure.

Other settings, all in `process_smoke.py`:

| Constant | Value | Meaning |
|---|---|---|
| `AREA_START_YEAR` | 2006 | start of the all-density area series |
| `DENSITY_START_YEAR` | 2011 | first year with usable `Density` |
| `END_YEAR` | 2025 | last complete calendar year |
| `AREA_CRS` | `EPSG:5070` | NAD83 / CONUS Albers Equal Area |
| grid | 0.1°, −125→−66 E, 24→50 N | 260 × 590 cells |

### Pipeline

**1 — Download and cache.** Annual bundles are fetched to `data/raw/` with 3 retries and
written via a `.part` temp file so an interrupted download cannot leave a truncated zip that
looks complete. Existing non-empty files are skipped, so reruns are cheap.

**2 — CONUS domain.** Natural Earth 50m admin-1 states (bundled with cartopy, no download),
filtered to the US minus Alaska and Hawaii, dissolved to one geometry. Held in both EPSG:4326
(for the grid mask) and EPSG:5070 (for area). Total 7,784,539 km².

Clipping to CONUS is essential, not cosmetic: HMS polygons extend far into Canada, Mexico
and offshore, and without the clip the series would not mean "over the U.S."

**3 — Grid.** 0.1° cell *centers*, an `shapely.STRtree` built over them once, and a boolean
`in_conus` mask so out-of-domain cells are never counted. 81,775 of 153,400 cells are inside
CONUS.

**4 — Per year:**

- Read the bundle with `on_invalid="ignore"` — some years (2015 especially) contain unclosed
  rings that are otherwise a hard read error. Those features yield `None` and are dropped and
  counted, not silently lost.
- Parse `Start` to a UTC calendar date; drop unparseable dates and stragglers from the
  adjacent year that occasionally appear in a bundle.
- Repair self-intersections with `shapely.make_valid`. HMS polygons are hand-drawn and
  invalid geometry would otherwise abort the union.
- Normalize `Density` with `.str.strip().str.title()` — casing and whitespace vary across the
  record.

**5 — Per UTC day:**

- **Dissolve** (`shapely.union_all`) all polygons of the class. Several satellites analyze
  the same plume, so overlapping polygons must count once, not three times.
- **Reproject** the dissolved geometry to EPSG:5070, repair again (reprojecting large
  hand-drawn polygons can itself produce side-location conflicts), **clip** to CONUS, take
  the area. Falls back to a zero-width buffer if the intersection still fails.
- **Grid**: query the STRtree with the day's dissolved geometry
  (`predicate="intersects"`) to find covered cell centers, AND with `in_conus`, and increment
  that year's day counter.

Areas are recorded for **all three classes plus all-density** each day; the selected
`SMOKE_CLASS` additionally gets the polygon-count diagnostics and the gridded counts.

**6 — Aggregate.** Daily areas are summed by calendar month. The monthly table is reindexed
against a complete year × month product so smoke-free months appear as explicit zeros rather
than missing rows.

### The metric

**Smoke-area-days** (km²·days): for each day, the dissolved, CONUS-clipped area of the class;
summed over the month.

It captures both extent and persistence — 1 M km² for 10 days and 10 M km² for 1 day both
score 10 M km²·days. `mean_daily_<cls>_km2` (the same quantity divided by days in month) is
also written if you prefer a plain area unit.

---

## Outputs

### `data/processed/`

**`daily_smoke_area.csv`** — 7,239 rows, one per day that had any smoke (2006–2025).

| Column | Meaning |
|---|---|
| `date` | UTC calendar date |
| `all_km2` | CONUS area under smoke of any density |
| `light_km2`, `medium_km2`, `heavy_km2` | area under at least that density (blank before 2011) |
| `n_<cls>_polys` | polygons of the selected class that day |
| `<cls>_max_poly_km2` | largest single polygon that day — the long-tail check |

**`monthly_<cls>_smoke_area.csv`** — 180 rows (15 years × 12 months), zero-filled.
Columns: `year`, `month`, `<cls>_km2_days`, `n_days_with_<cls>`, `all_km2_days`,
`mean_daily_<cls>_km2`.

**`monthly_all_smoke_area.csv`** — 240 rows (20 years × 12 months), zero-filled. Written by
`plot_smoke.py --class all`, not by `process_smoke.py`. Columns: `year`, `month`,
`all_km2_days`, `n_days`.

**`annual_<cls>_smoke_area.csv`** — one row per year, written by `plot_smoke.py`.

| Column | Meaning |
|---|---|
| `year` | |
| `km2_days` | annual total smoke-area-days |
| `n_days` | days that year with any of this class over CONUS |
| `peak_month_km2_days` | the largest single month's contribution |
| `mean_daily_km2` | `km2_days` / days in year — a plain area unit |
| `peak_month` | which calendar month peaked (1–12) |

**`<cls>_smoke_days.nc`** — `<cls>_smoke_days(year, lat, lon)`, int16, dims 15 × 260 × 590,
plus an `in_conus(lat, lon)` mask. The mask is what lets you tell a real zero ("no smoke")
from "outside the domain". Units are `"count"`, deliberately not `"days"` — a time-unit
string makes xarray decode the variable into `timedelta64` on read. Provenance, grid
definition, and the caveats above are recorded in the file attributes.

### `figures/`

| File | Content | `all` |
|---|---|---|
| `<cls>_smoke_days_panel.png` | facet of annual maps, shared color scale | — |
| `maps/<cls>_smoke_days_YYYY.png` | the same map, one file per year, same scale | — |
| `monthly_<cls>_smoke_area.png` | monthly km²·days | ✓ |
| `annual_<cls>_smoke_area.png` | annual totals, with median reference | ✓ |
| `<cls>_seasonality_and_trend.png` | monthly climatology + annual totals | ✓ |

The annual figure carries **no fitted trend line**, deliberately. The polygon-granularity
shift described under Caveats inflates the recent end of the series, so a regression through
it would assert a physical trend the data cannot support. The dashed line is the median.

Map design notes: Lambert Conformal projection; shared `vmax` from the 99th percentile of
nonzero cells so one extreme year does not flatten the rest; zero-day cells are masked rather
than drawn, because the lightest ramp step is still visibly blue and leaving zeros in tints
the whole country and destroys the low end of the scale.

---

## Caveats

**This is an analyst product, and analyst practice changed mid-record.** This is the most
important caveat here, and it is visible in the data:

| | 2011–2016 | 2021–2025 |
|---|---|---|
| median polygons per day | 27–61 | 12–33 |
| mean largest polygon per day | 0.75–1.2 M km² | 4.4–13.7 M km² |
| mean daily area | 0.5–1.1 M km² | 1.9–2.9 M km² |

Analysts shifted from drawing **many small plumes to fewer, far larger ones**. Total polygon
counts do *not* trend up over the record — they peak in 2018–2020 (41k–45k/yr) and fall to
12.5k in 2024 — but the largest single polygon per day grew roughly tenfold. In 2023 the
largest polygon on a typical day spanned 13.7 M km², **larger than CONUS itself** (7.8 M km²),
so it saturates the domain on its own.

The rising area trend is therefore confounded with a change in how smoke is delineated, and
the confound runs in the direction of *inflating* recent years. GOES-16/17 coming online in
2017–2018 plausibly drove part of this. **Do not read the annual totals as a clean physical
trend.** If you need one, work from the gridded day counts (less sensitive to polygon
granularity than area) and treat pre-2017 and post-2020 as potentially different regimes.

**Smoke aloft ≠ surface exposure.** These polygons are what the satellite sees from above.
For air-quality work, pair with surface PM2.5.

**Density is subjective** and applied inconsistently across analysts and years. The nesting
property held everywhere it was checked, but the class boundaries are judgment calls.

**Giant polygons are retained.** See property 3. Check `<cls>_max_poly_km2` before trusting
a spike.

**Day boundaries are UTC**, not local — a west-coast evening plume can land on the next day.

---

## Requirements

Python 3 with `geopandas`, `shapely` ≥ 2.0, `pyproj`, `pyogrio`, `pandas`, `numpy`, `xarray`,
`netCDF4`, `matplotlib`, `cartopy`, `requests`. No rasterio — gridding uses `shapely.STRtree`
instead.

Runtime is ~35 min for `SMOKE_CLASS = "Light"` (light polygons are the most numerous and the
largest); ~10 min for `"Heavy"`. Disk: ~97 MB raw, ~10 MB processed, ~5 MB figures.
