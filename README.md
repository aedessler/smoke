# Smoke over the contiguous United States

Two independent measurements of the same thing, kept side by side on purpose, and a paper
that reads one against the other.

| Folder | Product | Record | Source |
|---|---|---|---|
| [`noaa/`](noaa/) | Smoke-covered area and gridded smoke-days, from NOAA HMS smoke polygons | 2006–2025 | NOAA NESDIS Hazard Mapping System |
| [`modis/`](modis/) | Monthly 25 km AOD and smoke-condition statistics, from MODIS MAIAC | 2000-02 → 2025-07 | NASA `MCD19A2.061` |
| [`paper/`](paper/) | The manuscript on what the MAIAC record shows, its figures, and the code behind them | 2003–2024 window | both of the above |

The HMS polygons are **hand-drawn by NOAA analysts** tracing visible smoke in
satellite imagery. The MAIAC fields are remotely sensed values. The two
fail in unrelated ways, which is the entire point: analyst practice drifted mid-record (HMS
analysts shifted to fewer, continental-scale polygons around 2017–2020, which inflates recent
years), and that drift has no counterpart in MAIAC. MAIAC's own biases — cloud screening,
bright-surface retrieval failures, snow-covered winters, orbital sampling — have no counterpart
in HMS. Where the two agree, the signal is probably robust.

---

`noaa/` and `modis/` are self-contained — each has its own code, data, figures and README,
with paths resolved relative to itself, and neither imports from the other. `paper/` is the
one thing that spans them: it reads both archives and reports the MAIAC record with HMS as
an independent check. Start with:

- [`noaa/README.md`](noaa/README.md) — method, outputs, and the caveats that matter most
  (density is unusable before 2011; the density classes are nested, not disjoint; polygon area
  is extremely long-tailed).
- [`modis/README.md`](modis/README.md) — what the six variables actually mean, why
  `mean_smoke_aod_055` is *not* smoke-only AOD, and why every ratio must be read next to its
  sample-size weight.
- [`paper/README.md`](paper/README.md) — the headline numbers, the three decisions that shape
  all of them, and what the code computes that the paper no longer reports.

```bash
cd noaa  && python3 process_smoke.py && python3 plot_smoke.py
cd modis && python3 maiac/plot_month.py data/maiac/monthly/maiac_smoke_25km_2023_06.nc
cd paper && python3 scripts/make_all.py
```

**Not in this repository.** The data trees are outputs rather than source, and are large, so
they are not archived here: `noaa/data/`, `modis/data/`, and the derived `paper/results/`.
`noaa/figures/` is regenerated in a minute from `noaa/data/processed/` and is likewise left
out; the `modis/` and `paper/` figures *are* here. The manuscript itself
(`paper/maiac_conus_smoke_record.docx`) is not published. The practical consequence: a fresh
clone gets all the code and the paper's figures, but the first two commands above have to be
run — in that order — before the third has anything to read.

Common caveats for both products: **smoke aloft is not surface exposure** — these are top-down
views, so pair with surface PM2.5 for air-quality work — and **day boundaries are UTC**, so a
west-coast evening plume can land on the next day.
