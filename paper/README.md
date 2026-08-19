# Preprint: what the MODIS MAIAC smoke record shows

The manuscript, its figures, and everything that produced them.

```bash
python3 scripts/make_all.py     # statistics -> results/, figures -> figures/   (~30 s)
```

The manuscript itself is no longer generated. It is hand-edited in Word, as
`maiac_conus_smoke_record.docx`.

`make_all.py` reads the two archives in the sibling folders — `../modis/data/` and
`../noaa/data/processed/` — neither of which is in this repository. Build them first
(`cd ../noaa && python3 process_smoke.py`, and the `../modis/` pipeline) or the run
stops on the first missing file.

---

## What this folder is for

This folder is the paper about **what the MODIS record actually shows** — the trend,
its decomposition into occurrence and intensity, where it is, and when in the year it
is. HMS appears only in Section 4, only as an independent check, and only its *heavy*
density class.

The headline:

| | |
|---|---|
| CONUS smoke days, 2003–2013 → 2014–2024 | **3.07 → 6.99 per year** (×2.28, Welch p = 0.038) |
| Theil–Sen slope | +0.29 [+0.04, +0.51] days yr⁻¹ (p = 0.019) |
| Proportional rate | +6.6 % yr⁻¹ |
| Smoke AOD index | +8.9 % yr⁻¹ — **74 % occurrence, 26 % intensity** |
| Background (non-smoke) AOD | +0.01 % yr⁻¹, τ = 0.004 — flat |
| Share of the rise in Jul–Sep | 91 % |
| Agreement with HMS heavy | r = 0.960 monthly, 0.946 annual, slope 0.96 |

---

## Layout

```
scripts/
    common.py           loaders, regridding, smoke-day conversion, design tokens
    maiac_common.py     window, domain, trend machinery; re-exports common.py
    run_analysis.py     the statistics -> results/
    fig1…fig6_*.py      one script per figure
    make_all.py         runs the above in order, each in its own process
figures/                the six figures, 200 dpi PNG
results/                the audit trail (see below)

maiac_conus_smoke_record.docx   the manuscript, edited by hand (not in this repository)
```

**Definitions live in one place.** `maiac_common.py` imports the loaders, the regridding
and the smoke-day conversion from `common.py` rather than reimplementing them. What it
adds is specific to this paper: the analysis window, the domain, and the trend and
permutation machinery.

**Figures do not read `results/`.** Each figure script recomputes what it draws from the
archive. `results/` is the audit trail for the numbers in the text.

**The manuscript is edited by hand.** The figures in the `.docx` are embedded copies. If
you regenerate a PNG, re-insert it in Word — nothing does that for you, and nothing
checks that the embedded copy still matches the file on disk.

### `results/`

**Not in this repository.** Everything below is derived — `scripts/make_all.py` rewrites
all of it in about 30 s, given the two input archives — so it is regenerated rather than
archived. The table is the catalogue of what a run produces.

| File | Contents |
|---|---|
| `maiac_summary.json` | every scalar the analysis produces, nested by section |
| `conus_monthly.csv` | the full 306-month CONUS-aggregate series |
| `conus_annual.csv` | 2003–2024 annual totals, intensity and index |
| `regional_annual.csv` | the same by West / Central / East |
| `seasonal.csv` | monthly climatology, per-month trend, share of the change |
| `tail_area_fraction.csv` | grid fraction above each threshold, by year |
| `cell_trends.nc` | per-cell mean, Sen slope, τ, p, permutation p, FDR pass |
| `extreme_cell_years.csv` | the largest single cell-years in the record |

`run_analysis.py` computes more than the paper reports. `tail_area_fraction.csv`,
`extreme_cell_years.csv`, `regional_annual.csv` and the `fdr_significant` field of
`cell_trends.nc` are kept as audit trail but no longer feed anything in the manuscript.

---

## The three decisions that shape every number

**The window is 2003–2024.** Aqua joins Terra in mid-2002, and from 2003 a pixel-day has
two chances to be seen clear and two chances to be labelled smoke instead of one. Both
accumulators rise, so the ratio does not move mechanically — but it does not stay put
either. Valid pixel-days per month rise 24 % across the boundary. Extending the window
back across it *tightens* the apparent significance, which is exactly the artifact:
the Terra-only years look artificially clean. 2025 is out because the archive stops in
July, inside the peak.

**The domain is MODIS's own.** The 13,064 cells with a valid observation in every year of
the window. No external boundary file, which is what keeps Section 4 a check rather than
a tautology. Areal statistics are percentages of the grid, not of CONUS land area,
because coastal cells are counted whole.

**The unit is smoke days.** `(ΣD/ΣB) × days in month`, summed to the year. This
extrapolates the clear-sky frequency to every day of the month, which *is an assumption*
— smoke and cloud are not independent. Section 4 reports the scale factor against an
independent product rather than assuming the conversion is right.

## Analyses the code still runs that the paper no longer reports

Both of these were in earlier drafts. The code still produces them, so the numbers are
in `results/` if they are ever wanted back.

**Counting the signs of grid-cell trends is not a test.** 77 % of cells have a positive
slope, which sounds convincing and is not: shuffling the year labels of the whole field
(preserving its spatial correlation exactly) produces 89 % or more in one shuffle in
twenty. The observed 77 % does not clear that (p = 0.066). What does clear it is the
*count of locally significant cells*: 35 % against a null 95th percentile of 18 %,
field significance p = 0.009 (Livezey & Chen 1983). Meanwhile only 0.9 % of cells survive
a Benjamini–Hochberg FDR correction individually — the map carries a signal, the
individual 25 km cell does not. Only the stippling in Figure 4b survives into the paper.

**The trend depends on 2020 and 2021.** Dropping them halves the slope to +0.17 days
yr⁻¹ with a CI spanning zero (p = 0.086). `robustness()` in `run_analysis.py` reports the
slope under seven aggregation and window choices: it stays between +0.17 and +0.51 days
yr⁻¹ in every one, but its nominal significance moves between 0.02 and 0.09. The
direction and rough size are robust; a precise slope is not.

---

## Requirements

`numpy`, `pandas`, `xarray`, `netCDF4`, `scipy`, `matplotlib`, `cartopy`, `pyproj`. No
network access and no credentials: everything reads the archives already on disk.

There is no local `.docx → PDF` converter on this machine (no LibreOffice, no pandoc),
so open the `.docx` in Word to check pagination.
