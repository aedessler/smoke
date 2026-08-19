#!/usr/bin/env python3
"""Every number quoted in the MAIAC smoke-record manuscript.

    python3 run_analysis.py

Writes to ../results/:

    maiac_summary.json      every scalar in the text, nested by section
    conus_monthly.csv       the full 306-month CONUS-aggregate series
    conus_annual.csv        2003-2024, smoke days + intensity + index + tail
    regional_annual.csv     the same by West / Central / East
    seasonal.csv            monthly climatology and per-month trend
    cell_trends.nc          per-cell mean, Sen slope, tau, p, FDR pass
    extreme_cell_years.csv  the largest single cell-years in the record

Nothing here reads the figures and no figure reads this; both recompute from
the archive. The JSON is the audit trail for the manuscript, not an input to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import maiac_common as M
from maiac_common import C

OUT = M.RESULT_DIR
THRESHOLDS = (1, 5, 10, 20, 30)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# --------------------------------------------------------------------------
# Sections. The CONUS monthly and annual aggregates come from maiac_common so
# that the figures, which never read this file's output, build on exactly the
# same definitions.
# --------------------------------------------------------------------------

def record_and_trend(ann: pd.DataFrame) -> dict:
    a = ann[(ann.year >= M.FIRST_YEAR) & (ann.year <= M.LAST_YEAR)].set_index("year")
    yr = a.index.values.astype(float)
    sd = a["smoke_days"].values

    half = len(a) // 2
    early, late = sd[:half], sd[half:]
    tw = stats.ttest_ind(early, late, equal_var=False)

    keep = ~np.isin(a.index.values, [2020, 2021])
    ols = stats.linregress(yr, sd)

    order = np.argsort(sd)[::-1]
    top5 = a.index.values[order[:5]]
    last5 = a.loc[M.LAST_YEAR - 4:M.LAST_YEAR, "smoke_days"].sum()

    out = {
        "window": [int(M.FIRST_YEAR), int(M.LAST_YEAR)],
        "n_years": int(len(a)),
        "mean_smoke_days": float(sd.mean()),
        "median_smoke_days": float(np.median(sd)),
        "min": {"year": int(a["smoke_days"].idxmin()), "value": float(sd.min())},
        "max": {"year": int(a["smoke_days"].idxmax()), "value": float(sd.max())},
        "max_over_min": float(sd.max() / sd.min()),
        "theil_sen": C.theil_sen(yr, sd),
        "log_trend": M.log_trend(sd, yr),
        "ols": {"slope": float(ols.slope), "p": float(ols.pvalue),
                "r2": float(ols.rvalue ** 2)},
        "halves": {
            "early_years": [int(a.index[0]), int(a.index[half - 1])],
            "late_years": [int(a.index[half]), int(a.index[-1])],
            "early_mean": float(early.mean()), "late_mean": float(late.mean()),
            "ratio": float(late.mean() / early.mean()),
            "welch_t": float(tw.statistic), "welch_p": float(tw.pvalue),
        },
        "without_2020_2021": C.theil_sen(yr[keep], sd[keep]),
        "top5_years": [int(y) for y in top5],
        "top5_values": [float(a.loc[y, "smoke_days"]) for y in top5],
        "last5_share_of_total": float(last5 / sd.sum()),
        "last5_share_of_years": float(5 / len(a)),
        "ranked_years": {int(y): float(a.loc[y, "smoke_days"])
                         for y in a["smoke_days"].sort_values(ascending=False).index},
    }
    return out


def decomposition(ann: pd.DataFrame) -> dict:
    """Is the record getting smokier because smoke is more frequent, or thicker?

    smoke_aod_index = smoke_pixel_day_fraction x mean_smoke_AOD holds exactly in
    the archive, so in logs the growth rate of the index is the sum of the
    growth rates of frequency and intensity, and the split is a genuine
    decomposition rather than two loosely related numbers.
    """
    a = ann[(ann.year >= M.FIRST_YEAR) & (ann.year <= M.LAST_YEAR)].set_index("year")
    yr = a.index.values.astype(float)
    freq = M.log_trend(a["smoke_days"].values, yr)
    inten = M.log_trend(a["smoke_intensity"].values, yr)
    index = M.log_trend(a["smoke_aod_index"].values, yr)
    tot = freq["percent_per_year"] + inten["percent_per_year"]
    return {
        "frequency": freq,
        "intensity": inten,
        "index": index,
        "frequency_plus_intensity": float(tot),
        "closure_residual": float(index["percent_per_year"] - tot),
        "frequency_share_of_index_growth": float(freq["percent_per_year"] / tot),
        "intensity_theil_sen": C.theil_sen(yr, a["smoke_intensity"].values),
        "index_theil_sen": C.theil_sen(yr, a["smoke_aod_index"].values),
        "mean_aod_trend": M.log_trend(a["mean_aod"].values, yr),
        # Mean AOD over the pixel-days the retrieval did *not* label smoke:
        # (sum A - sum C) / (sum B - sum D). This is the closest thing the
        # archive offers to a background, and it inherits every weakness of the
        # aerosol-model bits, so it is reported as a contrast and not as a
        # measurement of non-smoke aerosol.
        "background_aod_trend": M.log_trend(
            ((a["sum_aod"] - a["sum_smoke_aod"])
             / (a["valid_pixel_days"] - a["smoke_pixel_days"])).values, yr),
        "background_aod_mean": float(
            ((a["sum_aod"] - a["sum_smoke_aod"])
             / (a["valid_pixel_days"] - a["smoke_pixel_days"])).mean()),
    }


def spatial(years, vals, lon, lat) -> dict:
    slope = M.sen_slope(vals, years)
    tau, p = M.kendall_stack(vals, years)
    sig = M.bh_fdr(p, 0.05)
    field = M.field_test(vals, years)
    mean = vals.mean(0)
    half = vals.shape[0] // 2
    early, late = vals[:half].mean(0), vals[half:].mean(0)

    regions = {}
    for name, f in C.REGIONS.items():
        sel = f(lon)
        ser = vals[:, sel].mean(1)
        regions[name] = {
            "n_cells": int(sel.sum()),
            "mean_smoke_days": float(ser.mean()),
            "early_mean": float(ser[:half].mean()),
            "late_mean": float(ser[half:].mean()),
            "ratio": float(ser[half:].mean() / ser[:half].mean()),
            "theil_sen": C.theil_sen(years, ser),
            "log_trend": M.log_trend(ser, years),
            "cell_frac_positive": float((slope[sel] > 0).mean()),
            "cell_frac_sig_positive": float(
                ((slope[sel] > 0) & (field["p_cell"][sel] < 0.05)).mean()),
            "peak_year": int(years[int(np.argmax(ser))]),
        }

    return {
        "n_cells": int(vals.shape[1]),
        "grid_area_km2": float(vals.shape[1] * M.CELL_AREA_KM2),
        "cell_mean_smoke_days": float(mean.mean()),
        "cell_mean_min": float(mean.min()), "cell_mean_max": float(mean.max()),
        "slope_median": float(np.median(slope)),
        "slope_p90": float(np.percentile(slope, 90)),
        "slope_max": float(slope.max()),
        "frac_positive": float((slope > 0).mean()),
        "frac_sig_positive_raw": float(((tau > 0) & (p < 0.05)).mean()),
        "frac_sig_negative_raw": float(((tau < 0) & (p < 0.05)).mean()),
        "frac_sig_positive_fdr": float(((tau > 0) & sig).mean()),
        "frac_sig_negative_fdr": float(((tau < 0) & sig).mean()),
        "field_test": {k: v for k, v in field.items()
                       if not isinstance(v, np.ndarray)},
        "early_mean": float(early.mean()), "late_mean": float(late.mean()),
        "max_slope_cell": {"lon": float(lon[int(np.argmax(slope))]),
                           "lat": float(lat[int(np.argmax(slope))]),
                           "slope": float(slope.max())},
        "regions": regions,
    }, slope, tau, p, sig, mean, field


def seasonality(mon: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    m = mon[(mon.year >= M.FIRST_YEAR) & (mon.year <= M.LAST_YEAR)]
    clim = m.groupby("month")["smoke_days"].mean()
    rows = []
    for mo in range(1, 13):
        s = m[m.month == mo].set_index("year")["smoke_days"]
        t = C.theil_sen(s.index.values.astype(float), s.values)
        rows.append({
            "month": mo, "name": M.MONTH_ABBR[mo - 1],
            "climatology_smoke_days": float(clim[mo]),
            "share_of_year": float(clim[mo] / clim.sum()),
            "sen_slope": t["slope"], "kendall_tau": t["kendall_tau"],
            "kendall_p": t["kendall_p"],
            "early_mean": float(s.loc[:M.FIRST_YEAR + 10].mean()),
            "late_mean": float(s.loc[M.FIRST_YEAR + 11:].mean()),
        })
    seas = pd.DataFrame(rows)
    # Which months the rise happened in. Differences of means add up to the
    # annual difference, which is why the seasonal attribution is done with
    # them; Theil-Sen slopes are medians and do not add.
    seas["change"] = seas["late_mean"] - seas["early_mean"]
    seas["share_of_change"] = seas["change"] / seas["change"].sum()

    # Timing: the smoke-weighted mean day of year, one value per year. Monthly
    # data cannot resolve a shift finer than a few weeks, which is why the test
    # below is reported with its confidence interval and not as a point value.
    mid = {mo: pd.Timestamp(2003, mo, 15).dayofyear for mo in range(1, 13)}
    g = m.assign(doy=m["month"].map(mid))
    cen = g.groupby("year").apply(
        lambda d: float(np.average(d["doy"], weights=np.maximum(d["smoke_days"], 0) + 1e-12)),
        include_groups=False)
    yr = cen.index.values.astype(float)

    peak = m.loc[m["smoke_days"].idxmax()]
    # Select by month number, never by row position: `seas` is month-ordered
    # today and a positional slice would silently shift if it ever were not.
    by_month = seas.set_index("month")
    out = {
        "climatology_total": float(clim.sum()),
        "jun_sep_share": float(clim.loc[6:9].sum() / clim.sum()),
        "peak_month": int(clim.idxmax()),
        "peak_month_share": float(clim.max() / clim.sum()),
        "total_change_days": float(seas["change"].sum()),
        "jul_sep_share_of_change": float(by_month.loc[[7, 8, 9], "share_of_change"].sum()),
        "aug_share_of_change": float(by_month.loc[8, "share_of_change"]),
        "months_declining": [int(r["month"]) for _, r in seas.iterrows()
                             if r["change"] < 0],
        "spring_change_days": float(by_month.loc[[3, 4, 5], "change"].sum()),
        "sum_of_monthly_sen_slopes": float(seas["sen_slope"].sum()),
        "centroid_trend": C.theil_sen(yr, cen.values),
        "centroid_mean_doy": float(cen.mean()),
        "largest_month": {"date": str(peak["date"])[:7],
                          "smoke_days": float(peak["smoke_days"]),
                          "fraction": float(peak["smoke_fraction"])},
    }
    return out, seas


def tail(years, vals) -> tuple[dict, pd.DataFrame]:
    """How much of the domain crosses each smoke-day threshold, year by year."""
    rows = []
    for i, y in enumerate(years):
        row = {"year": int(y)}
        for thr in THRESHOLDS:
            row[f"frac_ge_{thr}"] = float((vals[i] >= thr).mean())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("year")
    half = len(df) // 2
    out = {}
    for thr in THRESHOLDS:
        col = df[f"frac_ge_{thr}"].values * 100
        out[f"ge_{thr}_days"] = {
            "mean_percent": float(col.mean()),
            "early_percent": float(col[:half].mean()),
            "late_percent": float(col[half:].mean()),
            "ratio": float(col[half:].mean() / max(col[:half].mean(), 1e-9)),
            "theil_sen_percent_per_year": C.theil_sen(years, col),
            "max_year": int(df.index[int(np.argmax(col))]),
            "max_percent": float(col.max()),
        }
    return out, df.reset_index()


def extremes(years, vals, lon, lat) -> pd.DataFrame:
    flat = vals.ravel()
    idx = np.argsort(flat)[::-1][:15]
    iy, ic = np.unravel_index(idx, vals.shape)
    return pd.DataFrame({
        "rank": np.arange(1, len(idx) + 1),
        "year": years[iy].astype(int),
        "lat": np.round(lat[ic], 2),
        "lon": np.round(lon[ic], 2),
        "smoke_days": np.round(flat[idx], 2),
    })


def satellite_step(mon: pd.DataFrame) -> dict:
    """What the Terra-only years look like, and why they are excluded.

    Both accumulators rise when Aqua joins, so the *ratio* does not move
    mechanically -- but it does not stay put either, because the extra overpass
    adds smoke detections faster than it adds clear-sky pixel-days.
    """
    # 2001 is the only complete Terra-only calendar year in the archive: 2000
    # starts in February and Aqua data begin in July 2002.
    a = mon[mon.year == 2001]
    b = mon[(mon.year >= 2003) & (mon.year <= 2004)]
    per_month = lambda d: float(d["valid_pixel_days"].sum() / len(d))
    return {
        "terra_only_years": [2000, 2001, 2002],
        "valid_pixel_days_per_month_2001": per_month(a),
        "valid_pixel_days_per_month_2003_2004": per_month(b),
        "coverage_increase_percent": 100.0 * (per_month(b) / per_month(a) - 1.0),
        "smoke_days_2000_2002": {
            int(y): float(mon[mon.year == y]["smoke_days"].sum())
            for y in (2000, 2001, 2002)},
        "note": ("2000 covers February onward and 2025 stops in July; neither is a "
                 "complete year and neither enters an annual statistic."),
        "smoke_days_2025_partial": float(mon[mon.year == 2025]["smoke_days"].sum()),
    }


def robustness(ds, mask, mon, ann, years, vals) -> dict:
    """The trend under every aggregation and window choice worth arguing about.

    The headline number weights each cell by how much of it MODIS actually saw,
    which is the aggregation rule the archive is built on. It is not the only
    defensible one, and the alternatives are reported here rather than left for
    a reader to wonder about: an equal-area mean over cells, the same with the
    worst-observed cells removed, the window without the two extreme years, and
    the window extended back across the Terra-only years that Section 2.5
    argues should be excluded.
    """
    sub = ds.sel(time=slice(f"{M.FIRST_YEAR}-01", f"{M.LAST_YEAR}-12"))
    cov = sub["valid_pixel_day_weight"].sum("time").values[mask]
    q = cov / np.median(cov)

    def trend(ser, yr=years):
        t = C.theil_sen(yr, ser)
        return {"mean": float(np.mean(ser)), "slope": t["slope"],
                "slope_lo": t["slope_lo"], "slope_hi": t["slope_hi"],
                "kendall_p": t["kendall_p"], "n_years": t["n"]}

    a = M.window(ann)
    out = {
        "primary_coverage_weighted": trend(a["smoke_days"].values),
        "equal_area_cell_mean": trend(vals.mean(1)),
        "cell_mean_coverage_ge_20pct": trend(vals[:, q >= 0.2].mean(1)),
        "cell_mean_coverage_ge_50pct": trend(vals[:, q >= 0.5].mean(1)),
        "n_cells_dropped_20pct": int((q < 0.2).sum()),
        "n_cells_dropped_50pct": int((q < 0.5).sum()),
    }

    keep = ~np.isin(a["year"].values, [2020, 2021])
    out["without_2020_2021"] = trend(a["smoke_days"].values[keep],
                                     a["year"].values.astype(float)[keep])

    # The window the HMS comparison of Section 5 runs over, so that the two
    # sections can be read against each other.
    b = ann[(ann.year >= 2011) & (ann.year <= M.LAST_YEAR)]
    out["window_2011_2024"] = trend(b["smoke_days"].values,
                                    b["year"].values.astype(float))

    # Including the Terra-only years. Reported to show the size of the
    # inhomogeneity, not as an alternative estimate.
    c = ann[(ann.year >= 2001) & (ann.year <= M.LAST_YEAR)]
    out["window_2001_2024_inhomogeneous"] = trend(
        c["smoke_days"].values, c["year"].values.astype(float))

    # Coverage and smoke are correlated through latitude, so the screen above
    # is the check that the correlation is not the result.
    out["coverage_bins"] = [
        {"lo": lo, "hi": hi, "n_cells": int(((q >= lo) & (q < hi)).sum()),
         "mean_smoke_days": float(vals[:, (q >= lo) & (q < hi)].mean())}
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 5))
    ]
    out["coverage_vs_mean_spearman"] = float(
        stats.spearmanr(q, vals.mean(0)).statistic)
    return out


def validation_link() -> dict:
    """The numbers Section 5 borrows from the companion HMS cross-validation.

    Read from the HMS cross-validation's own audit trail rather than
    recomputed, so the two cannot quote different values for the same
    statistic. The file is a copy of the one that run_validation.py wrote.
    Only the *heavy* density class appears here. HMS density classes are nested
    contours, and the light class -- the outer envelope of any smoke at all --
    is delineated inconsistently across the record; it is not used as a
    reference for anything in this paper.
    """
    p = M.RESULT_DIR / "validation_summary.json"
    if not p.exists():
        return {"available": False}
    v = json.loads(p.read_text())
    heavy = v["spatial"]["hms_heavy"]
    return {
        "available": True,
        "source": "results/validation_summary.json",
        "reference_class": "HMS heavy",
        "window": v["window"],
        "n_months": v["n_months"],
        "monthly_r": v["conus_monthly"]["maiac_vs_hms_heavy"]["pearson_r"],
        "monthly_spearman": v["conus_monthly"]["maiac_vs_hms_heavy"]["spearman_r"],
        "annual_r": v["conus_annual"]["maiac_vs_hms_heavy"]["pearson_r"],
        "annual_spearman": v["conus_annual"]["maiac_vs_hms_heavy"]["spearman_r"],
        "index_monthly_r":
            v["conus_monthly_aod_index"]["maiac_index_vs_hms_heavy"]["pearson_r"],
        "seasonal_shape_r": v["seasonal_shape_r"]["maiac_vs_hms_heavy"],
        "per_cell_r_median": heavy["per_cell_r_median"],
        "per_cell_frac_gt_0p5": heavy["per_cell_frac_gt_0p5"],
        "spatial_r_by_year_mean": heavy["spatial_r_by_year_mean"],
        "n_cells": heavy["n_cells"],
        "conus_area_km2": heavy["conus_area_km2"],
        "slope": heavy["slope_theilsen"],
        "mean_days_maiac": heavy["mean_days_maiac"],
        "mean_days_hms": heavy["mean_days_hms"],
        "maiac_over_hms": heavy["maiac_over_hms"],
        "regions": {k: {"maiac_over_hms": r["maiac_over_hms"],
                        "annual_r": r["annual_r"],
                        "hms_mean_days": r["hms_mean_days"],
                        "maiac_mean_days": r["maiac_mean_days"]}
                    for k, r in heavy["regions"].items()},
        # Theil-Sen on the log of each annual series over 2011-2024, so these
        # are proportional rates -- multiply by 100 for percent per year. The
        # two products are in different units and only a unit-free rate can be
        # compared between them.
        "hms_heavy_log_trend": v["conus_annual_trend"]["hms_heavy"],
        "maiac_log_trend_2011_2024": v["conus_annual_trend"]["maiac_smoke_fraction"],
        "residual_drift_dex_per_year":
            v["drift"]["hms_heavy"]["residual_theilsen_dex_per_year"],
        "residual_drift_lo": v["drift"]["hms_heavy"]["residual_theilsen_lo"],
        "residual_drift_hi": v["drift"]["hms_heavy"]["residual_theilsen_hi"],
        "residual_drift_p": v["drift"]["hms_heavy"]["residual_kendall_p"],
        "cases": v["cases"],
    }


# --------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = C.load_maiac()

    print("domain and CONUS aggregates")
    mask = M.domain(ds).values
    mon = M.conus_monthly(ds, mask)
    ann = M.annual_from_monthly(mon)

    print("per-cell annual smoke days")
    ann_cell = M.annual_cell_days(ds)
    years = ann_cell["year"].values.astype(float)
    vals = ann_cell.values.reshape(ann_cell.shape[0], -1)[:, mask.ravel()]
    lon2d, lat2d = C.lonlat_of(ds)
    lon = lon2d.ravel()[mask.ravel()]
    lat = lat2d.ravel()[mask.ravel()]

    print("trends")
    summary = {
        "archive": {
            "span": [M.ARCHIVE_START, M.ARCHIVE_END],
            "n_months": int(ds["time"].size),
            "grid": "25 km, 130 \u00d7 242 cells",
            "n_domain_cells": int(mask.sum()),
            "domain_area_km2": float(mask.sum() * M.CELL_AREA_KM2),
        },
        "satellite_step": satellite_step(mon),
        "record": record_and_trend(ann),
        "decomposition": decomposition(ann),
    }

    sp, slope, tau, p, sig, cellmean, field = spatial(years, vals, lon, lat)
    summary["spatial"] = sp
    seas_sum, seas = seasonality(mon)
    summary["seasonality"] = seas_sum
    tail_sum, tail_df = tail(years, vals)
    summary["tail"] = tail_sum
    summary["robustness"] = robustness(ds, mask, mon, ann, years, vals)
    summary["validation"] = validation_link()

    print("writing results/")
    mon.to_csv(OUT / "conus_monthly.csv", index=False)
    ann.to_csv(OUT / "conus_annual.csv", index=False)
    seas.to_csv(OUT / "seasonal.csv", index=False)
    tail_df.to_csv(OUT / "tail_area_fraction.csv", index=False)
    extremes(years, vals, lon, lat).to_csv(OUT / "extreme_cell_years.csv", index=False)

    rows = []
    for name, f in C.REGIONS.items():
        sel = f(lon)
        for i, y in enumerate(years):
            rows.append({"region": name, "year": int(y),
                         "smoke_days": float(vals[i, sel].mean())})
    pd.DataFrame(rows).to_csv(OUT / "regional_annual.csv", index=False)

    def grid(v):
        g = np.full(mask.size, np.nan)
        g[mask.ravel()] = v
        return g.reshape(mask.shape)

    xr.Dataset(
        {"mean_smoke_days": (("y", "x"), grid(cellmean)),
         "sen_slope": (("y", "x"), grid(slope)),
         "kendall_tau": (("y", "x"), grid(tau)),
         "kendall_p": (("y", "x"), grid(p)),
         "spearman_rho": (("y", "x"), grid(field["rho"])),
         "permutation_p": (("y", "x"), grid(field["p_cell"])),
         "fdr_significant": (("y", "x"), grid(sig.astype(float)))},
        coords={"y": ds["y"].values, "x": ds["x"].values},
        attrs={"window": f"{M.FIRST_YEAR}-{M.LAST_YEAR}",
               "units": "days, days per year",
               "permutation_p": f"{field['n_perm']} year-label shuffles, two-sided",
               "fdr": "Benjamini-Hochberg on kendall_p, q = 0.05"},
    ).to_netcdf(OUT / "cell_trends.nc")

    (OUT / "maiac_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2))

    r = summary["record"]
    print(f"\n  {r['n_years']} years, mean {r['mean_smoke_days']:.2f} smoke days")
    print(f"  Theil-Sen {r['theil_sen']['slope']:+.3f} d/yr "
          f"[{r['theil_sen']['slope_lo']:+.3f}, {r['theil_sen']['slope_hi']:+.3f}], "
          f"p = {r['theil_sen']['kendall_p']:.4f}")
    print(f"  {r['log_trend']['percent_per_year']:+.1f} %/yr "
          f"(x{r['log_trend']['factor_over_window']:.1f} over the window)")
    print(f"  halves {r['halves']['early_mean']:.2f} -> {r['halves']['late_mean']:.2f} "
          f"days (x{r['halves']['ratio']:.2f}, Welch p = {r['halves']['welch_p']:.3f})")
    ft = sp["field_test"]
    print(f"  {sp['frac_positive']*100:.1f} % of {sp['n_cells']} cells rising "
          f"(null 95th pct {ft['frac_upward_null_p95']*100:.1f} %, "
          f"p = {ft['frac_upward_p']:.4f})")
    print(f"  {ft['frac_locally_significant']*100:.1f} % locally significant "
          f"(null 95th pct {ft['frac_locally_significant_null_p95']*100:.1f} %, "
          f"p = {ft['frac_locally_significant_p']:.4f}); "
          f"{sp['frac_sig_positive_fdr']*100:.1f} % survive FDR")
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
