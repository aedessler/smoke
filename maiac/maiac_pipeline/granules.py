"""Granule discovery (CMR), filename parsing, and duplicate resolution.

Kept free of any network-side dependency beyond `earthaccess` itself so the
parsing and de-duplication logic -- the parts with real scientific
consequences -- can be unit-tested without credentials or a network.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass

from . import config

# MCD19A2.A2022032.h13v12.061.2024226235541.hdf
_FILENAME_RE = re.compile(
    r"^(?P<product>MCD19A2)"
    r"\.A(?P<year>\d{4})(?P<doy>\d{3})"
    r"\.h(?P<h>\d{2})v(?P<v>\d{2})"
    r"\.(?P<collection>\d{3})"
    r"\.(?P<production>\d{13})"
)


@dataclass(frozen=True)
class GranuleId:
    """The parts of an MCD19A2 granule name that the pipeline reasons about."""

    year: int
    doy: int
    h: int
    v: int
    collection: str
    production: int

    @property
    def acquisition_key(self) -> tuple[int, int, int, int]:
        """(year, doy, h, v) -- the identity a duplicate shares."""
        return (self.year, self.doy, self.h, self.v)

    @property
    def date(self) -> dt.date:
        return dt.date(self.year, 1, 1) + dt.timedelta(days=self.doy - 1)

    @property
    def tile(self) -> str:
        return f"h{self.h:02d}v{self.v:02d}"


def parse_granule_name(name: str) -> GranuleId:
    """Parse an MCD19A2 filename or GranuleUR.

    Accepts a bare UR (no extension), a filename, or a full URL.
    """
    base = name.rsplit("/", 1)[-1]
    m = _FILENAME_RE.match(base)
    if m is None:
        raise ValueError(f"not an MCD19A2 granule name: {name!r}")
    return GranuleId(
        year=int(m["year"]),
        doy=int(m["doy"]),
        h=int(m["h"]),
        v=int(m["v"]),
        collection=m["collection"],
        production=int(m["production"]),
    )


def keep_latest_reprocessing(names: list[str]) -> list[str]:
    """Collapse duplicate granules to the newest production timestamp.

    LP DAAC reprocessed parts of 2022, leaving two granules with the same
    (acquisition day, tile) and different production timestamps. Downloading
    both wastes bandwidth and double-counts the day's observations, so the
    older one is dropped *before* any transfer happens (plan section 8).

    Ordering of the surviving names is preserved.
    """
    best: dict[tuple[int, int, int, int], tuple[int, str]] = {}
    order: list[tuple[int, int, int, int]] = []
    for name in names:
        gid = parse_granule_name(name)
        key = gid.acquisition_key
        if key not in best:
            order.append(key)
            best[key] = (gid.production, name)
        elif gid.production > best[key][0]:
            best[key] = (gid.production, name)
    return [best[k][1] for k in order]


def granule_name(granule) -> str:
    """Pull the granule name out of an earthaccess DataGranule.

    earthaccess returns UMM-JSON dicts; GranuleUR is the stable field. Falling
    back to the data link keeps this working if a future version reshapes the
    object.
    """
    try:
        return granule["umm"]["GranuleUR"]
    except (KeyError, TypeError):
        pass
    try:
        return granule.data_links(access="external")[0].rsplit("/", 1)[-1]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot determine granule name from {granule!r}") from exc


def granule_size_mb(granule) -> float:
    """Best-effort granule size in MB; 0.0 when the metadata omits it."""
    try:
        info = granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"]
    except (KeyError, TypeError):
        return 0.0
    total = 0.0
    for entry in info:
        size = entry.get("Size")
        if size is None:
            continue
        unit = (entry.get("SizeUnit") or "MB").upper()
        total += float(size) * {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}.get(unit, 1.0)
    return total


def month_bounds(month: str) -> tuple[str, str]:
    """'2023-06' -> ('2023-06-01', '2023-06-30'), inclusive."""
    year, mon = (int(p) for p in month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    return f"{year:04d}-{mon:02d}-01", f"{year:04d}-{mon:02d}-{last:02d}"


def months_in_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    y0, m0 = (int(p) for p in start.split("-"))
    y1, m1 = (int(p) for p in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def search_month(month: str, *, quiet: bool = False) -> list:
    """Query CMR for every MCD19A2 granule intersecting CONUS in `month`.

    Returns the de-duplicated list of earthaccess DataGranule objects.
    """
    import earthaccess  # imported lazily: unit tests must not need it

    start, end = month_bounds(month)
    granules = earthaccess.search_data(
        short_name=config.SHORT_NAME,
        version=config.VERSION,
        cloud_hosted=True,
        bounding_box=config.CONUS_BBOX,
        temporal=(start, end),
    )
    return dedupe_granules(granules)


def dedupe_granules(granules: list) -> list:
    """`keep_latest_reprocessing`, applied to earthaccess granule objects."""
    by_name = {}
    for g in granules:
        by_name.setdefault(granule_name(g), g)
    kept = keep_latest_reprocessing(list(by_name))
    return [by_name[n] for n in kept]


def group_by_day(granules: list) -> dict[str, list]:
    """Bucket granules by acquisition date -> {'2023-06-15': [granule, ...]}."""
    out: dict[str, list] = {}
    for g in granules:
        gid = parse_granule_name(granule_name(g))
        out.setdefault(gid.date.isoformat(), []).append(g)
    return dict(sorted(out.items()))
