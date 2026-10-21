"""LPJ-GUESS .out files: reader and land points to regular Gaussian (numpy).

Moved from ecemor/io/lpjg_out.py; used by ecemor and by the pycmor
EC-Earth4 steps.
"""
from __future__ import annotations

import calendar as _calendar
import gzip
from dataclasses import dataclass
from functools import lru_cache
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from .gaussian import ReducedGaussian

_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]

# Per-ring longitude counts, north to equator (ece2cmor3 lpjg2cmor.py).
_GAUSSIAN_RING_HALF: dict[int, list[int]] = {
    80: [
        18, 25, 36, 40, 45, 54, 60, 64, 72, 72, 80, 90, 96, 100, 108, 120,
        120, 128, 135, 144, 144, 150, 160, 160, 180, 180, 180, 192, 192, 200,
        200, 216, 216, 216, 225, 225, 240, 240, 240, 256, 256, 256, 256, 288,
        288, 288, 288, 288, 288, 288, 288, 288, 300, 300, 300, 300, 320, 320,
        320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320,
        320, 320, 320, 320, 320, 320, 320, 320,
    ],
    128: [
        18, 25, 36, 40, 45, 50, 60, 64, 72, 72, 80, 90, 90, 100, 108, 120,
        120, 125, 128, 144, 144, 150, 160, 160, 180, 180, 180, 192, 192, 200,
        216, 216, 216, 225, 240, 240, 240, 250, 250, 256, 270, 270, 288, 288,
        288, 300, 300, 320, 320, 320, 320, 324, 360, 360, 360, 360, 360, 360,
        360, 375, 375, 375, 375, 384, 384, 400, 400, 400, 400, 405, 432, 432,
        432, 432, 432, 432, 432, 450, 450, 450, 450, 450, 480, 480, 480, 480,
        480, 480, 480, 480, 480, 480, 486, 486, 486, 500, 500, 500, 500, 500,
        500, 500, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
    ],
}

GAUSSIAN_REDUCED_GRIDS: dict[int, list[int]] = {
    n: rings + rings[::-1] for n, rings in _GAUSSIAN_RING_HALF.items()
}

# Known land point counts; others resolved from the latitudes.
_GRIDSIZE_TO_N: dict[int, int] = {10407: 80, 25799: 128, 23102: 128}


@dataclass
class Table:
    """Land points of one year: lon, lat (n,), columns, values (n, k)."""
    lon: np.ndarray
    lat: np.ndarray
    columns: list
    values: np.ndarray

    def __len__(self) -> int:
        return self.lon.size


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


@lru_cache(maxsize=1)
def _parse(path: Path) -> tuple[list, np.ndarray]:
    """Header (lowercase) and all rows as float64 (n, ncols); last file cached."""
    with _open(path) as fh:
        header = fh.readline().lower().split()
        data = np.array(fh.read().split(), dtype="f8")
    return header, data.reshape(-1, len(header))


def detect_frequency(path: Path) -> str:
    """'yr' | 'mon' | 'day' from the file suffix, else the header."""
    stem = Path(str(path).removesuffix(".gz")).stem.lower()
    for suf, f in (("_yearly", "yr"), ("_monthly", "mon"), ("_daily", "day")):
        if stem.endswith(suf):
            return f
    with _open(path) as fh:
        header = fh.readline().lower().split()
    if "day" in header:
        return "day"
    return "mon" if "mth" in header or header[-12:] == _MONTH_NAMES else "yr"


def get_timespan(path: Path) -> tuple[int, int]:
    header, d = _parse(path)
    y = d[:, header.index("year")]
    return int(y.min()), int(y.max())


def _pivot(lon, lat, step, val, nstep) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (point, step, value) -> (lon, lat, values[point, step])."""
    pts, inv = np.unique(np.column_stack([lon, lat]), axis=0, return_inverse=True)
    out = np.full((pts.shape[0], nstep), np.nan)
    out[inv.ravel(), step.astype(int) - 1] = val
    return pts[:, 0], pts[:, 1], out


def read_year(path: Path, year: int, freq: str) -> Table:
    """One year of one .out file as a Table.

    yr: columns from the file; mon: jan..dec (both monthly layouts);
    day: day_001..day_NNN.
    """
    header, d = _parse(path)
    d = d[d[:, header.index("year")] == year]
    lon, lat = d[:, 0], d[:, 1]
    f = freq.lower()
    if f == "yr":
        cols = [c for c in header if c not in ("lon", "lat", "year")]
        return Table(lon, lat, cols, d[:, [header.index(c) for c in cols]])
    if f == "mon":
        if header[-12:] == _MONTH_NAMES:
            return Table(lon, lat, list(_MONTH_NAMES), d[:, -12:])
        vcol = header.index("total") if "total" in header else len(header) - 1
        lo, la, v = _pivot(lon, lat, d[:, header.index("mth")], d[:, vcol], 12)
        return Table(lo, la, list(_MONTH_NAMES), v)
    if f == "day":
        ndays = 366 if _calendar.isleap(year) else 365
        vcol = [i for i, c in enumerate(header) if c not in ("lon", "lat", "day", "year")][0]
        lo, la, v = _pivot(lon, lat, d[:, header.index("day")], d[:, vcol], ndays)
        return Table(lo, la, [f"day_{k:03d}" for k in range(1, ndays + 1)], v)
    raise ValueError(f"Unsupported frequency '{freq}'. Expected: yr, mon, day.")


def reconstruct_grid_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(lons [0, 360), lats) of the full reduced grid N=n, north to south."""
    rings = GAUSSIAN_REDUCED_GRIDS[n]
    lons = np.concatenate([np.linspace(0.0, 360.0, k, endpoint=False) for k in rings])
    x, _ = np.polynomial.legendre.leggauss(n * 2)
    ring_lat = -np.degrees(np.arcsin(x))
    lats = np.repeat(ring_lat, rings)
    return lons, lats


def _n_from_lats(lats: np.ndarray) -> Optional[int]:
    """N whose Gaussian latitudes contain all land latitudes (3 d.p.)."""
    got = set(np.round(np.asarray(lats, dtype="f8"), 3))
    for n in GAUSSIAN_REDUCED_GRIDS:
        x, _ = np.polynomial.legendre.leggauss(n * 2)
        if got <= set(np.round(np.degrees(np.arcsin(x)), 3)):
            return n
    return None


def _n_from_gridsize(n_points: int, lats: Optional[np.ndarray] = None) -> int:
    """N from the land point count, else from the latitudes."""
    if n_points in _GRIDSIZE_TO_N:
        return _GRIDSIZE_TO_N[n_points]
    n = _n_from_lats(lats) if lats is not None else None
    if n is not None:
        return n
    raise ValueError(
        f"Unknown LPJG grid size: {n_points} land points. "
        f"Supported sizes: {sorted(_GRIDSIZE_TO_N)}. "
        f"Check the target_grid setting in the experiment configuration."
    )


def _build_time_axis(
    freq: str,
    year: int,
    n_steps: int,
    ref_date: date,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Construct (time_vals, time_bnds, time_units) for one year of LPJG data.

    Returns time values and bounds in fractional days since ref_date.
    """
    ref_dt = date(ref_date.year, 1, 1)
    time_units = f"days since {ref_date.strftime('%Y-%m-%d')}"

    if freq.lower() == "yr":
        # Annual: midpoint = July 2 (conventional for CMIP yearly data).
        t0 = float((date(year, 1, 1)    - ref_dt).days)
        t1 = float((date(year + 1, 1, 1) - ref_dt).days)
        time_vals = np.array([(t0 + t1) / 2.0])
        time_bnds = np.array([[t0, t1]])

    elif freq.lower() == "mon":
        time_vals = np.empty(12, dtype=np.float64)
        time_bnds = np.empty((12, 2), dtype=np.float64)
        for m in range(1, 13):
            t0 = float((date(year, m, 1) - ref_dt).days)
            _, ndays = _calendar.monthrange(year, m)
            t1 = t0 + float(ndays)
            time_vals[m - 1] = (t0 + t1) / 2.0
            time_bnds[m - 1] = [t0, t1]

    else:  # day
        # n_steps is the number of days in the year (365 or 366).
        t_year_start = float((date(year, 1, 1) - ref_dt).days)
        time_vals = t_year_start + np.arange(n_steps, dtype=np.float64) + 0.5
        time_bnds = np.column_stack([
            t_year_start + np.arange(n_steps, dtype=np.float64),
            t_year_start + np.arange(1, n_steps + 1, dtype=np.float64),
        ])

    return time_vals, time_bnds, time_units


def _land_index(n: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Index of each land point in the full reduced grid N=n."""
    rings = np.asarray(GAUSSIAN_REDUCED_GRIDS[n])
    starts = np.concatenate([[0], np.cumsum(rings)[:-1]])
    x, _ = np.polynomial.legendre.leggauss(n * 2)
    ring_lat = -np.degrees(np.arcsin(x))
    r = np.abs(lat[:, None] - ring_lat[None, :]).argmin(axis=1)
    k = rings[r]
    pos = np.rint(np.mod(lon, 360.0) * k / 360.0).astype(int) % k
    bad = (np.abs(lat - ring_lat[r]) > 1e-3) | (
        np.abs(((np.mod(lon, 360.0) - pos * 360.0 / k) + 180.0) % 360.0 - 180.0) > 1e-3)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} of {lat.size} land points not on the N{n} grid")
    return starts[r] + pos


def to_regular_arrays(
    tab: Table,
    freq: str,
    year: int,
    ref_date: date,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    """Land points -> regular Gaussian, nearest along rows; ocean NaN.

    Returns (time, time_bnds, time_units, lats, lons, data[time, lat, lon]).
    """
    n = _n_from_gridsize(len(tab), tab.lat)
    full_lons, full_lats = reconstruct_grid_coords(n)
    full = np.full((tab.values.shape[1], full_lons.size), np.nan)
    full[:, _land_index(n, tab.lon, tab.lat)] = tab.values.T
    grid = ReducedGaussian(full_lats, full_lons)
    data = grid.to_regular(full, nearest=True)
    tv, tb, tu = _build_time_axis(freq, year, data.shape[0], ref_date)
    return tv, tb, tu, grid.lat, grid.lon, data
