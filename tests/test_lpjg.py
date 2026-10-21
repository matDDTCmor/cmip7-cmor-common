"""LPJG reader and regrid (numpy only). Samples via ECE4_SAMPLES_DIR."""
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cmip7_cmor_common import lpjg as L


def write(path, header, rows):
    with open(path, "w") as fh:
        fh.write(" ".join(header) + "\n")
        for r in rows:
            fh.write(" ".join(f"{v:.6f}" for v in r) + "\n")


def n80_points(step=7):
    lons, lats = L.reconstruct_grid_coords(80)
    pick = np.arange(0, lons.size, step)
    lon = np.where(lons[pick] > 180, lons[pick] - 360, lons[pick])
    return lon, lats[pick]


def test_yearly_regrid(tmp_path):
    lon, lat = n80_points()
    p = tmp_path / "x_yearly.out"
    write(p, ["Lon", "Lat", "Year", "Total"],
          [(a, b, 2000, i) for i, (a, b) in enumerate(zip(lon, lat))])
    assert L.detect_frequency(p) == "yr" and L.get_timespan(p) == (2000, 2000)
    tab = L.read_year(p, 2000, "yr")
    tv, tb, tu, la, lo, d = L.to_regular_arrays(tab, "yr", 2000, date(1850, 1, 1))
    assert d.shape == (1, 160, 320) and np.nanmax(d) == lon.size - 1
    assert np.allclose(tb, [[54786, 55152]])


def test_monthly_layouts_agree(tmp_path):
    lon, lat = n80_points(50)
    v = np.arange(lon.size * 12, dtype="f8").reshape(-1, 12)
    a = tmp_path / "a_monthly.out"
    write(a, ["Lon", "Lat", "Year"] + [m.capitalize() for m in L._MONTH_NAMES],
          [(x, y, 2000, *r) for x, y, r in zip(lon, lat, v)])
    b = tmp_path / "b_monthly.out"
    write(b, ["Lon", "Lat", "Year", "Mth", "Total"],
          [(x, y, 2000, m + 1, r[m]) for x, y, r in zip(lon, lat, v) for m in range(12)])
    ta, tb = L.read_year(a, 2000, "mon"), L.read_year(b, 2000, "mon")
    da = L.to_regular_arrays(ta, "mon", 2000, date(1850, 1, 1))[-1]
    db = L.to_regular_arrays(tb, "mon", 2000, date(1850, 1, 1))[-1]
    assert np.allclose(da, db, equal_nan=True)


def test_off_grid_point_rejected(tmp_path):
    lon, lat = n80_points()
    lon = lon.copy()
    lon[3] += 0.5
    tab = L.Table(lon, lat, ["total"], np.ones((lon.size, 1)))
    with pytest.raises(ValueError):
        L.to_regular_arrays(tab, "yr", 2000, date(1850, 1, 1))


SAMPLES = os.environ.get("ECE4_SAMPLES_DIR")


@pytest.mark.skipif(not SAMPLES, reason="ECE4_SAMPLES_DIR not set")
def test_t0i9_nbp_gz():
    p = Path(SAMPLES) / "t0i9" / "nbp_monthly.out.gz"
    assert L.detect_frequency(p) == "mon"
    tab = L.read_year(p, 1850, "mon")
    assert len(tab) == 23163
    d = L.to_regular_arrays(tab, "mon", 1850, date(1850, 1, 1))[-1]
    assert d.shape == (12, 256, 512)
    assert 0.15 < (~np.isnan(d[0])).mean() < 0.35
