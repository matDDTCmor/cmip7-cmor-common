"""Reduced Gaussian grid. Real sample with ECE4_SAMPLES_DIR (ecearth4-output-samples/samples)."""
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from cmip7_cmor_common.gaussian import GridError, ReducedGaussian

ROWS = (60.0, 20.0, -20.0, -60.0)
COUNTS = (4, 8, 8, 4)
SAMPLES = os.environ.get("ECE4_SAMPLES_DIR")


def grid_arrays():
    lat = np.concatenate([[la] * n for la, n in zip(ROWS, COUNTS)])
    lon = np.concatenate([np.arange(n) * 360.0 / n for n in COUNTS])
    return lat, lon


def test_regular_shape_and_exact_points():
    lat, lon = grid_arrays()
    g = ReducedGaussian(lat, lon)
    assert g.shape == (4, 8)
    f = np.cos(np.deg2rad(lon)) + lat / 100.0
    r = g.to_regular(f[np.newaxis])[0]
    assert np.allclose(r[1], f[4:12])
    assert np.allclose(r[0, ::2], f[:4])
    assert np.isclose(r[0, 1], (f[0] + f[1]) / 2)
    assert np.isclose(r[0, 7], (f[3] + f[0]) / 2)


def test_nan_propagates():
    lat, lon = grid_arrays()
    g = ReducedGaussian(lat, lon)
    f = np.ones(lat.size)
    f[1] = np.nan
    r = g.to_regular(f)
    assert np.isnan(r[0, 1]) and np.isnan(r[0, 2]) and np.isnan(r[0, 3])
    assert not np.isnan(r[0, 0])


def test_nearest_keeps_values():
    lat, lon = grid_arrays()
    g = ReducedGaussian(lat, lon)
    f = np.arange(lat.size, dtype=float)
    r = g.to_regular(f, nearest=True)
    assert set(np.unique(r)) <= set(f)


def test_bad_grids():
    lat, lon = grid_arrays()
    with pytest.raises(GridError):
        ReducedGaussian(lat[::-1], lon[::-1])
    lon2 = lon.copy()
    lon2[1] = 100.0
    with pytest.raises(GridError):
        ReducedGaussian(lat, lon2)
    with pytest.raises(GridError):
        ReducedGaussian(lat, lon).to_regular(np.ones(lat.size + 1))


@pytest.mark.skipif(not SAMPLES, reason="ECE4_SAMPLES_DIR not set")
def test_a9kk_tl255():
    path = Path(SAMPLES) / "a9kk" / "a9kk_atm_cmip6_1m_1992-1992_sample.nc"
    with nc.Dataset(path) as ds:
        g = ReducedGaussian(ds["lat"][:], ds["lon"][:])
        pr = ds["pr"][:].filled(np.nan)
    assert g.ncell == 88838
    assert g.shape == (256, 512)
    r = g.to_regular(pr)
    assert r.shape == (2, 256, 512)
    assert np.nanmin(r) >= np.nanmin(pr) and np.nanmax(r) <= np.nanmax(pr)
