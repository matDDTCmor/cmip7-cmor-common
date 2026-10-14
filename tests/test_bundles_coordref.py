"""Tests for bundle planning and the coordinate reference (no CMOR needed).
Run: python -m pytest tests/"""
import os
import numpy as np
import netCDF4
import pytest

from cmip7_cmor_common.bundles import (plan_bundles, nyears_for, bundle_bounds,
                                     close_year_from_date)
from cmip7_cmor_common.coordref import CoordReference, RefView, static_arrays


def _files(exp, first, last):
    return [f"/x/tas_Amon_S_{exp}_r1i1p1f1_gr_{y}01-{y}12.nc" for y in range(first, last + 1)]


@pytest.mark.parametrize("shape,freq,expected", [
    ((12, 256, 512), "mon", 50), ((365, 256, 512), "day", 10),
    ((12, 19, 256, 512), "mon", 10), ((365, 19, 256, 512), "day", 1),
    ((12, 75, 292, 362), "mon", 5), ((1460, 256, 512), "6hr", 1),
    ((12, 256, 512), "monPt", 50)])
def test_nyears_matches_upstream_formula(shape, freq, expected):
    nlevs = 1 if len(shape) <= 3 else shape[1]
    assert nyears_for(nlevs, shape[-2] * shape[-1], freq) == expected


def test_boundaries_scen7_offset_and_hist():
    assert bundle_bounds(2022, 10, "esm-scen7-h") == (2021, 2030)
    assert bundle_bounds(2031, 10, "esm-scen7-h") == (2031, 2040)
    assert bundle_bounds(1850, 10, "esm-hist") == (1850, 1859)
    assert bundle_bounds(2022, 1, "esm-scen7-h") == (2022, 2022)


def test_offline_plan_scen7_2022_2100():
    sel, still_open, done = plan_bundles(_files("esm-scen7-h", 2022, 2100), 10, "esm-scen7-h")
    labels = [b.label() for b in sel]
    assert labels[0] == "2022-2030" and labels[1] == "2031-2040" and labels[-1] == "2091-2100"
    assert not still_open and not done


def test_chunk_mode_closes_once_and_final_flushes():
    files = _files("esm-scen7-h", 2022, 2031)
    written = []
    for y in range(2022, 2032):
        sel, _, _ = plan_bundles([f for f in files if int(f[-16:-12]) <= y], 5, "esm-scen7-h",
                                 first_year=2022, close_year=y, final=(y == 2031))
        written += [b.label() for b in sel]
    assert written == ["2022-2025", "2026-2030", "2031-2031"]


def test_missing_year_reported():
    files = [f for f in _files("esm-scen7-h", 2022, 2025) if "2024" not in f]
    sel, _, _ = plan_bundles(files, 5, "esm-scen7-h", first_year=2022, close_year=2025)
    assert sel[0].missing_years() == [2024]


def test_close_year_both_autosubmit_conventions():
    assert close_year_from_date("20221231") == 2022
    assert close_year_from_date("20230101") == 2022
    assert close_year_from_date("20220701") is None


def _nc(path, lat_shift=0.0):
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", None)
        d.createDimension("lat", 3)
        d.createVariable("time", "f8", ("time",))[:] = [0.5]
        d.createVariable("lat", "f8", ("lat",))[:] = np.array([-45.0, 0.0, 45.0]) + lat_shift
        d.createVariable("tas", "f4", ("time", "lat"))[:] = 1.0


def test_reference_absorbs_last_bit_drift_and_rejects_real_differences(tmp_path):
    a, b, c = tmp_path / "a.nc", tmp_path / "b.nc", tmp_path / "c.nc"
    _nc(a)
    _nc(b, 1e-13)
    _nc(c, 1e-3)
    with netCDF4.Dataset(a) as d:
        ref = CoordReference(static_arrays(d, exclude={"tas"}), origin="a.nc")
    ref.save(str(tmp_path / "ref.npz"))
    ref = CoordReference.load(str(tmp_path / "ref.npz"))
    with netCDF4.Dataset(b) as d:
        aligned, errors = ref.compare(d, exclude={"tas"})
        assert [n for n, _ in aligned] == ["lat"] and not errors
        assert np.array_equal(RefView(d, ref).variables["lat"][:], ref.arrays["lat"])
        assert RefView(d, ref).variables["tas"][:].shape == (1, 3)
    with netCDF4.Dataset(c) as d:
        _, errors = ref.compare(d, exclude={"tas"})
        assert errors and "lat" in errors[0]


@pytest.mark.parametrize("name, years", [
    ("sidivvel_SIday_x_gn_20250102-20260101.nc", (2025, 2025)),
    ("sidivvel_SImon_x_gn_202502-202601.nc", (2025, 2025)),
    ("huss_3hr_x_gr_202501010300-202601010000.nc", (2025, 2025)),
    ("tas_day_x_gr_20250101-20251231.nc", (2025, 2025)),
    ("tas_Amon_x_gr_202501-202601.nc", (2025, 2026)),
    ("tas_day_x_gr_20240101-20260101.nc", (2024, 2026)),
])
def test_source_years_point_time_yearly_files(name, years):
    from cmip7_cmor_common.bundles import source_years
    assert source_years(name) == years


def test_new_reference_is_saved_only_on_commit(tmp_path):
    from cmip7_cmor_common.coordref import commit_coord_reference
    path = str(tmp_path / "ref" / "x.npz")
    ref = CoordReference({"lat": np.arange(3.0)}, origin="a.nc")
    ref.pending_path = path
    assert not os.path.exists(path)
    commit_coord_reference(ref)
    assert os.path.exists(path) and ref.pending_path is None
    commit_coord_reference(ref)     # second call is a no-op
    commit_coord_reference(None)
