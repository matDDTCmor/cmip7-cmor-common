"""Tests for the registry lookup. Run: python -m pytest tests/"""
import os

import pytest

from cmip7_cmor_common.registry import Registry, TargetError, read_exclude

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "registry-sample.xml")


@pytest.fixture(scope="module")
def reg():
    return Registry(SAMPLE)


def test_plain_row(reg):
    t = reg.target("atmos.tas.tavg-h2m-hxy-u.mon.glb")
    assert (t.branded_variable_name, t.grid_label, t.realm) == \
        ("tas_tavg-h2m-hxy-u", "g114", "atmos")
    assert t.dimensions[-1] == "height2m"


def test_heritage_mismatch_row_is_skipped(reg):
    t = reg.target("atmos.hus.tavg-p19-hxy-u.day.glb")
    assert t.cmip6_compound_name == "Eday.hus"
    assert [s.cmip6_compound_name for s in reg.sources(t.cmip7_compound_name)] == ["Eday.hus"]


def test_duplicate_target_row_is_skipped(reg):
    assert reg.target("landIce.snw.tavg-u-hxy-lnd.mon.glb").cmip6_compound_name == "LImon.snw"


def test_verified_wins_over_unverified(reg):
    t = reg.target("seaIce.siflsensbot.tavg-u-hxy-si.mon.glb")
    assert (t.cmip6_compound_name, t.out_name) == ("SImon.siflsensupbot", "siflsensupbot")


def test_two_verified_sources_same_target(reg):
    name = "land.mrro.tavg-u-hxy-lnd.mon.glb"
    assert sorted(s.cmip6_compound_name for s in reg.sources(name)) == \
        ["LPJGmon.mrro", "Lmon.mrro"]
    assert reg.target(name).grid_label == "g114"


def test_no_usable_row(reg):
    with pytest.raises(TargetError, match="no usable row"):
        reg.target("ocean.mlotst.tavg-u-hxy-sea.day.glb")


def test_unknown_target(reg):
    with pytest.raises(TargetError, match="not in registry"):
        reg.target("atmos.tas.tavg-h2m-hxy-u.yr.glb")


def test_missing_grid_label_is_empty(reg):
    assert reg.by_cmip6("Emon.rsdss")[0].grid_label == ""


def test_by_cmip6_keeps_skipped_rows(reg):
    assert reg.by_cmip6("day.hus")[0].skipped


def test_provenance(reg):
    p = reg.provenance()
    assert p["path"] == os.path.abspath(SAMPLE) and len(p["sha256"]) == 64


def test_read_exclude(tmp_path):
    f = tmp_path / "exclude.txt"
    f.write_text("# header\n\n3hr.uas\nday.hur  # plev8 only\n")
    assert read_exclude(f) == {"3hr.uas", "day.hur"}


@pytest.mark.skipif(not os.environ.get("CMIP7_REGISTRY_XML"),
                    reason="CMIP7_REGISTRY_XML not set")
def test_real_registry_every_target_resolves_or_fails_cleanly():
    reg = Registry(os.environ["CMIP7_REGISTRY_XML"])
    assert reg.rows
    for name in {t.cmip7_compound_name for t in reg.rows}:
        try:
            t = reg.target(name)
        except TargetError:
            continue
        assert t.branded_variable_name and not t.skipped
