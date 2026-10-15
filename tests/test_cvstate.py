"""Tests for the CV and tables staging. Run: python -m pytest tests/"""
import json
import os

import pytest

from cmip7_cmor_common import cvstate
from cmip7_cmor_common.cvstate import CVError

CV = {"CV": {
    "institution_id": {"EC-Earth-Consortium": "EC-Earth-Consortium"},
    "source_id": {"EC-Earth3-ESM-1-1": {"source": "x", "source_id": "EC-Earth3-ESM-1-1"}},
    "grid_label": {"g114": "d"},
    "activity_id": {"CMIP": "d"},
    "experiment_id": {"esm-hist": {"activity_id": ["CMIP"],
                                   "parent_experiment_id": ["esm-piControl"],
                                   "parent_activity_id": ["CMIP"]}},
    "realization_index": ["^r[[:digit:]]\\{1,\\}$"],
    "variant_label": [],
}}
ID = dict(source_id="EC-Earth3-ESM-1-1", institution_id="EC-Earth-Consortium",
          grid_label="g114", variant_label="r4i1p1f1", activity_id="CMIP",
          experiment_id="esm-hist")


@pytest.fixture
def tree(tmp_path):
    tables = tmp_path / "tables"
    cvs = tmp_path / "tables-cvs"
    tables.mkdir()
    cvs.mkdir()
    (tables / "CMIP7_atmos.json").write_text("{}")
    (cvs / "cmor-cvs.json").write_text(json.dumps(CV))
    return tmp_path


def test_known_identity_adds_no_stub():
    cv, notes = cvstate.augment(CV, **ID)
    assert notes == []
    assert "4" in cv["CV"]["realization_index"] and "4111" in cv["CV"]["variant_label"]
    assert "4" not in CV["CV"]["realization_index"]  # input untouched


def test_unknown_source_needs_entry_or_stub():
    ident = dict(ID, source_id="EC-Earth4")
    with pytest.raises(CVError):
        cvstate.augment(CV, **ident)
    cv, notes = cvstate.augment(CV, **ident, source_entry={"source": "EC-Earth4 (curated)"})
    assert cv["CV"]["source_id"]["EC-Earth4"]["source"] == "EC-Earth4 (curated)"
    cv, notes = cvstate.augment(CV, **ident, allow_stub=True)
    assert notes == ["stub source_id EC-Earth4"]


def test_unknown_grid_and_experiment_need_stub():
    for k, v in (("grid_label", "g999"), ("experiment_id", "x-exp")):
        with pytest.raises(CVError):
            cvstate.augment(CV, **dict(ID, **{k: v}))
    cv, notes = cvstate.augment(CV, **dict(ID, grid_label="g999"), allow_stub=True)
    assert "g999" in cv["CV"]["grid_label"]


def test_check_identity():
    assert cvstate.check_identity(CV, "esm-hist", "CMIP", "esm-piControl", "CMIP") == []
    assert cvstate.check_identity(CV, "esm-hist", "CMIP", "piControl")
    with pytest.raises(CVError):
        cvstate.check_identity(CV, "historical-x", "CMIP")
    assert cvstate.check_identity(CV, "historical-x", "CMIP", allow_stub=True)


def test_stage_and_publish(tree, tmp_path):
    state = tmp_path / "state"
    cv, path = cvstate.load_cv(str(tree / "tables"))
    assert path.endswith("tables-cvs/cmor-cvs.json")
    staging = cvstate.stage_tables(str(tree / "tables"), str(state))
    assert os.path.islink(os.path.join(staging, "CMIP7_atmos.json"))
    a = cvstate.publish(cvstate.augment(cv, **ID)[0], str(state), staging)
    b = cvstate.publish(cvstate.augment(cv, **ID)[0], str(state), staging)
    c = cvstate.publish(cvstate.augment(cv, **dict(ID, variant_label="r1i1p1f1"))[0],
                        str(state), staging)
    assert a == b != c
    assert json.load(open(os.path.join(staging, a)))["CV"]["variant_label"]
    assert not list((tree / "tables").glob(cvstate.CV_PREFIX + "*"))
