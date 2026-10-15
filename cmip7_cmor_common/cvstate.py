"""CMIP7 CV and CMOR tables staging in a state dir.

CMOR resolves tables and the CV file against CWD (inpath="."). The tables
tree can be read-only or shared, so it is mirrored by symlinks into
<state-dir>/.recmorise-tables-staging, and the augmented CV is written to
<state-dir>/.recmorise-cv and linked there. Names kept from recmorise-relay.

Anything missing from the CV is added only when the caller supplies a
curated entry, or with allow_stub=True. No silent stubs.
"""

import copy
import hashlib
import json
import os
import re

STAGING_DIR = ".recmorise-tables-staging"
CV_DIR = ".recmorise-cv"
CV_PREFIX = "recmorise-augmented-cv-"


class CVError(ValueError):
    """CV missing, or an identity value not in the CV without allow_stub."""


def find_cv(tables_dir):
    """cmor-cvs.json in the sibling tables-cvs/, else in tables_dir."""
    for p in (os.path.join(tables_dir, "..", "tables-cvs", "cmor-cvs.json"),
              os.path.join(tables_dir, "cmor-cvs.json")):
        if os.path.exists(p):
            return os.path.normpath(p)
    raise CVError(f"cmor-cvs.json not found near {tables_dir}")


def load_cv(tables_dir):
    path = find_cv(tables_dir)
    with open(path) as fh:
        return json.load(fh), path


def _top(cv):
    return cv.get("CV", cv)


def stage_tables(tables_dir, state_dir):
    """Symlink mirror of tables_dir in the state dir; returns its path.

    Marker-gated: many concurrent processes share it on GPFS.
    """
    tables_dir = os.path.abspath(tables_dir)
    staging = os.path.join(os.path.abspath(state_dir), STAGING_DIR)
    marker = os.path.join(staging, ".populated")
    os.makedirs(staging, exist_ok=True)
    if not os.path.exists(marker):
        for entry in os.listdir(tables_dir):
            if entry.startswith(CV_PREFIX):
                continue
            link = os.path.join(staging, entry)
            if not os.path.lexists(link):
                try:
                    os.symlink(os.path.join(tables_dir, entry), link)
                except FileExistsError:
                    pass
        open(marker, "a").close()
    return staging


def check_identity(cv, experiment_id, activity_id, parent_experiment_id=None,
                   parent_activity_id=None, allow_stub=False):
    """Raise CVError if experiment or activity are not in the CV (unless
    allow_stub). Returns warnings: parent or activity not in the experiment
    entry, or the stub notice."""
    top = _top(cv)
    missing = [f"{k} {v!r}" for k, v in (("experiment_id", experiment_id),
                                          ("activity_id", activity_id))
               if v not in top.get(k, {})]
    if missing:
        msg = f"{' and '.join(missing)} not in the CMIP7 CV"
        if not allow_stub:
            raise CVError(f"{msg}; use the CMIP7 name, or allow a stub")
        return [f"{msg}; adding a stub"]
    warns = []
    entry = top["experiment_id"][experiment_id]
    for key, val in (("activity_id", activity_id),
                     ("parent_experiment_id", parent_experiment_id),
                     ("parent_activity_id", parent_activity_id)):
        allowed = entry.get(key) or []
        allowed = [allowed] if isinstance(allowed, str) else allowed
        if val is not None and allowed and val not in allowed:
            warns.append(f"{key} {val!r} not in the CV entry of {experiment_id} {allowed}")
    return warns


def _need(top, section, key, allow_stub, what):
    if key in top.get(section, {}):
        return False
    if not allow_stub:
        raise CVError(f"{section} {key!r} not in the CMIP7 CV and no {what}; "
                      "add it, or allow a stub")
    return True


def augment(cv, *, source_id, institution_id, grid_label, variant_label,
            activity_id, experiment_id, source_entry=None, institution_name=None,
            allow_stub=False):
    """Copy of cv with this run's identity. Returns (cv, notes)."""
    cv = copy.deepcopy(cv)
    top = _top(cv)
    notes = []

    if institution_id not in top.setdefault("institution_id", {}):
        if institution_name:
            top["institution_id"][institution_id] = institution_name
        elif _need(top, "institution_id", institution_id, allow_stub, "long name"):
            top["institution_id"][institution_id] = institution_id
            notes.append(f"stub institution_id {institution_id}")

    if source_id not in top.setdefault("source_id", {}):
        if source_entry:
            top["source_id"][source_id] = copy.deepcopy(source_entry)
            notes.append(f"source_id {source_id} from the curated entry")
        elif _need(top, "source_id", source_id, allow_stub, "curated entry"):
            top["source_id"][source_id] = {"source": source_id, "source_id": source_id}
            notes.append(f"stub source_id {source_id}")

    if _need(top, "grid_label", grid_label, allow_stub, "entry"):
        top.setdefault("grid_label", {})[grid_label] = "stub grid label"
        notes.append(f"stub grid_label {grid_label}")

    if _need(top, "activity_id", activity_id, allow_stub, "entry"):
        top.setdefault("activity_id", {})[activity_id] = f"stub activity {activity_id}"
        notes.append(f"stub activity_id {activity_id}")

    if _need(top, "experiment_id", experiment_id, allow_stub, "entry"):
        top.setdefault("experiment_id", {})[experiment_id] = {
            "activity_id": [activity_id], "description": "stub",
            "end_year": "", "experiment": experiment_id,
            "experiment_id": experiment_id, "min_number_yrs_per_sim": 1,
            "parent_activity_id": ["CMIP"], "parent_experiment_id": ["no parent"],
            "start_year": "", "tier": 1,
        }
        notes.append(f"stub experiment_id {experiment_id}")

    # CMIP6-style source entries list activities; CMIP7 ones do not.
    part = top["source_id"][source_id].get("activity_participation")
    if isinstance(part, list) and activity_id not in part:
        part.append(activity_id)

    # The CMIP7 CV has regexes here; literal values are harmless (as the relay).
    m = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)$", variant_label or "")
    if not m:
        raise CVError(f"variant_label {variant_label!r} is not rNiNpNfN")
    for key, val in zip(("realization_index", "initialization_index",
                         "physics_index", "forcing_index"), m.groups()):
        sec = top.setdefault(key, [])
        if val not in sec:
            sec.append(val)
    sec = top.setdefault("variant_label", [])
    for vl in (variant_label, "".join(m.groups())):
        if vl not in sec:
            sec.append(vl)
    return cv, notes


def publish(cv, state_dir, staging_dir):
    """Write cv under the state dir, link it in staging; returns the bare name.

    Named by content hash: same content, one file; atomic write.
    """
    text = json.dumps(cv, indent=2, sort_keys=True)
    bare = CV_PREFIX + hashlib.sha1(text.encode()).hexdigest()[:12] + ".json"
    out_dir = os.path.join(os.path.abspath(state_dir), CV_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, bare)
    if not os.path.exists(out):
        tmp = f"{out}.tmp-{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, out)
    link = os.path.join(staging_dir, bare)
    if os.path.islink(link) and not os.path.exists(link):
        try:
            os.unlink(link)
        except FileNotFoundError:
            pass
    if not os.path.exists(link):
        try:
            os.symlink(out, link)
        except FileExistsError:
            pass
    return bare
