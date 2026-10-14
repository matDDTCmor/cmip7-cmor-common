"""
CMIP7 targets from the BSC variable registry (mapping XML).

The registry is content owned by bsc-variable-registry; this module only
reads it. Attributes read per row: cmip6_compound_name, cmip7_compound_name,
branded_variable_name, out_name, units, dimensions, frequency, region,
modeling_realm, grid_label, bsc_source_status.

One CMIP6 source maps to one or more CMIP7 targets (by_cmip6). One CMIP7
target can have several source rows (target): rows whose status says
another row produces the target are skipped, then the best status wins.
"""

import hashlib
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Statuses whose row is never written: another row produces the target.
SKIP_STATUS = ("duplicate_target", "dreq_heritage_mismatch")
# Lower is preferred; rows without a status rank between the two.
STATUS_RANK = {"verified": 0, "": 1, "unverified_upstream": 2}
# Fields that must agree among the rows chosen for one target.
TARGET_FIELDS = ("branded_variable_name", "out_name", "units", "dimensions",
                 "frequency", "region", "modeling_realm", "grid_label")


class TargetError(LookupError):
    """No usable row, or usable rows that disagree, for a CMIP7 target."""


@dataclass(frozen=True)
class Target:
    cmip6_compound_name: str
    cmip7_compound_name: str
    branded_variable_name: str
    out_name: str
    units: str
    dimensions: tuple
    frequency: str
    region: str
    modeling_realm: str
    grid_label: str
    status: str
    attrs: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def realm(self):
        """Realm of the CMIP7 table holding the variable (CMIP7_<realm>.json)."""
        return self.cmip7_compound_name.split(".", 1)[0]

    @property
    def skipped(self):
        return self.status in SKIP_STATUS

    @classmethod
    def from_element(cls, el):
        a = dict(el.attrib)
        return cls(
            cmip6_compound_name=a.get("cmip6_compound_name", ""),
            cmip7_compound_name=a.get("cmip7_compound_name", ""),
            branded_variable_name=a.get("branded_variable_name", ""),
            out_name=a.get("out_name", ""),
            units=a.get("units", ""),
            dimensions=tuple(a.get("dimensions", "").split()),
            frequency=a.get("frequency", ""),
            region=a.get("region", ""),
            modeling_realm=a.get("modeling_realm", ""),
            grid_label=a.get("grid_label", ""),
            status=a.get("bsc_source_status", ""),
            attrs=a,
        )


class Registry(object):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        root = ET.parse(self.path).getroot()
        self.rows = [Target.from_element(el) for el in root.iter("variable")]
        self._by6, self._by7 = {}, {}
        for t in self.rows:
            self._by6.setdefault(t.cmip6_compound_name, []).append(t)
            self._by7.setdefault(t.cmip7_compound_name, []).append(t)

    def by_cmip6(self, table_var):
        """All rows for a CMIP6 table.var (empty list if none)."""
        return list(self._by6.get(table_var, []))

    def by_cmip7(self, compound_name):
        """All rows for a CMIP7 compound name, skipped ones included."""
        return list(self._by7.get(compound_name, []))

    def sources(self, compound_name):
        """Usable rows of the best status for a CMIP7 target."""
        usable = [t for t in self.by_cmip7(compound_name) if not t.skipped]
        if not usable:
            return []
        best = min(STATUS_RANK.get(t.status, 1) for t in usable)
        return [t for t in usable if STATUS_RANK.get(t.status, 1) == best]

    def target(self, compound_name):
        """The row describing a CMIP7 target. Raises TargetError."""
        rows = self.by_cmip7(compound_name)
        if not rows:
            raise TargetError(f"{compound_name}: not in registry {self.path}")
        chosen = self.sources(compound_name)
        if not chosen:
            why = ", ".join(f"{t.cmip6_compound_name}={t.status}" for t in rows)
            raise TargetError(f"{compound_name}: no usable row ({why})")
        for f in TARGET_FIELDS:
            values = {getattr(t, f) for t in chosen}
            if len(values) > 1:
                rows_txt = ", ".join(t.cmip6_compound_name for t in chosen)
                raise TargetError(f"{compound_name}: rows {rows_txt} differ in {f}: "
                                  f"{sorted(map(str, values))}")
        return chosen[0]

    def provenance(self):
        """Path, sha256 and, inside a git checkout, the commit read."""
        with open(self.path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        info = {"path": self.path, "sha256": sha, "commit": None, "dirty": None}
        d = os.path.dirname(self.path)
        try:
            info["commit"] = subprocess.run(
                ["git", "-C", d, "rev-parse", "HEAD"], capture_output=True,
                text=True, check=True).stdout.strip()
            info["dirty"] = bool(subprocess.run(
                ["git", "-C", d, "status", "--porcelain", "--", self.path],
                capture_output=True, text=True, check=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
        return info


def read_exclude(path):
    """CMIP6 table.var (or CMIP7 compound names) from an exclude file: first
    token per line, blank lines and # comments ignored."""
    names = set()
    with open(path) as fh:
        for line in fh:
            tok = line.split("#", 1)[0].split()
            if tok:
                names.add(tok[0])
    return names
