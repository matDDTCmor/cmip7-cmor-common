"""
Persistent reference for the time-independent variables of a dataset.

Why: esg-publisher opens only the first and last file of a dataset and
outer-joins their coordinates. In abbd, 34 g114 datasets had files whose
lat / lat_bnds differed in the last bits (<= 1e-13) and two (day mrro,
mrso) whose lat_bnds differed by 7.8e-3; the publisher stalled on them and
the tree had to be repaired in place (fix_coord_consistency.py).

What: the first input file written for a dataset fixes the reference
(every numeric variable without a time dimension, except the data
variable itself: lat, lon, their bounds, plev, depth, 2D ORCA lat/lon and
vertices, ...). It is stored as a small .npz sidecar under the output root,
so it survives across bundles, chunks and jobs. Every later input file is
compared with it before anything is written:
  - equal: nothing to do;
  - differs by at most tol: the reference values are written instead, and
    the difference is logged (last-bit drift);
  - differs by more than tol, or has another shape/structure: the run
    stops with the variable, the file and the largest difference.
Nothing is ever copied silently.
"""

import os

import numpy as np

DEFAULT_TOL = 1e-10
_MASK_PREFIX = "__mask__"
_ORIGIN_KEY = "__origin__"


def _is_time_dependent(var):
    return any(d.startswith("time") for d in var.dimensions)


def static_arrays(ds, exclude=()):
    """name -> array for every numeric variable of ds with no time dimension."""
    out = {}
    for name, var in ds.variables.items():
        if name in exclude or _is_time_dependent(var):
            continue
        if var.dtype.kind not in "fiu":
            continue            # char label axes (landuse, basin names)
        out[name] = var[:]
    return out


def _max_diff(ref, new):
    """Largest absolute difference, np.inf when the masks disagree."""
    rm = np.ma.getmaskarray(ref)
    nm = np.ma.getmaskarray(new)
    if rm.shape != nm.shape or np.any(rm != nm):
        return np.inf
    r = np.ma.getdata(ref)[~rm].astype(np.float64)
    n = np.ma.getdata(new)[~nm].astype(np.float64)
    if r.size == 0:
        return 0.0
    return float(np.max(np.abs(r - n)))


class CoordReference(object):
    def __init__(self, arrays, origin=""):
        self.arrays = arrays
        self.origin = origin

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            origin = str(z[_ORIGIN_KEY]) if _ORIGIN_KEY in keys else ""
            arrays = {}
            for k in keys:
                if k.startswith(_MASK_PREFIX) or k == _ORIGIN_KEY:
                    continue
                a = z[k]
                mk = _MASK_PREFIX + k
                arrays[k] = np.ma.array(a, mask=z[mk]) if mk in keys else a
        return cls(arrays, origin)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {_ORIGIN_KEY: np.array(self.origin)}
        for k, a in self.arrays.items():
            payload[k] = np.ma.getdata(a)
            m = np.ma.getmaskarray(a)
            if m.any():
                payload[_MASK_PREFIX + k] = m
        tmp = f"{path}.tmp-{os.getpid()}.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, path)       # atomic: no reader sees half a file

    def compare(self, ds, exclude=(), tol=DEFAULT_TOL):
        """(aligned, errors) for ds against the reference.
        aligned: [(name, maxdiff)] within tol but not equal.
        errors: [str] above tol or structural differences."""
        new = static_arrays(ds, exclude)
        aligned, errors = [], []
        for name in sorted(set(self.arrays) | set(new)):
            if name not in new:
                errors.append(f"{name}: in the reference, missing from this file")
                continue
            if name not in self.arrays:
                errors.append(f"{name}: in this file, not in the reference")
                continue
            ref, arr = self.arrays[name], new[name]
            if np.shape(ref) != np.shape(arr):
                errors.append(f"{name}: shape {np.shape(arr)} vs reference {np.shape(ref)}")
                continue
            d = _max_diff(ref, arr)
            if d == 0.0:
                continue
            if d <= tol:
                aligned.append((name, d))
            else:
                errors.append(f"{name}: max abs difference {d:.3e} > tol {tol:g}"
                              + (" (mask differs)" if d == np.inf else ""))
        return aligned, errors


def reference_path(output_dir, parts):
    """Sidecar path for one dataset; parts identify it (source, experiment,
    variant, region, frequency, branded name, grid label)."""
    safe = "_".join(str(p).replace(os.sep, "-") for p in parts)
    # Directory name kept for references already on disk.
    return os.path.join(output_dir, ".recmorise-coord-ref", safe + ".npz")


class _VarView(object):
    """A netCDF4 variable whose values come from the reference when given."""

    def __init__(self, var, ref):
        self._var = var
        self._ref = ref

    def __getitem__(self, key):
        return self._ref[key] if self._ref is not None else self._var[key]

    def __len__(self):
        return len(self._var)

    def __getattr__(self, name):
        return getattr(self._var, name)


class RefView(object):
    """Read-only view of a netCDF4.Dataset in which every static variable
    known to the reference reads the reference values. Everything else
    (data, time, attributes, dimensions) comes from the file itself."""

    def __init__(self, ds, reference=None):
        self._ds = ds
        arrays = reference.arrays if reference is not None else {}
        self.variables = {n: _VarView(v, arrays.get(n))
                          for n, v in ds.variables.items()}

    def __getattr__(self, name):
        return getattr(self._ds, name)


def commit_coord_reference(ref):
    """Save a reference created in this run, once its first file is written."""
    path = getattr(ref, "pending_path", None) if ref is not None else None
    if not path:
        return
    ref.save(path)
    ref.pending_path = None
    print(f"  coord-ref: saved {os.path.basename(path)} (from {ref.origin})")
