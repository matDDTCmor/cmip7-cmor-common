"""Reduced Gaussian grids: rows north to south, linear in longitude to regular.

Moved from ecemor/io/xios.py; used by ecemor and by the pycmor
EC-Earth4 steps. numpy only.
"""
from __future__ import annotations

import numpy as np


class GridError(ValueError):
    """Grid not usable as a reduced Gaussian grid."""


class ReducedGaussian:
    """Reduced Gaussian rows; linear in longitude to regular."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        lat = np.asarray(lat, dtype="f8")
        lon = np.asarray(lon, dtype="f8")
        brk = np.flatnonzero(np.abs(np.diff(lat)) > 1e-6) + 1
        starts = np.concatenate([[0], brk])
        counts = np.diff(np.concatenate([starts, [lat.size]]))
        rlat = lat[starts]
        if np.unique(np.round(rlat, 6)).size != rlat.size:
            raise GridError("latitude rows are not contiguous")
        if not np.all(np.diff(rlat) < 0):
            raise GridError("latitude rows are not north to south")
        nlon = int(counts.max())
        for s, n in zip(starts, counts):
            want = np.arange(n) * 360.0 / n
            got = np.mod(lon[s:s + n], 360.0)
            if not np.allclose(got, want, atol=1e-4):
                raise GridError(f"row at lat {lat[s]:.4f}: longitudes not uniform from 0")
        self.lat = rlat
        self.lon = np.arange(nlon) * 360.0 / nlon
        self.ncell = lat.size
        i0, i1, w = [], [], []
        for s, n in zip(starts, counts):
            x = self.lon * n / 360.0
            j = np.floor(x).astype(int)
            i0.append(s + j % n)
            i1.append(s + (j + 1) % n)
            w.append(x - j)
        self._i0 = np.concatenate(i0)
        self._i1 = np.concatenate(i1)
        self._w = np.concatenate(w)
        self._near = np.where(self._w < 0.5, self._i0, self._i1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.size, self.lon.size

    def to_regular(self, data: np.ndarray, nearest: bool = False) -> np.ndarray:
        """(..., cell) -> (..., nlat, nlon); nearest keeps land/sea edges."""
        if data.shape[-1] != self.ncell:
            raise GridError(f"last dimension {data.shape[-1]} is not {self.ncell} cells")
        if nearest:
            return data[..., self._near].reshape(data.shape[:-1] + self.shape)
        a, b = data[..., self._i0], data[..., self._i1]
        # w == 0: exact point, ignore neighbour
        out = np.where(self._w == 0.0, a, a + self._w * (b - a))
        return out.reshape(data.shape[:-1] + self.shape)
