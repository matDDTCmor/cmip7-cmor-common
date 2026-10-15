# cmip7-cmor-common

Shared, CMOR-free code for CMIP7 cmorization at BSC, used by the relay
(CMIP6 to CMIP7) and by direct cmorization from model output.

- `bundles`: years per output file and bundle planning.
- `coordref`: persistent reference for the time-independent variables of a
  dataset (lat, lon, bounds, levels), compared with tolerance.
- `registry`: CMIP7 targets read from the BSC variable registry.
- `cvstate`: tables staging and augmented CV in a state dir; no silent
  CV stubs.

Content (the mapping XML) lives in bsc-variable-registry and is passed by
path; this package reads it and never ships a copy.

## Install

    pip install -e .            # development
    pip install .               # deployment

## Tests

    pip install -e .[dev]
    python -m pytest tests/

`CMIP7_REGISTRY_XML=<path>` also runs the checks against a real registry.
