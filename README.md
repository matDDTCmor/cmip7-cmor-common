# cmip7-cmor-common

Code shared by BSC's CMIP7 CMORization tools that does not depend on the
writer: the relay (CMIP6 to CMIP7, CMOR), ecemor (model output, CMOR) and
the EC-Earth4 steps in pycmor (no CMOR). None of them can import from
another without pulling in its writer, so shared code lives here.

| Consumer | Writer | Uses |
|---|---|---|
| recmorise-relay | CMOR | bundles, coordref |
| ecemor | CMOR | registry, cvstate, gaussian; bundles, coordref planned |
| pycmor (dev/cmip7-ecearth4) | xarray | gaussian; registry, cvstate planned |

## What belongs here

All three hold:

1. Used by at least two of the tools, or planned for a second one with
   a decision record.
2. Does not call the writer (no CMOR, no pycmor).
3. Light dependencies only (numpy, netCDF4).

Writer calls, pycmor step wrappers and code used by one tool stay in that
tool.

## Modules

Content: what to produce and with which tables and CVs.

- `registry`: CMIP7 targets read from the BSC variable registry.
- `cvstate`: tables staging and augmented CV in a state dir; no silent
  CV stubs.

Content (the mapping XML, tables, CVs) is passed by path; this package
reads it and never ships a copy.

Model output: getting data ready for any writer.

- `gaussian`: reduced Gaussian rows to a regular grid (linear in
  longitude, or nearest for land/sea fields); OIFS and LPJG output.
- `bundles`: years per output file and bundle planning.
- `coordref`: persistent reference for the time-independent variables of
  a dataset (lat, lon, bounds, levels), compared with tolerance.

The two halves change for different reasons (content versions vs model
grids and formats). They split into two packages when the model-output
half grows (e.g. a NEMO reader) or the halves need different
dependencies.

## Install

    pip install -e .            # development
    pip install .               # deployment

## Tests

    pip install -e .[dev]
    python -m pytest tests/

`CMIP7_REGISTRY_XML=<path>` also runs the checks against a real registry.
