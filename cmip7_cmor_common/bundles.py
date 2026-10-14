"""
Multi-year output files ("bundles").

Years per file follow ece2cmor3's recmorise-cmip6-to-cmip7.py (nyears from
frequency, number of levels and horizontal size: 1, 5, 10, 20 or 50 years)
with the conventions of offline-recmorise-abbd rebundle_cmip7.py, which
produced the published abbd tree: bundles aligned to multiples of nyears,
shifted by +1 year for scen7 experiments.

A bundle is identified by its nominal range [a, b]. Its expected input
years are the years of [a, b] inside the run (first_year .. last year
known), and it closes (is written) at the year it ends: b, or earlier at
the end of the run (--final). Bundle boundaries do not depend on which
years are present, so a relay that only sees some chunk directories plans
the same bundles as an offline run over the whole series.

No CMOR here: pure planning, testable on its own.
"""

import datetime
import re

# Time steps per year by CMIP7 frequency (base names, see base_frequency).
TSTEPS = {"yr": 1, "dec": 1, "fx": 1, "mon": 12, "day": 365.25,
          "6hr": 365.25 * 4, "3hr": 365.25 * 8, "1hr": 365.25 * 24}

# Start year of a CMIP6 source file from its name (..._185001-185012.nc).
_SOURCE_YEAR_RE = re.compile(r"_(\d{4})(\d*)-(\d{4})(\d*)\.nc$")

# End stamp at the very start of a year: 0101, 01010000, or 01 for monthly
# stamps. Point-in-time (tpt) yearly files end there, e.g. SIday
# 20250102-20260101 or SImon 202502-202601, and still hold one year of data.
_YEAR_START_STAMP_RE = re.compile(r"01(01)?0*")


def base_frequency(freq):
    """CMIP7 frequency without point/climatology suffixes (monPt -> mon,
    1hrCM -> 1hr, monC -> mon); None when unknown."""
    for suffix in ("Pt", "CM", "C"):
        if freq.endswith(suffix) and freq[:-len(suffix)] in TSTEPS:
            return freq[:-len(suffix)]
    return freq if freq in TSTEPS else None


def nyears_for(nlevs, grdpts, freq):
    """Years per output file, as upstream (L385-401) and rebundle_cmip7.py.
    nlevs: levels of the output variable (1 for 2D); grdpts: product of its
    last two dimensions. The time length does not enter the formula."""
    base = base_frequency(freq)
    if base is None:
        raise ValueError(f"unknown frequency {freq!r} for multi-year files")
    n = int((12 * 75 * 362 * 262) / (nlevs * grdpts * TSTEPS[base]) * 7)
    for lim, val in ((5, 1), (10, 5), (20, 10), (50, 20)):
        if n < lim:
            return val
    return 50


def bundle_bounds(year, nyears, experiment_id):
    """Nominal [a, b] of the bundle holding year."""
    off = 1 if "scen7" in (experiment_id or "") else 0
    if nyears == 1:
        return year, year
    a = ((year - off) // nyears) * nyears + off
    return a, a + nyears - 1


def source_years(path):
    """(start year, end year) of a CMIP6 source file from its name, or None."""
    m = _SOURCE_YEAR_RE.search(path)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(3))
    # A yearly point-in-time file whose last value falls at the start of the
    # next year belongs to its start year.
    if (b == a + 1 and m.group(2) and m.group(2) != m.group(4)
            and _YEAR_START_STAMP_RE.fullmatch(m.group(4))):
        b = a
    return (a, b)


def close_year_from_date(date_str):
    """Last complete year when a chunk ends at date_str (YYYYMMDD[hh]).
    Accepts both Autosubmit conventions: the inclusive last day (YYYY1231)
    and the exclusive end ((YYYY+1)0101). Any other date ends mid-year:
    returns None (no bundle closes)."""
    d = datetime.datetime.strptime(date_str[:8], "%Y%m%d").date()
    if (d.month, d.day) == (12, 31):
        return d.year
    if (d.month, d.day) == (1, 1):
        return d.year - 1
    return None


class Bundle(object):
    def __init__(self, a, b, files):
        self.a, self.b = a, b
        self.files = files          # [(path, start_year)], sorted by year
        self.expected = []          # years this bundle must cover
        self.close_at = b           # year at which it is written

    @property
    def years(self):
        return [y for _, y in self.files]

    @property
    def paths(self):
        return [p for p, _ in self.files]

    def label(self):
        if not self.files:
            return f"{self.a}-{self.b} (no inputs)"
        return f"{self.years[0]}-{self.years[-1]}"

    def missing_years(self):
        have = set(self.years)
        return [y for y in self.expected if y not in have]


def plan_bundles(files, nyears, experiment_id, first_year=None, last_year=None,
                 close_year=None, final=False, catch_up=False):
    """Group input files into bundles and select the ones to write now.

    files: input paths (one year each, CMIP6 relay inputs are yearly).
    first_year: first year of the run (default: first input year).
    last_year: last year of the run, only meaningful with final.
    close_year: None -> every bundle is selected (offline run over the whole
        series, incomplete ones included at the series end). Otherwise only
        bundles closing at close_year are selected (online, one chunk).
    final: this is the end of the run: bundles still open at close_year
        (or at the last input year) close now, shortened.
    catch_up: with close_year, also select bundles that closed earlier
        (for --resume, which then skips the ones already written and sound).

    Returns (selected, still_open, done): lists of Bundle. done: closed at
    an earlier chunk and not selected (their staged inputs can go).
    Raises ValueError on a file without a year or with several years.
    """
    with_years = []
    for p in files:
        yy = source_years(p)
        if yy is None:
            raise ValueError(f"no year range in file name: {p}")
        if yy[0] != yy[1]:
            raise ValueError(f"input spans several years ({yy[0]}-{yy[1]}), "
                             f"multi-year output expects yearly inputs: {p}")
        with_years.append((p, yy[0]))
    with_years.sort(key=lambda t: (t[1], t[0]))
    if not with_years:
        return [], [], []

    years_present = [y for _, y in with_years]
    dup = sorted({y for y in years_present if years_present.count(y) > 1})
    if dup:
        raise ValueError(f"several input files for year(s) {dup} "
                         f"(overlapping CMIP6 versions or chunk dirs?)")

    run_first = first_year if first_year is not None else years_present[0]
    if final:
        run_last = last_year if last_year is not None else (
            close_year if close_year is not None else years_present[-1])
    elif close_year is None:
        run_last = years_present[-1]    # offline: the series ends here
    else:
        run_last = None                 # online, not final: end unknown

    groups = {}
    for p, y in with_years:
        if y < run_first:
            continue
        groups.setdefault(bundle_bounds(y, nyears, experiment_id), []).append((p, y))

    selected, still_open, done = [], [], []
    for (a, b), fl in sorted(groups.items()):
        bd = Bundle(a, b, fl)
        start = max(a, run_first)
        end = b if run_last is None else min(b, run_last)
        bd.close_at = end
        bd.expected = list(range(start, end + 1))
        if close_year is None:
            selected.append(bd)
        elif bd.close_at == close_year or (catch_up and bd.close_at < close_year):
            selected.append(bd)
        elif bd.close_at > close_year:
            still_open.append(bd)
        else:
            done.append(bd)     # written by an earlier chunk
    return selected, still_open, done
