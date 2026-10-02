"""
utils/data_loader.py
====================
Access layer for the *MTA Subway Hourly Ridership: 2020-2024* dataset
(Socrata view ``wujg-7c2s``).

The dataset holds roughly **121 million raw rows**, so nothing in this project
ever reads it. Every request issued here is a *server-side aggregate* (SoQL
``$select`` / ``$group``) whose result is small enough to live in memory:

===========================  ==================  =========================
Cube                         Grouped by           Rows over 2020-2024
===========================  ==================  =========================
``hourly``                   transit_timestamp    43,848
``station_daily``            station, calendar d  ~780,000
``station_totals``           station              428
``borough_totals``           borough              5
``payment_totals``           payment_method       2
``fare_totals``              fare_class_category  12
``station_hourly`` (1 stat.) transit_timestamp    43,848
===========================  ==================  =========================

Three rules keep the dashboard fast:

1. **Aggregate remotely, never download.** ``fetch_aggregate`` always sends a
   ``$group``; the browser/Python side only ever sees the grouped result.
2. **Chunk by year.** A single 5-year aggregate is both slow *and* large
   enough that the endpoint drops the connection, so long ranges are split into
   per-year chunks and concatenated. Each chunk response stays tiny.
3. **Cache to Parquet on disk.** ``AggregateCache`` memoises every aggregate
   under ``data/processed/`` keyed by a hash of its parameters, so a repeated
   filter combination costs a file read instead of a network round trip - and
   the cache survives a restart of the app.

Every network failure (timeout, reset connection, 4xx/5xx, malformed body)
surfaces as :class:`ApiUnavailableError` with a message written for
``st.error`` rather than as a raw traceback.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------

#: Socrata view id for "MTA Subway Hourly Ridership: 2020-2024".
DATASET_ID = "wujg-7c2s"

#: CSV flavour of the SODA endpoint. JSON is used for the scalar probes.
CSV_ENDPOINT = f"https://data.ny.gov/resource/{DATASET_ID}.csv"
JSON_ENDPOINT = f"https://data.ny.gov/resource/{DATASET_ID}.json"

#: Observed coverage of the dataset (verified against the API).
DATA_START = pd.Timestamp("2020-01-01 00:00:00")
DATA_END = pd.Timestamp("2024-12-31 23:00:00")

#: Default analysis window: the last three complete years in the dataset.
DEFAULT_START = pd.Timestamp("2022-01-01 00:00:00")
DEFAULT_END = pd.Timestamp("2024-12-31 23:00:00")

#: Project layout. ``data/processed`` holds the Parquet aggregate cache.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

#: Network behaviour. The endpoint needs 30-130s for a full-range aggregate, so
#: the timeout is generous; retries ride on top of it for transient resets.
REQUEST_TIMEOUT = 300
MAX_RETRIES = 3
RETRY_BACKOFF = 4.0

#: SODA caps a response at 1000 rows unless told otherwise, silently. Every
#: request therefore carries an explicit, generous limit.
ROW_LIMIT = 1_000_000

#: Aggregate chunks never span more than this, which keeps each HTTP response
#: small enough that the endpoint does not reset the connection.
CHUNK = pd.Timedelta(days=366)

#: The hourly cube is special-cased to a month per request. Measured against
#: the live endpoint, grouping by ``transit_timestamp`` costs roughly
#: proportional to the number of hourly groups returned: one month (~744
#: groups) takes ~24s, while a full year (~8,760 groups) reliably exceeds the
#: request timeout. Monthly chunks keep every response small and let a partial
#: build resume instead of losing an hour of work.
HOURLY_CHUNK = pd.Timedelta(days=31)

_UA = {"User-Agent": "mta-ridership-dashboard/1.0", "Accept": "text/csv,*/*"}


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class DataLoadError(Exception):
    """Raised when the dataset cannot be prepared for the dashboard.

    The message is written to be shown directly inside ``st.error``.
    """


class ApiUnavailableError(DataLoadError):
    """Raised when the Socrata endpoint cannot be reached or misbehaves."""


# --------------------------------------------------------------------------
# Filter model
# --------------------------------------------------------------------------


def _quote(value: str) -> str:
    """Escape a value for a SoQL single-quoted string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quote_many(values: Sequence[str]) -> str:
    return "(" + ", ".join(_quote(v) for v in values) + ")"


def _ts(value: Any) -> str:
    """Format a timestamp as a SoQL *date* literal.

    ``str(pd.Timestamp(...))`` yields ``2020-06-01 00:00:00``, which SoQL treats
    as text and then rejects inside a ``between`` comparison, so the literal is
    always built explicitly.
    """
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True)
class Filters:
    """The sidebar selection, in the shape the SoQL ``$where`` clause needs.

    ``None`` means "no restriction" for a dimension; an empty tuple means the
    user explicitly deselected everything and should see an empty state rather
    than silently falling back to the full dataset.
    """

    start: pd.Timestamp = DEFAULT_START
    end: pd.Timestamp = DEFAULT_END
    boroughs: Optional[Tuple[str, ...]] = None
    stations: Optional[Tuple[str, ...]] = None  # station_complex_id strings
    modes: Optional[Tuple[str, ...]] = None
    payments: Optional[Tuple[str, ...]] = None
    fares: Optional[Tuple[str, ...]] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        start: Any,
        end: Any,
        boroughs: Optional[Sequence[str]] = None,
        stations: Optional[Sequence[str]] = None,
        modes: Optional[Sequence[str]] = None,
        payments: Optional[Sequence[str]] = None,
        fares: Optional[Sequence[str]] = None,
    ) -> "Filters":
        """Coerce loose sidebar values (dates, ``None``, lists) into a Filters."""
        return cls(
            start=_coerce_ts(start, DEFAULT_START),
            end=_coerce_ts(end, DEFAULT_END),
            boroughs=_opt_tuple(boroughs),
            stations=_opt_tuple(stations),
            modes=_opt_tuple(modes),
            payments=_opt_tuple(payments),
            fares=_opt_tuple(fares),
        )

    # -- SoQL ---------------------------------------------------------------

    @property
    def lo(self) -> pd.Timestamp:
        return min(self.start, self.end).normalize()

    @property
    def hi(self) -> pd.Timestamp:
        return max(self.start, self.end).normalize()

    def is_empty_selection(self) -> bool:
        """True when a dimension was explicitly deselected to zero options."""
        return any(
            dim == ()
            for dim in (
                self.boroughs,
                self.stations,
                self.modes,
                self.payments,
                self.fares,
            )
        )

    def spans(self, chunk: Optional[pd.Timedelta] = None) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Split the window into chunks of at most ``chunk`` (default :data:`CHUNK`)."""
        step = chunk or CHUNK
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = self.lo
        hi = self.hi
        while cursor <= hi:
            stop = min(cursor + step - pd.Timedelta(days=1), hi)
            out.append((cursor, stop))
            cursor = stop + pd.Timedelta(days=1)
        return out or [(self.lo, self.hi)]

    def where(self, extra: str = "") -> str:
        """Build the ``$where`` clause for this filter set."""
        clauses = [
            f"transit_timestamp between {_quote(self._ts(self.lo))} "
            f"and {_quote(self._ts(self.hi + pd.Timedelta(hours=23)))}"
        ]
        for column, values in (
            ("borough", self.boroughs),
            ("station_complex_id", self.stations),
            ("transit_mode", self.modes),
            ("payment_method", self.payments),
            ("fare_class_category", self.fares),
        ):
            if values is not None:
                clauses.append(f"{column} in {_quote_many(values)}")
        if extra:
            clauses.append(f"({extra})")
        return " and ".join(clauses)

    def scoped(self, start: pd.Timestamp, end: pd.Timestamp) -> "Filters":
        """A copy of these filters restricted to one chunk (inclusive days)."""
        return replace(
            self,
            start=pd.Timestamp(start).normalize(),
            end=pd.Timestamp(end).normalize() + pd.Timedelta(hours=23),
        )

    # -- caching -------------------------------------------------------------

    def signature(self) -> str:
        """Stable short hash of everything that changes the query result."""
        parts = [
            self._ts(self.lo),
            self._ts(self.hi),
            repr(self.boroughs),
            repr(self.stations),
            repr(self.modes),
            repr(self.payments),
            repr(self.fares),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        """Human-readable summary used in captions and the cache manifest."""
        bits = [f"{self.lo:%d %b %Y} to {self.hi:%d %b %Y}"]
        for label, values in (
            ("borough", self.boroughs),
            ("station", self.stations),
            ("mode", self.modes),
            ("payment", self.payments),
            ("fare", self.fares),
        ):
            if values is None:
                continue
            bits.append(f"{label}={list(values) if len(values) <= 3 else f'{len(values)} selected'}")
        return "; ".join(bits)

    @staticmethod
    def _ts(value: pd.Timestamp) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


def _coerce_ts(value: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if isinstance(value, date):
        return pd.Timestamp(value)
    try:
        return pd.Timestamp(str(value))
    except Exception as exc:  # noqa: BLE001 - any parse failure is reportable
        raise DataLoadError(f"Could not interpret {value!r} as a date.") from exc


def _opt_tuple(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """``None`` -> unrestricted; ``[]`` -> explicitly empty; else a tuple."""
    if values is None:
        return None
    return tuple(str(v) for v in values)


def default_filters() -> Filters:
    """The documented default window: Jan 2022 -> Dec 2024, no dimension filter."""
    return Filters(start=DEFAULT_START, end=DEFAULT_END)


def dataset_span() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Full extent of the dataset (constants verified against the API)."""
    return DATA_START, DATA_END


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

_TRANSIENT = (
    urllib.error.URLError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    socket.timeout,
    ConnectionError,
    TimeoutError,
)


def _request(endpoint: str, params: Dict[str, str]) -> str:
    """GET a SODA endpoint with retries, returning the raw body as text.

    Raises :class:`ApiUnavailableError` with a user-facing message for every
    failure mode: unreachable host, timeout, HTTP error, or empty body.
    """
    url = endpoint + "?" + urllib.parse.urlencode(params)
    last: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if not body.strip():
                raise ApiUnavailableError(
                    "The NYC Open Data API returned an empty response. "
                    "This usually means the query matched no rows - try widening "
                    "the date range or clearing a filter."
                )
            return body
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001 - body may already be consumed
                pass
            # 4xx means the query itself is wrong; retrying will not help.
            if 400 <= exc.code < 500:
                raise ApiUnavailableError(
                    f"The NYC Open Data API rejected the query "
                    f"(HTTP {exc.code} {exc.reason}). {detail}"
                ) from exc
            last = exc
        except ApiUnavailableError:
            raise
        except _TRANSIENT as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - never let the app crash
            raise ApiUnavailableError(
                f"Unexpected error while calling the NYC Open Data API: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise ApiUnavailableError(
        "The NYC Open Data API did not respond after "
        f"{MAX_RETRIES} attempts ({type(last).__name__}: {last}). "
        "The service may be temporarily unavailable - please retry in a moment."
    )


# --------------------------------------------------------------------------
# Parquet-backed aggregate cache
# --------------------------------------------------------------------------


class AggregateCache:
    """Memoises aggregate results as Parquet under ``data/processed``.

    Two caches are layered, because Streamlit's own cache is process-local:

    * the **disk cache** survives restarts, which is what makes a rebuilt
      dashboard load instantly;
    * ``@st.cache_data`` (applied by the app) avoids even the disk read.

    An in-flight ``threading.Lock`` per key prevents two concurrent Streamlit
    sessions from issuing the same slow query twice.
    """

    def __init__(self, directory: Path = PROCESSED_DIR) -> None:
        self.directory = Path(directory)
        self._locks: Dict[str, Any] = {}

    # -- paths ---------------------------------------------------------------

    def path_for(self, cube: str, key: str) -> Path:
        return self.directory / f"{cube}__{key}.parquet"

    # -- public API ----------------------------------------------------------

    def get_or_build(self, cube: str, key: str, builder) -> pd.DataFrame:
        """Return the cached frame for ``(cube, key)``, calling ``builder`` once."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(cube, key)
        if target.exists():
            try:
                return pd.read_parquet(target)
            except Exception:  # noqa: BLE001 - a corrupt file is simply rebuilt
                target.unlink(missing_ok=True)

        lock = self._locks.setdefault(f"{cube}:{key}", _thread_lock())
        with lock:
            if target.exists():
                try:
                    return pd.read_parquet(target)
                except Exception:  # noqa: BLE001
                    target.unlink(missing_ok=True)
            frame = builder()
            if frame is not None and not frame.empty:
                frame.to_parquet(target, index=False)
            return frame

    def clear(self) -> int:
        """Delete every cached cube. Returns the number of files removed."""
        removed = 0
        if self.directory.exists():
            for path in self.directory.glob("*.parquet"):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def inventory(self) -> List[Dict[str, Any]]:
        """One row per cached cube, for the data-quality tab."""
        rows: List[Dict[str, Any]] = []
        if not self.directory.exists():
            return rows
        for path in sorted(self.directory.glob("*.parquet")):
            try:
                info = pd.read_parquet(path)
                rows.append(
                    {
                        "Cube": path.stem.split("__")[0],
                        "Rows": int(len(info)),
                        "Size (KB)": round(path.stat().st_size / 1024, 1),
                    }
                )
            except Exception:  # noqa: BLE001 - inventory must never crash
                continue
        return rows


def _thread_lock():
    import threading

    return threading.Lock()


#: Module-level cache shared by every loader in this module.
CACHE = AggregateCache()


def cache_status() -> pd.DataFrame:
    """Inventory of the on-disk aggregate cache."""
    rows = CACHE.inventory()
    return pd.DataFrame(rows, columns=["Cube", "Rows", "Size (KB)"])


def _remember(**entries: Any) -> None:
    """Record provenance in ``data/processed/manifest.json``."""
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {}
        if MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest.update(entries)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass


# --------------------------------------------------------------------------
# Aggregate fetching
# --------------------------------------------------------------------------


def _fetch(
    select: str,
    group: str,
    filters: Filters,
    order: str = "",
    cube: str = "adhoc",
    extra_where: str = "",
    full_table: bool = False,
    chunk: Optional[pd.Timedelta] = None,
    key_cols: Optional[Sequence[str]] = None,
    single: bool = False,
) -> pd.DataFrame:
    """Run a server-side aggregate, chunk by year, and cache the concatenation.

    Chunking is what keeps this reliable: a single five-year ``$group`` is slow
    *and* large enough that the endpoint drops the connection, whereas monthly
    chunks return in seconds with a few hundred rows each.

    ``full_table=True`` skips the date predicate and sends one request. That is
    the cheaper path for the *reference* cubes (domain values, station names and
    coordinates) because they are grouped by a handful of low-cardinality keys
    and therefore always come back small.
    """
    key = hashlib.sha1(
        f"{select}|{group}|{order}|{extra_where}|{full_table}|{chunk}|{single}"
        f"|{filters.signature()}".encode("utf-8")
    ).hexdigest()[:12]

    def build() -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        windows: List[Optional[Tuple[pd.Timestamp, pd.Timestamp]]] = (
            [None]
            if full_table
            else ([(filters.lo, filters.hi)] if single else filters.spans(chunk))
        )
        for window in windows:
            params: Dict[str, str] = {
                "$select": select,
                "$group": group,
                "$limit": str(ROW_LIMIT),
            }
            if window is not None:
                params["$where"] = filters.scoped(*window).where(extra_where)
            elif extra_where:
                params["$where"] = extra_where
            elif not full_table and not single:
                params["$where"] = filters.where(extra_where)
            if order:
                params["$order"] = order
            body = _request(CSV_ENDPOINT, params)
            part = pd.read_csv(pd.io.common.StringIO(body))
            if not part.empty:
                frames.append(part)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        # Chunks cover disjoint date ranges, so each key appears once *per
        # chunk*: the measures must be summed across chunks. Dropping or
        # overwriting duplicates would silently report only one chunk's worth
        # of ridership (or, for a dimension cube, repeat it three times).
        keys = [c for c in (key_cols or []) if c in out.columns]
        if keys:
            measures = [c for c in out.columns if c not in keys]
            out = out.groupby(keys, as_index=False, dropna=False)[measures].sum()
        else:
            out = out.drop_duplicates()
        return out.reset_index(drop=True)

    frame = CACHE.get_or_build(cube, key, build)
    _remember(
        dataset_id=DATASET_ID,
        last_cube=cube,
        last_filters=filters.describe(),
    )
    return frame


def _scalar(params: Dict[str, str]) -> Any:
    """Run a single-row JSON query (used for counts and extents)."""
    body = _request(JSON_ENDPOINT, params)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiUnavailableError(
            "The NYC Open Data API returned a malformed response that could not "
            f"be parsed as JSON. First 200 characters: {body[:200]!r}"
        ) from exc
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return payload


# --------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------


def _fetch_hourly(filters: Filters) -> pd.DataFrame:
    """Fetch the hourly series one month at a time, caching each month.

    Each month is memoised under its own Parquet key, so a build that is
    interrupted half-way keeps its completed months and only re-fetches what is
    missing. A three-year window is 36 such chunks.
    """
    select = "transit_timestamp, sum(ridership) as ridership, sum(transfers) as transfers"
    group = "transit_timestamp"
    base = hashlib.sha1(
        f"{select}|{group}|{filters.signature()}".encode("utf-8")
    ).hexdigest()[:12]

    frames: List[pd.DataFrame] = []
    for index, (lo, hi) in enumerate(filters.spans(HOURLY_CHUNK)):

        def build(lo=lo, hi=hi) -> pd.DataFrame:
            params = {
                "$select": select,
                "$group": group,
                "$where": filters.scoped(lo, hi).where(),
                "$order": "transit_timestamp",
                "$limit": str(ROW_LIMIT),
            }
            body = _request(CSV_ENDPOINT, params)
            return pd.read_csv(pd.io.common.StringIO(body))

        chunk = CACHE.get_or_build("hourly", f"{base}_{index:02d}", build)
        if chunk is not None and not chunk.empty:
            frames.append(chunk)

    _remember(dataset_id=DATASET_ID, last_cube="hourly", last_filters=filters.describe())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_hourly(filters: Filters) -> pd.DataFrame:
    """CITYWIDE hourly series: one row per ``transit_timestamp``.

    This is the primary time-series target. ``sum(ridership)`` here is taken
    **server-side across every payment_method and fare_class_category**, which
    is exactly what stops the raw rows from being counted as independent
    station-hour observations.
    """
    frame = _fetch_hourly(filters)
    return _finalise_hourly(frame)


def load_station_hourly(filters: Filters, station_id: str) -> pd.DataFrame:
    """Hourly series for a single station complex (used by the drill-down)."""
    if not filters.stations:
        scoped = replace(filters, stations=(str(station_id),))
    else:
        chosen = {str(s) for s in filters.stations}
        chosen.add(str(station_id))
        scoped = replace(filters, stations=tuple(sorted(chosen)))
    return load_hourly(scoped)


def load_station_daily(filters: Filters) -> pd.DataFrame:
    """STATION x DAY cube (782k rows over five years).

    ``date_extract_*`` is used because Socrata rejects a function call inside
    ``$group``; the three components are stitched back into a date locally.
    """
    frame = _fetch(
        select="station_complex_id, date_extract_y(transit_timestamp) as y, "
               "date_extract_m(transit_timestamp) as m, "
               "date_extract_d(transit_timestamp) as d, "
               "sum(ridership) as ridership, sum(transfers) as transfers",
        group="station_complex_id, date_extract_y(transit_timestamp), "
              "date_extract_m(transit_timestamp), date_extract_d(transit_timestamp)",
        filters=filters,
        cube="station_daily",
        key_cols=["station_complex_id", "y", "m", "d"],
    )
    if frame.empty:
        return pd.DataFrame(columns=["station_complex_id", "date", "ridership", "transfers"])
    frame["date"] = pd.to_datetime(
        frame["y"].astype(int).astype(str)
        + "-"
        + frame["m"].astype(int).astype(str).str.zfill(2)
        + "-"
        + frame["d"].astype(int).astype(str).str.zfill(2),
        errors="coerce",
    )
    frame = frame.dropna(subset=["date"])
    return frame[["station_complex_id", "date", "ridership", "transfers"]].reset_index(
        drop=True
    )


def load_station_totals(filters: Filters) -> pd.DataFrame:
    """One row per station complex: totals, transfers and raw-row count.

    Note on ``raw_rows``: it is *not* a station-hour count. The raw dataset
    splits a station-hour across payment methods and fare classes, so this is
    roughly ten times the number of observed hours. It is reported so the data
    quality tab can state how many dataset rows were aggregated, and it is
    never used as a divisor.
    """
    frame = _fetch(
        select="station_complex_id, sum(ridership) as ridership, "
               "sum(transfers) as transfers, count(*) as raw_rows",
        group="station_complex_id",
        filters=filters,
        order="ridership DESC",
        cube="station_totals",
        key_cols=["station_complex_id"],
        single=True,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["station_complex_id", "ridership", "transfers", "raw_rows"]
        )
    # Station ids arrive as integers but are compared against sidebar strings
    # and joined to the reference table, so they are normalised once here.
    frame["station_complex_id"] = frame["station_complex_id"].astype(str)
    return frame


def load_borough_totals(filters: Filters) -> pd.DataFrame:
    """One row per borough, including how many station complexes it contains."""
    frame = _fetch(
        select="borough, sum(ridership) as ridership, sum(transfers) as transfers, "
               "count(*) as raw_rows",
        group="borough",
        filters=filters,
        order="ridership DESC",
        cube="borough_totals",
        key_cols=["borough"],
        single=True,
    )
    if frame.empty:
        return pd.DataFrame(columns=["borough", "ridership", "transfers", "raw_rows"])
    return frame


def load_payment_totals(filters: Filters) -> pd.DataFrame:
    """Ridership by ``payment_method`` - categories are read, never hard-coded."""
    frame = _fetch(
        select="payment_method, sum(ridership) as ridership, "
               "sum(transfers) as transfers, count(*) as raw_rows",
        group="payment_method",
        filters=filters,
        order="ridership DESC",
        cube="payment_totals",
        key_cols=["payment_method"],
        single=True,
    )
    if frame.empty:
        return pd.DataFrame(columns=["payment_method", "ridership", "transfers", "raw_rows"])
    return frame


def load_fare_totals(filters: Filters) -> pd.DataFrame:
    """Ridership by ``fare_class_category`` - categories are read, never hard-coded."""
    frame = _fetch(
        select="fare_class_category, sum(ridership) as ridership, "
               "sum(transfers) as transfers, count(*) as raw_rows",
        group="fare_class_category",
        filters=filters,
        order="ridership DESC",
        cube="fare_totals",
        key_cols=["fare_class_category"],
        single=True,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["fare_class_category", "ridership", "transfers", "raw_rows"]
        )
    return frame


def load_mode_totals(filters: Filters) -> pd.DataFrame:
    """Ridership by ``transit_mode`` (subway / Staten Island Railway / tram)."""
    frame = _fetch(
        select="transit_mode, sum(ridership) as ridership, "
               "sum(transfers) as transfers, count(*) as raw_rows",
        group="transit_mode",
        filters=filters,
        order="ridership DESC",
        cube="mode_totals",
        key_cols=["transit_mode"],
        single=True,
    )
    if frame.empty:
        return pd.DataFrame(columns=["transit_mode", "ridership", "transfers", "raw_rows"])
    return frame


#: Sample months used to build the station reference table. Station names,
#: boroughs and coordinates are effectively static, and a single month already
#: contains ~427 of the 428 complexes, so three slices cover the whole network
#: in ~30s instead of the ~8 minutes a full five-year scan would cost.
META_SAMPLE_MONTHS: Tuple[str, ...] = ("2020-06", "2022-06", "2024-06")


def load_station_meta() -> pd.DataFrame:
    """Static station reference: name, borough and coordinates.

    Coordinates come from the dataset's own ``latitude``/``longitude`` columns -
    nothing is geocoded externally.

    The query groups by ``station_complex_id`` (428 groups) and takes ``max``
    of each descriptive field rather than grouping by all five columns: grouping
    by the long text columns as well makes the aggregate engine far slower for
    an identical result. A single unrestricted five-year request times out
    entirely, which is why the coverage is sampled month-by-month and unioned.
    """
    columns = ["station_complex_id", "station_complex", "borough", "latitude", "longitude"]

    def build() -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for month in META_SAMPLE_MONTHS:
            start = pd.Timestamp(f"{month}-01")
            end = start + pd.offsets.MonthEnd(0)
            params = {
                "$select": "station_complex_id, max(station_complex) as station_complex, "
                           "max(borough) as borough, min(latitude) as latitude, "
                           "min(longitude) as longitude",
                "$group": "station_complex_id",
                "$where": f"transit_timestamp between {_quote(_ts(start))} and "
                          f"{_quote(_ts(end + pd.Timedelta(hours=23)))}",
                "$limit": str(ROW_LIMIT),
            }
            body = _request(CSV_ENDPOINT, params)
            chunk = pd.read_csv(pd.io.common.StringIO(body))
            if not chunk.empty:
                frames.append(chunk)
        if not frames:
            return pd.DataFrame(columns=columns)
        out = pd.concat(frames, ignore_index=True)
        out["station_complex_id"] = out["station_complex_id"].astype(str)
        out = out.dropna(subset=["station_complex_id", "latitude", "longitude"])
        return out.drop_duplicates(subset="station_complex_id", keep="first")[
            columns
        ].reset_index(drop=True)

    return CACHE.get_or_build("station_meta", "all", build)


def load_domains() -> Dict[str, List[str]]:
    """Every categorical value the dataset actually contains.

    Nothing in the dashboard hard-codes a payment method, fare class, borough or
    transit mode; the sidebar options and every "unknown category" fallback are
    derived from these lists.
    """
    frame = CACHE.get_or_build("domains", "all", _build_domains)
    if frame.empty:
        return {k: [] for k in ("borough", "payment_method", "fare_class_category", "transit_mode")}
    out: Dict[str, List[str]] = {}
    for column in ("borough", "payment_method", "fare_class_category", "transit_mode"):
        if column in frame.columns:
            out[column] = sorted(str(v) for v in frame[column].dropna().unique())
    return out


def _build_domains() -> pd.DataFrame:
    """One grouped query per categorical dimension, over the whole dataset."""
    columns = ["borough", "payment_method", "fare_class_category", "transit_mode"]
    full = Filters(start=DATA_START, end=DATA_END)
    frames: Dict[str, pd.DataFrame] = {}
    for column in columns:
        frame = _fetch(
            select=f"{column}, count(*) as n",
            group=column,
            filters=full,
            order="n DESC",
            cube=f"domain_{column}",
            full_table=True,
        )
        frames[column] = frame[[column]].dropna() if not frame.empty else pd.DataFrame(columns=[column])

    rows = []
    for i in range(max(len(f) for f in frames.values())):
        row: Dict[str, Any] = {}
        for column, frame in frames.items():
            row[column] = frame[column].iloc[i] if i < len(frame) else None
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def load_raw_row_count() -> int:
    """Total rows in the raw dataset - reported as *context*, never as memory.

    Used only so the data-quality tab can state honestly how many raw records
    exist versus how many aggregated records this dashboard actually analysed.
    """
    key = "raw_row_count"

    def build() -> pd.DataFrame:
        row = _scalar({"$select": "count(*) as n", "$limit": "1"})
        return pd.DataFrame({"raw_row_count": [int(row["n"])]})

    frame = CACHE.get_or_build("meta", key, build)
    if frame.empty:
        return 0
    return int(frame["raw_row_count"].iloc[0])


def load_observed_span(filters: Optional[Filters] = None) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """First and last ``transit_timestamp`` present in the raw dataset."""
    frame = CACHE.get_or_build(
        "meta",
        "dataset_span",
        lambda: pd.DataFrame(
            [
                _scalar(
                    {
                        "$select": "min(transit_timestamp) as mn, max(transit_timestamp) as mx",
                        "$limit": "1",
                    }
                )
            ]
        )
    )
    if frame.empty:
        return None, None
    row = frame.iloc[0]
    return pd.to_datetime(row["mn"]), pd.to_datetime(row["mx"])


# --------------------------------------------------------------------------
# Post-processing shared by every cube
# --------------------------------------------------------------------------


def _finalise_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse, de-duplicate and sort a raw hourly aggregate."""
    columns = ["timestamp", "ridership", "transfers"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    out = frame.rename(columns={"transit_timestamp": "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["ridership"] = pd.to_numeric(out.get("ridership"), errors="coerce").fillna(0.0)
    if "transfers" in out.columns:
        out["transfers"] = pd.to_numeric(out["transfers"], errors="coerce").fillna(0.0)
    else:
        out["transfers"] = 0.0

    out = out.dropna(subset=["timestamp"])
    # One row per timestamp is the contract of this cube; duplicates would
    # double count, so they are collapsed rather than summed blindly.
    out = out.groupby("timestamp", as_index=False)[["ridership", "transfers"]].sum()
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)[columns]
