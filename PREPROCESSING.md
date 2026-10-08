# Preprocessing Pipeline — Every Step, With File References

This is the ordered walkthrough of what happens to the data between the raw
NYC Open Data endpoint and a chart pixel. Nothing is left implicit: each stage
names the **file**, the **function**, and the **reason** the step exists.

> **The headline rule:** the raw dataset has **120,855,567 rows** and is
> **never loaded into Pandas**. Preprocessing in this project therefore begins
> *inside* the API — the first thing Pandas ever sees is already a grouped,
> summed, handful-of-rows aggregate.

---

## The pipeline at a glance

```
 raw endpoint (120,855,567 rows)
        │  never downloaded
        ▼
 [1] Filter construction ──────────── utils/data_loader.py :: Filters
        │  builds SoQL $where
        ▼
 [2] Server-side aggregation ──────── utils/data_loader.py :: _fetch, _fetch_hourly
        │  $select + $group → small cubes
        ▼
 [3] Chunking + combine-by-key ────── utils/data_loader.py :: _fetch
        │  chunks summed, never deduped
        ▼
 [4] Two-layer caching ────────────── utils/data_loader.py :: AggregateCache
        │  Parquet  +  app.py @st.cache_data
        ▼
 [5] Local finalisation ───────────── utils/data_loader.py :: _finalise_hourly
        │  coerce, de-duplicate, sort
        ▼
 [6] Calendar derivation ──────────── utils/preprocessing.py :: add_calendar_columns
        │  timestamp → 13 derived fields
        ▼
 [7] Grain selection ──────────────── app.py :: resolve_aggregation
        │  window length → Monthly / Daily / Hourly
        ▼
 [8] Resampling to display grain ──── utils/preprocessing.py :: resample_series
        │  gaps left as gaps
        ▼
 [9] Plot down-sampling ───────────── utils/preprocessing.py :: downsample_for_plot
        │  ≤ 6,000 points, last row kept
        ▼
 [10] Descriptive aggregation ─────── utils/analytics.py
        │  KPIs, profiles, rankings, map, observations
        ▼
 [11] Phase 2/3 hand-off ──────────── utils/preprocessing.py :: build_forecast_table
```

Two entry points then consume the result:

| Consumer | File | Note |
|---|---|---|
| Streamlit app | `app.py :: build_view` | Builds every cube once per rerun into a shared `View` dataclass |
| Headless suites | `verify.py`, `verify_filters.py`, `check_windows.py` | Assert the invariants each stage promises |

---

## Stage 1 — Filter construction

**File:** `utils/data_loader.py` — class `Filters` (line ~147)

The sidebar never returns loose values; they are normalised into one frozen
`dataclass` that is simultaneously the cache key and the `$where` clause.

| Method | What it does | Why it exists |
|---|---|---|
| `Filters.build(...)` | Coerces dates, `None` and lists into typed fields | Sidebar widgets can hand back `date` or `str`; one place does the coercion |
| `Filters.lo` / `Filters.hi` | Ordered, normalized window bounds | The user may drag the date pickers backwards |
| `Filters.where(extra)` | Emits the SoQL `$where` clause | Filtering must happen server-side — see Stage 2 |
| `Filters.spans(chunk)` | Splits the window into ≤ `chunk` ranges | A single five-year `$group` times out |
| `Filters.scoped(lo, hi)` | Copy of the filters narrowed to one chunk | Each chunk needs its own predicate |
| `Filters.signature()` | 12-char SHA1 of every value that changes the result | Stable, collision-resistant cache key |
| `Filters.describe()` | Human-readable summary | Used in captions and `manifest.json` |
| `Filters.is_empty_selection()` | True when a dimension was explicitly emptied | Distinguishes "no rows" from "no restriction" — `None` means unrestricted, `()` means show an empty state |

**Two constants matter here:**

```python
CHUNK        = pd.Timedelta(days=366)   # general dimension cubes
HOURLY_CHUNK = pd.Timedelta(days=31)    # the expensive hourly cube
```

**Why server-side filtering:** a `$where` predicate on `borough` plus a date
range is evaluated by Socrata over the full 121M-row table. Pushing it down
means the network only ever carries grouped results.

---

## Stage 2 — Server-side aggregation

**File:** `utils/data_loader.py` — `_fetch(...)` (line ~496), `_fetch_hourly(...)`
(line ~593)

Every request is `$select <agg> $group <key> $where <filters> $limit <explicit>`.
The `$limit` is mandatory: Socrata **silently truncates a response at 1,000
rows** when `$limit` is absent, which would quietly corrupt every total.

### The cubes and their exact SoQL

| Loader (`utils/data_loader.py`) | `$select` | `$group` | Rows returned |
|---|---|---|---|
| `load_hourly` | `transit_timestamp, sum(ridership), sum(transfers)` | `transit_timestamp` | 43,848 |
| `load_station_totals` | `station_complex_id, sum(ridership), sum(transfers), count(*)` | `station_complex_id` | 428 |
| `load_borough_totals` | `borough, sum(ridership), sum(transfers), count(*)` | `borough` | 5 |
| `load_payment_totals` | `payment_method, sum(ridership), ...` | `payment_method` | 2 |
| `load_fare_totals` | `fare_class_category, sum(ridership), ...` | `fare_class_category` | 12 |
| `load_mode_totals` | `transit_mode, sum(ridership), ...` | `transit_mode` | 3 |
| `load_station_hourly` | same as `load_hourly`, scoped to one id | `transit_timestamp` | 43,848 |
| `load_station_daily` | `station_complex_id, date_extract_y/m/d, sum(ridership)` | id + y + m + d | 782k |
| `load_station_meta` | `station_complex_id, station_complex, borough, latitude, longitude` | — | 428 |

### Three Socrata quirks that shaped this code

1. **`date(transit_timestamp)` is rejected inside `$group`.** So is
   `date_extract_h` / `date_extract_hour` — neither function exists in SoQL.
   `date_extract_y` / `_m` / `_d` *do* work, so `load_station_daily` groups on
   the three numeric components and stitches the date back together locally
   with `str.zfill(2)`.
2. **`full_table=True`** skips the date predicate entirely and sends one
   request. This is the cheap path for reference cubes (domain values, station
   names and coordinates) because they are grouped on a handful of
   low-cardinality keys and always return small.
3. **`count(distinct transit_timestamp)` times out.** This is why
   `station_summary` cannot compute each station's own observed-hour count and
   divides by the window length instead (Stage 10).

### The aggregation rules, expressed as code

`ridership` is a **measure, not a count**. One station-hour is split across
`payment_method` × `fare_class_category` — roughly ten raw rows per
station-hour. Therefore:

```
citywide   GROUP BY transit_timestamp        → SUM(ridership)    load_hourly
station    GROUP BY station_complex_id       → SUM(ridership)    load_station_totals
borough    GROUP BY borough                  → SUM(ridership)    load_borough_totals
```

`MEAN(ridership)` and `COUNT(*)` are **never** applied to raw rows.
`count(*)` does appear in `load_station_totals`, but only as the `Raw_Rows`
column — reported by the Data Quality tab and explicitly never used as a
divisor.

**Averages are computed from the collapsed series, not from raw rows.** This is
what makes `Average Ridership by Hour of Day` a mean of *hourly totals* with
one observation per timestamp, rather than a mean over payment rows.

---

## Stage 3 — Chunking and cross-chunk combination

**File:** `utils/data_loader.py` — inside `_fetch(...)`

`$group` cost scales with the number of groups returned:

* one month grouped by `transit_timestamp` (744 groups) ≈ **24 seconds**;
* a full year (~8,760 groups) reliably **exceeds the request timeout**;
* a single five-year request is large enough that the endpoint **drops the
  connection**.

So the hourly cube is fetched **one month per request**, each month cached
under its own Parquet key (`hourly__<hash>_00`, `..._01`, …). A three-year
window is 36 chunks, and an interrupted build resumes rather than restarting.

### The combination rule — the single most important line

```python
out = out.groupby(keys, as_index=False, dropna=False)[measures].sum()
```

Chunks cover **disjoint date ranges**, so every key legitimately appears once
*per chunk*. They are **summed by key, never deduplicated**.

> Deduplicating instead of summing silently reports roughly **one third** of
> true ridership for a three-year window, with no error and no warning. This
> was a real defect caught during the build.

When `key_cols` is not supplied (reference cubes such as `domains`), the
concatenation is `drop_duplicates()`, because those rows are identical across
chunks rather than additive.

---

## Stage 4 — Two-layer caching

**Layer 1 — disk.** `utils/data_loader.py :: AggregateCache` (line ~387)

* `get_or_build(cube, key, builder)` writes
  `data/processed/<cube>__<key>.parquet` and returns the frame.
* `path_for()` derives the path from the cube name and the SHA1 key.
* `_thread_lock()` — a per-key `threading.Lock` so two concurrent Streamlit
  sessions do not issue the same 24-second query twice.
* `inventory()` lists what is cached; `clear()` wipes it. Both are surfaced in
  the **Data Quality** tab.
* `_remember(...)` appends provenance (`dataset_id`, last cube, last filter
  description) to `data/processed/manifest.json`, best-effort — a failure
  there is swallowed because provenance must never break a load.

**Layer 2 — memory.** `app.py` lines ~748-800

```python
@st.cache_data(show_spinner=False)
def get_hourly(filters: Filters) -> pd.DataFrame: ...
```

Every `get_*` loader is wrapped in `@st.cache_data`, keyed on the `Filters`
object, so a rerun does not even re-read Parquet. Changing any sidebar widget
changes the `Filters` hash and invalidates only that entry.

**Why both:** Parquet survives a process restart; `st.cache_data` survives the
~20 Streamlit reruns per user session.

---

## Stage 5 — Local finalisation

**File:** `utils/data_loader.py` — `_finalise_hourly(...)` (line ~923)

Applied to every hourly frame before it leaves the loader:

1. `transit_timestamp` → `timestamp`, parsed with `pd.to_datetime(..., errors="coerce")`.
2. `ridership` / `transfers` → numeric via `pd.to_numeric(errors="coerce")`,
   missing filled with `0.0`.
3. Rows with an unparseable timestamp dropped.
4. `groupby("timestamp")[["ridership", "transfers"]].sum()` — the cube's
   contract is **exactly one row per timestamp**; a duplicate would double
   count, so it is collapsed rather than summed blindly or silently kept.
5. Sorted by timestamp, stable, index reset.

The returned frame is always exactly
`[timestamp, ridership, transfers]`.

---

## Stage 6 — Calendar derivation

**File:** `utils/preprocessing.py` — `add_calendar_columns(...)` (line ~80)

The one place where raw `timestamp` becomes usable calendar features. Required
by the project brief:

```python
timestamp -> date, hour, day, day_of_week, week, month, month_name,
             quarter, year, is_weekend
```

Plus three extra labels derived alongside them: `day_name`, `year_month`,
`year_quarter`.

| Output column | Source | dtype |
|---|---|---|
| `date` | `ts.dt.normalize()` | datetime64 (midnight) |
| `hour` | `ts.dt.hour` | int64 |
| `day` | `ts.dt.day` | int64 |
| `day_of_week` | `ts.dt.dayofweek` — **0 = Monday** | int64 |
| `day_name` | `day_of_week` mapped through `DOW_NUMBER_TO_NAME` | str |
| `week` | `ts.dt.isocalendar().week` | int64 |
| `month` | `ts.dt.month` | int64 |
| `month_name` | `ts.dt.month_name()` | str |
| `quarter` | `ts.dt.quarter` | int64 |
| `year` | `ts.dt.year` | int64 |
| `is_weekend` | `day_of_week ∈ {5, 6}` (Sat/Sun) | bool |
| `year_month` | `ts.dt.to_period("M")` | str, e.g. `2024-07` |
| `year_quarter` | `ts.dt.to_period("Q")` | str, e.g. `2024Q3` |

### Three behaviours worth knowing

* **Invalid timestamps are dropped, not raised.** The count is recorded in
  `out.attrs["dropped_invalid_timestamps"]` so the Data Quality tab can report
  it. An unparseable timestamp cannot be placed on a time axis, so failing the
  whole render would be worse than reporting it.
* **Every original column survives.** `extra_dims` names grouping columns such
  as `station_complex` that must ride along, and measures like `ridership` are
  preserved. An earlier version of this function silently dropped measure
  columns — the current column-ordering block exists specifically to prevent
  that regression.
* **`extra_dims` is optional.** Calling it as `add_calendar_columns(hourly,
  "timestamp")` on a simple series works with no second argument.

**Constants defined alongside it:**

```python
DOW_ORDER        = ("Monday", ..., "Sunday")   # Monday-first, as the brief requires
DOW_NUMBER_TO_NAME = {0: "Monday", ..., 6: "Sunday"}
WEEKEND_DOW_NUMBERS = (5, 6)
CALENDAR_COLUMNS = ("date", "hour", "day", ... "is_weekend")
```

---

## Stage 7 — Grain selection by window length

**File:** `app.py` — `resolve_aggregation(requested, days)` (line ~965),
`_window_days(filters)` (line ~1021)

Density is what makes a long-range chart unreadable: a daily line over three
years is ~1,100 near-identical points. The **main** trend chart therefore picks
its grain from the number of calendar days in the window:

| Window | Grain | `resample` freq | Chart title |
|---|---|---|---|
| `> 90` days | **Monthly** | `MS` | `MTA Monthly Ridership Trend` |
| `15 – 90` days | **Daily** | `D` | `MTA Daily Ridership Trend` |
| `≤ 14` days | **Hourly** | `h` | `MTA Hourly Ridership Trend` |

Boundaries are pinned by `check_windows.py`: 14 → Hourly, 15 → Daily,
90 → Daily, 91 → Monthly.

**An explicit sidebar choice always wins.** The sidebar *Time aggregation*
radio offers `Auto | Hourly | Daily | Weekly | Monthly`, so no finer scale is
ever out of reach. A collapsed expander on the Overview additionally keeps the
daily, monthly and hourly charts one click away whatever `Auto` resolved to.

**Per-grain hover formats** live in `app.py :: TREND_HOVER`, so a monthly
chart shows `Jan 2024` and an hourly one shows `29 Oct 2024 17:00` rather than
both showing a bare date.

---

## Stage 8 — Resampling to the display grain

**File:** `utils/preprocessing.py` — `resample_series(...)` (line ~134)

```python
AGGREGATION_FREQ = {
    "Hourly":  "h",       # transit_timestamp -> SUM
    "Daily":   "D",       # date              -> SUM
    "Weekly":  "W-MON",   # week start        -> SUM
    "Monthly": "MS",      # month start       -> SUM
}
```

Implementation: `set_index(timestamp).resample(freq).agg({value: "sum"})`,
then `dropna(subset=[value])`, `reset_index`, rename the index to
`period_start`, stable sort.

* **Gaps are deliberately not filled.** A missing bucket stays missing so
  absence of data is never disguised as a genuine zero. The Plotly traces are
  built with `connectgaps=False`, turning those gaps into visible breaks.
* `extra_cols` can carry additional columns through with `agg("mean")` — used
  for optional secondary measures, **never** for `ridership`.
* An unknown aggregation label raises `ValueError` naming the valid set.

`check_windows.py` asserts resampling is **lossless**: the sum at every grain
equals the hourly citywide total to the unit. For the default window that is
`3,395,439,569` at Monthly, Daily *and* Hourly.

---

## Stage 9 — Plot down-sampling

**File:** `utils/preprocessing.py` — `downsample_for_plot(series, max_points=4000)`
(line ~176)

Even after resampling, a multi-year hourly series is 26,304 points. The app
caps charts at **`HOURLY_POINT_BUDGET = 6000`** (`app.py` line 47).

```python
step  = ceil(len(series) / max_points)
taken = list(range(0, len(series), step))
if taken[-1] != len(series) - 1:
    taken.append(len(series) - 1)      # ← always keep the last row
```

**The last row is appended explicitly.** A plain `iloc[::step]` slice silently
truncates the final observation, which would drop the most recent reading from
every long-range chart. `verify_filters.py` asserts both endpoints survive.

This step is **presentation only** — it never feeds a KPI, a total or an
observation. Every number the dashboard reports is computed from the full
frame.

---

## Stage 10 — Descriptive aggregation

**File:** `utils/analytics.py`

Pure functions over a DataFrame, importable from a notebook without
Streamlit.

### Profiles — all from the *collapsed* hourly series

Each calls `add_calendar_columns(hourly, "timestamp")` first, then groups.

| Function | Grouping | Output |
|---|---|---|
| `hourly_profile` (L85) | `groupby("hour")` → `mean` / `sum` / `size` | 24 rows, `reindex(range(24))` |
| `dow_profile` (L106) | `groupby("day_of_week")` → `mean` / `sum` / `size` | 7 rows, `reindex(range(7))`, Monday-first |
| `hour_day_matrix` (L127) | `pivot_table(index=day_of_week, columns=hour, aggfunc="mean")` | 7 × 24 |
| `year_trend` (L141) | `groupby("year")` | chronological |
| `monthly_trend` (L159) | `groupby("year_month")` | sorted on the **string**, so Plotly keeps it a label |

`reindex(...)` is why an unobserved hour renders as a **gap (NaN)** rather than
shifting the axis or fabricating a zero.

### Rankings

| Function | What it produces |
|---|---|
| `station_summary(station_totals, station_meta, window_hours)` (L196) | 428 rows joined to name, borough, lat/lon; sorted by `Total_Ridership` desc |
| `top_stations(summary, n)` (L262) | `nlargest(n, "Total_Ridership")` — the Top 5/10/15/20 selector |
| `borough_summary(borough_totals, station_totals)` (L269) | Per-borough totals **plus** station counts |
| `map_points(summary, avg_hourly)` (L313) | Coordinate frame sized by volume for the scattergeo map |

**`borough_summary` must receive the *joined* station summary.** The raw
station cube groups by id alone and has no `borough` column; only
`station_summary`'s output carries one. Passing the unjoined frame yields
`Stations = 0` for every borough — a defect fixed during the build.

**`station_summary` — the window-length divisor.** `Avg_Hourly_Ridership` is
`Total_Ridership / window_hours`, where `window_hours = len(hourly)`.
Computing each station's own observed hours needs `count(distinct
transit_timestamp)`, which exceeds the endpoint's request timeout. The
approximation is **exact** for complexes reporting through the whole window
and a slight overstatement for ones that opened or closed inside it — it is
stated on the chart rather than silently applied.

**`map_points` — two fixes worth preserving:**

1. Duplicate coordinates are collapsed by rounding lat/lon to 5 dp and grouping
   on `(station_complex, borough, latitude, longitude)` — **summing** volume,
   so a shared point never draws two markers.
2. When two complexes merge, their `Avg_Hourly_Ridership` is combined as a
   **plain mean**, not weighted by ridership. Every complex was already
   averaged over the *same* window length, so weighting divides a rate by a
   volume and collapses every marker to ~0 (observed: ~3.8e-05).

### KPIs and observations

* `compute_kpis(hourly, station_totals, borough_totals)` (L26) — sums the
  **filtered** hourly cube, so every card respects the sidebar. The docstring
  states the contract: `hourly` is already one row per timestamp with
  ridership summed across payment methods and fare classes, which is what stops
  raw rows being treated as independent observations.
* `build_observations(...)` (L379) — generates the Initial Observations list.
  Every statement is prefixed with `Within {period},`, where `period` comes from
  `app.py :: _period_label(filters)` (the date window **plus** any active
  dimension filter). Descriptive only: peak and trough hour, highest and lowest
  day of week, highest and lowest station, highest and lowest borough, highest
  calendar month, payment share, observation count. No causal claims, no
  hardcoded values.

---

## Stage 11 — Phase 2 / Phase 3 hand-off

**File:** `utils/preprocessing.py`

| Function | Returns | Computes |
|---|---|---|
| `build_forecast_table(hourly, include_calendar=True)` (L233) | `timestamp, total_ridership` (+ calendar) | Sorted, one row per observation |
| `build_station_forecast_table(station_daily, station_meta)` (L256) | `timestamp, station_complex, borough, ridership, station_complex_id` | Station-level panel |

Both perform **no modelling** and compute **no lag or rolling features**. The
docstring is explicit that Phase 2 adds `lag_1` / `lag_24` / `lag_168` /
`rolling_mean_24` / `rolling_mean_168` on top of this frame — that is
forecasting work, not preprocessing.

The dashboard exposes `build_forecast_table` in a collapsed **Data Quality**
expander, where it reports only the row count and date range.

> `build_station_forecast_table` currently has **no callers**. It is kept as
> the documented seam for the station panel in Phase 2.

---

## `utils/preprocessing.py :: filter_frame` — exported, unused by the app

```python
def filter_frame(df, start=None, end=None, hours=None, days=None, column="timestamp")
```

Restricts a prepared frame to a window and/or an hour/weekday subset, using the
*calendar day* so the final hour of the last selected day is never excluded.

It is exported from `utils/__init__.py` for notebook use but has **no call
site** in `app.py`, `verify*.py` or `scripts/`. All dashboard filtering happens
server-side in `Filters.where()`. It is left in place as public library API,
not because anything depends on it.

---

## Column reference: what each stage produces

| After stage | Columns |
|---|---|
| `_fetch_hourly` (raw from API) | `transit_timestamp, ridership, transfers` |
| `_finalise_hourly` | `timestamp, ridership, transfers` |
| `add_calendar_columns` | + `date, hour, day, day_of_week, day_name, week, month, month_name, quarter, year, is_weekend, year_month, year_quarter` |
| `resample_series` | `period_start, ridership` (+ optional extra) |
| `build_forecast_table` | `timestamp, total_ridership` + calendar |

---

## What this pipeline deliberately does *not* do

| Not done | Why |
|---|---|
| Impute or fill missing hours | Absence of data must not become a fabricated zero; gaps render as gaps |
| Download the raw table | 120.8M rows; every figure is a server-side aggregate |
| Count raw rows as observations | They are `payment_method` × `fare_class_category` splits of one station-hour |
| Forecast, predict, or fit a model | Phase 1 scope only — CI fails the build if one is introduced |
| Geocode station names | The dataset already supplies `latitude` / `longitude`; no external requests |
| Scale partial years | The year-on-year comparison compares raw totals and says so |
| Weight merged map averages by ridership | Would divide a rate by a volume and collapse every marker |

---

## Which test covers which stage

| Stage | Covered by |
|---|---|
| 1 Filters | `verify_filters.py` §1–§6 |
| 2 Aggregation rules | `verify.py` — all five dimension cubes reconcile to `3,395,439,569`, diff 0.00 |
| 3 Chunk combination | `verify_filters.py` §3 (reconciliation **under** filters) |
| 4 Caching | `verify.py` cache inventory; `data/processed/manifest.json` |
| 5 Finalisation | `verify_filters.py` §10 (invalid timestamp dropped) |
| 6 Calendar derivation | `verify.py` — Monday-first DOW order, heatmap shape 7×24, no NaN |
| 7 Grain selection | `check_windows.py` boundaries + end-to-end `build_view` |
| 8 Resampling | `check_windows.py` — grain sum == hourly total == KPI total |
| 9 Down-sampling | `verify_filters.py` §9 — ≤ budget, **first and last row preserved** |
| 10 Analytics | `verify.py` — KPIs, profiles, rankings, 428 map points, memory ≤ 400 MB (actual **6.8 MB**) |
| 11 Phase 2/3 export | `verify.py` — forecast table 26,302 rows, correct columns |

Run them with:

```bash
python verify.py          # stages 2, 6, 10, 11 + memory bound
python verify_filters.py  # stages 1, 3, 5, 9
python check_windows.py   # stages 7, 8
```

---

## See also

* [README.md](README.md) §4 — how the data is loaded, chunking, caching
* [README.md](README.md) §5 — the no-double-counting rules
* [README.md](README.md) §8 — performance and automatic granularity
* [CONTRIBUTING.md](CONTRIBUTING.md) — the two rules a contribution must respect
* `utils/data_loader.py`, `utils/preprocessing.py`, `utils/analytics.py`,
  `app.py` — the source, in pipeline order
