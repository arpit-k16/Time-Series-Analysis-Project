# MTA Subway Ridership Analytics

[![CI](https://github.com/arpit-k16/Time-Series-Analysis-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/arpit-k16/Time-Series-Analysis-Project/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](requirements.txt)
[![Streamlit](https://img.shields.io/badge/streamlit-1.43%2B-FF4B25.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Temporal, Station & Geographic Analysis — 2020–2024**

A Phase-1 exploratory dashboard for a time-series project, built on the
**[MTA Subway Hourly Ridership: 2020–2024](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s)**
dataset (NYC Open Data / Socrata view `wujg-7c2s`).

> **120,855,567 raw rows · 428 stations · 5 years · 6.8 MB peak memory.**
> Every KPI, chart and map point is a server-side aggregate. The raw dataset is
> never downloaded, not even once.

> **The raw dataset contains 120,855,567 rows. Nothing in this project ever
> downloads it.** Every figure is built from a *server-side aggregate* whose
> result is small enough to hold in memory, and each result is memoised to a
> Parquet file so a repeated query costs a disk read instead of a network round
> trip.

**Contents** — [Quick start](#2-quick-start) · [How the data is loaded](#4-how-the-data-is-loaded)
· [Aggregation rules](#5-how-ridership-is-aggregated-and-not-double-counted)
· [**Preprocessing pipeline, stage by stage**](PREPROCESSING.md)
· [Verification](#verification-scripts) · [Limitations](#11-limitations-and-assumptions)

---

## 1. Scope

This dashboard is **Phase 1 only**: descriptive, temporal and geographic
analysis. It deliberately contains **no** forecasting model — no ARIMA, SARIMA,
Prophet, XGBoost, LSTM, GRU, no predictions, no RMSE/MAE comparison and no
train/test split. Those belong to Phases 2 and 3.

The pipeline is built so those later phases can reuse it directly:
`utils/preprocessing.build_forecast_table()` returns a clean
`timestamp, total_ridership` frame, and `build_station_forecast_table()` returns
a `timestamp, station_complex, borough, ridership` panel. No lag or rolling
features are computed yet — that is forecasting work, not preprocessing.

---

## 2. Quick start

```bash
pip install -r requirements.txt

# Optional but recommended: pre-build the aggregate cache so the first page
# load is instant (see "Cache warm-up" below for the expected duration).
python scripts/build_cache.py

streamlit run app.py
```

Then open <http://localhost:8501>.

No API key is needed — the NYC Open Data endpoint is public.

### Verification scripts

```bash
python verify.py           # cubes reconcile, analytics correct, memory bounded
python verify_filters.py   # filters, aggregation, drill-down, error handling
python check_windows.py    # auto grain follows the window; resampling is lossless
```

Each prints `PASSED` only when every assertion holds. All three hit the live
API and populate `data/processed/`, so allow a few minutes on a cold cache.
[CONTRIBUTING.md](CONTRIBUTING.md) explains what each suite protects.

---

## 3. Project structure

```
app.py                     Streamlit UI: 6 tabs, sidebar filters, 24 charts
utils/
    __init__.py            package re-exports
    data_loader.py         Socrata client, chunked aggregates, Parquet cache
    preprocessing.py       calendar derivation, resampling, Phase 2/3 tables
    analytics.py           KPIs, profiles, rankings, map points, observations
scripts/
    build_cache.py         pre-builds the Parquet cache for a given window
.streamlit/config.toml     theme (committed); secrets.toml is gitignored
data/processed/            generated Parquet cache - gitignored, rebuildable
verify.py                  reconciliation, analytics and memory suite
verify_filters.py          filter, drill-down and error-handling suite
check_windows.py           window-to-grain rule and lossless-resampling suite
.github/workflows/         ci.yml (offline, every push) + data-tests.yml (live API)
```

For a **stage-by-stage walkthrough of every preprocessing step**, naming the
file, function and reason for each, see [**PREPROCESSING.md**](PREPROCESSING.md).

`data/processed/` is **generated output, not source**. It is excluded from git
and rebuilt with `python scripts/build_cache.py`, so a fresh clone stays small
and a cached query costs a disk read rather than a network round trip.

---

## 4. How the data is loaded

### 4.1 The rule: aggregate remotely, never download

Every request is a SoQL aggregate (`$select` + `$group`). The Python side only
ever receives the grouped result:

| Cube | Grouped by | Rows (2020–2024) |
|---|---|---|
| `hourly` | `transit_timestamp` | 43,848 |
| `station_totals` | `station_complex_id` | 428 |
| `borough_totals` | `borough` | 5 |
| `payment_totals` | `payment_method` | 2 |
| `fare_totals` | `fare_class_category` | 12 |
| `mode_totals` | `transit_mode` | 3 |
| `station_meta` | `station_complex_id` | 428 |
| `station_hourly` (one complex) | `transit_timestamp` | 43,848 |

For the default Jan 2022 – Dec 2024 window the whole dashboard runs on
**26,302 hourly records (≈ 7 MB)** versus the 77.2 million raw rows behind that
window — a ~2,900× reduction, and 6.8 MB peak Python allocation end-to-end.

### 4.2 Chunking

`$group` cost on this endpoint scales with the number of groups returned:

* grouping by `transit_timestamp` for one month (744 groups) takes ~24 s;
* a full year (~8,760 groups) reliably exceeds the request timeout;
* a single five-year request is large enough that the endpoint drops the
  connection.

So the hourly cube is fetched **one month per request**, and every month is
cached under its own Parquet key. An interrupted build resumes from where it
stopped instead of redoing hours of work. The small dimension cubes
(borough/payment/fare/mode/station) are sent as a single request — their
responses are a handful of rows.

Socrata also silently caps a response at 1,000 rows unless `$limit` is set, so
every request carries an explicit limit.

### 4.3 Caching — two layers

1. `data_loader.AggregateCache` writes each aggregate to
   `data/processed/<cube>__<hash>.parquet`. This survives restarts.
2. `@st.cache_data` in `app.py` avoids even the disk read on rerun.

A per-key `threading.Lock` stops two concurrent sessions issuing the same slow
query twice. `st.cache_data` is keyed on the `Filters` object, so it
invalidates automatically when any filter changes.

### 4.4 Cache warm-up

```bash
python scripts/build_cache.py                            # default window
python scripts/build_cache.py --start 2024-01-01 --end 2024-12-31
```

Measured against the live endpoint, a cold three-year build takes roughly
**15–20 minutes**, almost all of it the 36 hourly month-chunks. Afterwards the
dashboard loads from Parquet in well under a second. Selecting a *new* filter
combination that has never been cached pays that cost once, which is why the
app shows an explicit spinner and the Data Quality tab lists exactly what is
cached.

---

## 5. How ridership is aggregated (and not double counted)

The raw dataset splits a single station-hour across `payment_method` **and**
`fare_class_category`. In January 2022 the top station complex alone produced
5,726 raw rows. Treating those as independent station-hour observations would
inflate every count roughly ten-fold.

Two rules prevent that:

1. **Server-side summing.** The citywide series is
   `sum(ridership) GROUP BY transit_timestamp`, so every payment method and
   fare class is collapsed before a single byte reaches the app.
2. **Averages are computed on the collapsed series.** `Average Ridership by
   Hour of Day` is the mean of the *citywide hourly totals*, not the mean of raw
   rows — so the denominator is one observation per timestamp.

`verify.py` proves this: for the default window, the station, borough, payment,
fare and mode cubes each sum to **exactly** the citywide total
(3,395,439,569 — difference 0.00), and the same holds under a Manhattan /
subway / OMNY filter.

Chunks cover disjoint date ranges, so they are **summed by key** — never
deduplicated — when concatenated.

> The full pipeline — filter construction, SoQL shapes, chunk combination,
> caching, calendar derivation, grain selection, resampling, down-sampling and
> the descriptive analytics — is documented step by step in
> [**PREPROCESSING.md**](PREPROCESSING.md), naming the file and function behind
> each stage.

---

## 6. Station-level analysis

* **Top-N bars and ranking table** come from `station_totals` (428 rows),
  joined to the station reference for name, borough and coordinates.
* **Borough focus** narrows both the ranking and the drill-down list.
* **Drill-down** loads `station_hourly` for the chosen complex alone — an extra
  aggregate constrained by `station_complex_id`, cached per (station, filter).
  Its hourly, daily, hour-of-day, day-of-week and monthly charts are all
  derived from that one series.
* `Avg_Hourly_Ridership` is a station's total divided by the number of hourly
  timestamps in the selected window. Counting each station's *own* distinct
  hours would need `count(distinct transit_timestamp)`, which the endpoint
  cannot evaluate inside the request timeout; this approximation is documented
  on the chart rather than silently applied.

---

## 7. The map

`go.Scattergeo` with the built-in geographic projection — **no Mapbox token is
required**. One marker per station complex, sized so that marker *area* is
proportional to total ridership, coloured by borough with one trace per borough
so the legend is clickable. Hover shows station, borough, total ridership,
average hourly, latitude and longitude.

Coordinates come from the dataset's own `latitude`/`longitude`; nothing is
geocoded externally. The view is framed on the NYC bounding box
(lat 40.45–41.05, lon −74.30…−73.65) so the city fills the plot.

Complexes sharing a coordinate (rounded to 5 decimals) are merged into a single
marker with their volumes summed, so no point is drawn twice. In the default
window all 428 complexes have distinct coordinates, so the merge is a no-op
here — but it is applied unconditionally and the app reports it when it fires.

---

## 8. Performance

* Cached server-side aggregates; no raw download, ever.
* Aggregate before visualizing — the largest frame is 26,302 rows.
* **Automatic granularity.** The main trend chart picks its grain from the
  length of the selected window, because density is what makes a long-range
  chart unreadable (a daily line over three years is ~1,100 near-identical
  points). `Auto` resolves to **Monthly** above 90 days, **Daily** for 15-90
  days and **Hourly** for 14 days or fewer, so a multi-year window never hands
  Plotly 26k points. The title follows the grain — "MTA Monthly Ridership
  Trend", "MTA Daily Ridership Trend", "MTA Hourly Ridership Trend" — and an
  explicit sidebar choice always wins, so no finer scale is ever out of reach.
  A collapsed expander on the Overview keeps the daily, monthly and hourly
  charts one click away regardless of the auto-selected grain.
* Plot-level down-sampling to ~6,000 points, **keeping the last row** (a plain
  stride would silently drop the most recent reading).
* Hour-of-day bars reindex to a fixed 24 rows, so an absent hour renders as a
  gap rather than shifting the axis.

---

## 9. Error handling

Every network failure surfaces as a friendly `st.error`, never a traceback:

| Failure | Behaviour |
|---|---|
| API unreachable / 5xx | 3 attempts with backoff, then a readable message |
| Request timeout | 300 s per attempt |
| HTTP 4xx (bad query) | Reported immediately with the API's own explanation |
| Malformed / empty body | Treated as an error, not parsed |
| Missing columns | Named explicitly; optional columns degrade to defaults |
| Invalid dates | `DataLoadError` with the offending value |
| Empty filter result | Empty-state message, no crash, no silent fallback |
| Duplicate timestamps | Summed on load, so a duplicate cannot double count |
| Missing values | Reported per column in Data Quality, never imputed |

Gaps in the time series are shown as breaks in the lines (`connectgaps=False`)
and **never interpolated or filled**.

---

## 10. Tabs

1. **Overview** — KPIs, overall series, daily and monthly trends, Initial Observations.
2. **Temporal Analysis** — average by hour, by day of week (Monday-first), day × hour heatmap, monthly/yearly trend.
3. **Station Analysis** — Top 5/10/15/20 bars, ranking table, per-station drill-down with 5 charts.
4. **Geographic Analysis** — ridership/transfers/station counts by borough, station map, largest stations table.
5. **Payment & Fare Analysis** — ridership and share by payment method, by fare class, transit modes.
6. **Data Quality** — analysis records vs raw dataset size, continuity, missing values, cache inventory, Phase 2/3 export.

---

## 11. Limitations and assumptions

* **First load of an uncached filter combination is slow** (up to ~15 min for a
  three-year window) because the endpoint charges roughly 24 s per month of
  hourly aggregation. Run `scripts/build_cache.py` ahead of time.
* **First load requires network access.** There is no offline bundle.
* **`Avg_Hourly_Ridership` per station** uses the window length as the divisor,
  not the station's own observed-hour count (see §6). Exact for complexes
  reporting all window; a slight overstatement otherwise.
* **Station names, boroughs and coordinates** are sampled from three months
  (2020-06, 2022-06, 2024-06) and unioned — a full five-year reference scan
  times out. Complexes that existed only between those samples would be missing.
* **`transit_mode` is not subway-only.** The view includes Staten Island Railway
  and tram rows; use the Transit mode filter for a pure subway series. Totals
  across all modes are still internally consistent.
* **Partial years are not scaled.** The year-on-year comparison compares raw
  totals and says so.
* **`transfers`** is the recorded value from the dataset, not a derived
  inter-station transfer count.
* 2 of 26,304 hours in the default window are absent from the source; the app
  reports this rather than filling it.

---

## 12. License and data use

This project is released under the [MIT License](LICENSE) — see
[LICENSE](LICENSE) for the full text.

The code is mine; the data is not. The MTA Subway Hourly Ridership dataset is
published by **NYC Open Data** and is *not* redistributed in this repository.
Queries go straight to the public Socrata endpoint at request time, so nothing
is committed and no data licence travels with the clone. Attribution and terms
of use belong to the City of New York — see the
[source dataset page](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s).

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) first,
especially the aggregation rule in §2 that the whole project depends on.