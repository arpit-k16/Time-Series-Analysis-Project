# Contributing

Thanks for looking at this project. It is a Phase-1 exploratory dashboard, and
most of the interesting constraints come from the size of the underlying
dataset rather than from the UI.

## Before you start

This project analyses the
[MTA Subway Hourly Ridership: 2020–2024](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s)
dataset (NYC Open Data, Socrata view `wujg-7c2s`) — **120,855,567 raw rows**.
Every contribution has to respect that.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_cache.py     # optional but recommended, see README §4.4
streamlit run app.py
```

## The two rules that matter most

**1. Never aggregate in Pandas what the API can aggregate for you.**
The raw dataset is never downloaded. Requests are built as SoQL `$select
... $group` queries and return small cubes. If you find yourself writing code
that iterates over raw rows, it belongs in a `$group` clause instead.

**2. `ridership` is a measure, not a count.**
Every station-hour appears several times — once per `payment_method` ×
`fare_class_category` combination. The invariant is:

```
overall   transit_timestamp        -> SUM(ridership)
station   transit_timestamp + complex -> SUM(ridership)
borough   transit_timestamp + borough  -> SUM(ridership)
```

Never `COUNT(*)` or `MEAN(ridership)` on the raw rows. `verify.py` asserts
that all five dimension cubes reconcile to the same citywide total; if your
change breaks that, the change is wrong.

## Scope: descriptive only

Phase 1 is exploratory. Please do **not** add ARIMA, SARIMA, Prophet,
XGBoost, LSTM, GRU, predictions, train/test splits, RMSE/MAE/MAPE or model
comparisons. Phases 2 and 3 own that work. `build_forecast_table()` in
`utils/preprocessing.py` is the seam designed for it: it returns a clean
`timestamp, total_ridership` frame and deliberately computes no lag or rolling
features.

## Verifying your change

Three headless suites, no browser required. Run the fast one first.

```bash
python verify.py          # cubes reconcile, analytics correct, memory bounded (~2 min)
python verify_filters.py  # filters, drill-down, error handling (~3 min)
python check_windows.py   # grain rule + lossless resampling (~2 min)
```

All three hit the live API and write into `data/processed/`. A PR is not ready
until all three print `PASSED`, and a new behaviour should come with a new
assertion in one of them.

## Coding conventions

* Follow the surrounding style: `snake_case` functions, `PascalCase` classes,
  dataclasses for structured returns, type hints on public functions.
* Docstrings explain **why** a decision was made, not what the next line does.
  Several carry a note about a Socrata API limit that is not obvious from the
  code — that context is the point.
* UI text in `app.py` is written for a reader: no emoji, sentence case
  headings, consistent number formatting via the `fmt_*` helpers.
* Prefer editing an existing file over adding a new module.

## Commit and pull request

1. Branch from `main`: `git checkout -b short-description`
2. Make the change, then run the three suites.
3. Commit with a message explaining the **why**: `python -m compileall app.py utils`
   must be clean before you push.
4. Open a PR describing the change, how you verified it, and any change in
   cache size or runtime.

## Reporting a bug

Open an issue with the filter selection (date range, borough, mode, payment
method, fare class) that reproduces it, the tab, and what you expected. If the
Data Quality tab shows a non-zero gap or a failed reconciliation, include its
numbers — that usually points straight at the cause.