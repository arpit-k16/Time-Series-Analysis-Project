"""
utils/preprocessing.py
======================
Reusable, model-free preparation of MTA Subway ridership observations.

This module is deliberately standalone: Phase 2 / Phase 3 notebooks can import
it and obtain the same cleaned series the dashboard plots, which guarantees
that whatever is explored here is exactly what will later be forecast.

Derived calendar fields (as required by the project brief)::

    timestamp -> date, hour, day, day_of_week, week, month, month_name,
                 quarter, year, is_weekend

Frequency aliases use the modern pandas spelling (``'h'``/``'MS'``/``'QS'``)
so the module works on pandas 2.x and 3.x alike.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Monday-first ordering, as required by the day-of-week views.
DOW_ORDER: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

#: ``DayOfWeek`` integer -> display label (0 == Monday).
DOW_NUMBER_TO_NAME: Dict[int, str] = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

WEEKEND_DOW_NUMBERS: Tuple[int, ...] = (5, 6)  # Saturday, Sunday

#: Sidebar aggregation label -> pandas resampling frequency.
AGGREGATION_FREQ: Dict[str, str] = {
    "Hourly": "h",
    "Daily": "D",
    "Weekly": "W-MON",
    "Monthly": "MS",
}

AGGREGATION_ORDER: Tuple[str, ...] = ("Hourly", "Daily", "Weekly", "Monthly")

#: The calendar fields derived for every observation.
CALENDAR_COLUMNS: Tuple[str, ...] = (
    "date",
    "hour",
    "day",
    "day_of_week",
    "week",
    "month",
    "month_name",
    "quarter",
    "year",
    "is_weekend",
)


# --------------------------------------------------------------------------
# Calendar derivation
# --------------------------------------------------------------------------


def add_calendar_columns(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    extra_dims: Sequence[str] = (),
) -> pd.DataFrame:
    """Attach the descriptive calendar fields used across the dashboard.

    ``extra_dims`` names additional grouping columns (e.g. ``station_complex``)
    that should survive the derivation untouched. Rows with an unparseable
    timestamp are dropped and reported through ``attrs`` rather than raising,
    because an invalid timestamp cannot be placed on a time axis.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    dropped = int(out[timestamp_col].isna().sum())
    out = out.dropna(subset=[timestamp_col])

    ts = out[timestamp_col]
    out["date"] = ts.dt.normalize()
    out["hour"] = ts.dt.hour.astype("int64")
    out["day"] = ts.dt.day.astype("int64")
    out["day_of_week"] = ts.dt.dayofweek.astype("int64")
    out["day_name"] = out["day_of_week"].map(DOW_NUMBER_TO_NAME)
    out["week"] = ts.dt.isocalendar().week.astype("int64")
    out["month"] = ts.dt.month.astype("int64")
    out["month_name"] = ts.dt.month_name()
    out["quarter"] = ts.dt.quarter.astype("int64")
    out["year"] = ts.dt.year.astype("int64")
    out["is_weekend"] = out["day_of_week"].isin(WEEKEND_DOW_NUMBERS)
    out["year_month"] = ts.dt.to_period("M").astype(str)
    out["year_quarter"] = ts.dt.to_period("Q").astype(str)

    # Every original column is preserved: the derived calendar fields are
    # *added* to the frame, so a measure such as `ridership` survives and the
    # caller can still group by it.
    keep = [c for c in out.columns if c not in CALENDAR_COLUMNS]
    keep += [c for c in CALENDAR_COLUMNS if c in out.columns]
    keep += ["day_name", "year_month", "year_quarter"]
    ordered: list = []
    for column in keep:
        if column not in ordered:
            ordered.append(column)
    out.attrs["dropped_invalid_timestamps"] = dropped
    return out[ordered].reset_index(drop=True)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def resample_series(
    df: pd.DataFrame,
    aggregation: str = "Daily",
    value_col: str = "ridership",
    timestamp_col: str = "timestamp",
    extra_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Aggregate a series to the requested sidebar frequency.

    Gaps are **not** filled. A missing bucket stays missing so that absence of
    data is never disguised as a genuine zero; ``connectgaps=False`` on the
    Plotly traces turns those gaps into visible breaks in the line.
    """
    if df.empty:
        return pd.DataFrame(columns=[timestamp_col, value_col])

    freq = AGGREGATION_FREQ.get(aggregation)
    if freq is None:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}; expected one of {AGGREGATION_ORDER}."
        )

    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, value_col])

    aggs = {value_col: "sum"}
    for source, target in (extra_cols or {}).items():
        if source in work.columns:
            aggs[target] = "mean"

    out = (
        work.set_index(timestamp_col)
        .resample(freq)
        .agg(aggs)
        .dropna(subset=[value_col])
        .reset_index()
        .rename(columns={timestamp_col: "period_start"})
    )
    return out.sort_values("period_start", kind="stable").reset_index(drop=True)


def downsample_for_plot(series: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    """Cap a series at ``max_points`` rows for plotting without losing shape.

    Long ranges (multi-year hourly windows) would otherwise hand Plotly tens of
    thousands of points. Down-sampling keeps every ``nth`` point so the line's
    envelope is preserved, and the **last row is always kept** - a plain
    ``iloc[::step]`` slice silently truncates the final observation, which would
    drop the most recent reading from every long-range chart.
    """
    if series.empty or len(series) <= max_points:
        return series.reset_index(drop=True)
    step = int(np.ceil(len(series) / max_points))
    taken = list(range(0, len(series), step))
    if taken[-1] != len(series) - 1:
        taken.append(len(series) - 1)
    return series.iloc[taken].reset_index(drop=True)


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def filter_frame(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    hours: Optional[Sequence[int]] = None,
    days: Optional[Sequence[int]] = None,
    column: str = "timestamp",
) -> pd.DataFrame:
    """Restrict a prepared frame to a window and/or an hour/weekday subset.

    Date filtering uses the *calendar day* of each observation so the final
    hour of the last selected day is never excluded.
    """
    if df.empty:
        return df

    out = df
    if start is not None:
        out = out.loc[out[column] >= pd.Timestamp(start).normalize()]
    if end is not None:
        stop = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        out = out.loc[out[column] < stop]
    if hours:
        out = out.loc[out["hour"].isin(list(hours))]
    if days:
        out = out.loc[out["day_of_week"].isin(list(days))]
    return out.copy()


# --------------------------------------------------------------------------
# Phase 2 / Phase 3 hand-off
# --------------------------------------------------------------------------


def build_forecast_table(
    hourly: pd.DataFrame,
    include_calendar: bool = True,
) -> pd.DataFrame:
    """The clean ``timestamp -> total_ridership`` table handed to Phase 2/3.

    This performs **no** modelling and computes **no** lag or rolling features:
    it only guarantees a strictly increasing, gap-free-by-observation series.
    Phase 2 may add ``lag_1`` / ``lag_24`` / ``lag_168`` / ``rolling_mean_24``
    / ``rolling_mean_168`` on top of this frame.
    """
    if hourly.empty:
        return pd.DataFrame(columns=["timestamp", "total_ridership"])

    out = hourly[["timestamp", "ridership"]].rename(
        columns={"ridership": "total_ridership"}
    )
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if include_calendar:
        out = add_calendar_columns(out, "timestamp")
    return out


def build_station_forecast_table(
    station_daily: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """``timestamp, station_complex, borough, ridership`` panel for Phase 2/3."""
    if station_daily.empty:
        return pd.DataFrame(columns=["timestamp", "station_complex", "borough", "ridership"])

    out = station_daily.rename(columns={"date": "timestamp"}).copy()
    if not station_meta.empty:
        names = station_meta[["station_complex_id", "station_complex", "borough"]]
        out = out.merge(names, on="station_complex_id", how="left")
    else:
        out["station_complex"] = out["station_complex_id"].astype(str)
        out["borough"] = None
    out = out.rename(columns={"ridership": "ridership"})
    return out[["timestamp", "station_complex", "borough", "ridership", "station_complex_id"]]
