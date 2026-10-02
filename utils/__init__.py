"""
utils/
=====
Helper package for the *MTA Subway Ridership Analytics* dashboard.

Split by responsibility so each stage of the pipeline can be reused (and unit
tested, or lifted into a forecasting notebook) on its own:

``data_loader``
    Access layer for the Socrata dataset ``wujg-7c2s``. Every call is a
    server-side aggregate; nothing ever reads the ~121M raw rows.
``preprocessing``
    Turns raw API columns into the descriptive calendar fields the Phase 1
    views need, plus the aggregation/resampling helpers.
``analytics``
    KPI, profile, ranking and observation builders used by the Streamlit tabs.
"""

from __future__ import annotations

from .analytics import (
    borough_summary,
    build_observations,
    compute_kpis,
    dow_profile,
    hourly_profile,
    hour_day_matrix,
    map_points,
    monthly_trend,
    station_summary,
    top_stations,
    year_trend,
)
from .data_loader import (
    ApiUnavailableError,
    AggregateCache,
    DataLoadError,
    Filters,
    dataset_span,
    default_filters,
    cache_status,
    load_borough_totals,
    load_domains,
    load_fare_totals,
    load_hourly,
    load_mode_totals,
    load_observed_span,
    load_payment_totals,
    load_raw_row_count,
    load_station_daily,
    load_station_hourly,
    load_station_meta,
    load_station_totals,
)
from .preprocessing import (
    DOW_NUMBER_TO_NAME,
    DOW_ORDER,
    add_calendar_columns,
    filter_frame,
    resample_series,
)

__all__ = [
    "AggregateCache",
    "ApiUnavailableError",
    "DataLoadError",
    "DOW_NUMBER_TO_NAME",
    "DOW_ORDER",
    "Filters",
    "add_calendar_columns",
    "borough_summary",
    "build_observations",
    "cache_status",
    "compute_kpis",
    "dataset_span",
    "default_filters",
    "dow_profile",
    "downsample_for_plot",
    "filter_frame",
    "hour_day_matrix",
    "hourly_profile",
    "load_borough_totals",
    "load_domains",
    "load_fare_totals",
    "load_hourly",
    "load_mode_totals",
    "load_observed_span",
    "load_payment_totals",
    "load_raw_row_count",
    "load_station_daily",
    "load_station_hourly",
    "load_station_meta",
    "load_station_totals",
    "map_points",
    "monthly_trend",
    "resample_series",
    "station_summary",
    "top_stations",
    "year_trend",
]
