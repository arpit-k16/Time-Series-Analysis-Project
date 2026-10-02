"""
utils/analytics.py
==================
Descriptive aggregations behind every tab of the dashboard.

Nothing here forecasts, imputes or models: each function turns an already
aggregated cube into the exact frame a chart or KPI card needs. All of them are
pure functions over a DataFrame, so they are reusable from a Phase 2 / Phase 3
notebook without touching Streamlit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .preprocessing import DOW_NUMBER_TO_NAME, DOW_ORDER, add_calendar_columns

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------


def compute_kpis(
    hourly: pd.DataFrame,
    station_totals: Optional[pd.DataFrame] = None,
    borough_totals: Optional[pd.DataFrame] = None,
    transfers: Optional[float] = None,
) -> Dict[str, Any]:
    """Every Section 1 KPI, recomputed from the *filtered* cubes.

    ``hourly`` is the citywide series (one row per timestamp, ridership already
    summed across stations, payment methods and fare classes), so its sums are
    the true network totals - no raw row is ever treated as an independent
    observation.
    """
    out: Dict[str, Any] = {
        "total_ridership": 0.0,
        "avg_hourly": float("nan"),
        "max_hourly": float("nan"),
        "max_hourly_timestamp": pd.NaT,
        "min_timestamp": pd.NaT,
        "max_timestamp": pd.NaT,
        "n_observations": 0,
        "n_stations": 0,
        "n_boroughs": 0,
        "total_transfers": 0.0,
        "avg_transfers": float("nan"),
    }
    if not hourly.empty:
        totals = hourly["ridership"]
        idx = totals.idxmax()
        out.update(
            total_ridership=float(totals.sum()),
            avg_hourly=float(totals.mean()),
            max_hourly=float(totals.max()),
            max_hourly_timestamp=hourly.loc[idx, "timestamp"],
            min_timestamp=hourly["timestamp"].min(),
            max_timestamp=hourly["timestamp"].max(),
            n_observations=int(len(hourly)),
            total_transfers=float(hourly["transfers"].sum())
            if "transfers" in hourly.columns
            else 0.0,
            avg_transfers=float(hourly["transfers"].mean())
            if "transfers" in hourly.columns
            else float("nan"),
        )
    if transfers is not None:
        out["total_transfers"] = float(transfers)

    if station_totals is not None and not station_totals.empty:
        out["n_stations"] = int(station_totals["station_complex_id"].nunique())
    if borough_totals is not None and not borough_totals.empty:
        out["n_boroughs"] = int(len(borough_totals))
    return out


# --------------------------------------------------------------------------
# Temporal patterns
# --------------------------------------------------------------------------


def hourly_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average *citywide hourly* ridership for each hour of day (0-23)."""
    columns = ["hour", "Avg_Ridership", "Total_Ridership", "Observations"]
    if hourly.empty:
        return pd.DataFrame(columns=columns)
    out = add_calendar_columns(hourly, "timestamp")
    out = (
        out.groupby("hour", observed=True)
        .agg(
            Avg_Ridership=("ridership", "mean"),
            Total_Ridership=("ridership", "sum"),
            Observations=("ridership", "size"),
        )
        .reindex(range(24))
        .rename_axis("hour")
        .reset_index()
    )
    # An unobserved hour is a gap (NaN), never a fabricated zero.
    return out[columns]


def dow_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average citywide hourly ridership per weekday, Monday-first."""
    columns = ["DayOfWeek", "DayName", "Avg_Ridership", "Total_Ridership", "Observations"]
    if hourly.empty:
        return pd.DataFrame(columns=columns)
    out = add_calendar_columns(hourly, "timestamp")
    out = (
        out.groupby("day_of_week", observed=True)
        .agg(
            Avg_Ridership=("ridership", "mean"),
            Total_Ridership=("ridership", "sum"),
            Observations=("ridership", "size"),
        )
        .reindex(range(7))
        .rename_axis("DayOfWeek")
        .reset_index()
    )
    out["DayName"] = out["DayOfWeek"].map(DOW_NUMBER_TO_NAME)
    return out[columns]


def hour_day_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week x hour-of-day matrix of average citywide hourly ridership."""
    index = pd.Index(range(7), name="DayOfWeek")
    columns = pd.Index(range(24), name="hour")
    if hourly.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)

    out = add_calendar_columns(hourly, "timestamp")
    pivot = out.pivot_table(
        index="day_of_week", columns="hour", values="ridership", aggfunc="mean", observed=True
    )
    return pivot.reindex(index=index, columns=columns)


def year_trend(hourly: pd.DataFrame) -> pd.DataFrame:
    """Ridership per calendar year - the multi-year depth this dataset adds."""
    columns = ["year", "Total_Ridership", "Avg_Hourly_Ridership", "Observations"]
    if hourly.empty:
        return pd.DataFrame(columns=columns)
    out = add_calendar_columns(hourly, "timestamp")
    grouped = (
        out.groupby("year", observed=True)
        .agg(
            Total_Ridership=("ridership", "sum"),
            Avg_Hourly_Ridership=("ridership", "mean"),
            Observations=("ridership", "size"),
        )
        .reset_index()
    )
    return grouped.sort_values("year").reset_index(drop=True)[columns]


def monthly_trend(hourly: pd.DataFrame) -> pd.DataFrame:
    """Ridership per calendar month, ordered chronologically."""
    columns = ["YearMonth", "Total_Ridership", "Avg_Hourly_Ridership", "Observations"]
    if hourly.empty:
        return pd.DataFrame(columns=columns)
    out = add_calendar_columns(hourly, "timestamp")
    grouped = (
        out.groupby("year_month", observed=True)
        .agg(
            Total_Ridership=("ridership", "sum"),
            Avg_Hourly_Ridership=("ridership", "mean"),
            Observations=("ridership", "size"),
        )
        .reset_index()
        .rename(columns={"year_month": "YearMonth"})
    )
    # Plotly parses "2024-07" as a date; sorting on the string keeps it a label.
    return grouped.sort_values("YearMonth").reset_index(drop=True)[columns]


# --------------------------------------------------------------------------
# Station & borough
# --------------------------------------------------------------------------

_STATION_COLUMNS = [
    "station_complex_id",
    "station_complex",
    "borough",
    "latitude",
    "longitude",
    "Total_Ridership",
    "Avg_Hourly_Ridership",
    "Total_Transfers",
    "Raw_Rows",
]


def station_summary(
    station_totals: pd.DataFrame,
    station_meta: Optional[pd.DataFrame] = None,
    window_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Per-station totals joined to names, borough and coordinates.

    ``window_hours`` is the number of hourly timestamps in the selected
    window; the station's average hourly ridership is its total divided by
    that. Counting each station's own distinct hours server-side would need
    ``count(distinct transit_timestamp)``, which the endpoint cannot evaluate
    inside the request timeout, so the window length is used instead. This is
    exact for complexes reporting through the whole window and a slight
    overstatement for complexes that opened or closed during it - documented as
    an assumption rather than silently applied.
    """
    if station_totals is None or station_totals.empty:
        return pd.DataFrame(columns=_STATION_COLUMNS)

    out = station_totals.rename(
        columns={
            "ridership": "Total_Ridership",
            "transfers": "Total_Transfers",
            "raw_rows": "Raw_Rows",
        }
    ).copy()
    out["station_complex_id"] = out["station_complex_id"].astype(str)

    if station_meta is not None and not station_meta.empty:
        meta = station_meta.copy()
        meta["station_complex_id"] = meta["station_complex_id"].astype(str)
        keep = ["station_complex_id"]
        for column in ("station_complex", "borough", "latitude", "longitude"):
            if column in meta.columns:
                keep.append(column)
        meta = meta[keep].drop_duplicates(subset="station_complex_id", keep="first")
        out = out.merge(meta, on="station_complex_id", how="left")

    for column, default in (
        ("station_complex", None),
        ("borough", "Unknown"),
        ("latitude", np.nan),
        ("longitude", np.nan),
    ):
        if column not in out.columns:
            out[column] = default
    out["station_complex"] = out["station_complex"].fillna(
        "Complex " + out["station_complex_id"]
    )
    out["borough"] = out["borough"].fillna("Unknown")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    for column in ("Total_Ridership", "Total_Transfers", "Raw_Rows"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce").fillna(0.0)

    if window_hours:
        out["Avg_Hourly_Ridership"] = out["Total_Ridership"] / float(window_hours)
    else:
        out["Avg_Hourly_Ridership"] = np.nan

    return out.sort_values("Total_Ridership", ascending=False).reset_index(drop=True)[
        _STATION_COLUMNS
    ]


def top_stations(summary: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top ``n`` station complexes by total ridership, longest label last."""
    if summary.empty:
        return summary
    return summary.nlargest(n, "Total_Ridership")


def borough_summary(
    borough_totals: Optional[pd.DataFrame],
    station_totals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-borough totals plus the number of station complexes it contains.

    ``station_totals`` must carry a ``borough`` column - pass the output of
    :func:`station_summary` (which joins the reference table), because the raw
    station cube groups by id alone.
    """
    columns = ["borough", "Total_Ridership", "Total_Transfers", "Raw_Rows", "Stations"]
    if borough_totals is None or borough_totals.empty:
        return pd.DataFrame(columns=columns)

    out = borough_totals.rename(
        columns={
            "ridership": "Total_Ridership",
            "transfers": "Total_Transfers",
            "raw_rows": "Raw_Rows",
        }
    ).copy()

    if station_totals is not None and not station_totals.empty and "borough" in station_totals:
        counts = (
            station_totals[["station_complex_id", "borough"]]
            .dropna(subset=["borough"])
            .drop_duplicates()
            .groupby("borough", observed=True)
            .size()
            .rename("Stations")
        )
        out = out.merge(counts, on="borough", how="left")
    if "Stations" not in out.columns:
        out["Stations"] = 0
    out["Stations"] = out["Stations"].fillna(0).astype("int64")

    for column in ("Total_Ridership", "Total_Transfers", "Raw_Rows"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce").fillna(0.0)

    return out.sort_values("Total_Ridership", ascending=False).reset_index(drop=True)[
        columns
    ]


def map_points(
    summary: pd.DataFrame,
    avg_hourly: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Station coordinates with the volume used to size each map marker.

    Duplicate coordinates (two complexes sharing a point, or one complex listed
    at slightly different coordinates) are collapsed by averaging position and
    summing volume, so the map never double-draws a marker.
    """
    columns = [
        "station_complex",
        "borough",
        "Total_Ridership",
        "Avg_Hourly_Ridership",
        "latitude",
        "longitude",
        "Stations_Merged",
    ]
    if summary is None or summary.empty:
        return pd.DataFrame(columns=columns)

    out = summary.copy()
    out = out.dropna(subset=["latitude", "longitude"])
    if out.empty:
        return pd.DataFrame(columns=columns)

    if "Avg_Hourly_Ridership" not in out.columns:
        out["Avg_Hourly_Ridership"] = np.nan

    # `station_summary` already divides each complex's total by the number of
    # hourly timestamps in the selected window, so the map hover carries a real
    # average rather than a placeholder.
    out["_avg"] = pd.to_numeric(out["Avg_Hourly_Ridership"], errors="coerce").fillna(0.0)
    # Each complex's average is already a rate over the *same* window length,
    # so merging two of them is a plain mean. Weighting by ridership would
    # divide a rate by a volume and collapse every marker to ~0.
    out["_weight"] = 1.0

    merged = (
        out.assign(latitude=out["latitude"].round(5), longitude=out["longitude"].round(5))
        .groupby(["station_complex", "borough", "latitude", "longitude"], as_index=False)
        .agg(
            Total_Ridership=("Total_Ridership", "sum"),
            _num=("_avg", "sum"),
            _weight=("_weight", "sum"),
            Stations_Merged=("station_complex_id", "count"),
        )
    )
    # Plain mean over the merged complexes, which is correct because every one
    # of them was averaged over the same number of hours.
    merged["Avg_Hourly_Ridership"] = np.where(
        merged["_weight"] > 0, merged["_num"] / merged["_weight"], np.nan
    )
    merged = merged.drop(columns=["_num", "_weight"])

    return merged.sort_values("Total_Ridership", ascending=False).reset_index(drop=True)[
        columns
    ]


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def build_observations(
    kpis: Dict[str, Any],
    hour_prof: pd.DataFrame,
    dow_prof: pd.DataFrame,
    stations: pd.DataFrame,
    boroughs: pd.DataFrame,
    years: pd.DataFrame,
    payments: Optional[pd.DataFrame] = None,
    focus_station: Optional[str] = None,
    months: Optional[pd.DataFrame] = None,
    period: str = "the selected period",
) -> List[str]:
    """Factual observations computed from the current selection.

    Every sentence is derived from a value calculated above the call site and
    uses descriptive language only - no causal claim is made anywhere. Each one
    is explicitly scoped to the active filter/date selection so it can never be
    read as a statement about the whole dataset.
    """
    notes: List[str] = []

    if kpis.get("n_observations", 0) == 0:
        return [f"No observations match the current filter selection ({period})."]

    within = f"Within {period}"

    # 1. Peak / trough hour of day.
    valid = hour_prof.dropna(subset=["Avg_Ridership"])
    if not valid.empty:
        busy = valid.loc[valid["Avg_Ridership"].idxmax()]
        quiet = valid.loc[valid["Avg_Ridership"].idxmin()]
        notes.append(
            f"{within}, the highest average hourly ridership occurs at "
            f"**{int(busy['hour']):02d}:00**, averaging "
            f"**{busy['Avg_Ridership']:,.0f}** riders per hour, and the lowest at "
            f"**{int(quiet['hour']):02d}:00** "
            f"({quiet['Avg_Ridership']:,.0f} riders per hour)."
        )

    # 2. Busiest / quietest day of week.
    valid_dow = dow_prof.dropna(subset=["Avg_Ridership"])
    if not valid_dow.empty:
        busy = valid_dow.loc[valid_dow["Avg_Ridership"].idxmax()]
        quiet = valid_dow.loc[valid_dow["Avg_Ridership"].idxmin()]
        notes.append(
            f"{within}, **{busy['DayName']}** has the highest average hourly ridership "
            f"({busy['Avg_Ridership']:,.0f}) and **{quiet['DayName']}** the lowest "
            f"({quiet['Avg_Ridership']:,.0f})."
        )

    # 3. Top / bottom station.
    if not stations.empty:
        top = stations.iloc[0]
        total = float(kpis.get("total_ridership", 0.0))
        share = 100.0 * float(top["Total_Ridership"]) / total if total else float("nan")
        quiet = stations.iloc[-1]
        notes.append(
            f"{within}, **{top['station_complex']}** ({top['borough']}) records the "
            f"highest total ridership at **{top['Total_Ridership']:,.0f}** riders "
            f"(**{share:.1f}%** of the period total); the lowest-ranked complex is "
            f"**{quiet['station_complex']}** ({quiet['borough']}) at "
            f"**{quiet['Total_Ridership']:,.0f}** riders."
        )

    # 4. Top borough.
    if not boroughs.empty:
        top = boroughs.iloc[0]
        total = float(kpis.get("total_ridership", 0.0))
        share = 100.0 * float(top["Total_Ridership"]) / total if total else float("nan")
        quiet = boroughs.iloc[-1]
        notes.append(
            f"{within}, **{top['borough']}** contributes the most ridership "
            f"({top['Total_Ridership']:,.0f} riders, {share:.1f}% of the period total) "
            f"across **{int(top['Stations'])}** station complexes, while "
            f"**{quiet['borough']}** contributes the least "
            f"({quiet['Total_Ridership']:,.0f} riders)."
        )

    # 5. Busiest month.
    if months is not None and not months.empty:
        best = months.loc[months["Total_Ridership"].idxmax()]
        notes.append(
            f"{within}, the highest-ridership calendar month on record is "
            f"**{best['YearMonth']}** at **{best['Total_Ridership']:,.0f}** riders."
        )

    # 6. Year-on-year movement.
    if years is not None and not years.empty and len(years) >= 2:
        first, last = years.iloc[0], years.iloc[-1]
        if first["Total_Ridership"] > 0:
            change = 100.0 * (
                last["Total_Ridership"] - first["Total_Ridership"]
            ) / first["Total_Ridership"]
            direction = "increased" if change >= 0 else "decreased"
            notes.append(
                f"{within}, total recorded ridership {direction} by "
                f"**{abs(change):.1f}%** between {int(first['year'])} "
                f"({first['Total_Ridership']:,.0f}) and {int(last['year'])} "
                f"({last['Total_Ridership']:,.0f}). Partial years are not adjusted "
                "for length."
            )

    # 7. Leading payment method.
    if payments is not None and not payments.empty:
        top = payments.iloc[0]
        # The share is against every payment method, not just the leading one.
        total = float(pd.to_numeric(payments["Total_Ridership"], errors="coerce").fillna(0).sum())
        share = 100.0 * float(top["Total_Ridership"]) / total if total else float("nan")
        notes.append(
            f"{within}, **{top['payment_method']}** accounts for the largest observed "
            f"ridership ({top['Total_Ridership']:,.0f} riders, {share:.1f}% of the "
            "period total)."
        )

    # 8. Coverage of the window.
    notes.append(
        f"{within}, the analysis covers **{kpis['n_observations']:,} hourly observations** "
        f"across **{kpis['n_stations']}** station complexes and "
        f"**{kpis['n_boroughs']}** borough(s), with a peak hourly total of "
        f"**{kpis['max_hourly']:,.0f}** riders at "
        f"{pd.Timestamp(kpis['max_hourly_timestamp']):%d %b %Y %H:%M}."
    )

    # 9. Focus station, when one is selected.
    if focus_station and not stations.empty and focus_station in set(stations["station_complex"]):
        row = stations.loc[stations["station_complex"] == focus_station].iloc[0]
        rank = int(stations.index[stations["station_complex"] == focus_station][0]) + 1
        notes.append(
            f"{within}, the selected complex **{focus_station}** ranks **{rank}** of "
            f"{len(stations)} complexes, recording {row['Total_Ridership']:,.0f} riders "
            f"and {row['Total_Transfers']:,.0f} transfers."
        )

    return notes
