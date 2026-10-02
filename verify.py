"""End-to-end verification of the MTA pipeline for the default window.

Checks that every cube reconciles with the citywide total (proving the
payment_method / fare_class_category split is summed rather than double
counted), that the analytics builders work, and that memory stays small.
"""
from __future__ import annotations

import sys
import tracemalloc

import pandas as pd

from utils import analytics as an
from utils import data_loader as dl
from utils import preprocessing as pp

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}", flush=True)
    if not condition:
        FAILURES.append(label)


def main() -> int:
    tracemalloc.start()
    f = dl.default_filters()
    print(f"Window: {f.lo:%d %b %Y} -> {f.hi:%d %b %Y}\n")

    print("Loading cubes from the Parquet cache (no API call expected)...")
    hourly = dl.load_hourly(f)
    stations_raw = dl.load_station_totals(f)
    boroughs_raw = dl.load_borough_totals(f)
    payments_raw = dl.load_payment_totals(f)
    fares_raw = dl.load_fare_totals(f)
    modes_raw = dl.load_mode_totals(f)
    meta = dl.load_station_meta()

    city = float(hourly["ridership"].sum())
    city_transfers = float(hourly["transfers"].sum())

    print("\n--- Row counts ---")
    print(f"  hourly           {len(hourly):>8,}")
    print(f"  station_totals   {len(stations_raw):>8,}")
    print(f"  borough_totals   {len(boroughs_raw):>8,}")
    print(f"  payment_totals   {len(payments_raw):>8,}")
    print(f"  fare_totals      {len(fares_raw):>8,}")
    print(f"  mode_totals      {len(modes_raw):>8,}")
    print(f"  station_meta     {len(meta):>8,}")
    print(f"  citywide total   {city:>8,.0f} riders")

    print("\n--- Reconciliation (no double counting) ---")
    # Every dimension cube must partition the citywide total exactly. This is
    # the test that matters: the raw dataset splits each station-hour across
    # payment_method and fare_class_category, so any cube that forgot to sum
    # would disagree here.
    for label, frame, key in (
        ("station", stations_raw, None),
        ("borough", boroughs_raw, None),
        ("payment", payments_raw, None),
        ("fare", fares_raw, None),
        ("mode", modes_raw, None),
    ):
        total = float(pd.to_numeric(frame["ridership"], errors="coerce").fillna(0).sum())
        diff = total - city
        check(
            f"{label} totals == citywide total",
            abs(diff) < 1.0,
            f"{total:,.0f} vs {city:,.0f} (diff {diff:,.2f})",
        )

    transfers_check = float(pd.to_numeric(stations_raw["transfers"], errors="coerce").fillna(0).sum())
    check(
        "station transfers == citywide transfers",
        abs(transfers_check - city_transfers) < 1.0,
        f"{transfers_check:,.0f} vs {city_transfers:,.0f}",
    )

    check(
        "hourly timestamps unique",
        int(hourly["timestamp"].duplicated().sum()) == 0,
    )
    check("hourly sorted ascending", hourly["timestamp"].is_monotonic_increasing)
    expected = 1096 * 24  # 2022-01-01 .. 2024-12-31
    print(
        f"\n  hourly rows {len(hourly):,} vs {expected:,} expected hours "
        f"({expected - len(hourly)} absent)"
    )

    print("\n--- Raw rows vs analysis rows ---")
    raw_rows = int(pd.to_numeric(stations_raw["raw_rows"], errors="coerce").fillna(0).sum())
    print(f"  raw dataset rows behind this window : {raw_rows:>12,}")
    print(f"  analysis records held in memory     : {len(hourly):>12,}")
    print(f"  reduction factor                    : {raw_rows / max(len(hourly), 1):>11,.0f}x")

    print("\n--- Analytics ---")
    summary = an.station_summary(stations_raw, meta, window_hours=len(hourly))
    check("station summary rows", len(summary) == 428, f"{len(summary)}")
    check("station summary sorted by total", bool(
        summary["Total_Ridership"].is_monotonic_decreasing))
    check("station summary no NaN", not summary.isna().any().any(),
          str({c: int(summary[c].isna().sum()) for c in summary.columns if summary[c].isna().any()}))
    check("station summary has coordinates", summary[["latitude", "longitude"]].notna().all().all())
    check("station summary boroughs known",
          set(summary["borough"]) <= {"Manhattan", "Brooklyn", "Queens", "Bronx",
                                      "Staten Island", "Unknown"},
          str(sorted(set(summary["borough"]))))

    boroughs = an.borough_summary(boroughs_raw, summary)
    check("borough summary rows", len(boroughs) == 5, f"{len(boroughs)}")
    check("borough station counts > 0", bool((boroughs["Stations"] > 0).all()),
          str(list(boroughs["Stations"])))
    print(boroughs.to_string(index=False))

    kpis = an.compute_kpis(hourly, stations_raw, boroughs_raw)
    check("KPI total matches", abs(kpis["total_ridership"] - city) < 1.0)
    check("KPI stations", kpis["n_stations"] == 428, str(kpis["n_stations"]))
    check("KPI boroughs", kpis["n_boroughs"] == 5, str(kpis["n_boroughs"]))
    check("KPI observations", kpis["n_observations"] == len(hourly))
    print(f"\n  peak {kpis['max_hourly']:,.0f} at {kpis['max_hourly_timestamp']}")
    print(f"  avg hourly {kpis['avg_hourly']:,.0f}")
    print(f"  total transfers {kpis['total_transfers']:,.0f}")

    hp = an.hourly_profile(hourly)
    dp = an.dow_profile(hourly)
    mat = an.hour_day_matrix(hourly)
    years = an.year_trend(hourly)
    check("hourly profile 24 rows", len(hp) == 24)
    check("dow profile 7 rows", len(dp) == 7)
    check("dow order Monday-first",
          list(dp.sort_values("DayOfWeek")["DayName"]) == list(pp.DOW_ORDER))
    check("heatmap shape 7x24", mat.shape == (7, 24), str(mat.shape))
    check("heatmap no NaN", bool(mat.notna().all().all()))
    check("year trend covers 2022-2024", list(years["year"]) == [2022, 2023, 2024],
          str(list(years["year"])))
    print("\n" + years.to_string(index=False))

    points = an.map_points(summary)
    check("map points present", len(points) > 0, f"{len(points)} markers")
    check("map avg hourly > 0", bool((points["Avg_Hourly_Ridership"].fillna(0) > 0).all()),
          f"min={points['Avg_Hourly_Ridership'].min():.1f} "
          f"max={points['Avg_Hourly_Ridership'].max():.1f}")
    check("summary avg hourly > 0", bool((summary["Avg_Hourly_Ridership"] > 0).all()))
    check("map lat/lon numeric",
          pd.to_numeric(points["latitude"], errors="coerce").notna().all())

    # --- resampling / forecast table ---------------------------------
    for agg in ("Daily", "Weekly", "Monthly"):
        out = pp.resample_series(hourly, agg)
        check(f"resample {agg}", len(out) > 0, f"{len(out)} buckets")
    ft = pp.build_forecast_table(hourly)
    check("forecast table", len(ft) == len(hourly), f"{len(ft)} rows")
    check("forecast columns",
          {"timestamp", "total_ridership", "hour", "day_of_week", "month",
           "year", "is_weekend"} <= set(ft.columns))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n--- Memory (python allocations during load + analytics) ---")
    print(f"  peak traced : {peak / 1e6:.1f} MB")
    check("peak memory under 400 MB", peak / 1e6 < 400, f"{peak / 1e6:.1f} MB")

    print("\n--- Observations (dynamic) ---")
    payments = payments_raw.rename(columns={
        "ridership": "Total_Ridership", "transfers": "Total_Transfers",
        "raw_rows": "Raw_Rows"})
    for note in an.build_observations(kpis, hp, dp, summary, boroughs, years, payments, None,
                                   an.monthly_trend(hourly), "the selected period"):
        print(f"  - {note}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
