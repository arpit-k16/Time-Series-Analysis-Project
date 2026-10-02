"""Filter / aggregation / drill-down / error-handling tests.

Drives the *same* functions the Streamlit UI calls (`Filters.build`, the
loaders, the analytics builders and the aggregation resolver), on a short
window so the server-side aggregates stay cheap to fetch.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from utils import analytics as an
from utils import data_loader as dl
from utils import preprocessing as pp

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          f"{(' - ' + detail) if detail else ''}", flush=True)
    if not condition:
        FAILURES.append(label)


def reconciliation(label, hourly, **cubes):
    city = float(hourly["ridership"].sum())
    for name, frame in cubes.items():
        if frame is None or frame.empty:
            continue
        total = float(pd.to_numeric(frame["ridership"], errors="coerce").fillna(0).sum())
        diff = total - city
        check(f"{label}: {name} == citywide", abs(diff) < 1.0,
              f"{total:,.0f} vs {city:,.0f} (diff {diff:,.2f})")


def main() -> int:
    # ---------------------------------------------------------------- filters
    print("1. Sidebar state -> Filters (mirrors app.make_filters)")
    sidebar = {
        "dates": (__import__("datetime").date(2024, 10, 1), __import__("datetime").date(2024, 10, 7)),
        "boroughs": ["Manhattan"],
        "stations": [],
        "modes": ["subway"],
        "payments": ["omny"],
        "fares": [],
        "aggregation": "Auto",
    }
    f = dl.Filters.build(
        start=sidebar["dates"][0], end=sidebar["dates"][1],
        boroughs=sidebar["boroughs"] or None,
        stations=sidebar["stations"] or None,
        modes=sidebar["modes"] or None,
        payments=sidebar["payments"] or None,
        fares=sidebar["fares"] or None,
    )
    print(f"   where: {f.where()}")
    check("borough in $where", "borough in ('Manhattan')" in f.where())
    check("mode in $where", "transit_mode in ('subway')" in f.where())
    check("payment in $where", "payment_method in ('omny')" in f.where())
    check("unselected dimension omitted", "fare_class_category in" not in f.where())
    check("signature stable", f.signature() == dl.Filters.build(
        start=sidebar["dates"][0], end=sidebar["dates"][1],
        boroughs=["Manhattan"], modes=["subway"], payments=["omny"]).signature())

    # injection attempt: a value containing a quote must be escaped
    nasty = dl.Filters.build(start="2024-10-01", end="2024-10-07",
                             boroughs=["Manha'tan"])
    check("quotes escaped in $where", "''" in nasty.where(), nasty.where()[:90])

    print("\n2. Filtered load (short window keeps the API cheap)")
    t = time.time()
    hourly = dl.load_hourly(f)
    stations = dl.load_station_totals(f)
    boroughs = dl.load_borough_totals(f)
    payments = dl.load_payment_totals(f)
    fares = dl.load_fare_totals(f)
    modes = dl.load_mode_totals(f)
    print(f"   loaded in {time.time() - t:.1f}s")
    print(f"   hourly={len(hourly)} stations={len(stations)} boroughs={len(boroughs)} "
          f"payments={len(payments)} fares={len(fares)} modes={len(modes)}")
    check("filtered hourly non-empty", len(hourly) > 0)
    check("only Manhattan stations", set(stations.get("station_complex_id", pd.Series(dtype=str)))
          <= set(stations["station_complex_id"]))
    check("only one borough", list(boroughs["borough"]) == ["Manhattan"], str(list(boroughs["borough"])))
    check("only subway mode", list(modes["transit_mode"]) == ["subway"], str(list(modes["transit_mode"])))
    check("only omny payment", list(payments["payment_method"]) == ["omny"],
          str(list(payments["payment_method"])))
    check("7 days x 24 hours", len(hourly) <= 7 * 24, f"{len(hourly)}")

    print("\n3. Reconciliation under filters (no double counting)")
    reconciliation("filtered", hourly, station=stations, borough=boroughs,
                   payment=payments, fare=fares, mode=modes)

    print("\n4. Aggregation resolution")
    # The resolver takes a WINDOW LENGTH IN DAYS, not a point count:
    # >90d -> Monthly, 15-90d -> Daily, <=14d -> Hourly.
    cases = [
        (1096, "Monthly"), (91, "Monthly"),
        (90, "Daily"), (89, "Daily"), (15, "Daily"),
        (14, "Hourly"), (7, "Hourly"), (1, "Hourly"),
    ]
    for days, expected in cases:
        got = _resolve(days)
        check(f"Auto({days}d) -> {expected}", got == expected, got)
    check("explicit Hourly honoured over 3 years", _resolve(1096, "Hourly") == "Hourly")
    check("explicit Daily honoured over 3 years", _resolve(1096, "Daily") == "Daily")
    check("explicit Monthly honoured", _resolve(7, "Monthly") == "Monthly")

    print("\n5. Derived series on the filtered window")
    kpis = an.compute_kpis(hourly, stations, boroughs)
    hp = an.hourly_profile(hourly)
    dp = an.dow_profile(hourly)
    mat = an.hour_day_matrix(hourly)
    summary = an.station_summary(stations, dl.load_station_meta(), window_hours=len(hourly))
    bsum = an.borough_summary(boroughs, summary)
    check("KPIs", kpis["n_observations"] == len(hourly))
    check("hour profile 24", len(hp) == 24)
    check("dow profile 7", len(dp) == 7)
    check("heatmap 7x24", mat.shape == (7, 24))
    check("station summary non-empty", len(summary) > 0)
    check("borough station count > 0", bool((bsum["Stations"] > 0).all()), str(list(bsum["Stations"])))
    check("filtered total < full total",
          kpis["total_ridership"] < 3_395_439_569,
          f"{kpis['total_ridership']:,.0f} vs 3,395,439,569")
    print(f"   filtered total = {kpis['total_ridership']:,.0f} across {kpis['n_stations']} stations")

    print("\n6. Station drill-down (adds a station to the same filter set)")
    top_id = str(summary.iloc[0]["station_complex_id"])
    series = dl.load_station_hourly(f, top_id)
    check("drill-down series non-empty", len(series) > 0, f"{len(series)} hours for {top_id}")
    check("drill-down respects window", len(series) <= 7 * 24)
    if len(series):
        prepared = pp.add_calendar_columns(series, "timestamp")
        check("drill-down calendar columns",
              {"hour", "day_of_week", "is_weekend", "month", "year"} <= set(prepared.columns))
        check("drill-down ridership preserved", "ridership" in prepared.columns)
        check("drill-down daily resample", len(pp.resample_series(series, "Daily")) > 0)
        check("drill-down monthly resample", len(pp.resample_series(series, "Monthly")) > 0)
        pts = an.map_points(summary)
        check("map points for filtered set", len(pts) > 0, f"{len(pts)} markers")
        check("map avg hourly positive", bool((pts["Avg_Hourly_Ridership"].fillna(0) > 0).all()))

    print("\n7. Empty selection (friendly empty state, no crash)")
    empty = dl.Filters.build(start="2024-10-01", end="2024-10-07",
                             boroughs=["Manhattan"], stations=["999999999"])
    try:
        eh = dl.load_hourly(empty)
        check("empty filter returns empty frame", eh.empty, f"{len(eh)} rows")
    except Exception as exc:  # noqa: BLE001
        check("empty filter returns empty frame", False, f"{type(exc).__name__}: {exc}")

    print("\n8. Invalid date handling")
    try:
        bad = dl.Filters.build(start="not-a-date", end="2024-10-07")
        check("invalid date raises DataLoadError", False, "no exception raised")
    except dl.DataLoadError:
        check("invalid date raises DataLoadError", True)
    except Exception as exc:  # noqa: BLE001
        check("invalid date raises DataLoadError", False, f"wrong type {type(exc).__name__}")

    print("\n9. Downsampling keeps figures small")
    big = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=26302, freq="h"),
                        "ridership": range(26302)})
    small = pp.downsample_for_plot(big, 6000)
    check("downsampled <= budget", len(small) <= 6000, f"{len(small)} of {len(big)}")
    check("downsample preserves first/last ends",
          small["timestamp"].iloc[0] == big["timestamp"].iloc[0]
          and small["timestamp"].iloc[-1] == big["timestamp"].iloc[-1])

    print("\n10. Missing-value / duplicate resilience")
    messy = pd.DataFrame({
        "transit_timestamp": ["2024-10-01T00:00:00", "2024-10-01T00:00:00", "bogus"],
        "ridership": [10.0, 5.0, 1.0],
    })
    out = dl._finalise_hourly(messy)
    check("duplicates summed not dropped", len(out) == 1 and out["ridership"].iloc[0] == 15.0,
          str(out.to_dict("records")))
    check("invalid timestamp dropped", pd.notna(out["timestamp"].iloc[0]))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("ALL FILTER CHECKS PASSED")
    return 0


def _resolve(days, requested="Auto"):
    from app import resolve_aggregation

    return resolve_aggregation(requested, days)


if __name__ == "__main__":
    sys.exit(main())