"""End-to-end check: the main trend grain follows the length of the window.

Complements the unit-level ``resolve_aggregation`` cases in verify_filters.py by
(a) pinning the exact day-count boundaries of the auto rule and (b) driving the
real ``build_view`` for a long, medium and short window to confirm the resolved
grain reaches the chart and that resampling stays lossless.
"""

import sys

import pandas as pd

from app import (
    TREND_HOVER,
    _window_days,
    build_view,
    make_filters,
    pp,
    resolve_aggregation,
)

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def trend_title(grain):
    """Same expression the Overview uses for the main chart title."""
    return f"MTA {grain} Ridership Trend"


def sidebar(start, end, aggregation="Auto"):
    return {
        "dates": (pd.Timestamp(start), pd.Timestamp(end)),
        "boroughs": [],
        "stations": [],
        "modes": [],
        "payments": [],
        "fares": [],
        "aggregation": aggregation,
    }


def check_boundaries():
    print("\nboundaries (unit)")
    for days, expected in ((14, "Hourly"), (15, "Daily"), (90, "Daily"),
                           (91, "Monthly"), (365, "Monthly"), (1096, "Monthly")):
        got = resolve_aggregation("Auto", days)
        check(f"{days} days -> {expected}", got == expected, got)
    for requested in ("Hourly", "Daily", "Monthly", "Weekly"):
        check(f"explicit {requested} wins over 1096 days",
              resolve_aggregation(requested, 1096) == requested)


def check_windows():
    cases = [
        ("long   2022-01-01 -> 2024-12-31", "2022-01-01", "2024-12-31", "Monthly"),
        ("medium 2024-01-01 -> 2024-03-29", "2024-01-01", "2024-03-29", "Daily"),
        ("short  2024-10-01 -> 2024-10-07", "2024-10-01", "2024-10-07", "Hourly"),
    ]
    for label, start, end, expected in cases:
        print(f"\n{label}")
        sb = sidebar(start, end)
        filters = make_filters(sb)
        days = _window_days(filters)
        grain = resolve_aggregation(sb["aggregation"], days)

        check(f"{days} days resolves to {expected}", grain == expected, grain)
        check("hover format defined for grain", grain in TREND_HOVER)
        check("title follows the grain", trend_title(grain) ==
              f"MTA {expected} Ridership Trend", trend_title(grain))

        v = build_view(sb)
        series = pp.resample_series(v.hourly, grain)

        check("view agrees with resolver", v.aggregation == grain, v.aggregation)
        check("series is non-empty", not series.empty, f"{len(series)} points")
        # Resampling must be lossless: a finer-to-coarser sum has to reproduce
        # the hourly citywide total exactly.
        total_hourly = float(
            pd.to_numeric(v.hourly["ridership"], errors="coerce").fillna(0).sum()
        )
        total_series = float(series["ridership"].sum())
        check("grain sum equals hourly total (no double counting)",
              abs(total_hourly - total_series) < 1.0,
              f"{total_series:,.0f} vs {total_hourly:,.0f}")
        check("grain total matches the KPI total",
              abs(total_series - float(v.kpis["total_ridership"])) < 1.0,
              f"{float(v.kpis['total_ridership']):,.0f}")
        # A coarser grain can never have more points than the hourly source.
        check("point count fits the grain", len(series) <= len(v.hourly),
              f"{len(series)} <= {len(v.hourly)}")

    print("\noverride (end to end)")
    long_sb = sidebar("2022-01-01", "2024-12-31")
    long_sb["aggregation"] = "Hourly"
    check("explicit Hourly over 3 years survives build_view",
          build_view(long_sb).aggregation == "Hourly")

    print("\nfiner scales remain reachable")
    v = build_view(sidebar("2022-01-01", "2024-12-31"))
    for grain in ("Daily", "Monthly", "Hourly"):
        s = pp.resample_series(v.hourly, grain)
        check(f"{grain} series buildable", not s.empty, f"{len(s)} points")


def main():
    check_boundaries()
    check_windows()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("ALL WINDOW CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())