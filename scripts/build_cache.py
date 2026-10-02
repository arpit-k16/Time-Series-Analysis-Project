"""
scripts/build_cache.py
=====================
Pre-builds the Parquet aggregate cache so the dashboard starts instantly and
never blocks on the API for the default analysis period.

The raw dataset holds ~121 million rows and is never downloaded. Each cube
below is a *server-side aggregate* whose result is memoised to
``data/processed/`` by ``utils.data_loader``.

Usage::

    python scripts/build_cache.py                # default period, all cubes
    python scripts/build_cache.py --start 2024-01-01 --end 2024-12-31
    python scripts/build_cache.py --hourly-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data_loader as dl  # noqa: E402


def _run(label, fn, log):
    start = time.time()
    try:
        frame = fn()
        rows = len(frame) if hasattr(frame, "__len__") else "-"
        log(f"  [OK  ] {label:<22} rows={rows:<8} {time.time() - start:6.1f}s")
        return frame
    except Exception as exc:  # noqa: BLE001 - the builder reports, never crashes
        log(f"  [FAIL] {label:<22} {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-build the MTA aggregate cache.")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--hourly-only", action="store_true")
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    filters = dl.Filters.build(start=args.start, end=args.end)

    log(f"MTA aggregate cache builder")
    log(f"Dataset : {dl.DATASET_ID}")
    log(f"Window  : {filters.lo:%d %b %Y} -> {filters.hi:%d %b %Y}")
    log(f"Chunks  : {len(filters.spans())}")
    log("")

    log("Reference data")
    _run("domains", dl.load_domains, log)
    meta = _run("station_meta", dl.load_station_meta, log)
    _run("raw_row_count", dl.load_raw_row_count, log)
    _run("observed_span", dl.load_observed_span, log)
    log("")

    log("Analysis cubes")
    _run("hourly", lambda: dl.load_hourly(filters), log)
    if not args.hourly_only:
        _run("station_totals", lambda: dl.load_station_totals(filters), log)
        _run("borough_totals", lambda: dl.load_borough_totals(filters), log)
        _run("payment_totals", lambda: dl.load_payment_totals(filters), log)
        _run("fare_totals", lambda: dl.load_fare_totals(filters), log)
        _run("mode_totals", lambda: dl.load_mode_totals(filters), log)

    if meta is not None and not meta.empty:
        first = str(meta.iloc[0]["station_complex_id"])
        _run("station_hourly (demo)", lambda: dl.load_station_hourly(filters, first), log)

    log("")
    log("Cache inventory")
    inventory = dl.cache_status()
    for _, row in inventory.iterrows():
        log(f"  {row['Cube']:<24} rows={row['Rows']:<9} {row['Size (KB)']:8.1f} KB")
    log(f"  TOTAL {len(inventory)} files, "
        f"{float(inventory['Size (KB)'].sum()) / 1024:.2f} MB")
    log("")
    log("Done. Start the dashboard with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
