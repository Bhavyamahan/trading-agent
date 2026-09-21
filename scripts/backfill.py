"""Download NSE history year by year and store it in Supabase Storage.

Output files (raw, unadjusted prices; adjustment happens when data is loaded):
  equity/<year>.parquet   every EQ/BE/BZ stock, every trading day of that year
  indices/<year>.parquet  every NSE index close (Nifty 500, sector indices, ...)

Years already stored are skipped unless --force is given, so a stopped run can
simply be started again. The current year is saved up to yesterday.
"""
import argparse
import datetime as dt
import sys
import time

import pandas as pd

from pipeline import nse, storage

PAUSE_SECONDS = 0.35


def ist_today() -> dt.date:
    return (dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).date()


def backfill_year(year: int, session) -> dict:
    last_day = min(dt.date(year, 12, 31), ist_today() - dt.timedelta(days=1))
    day = dt.date(year, 1, 1)
    equity_frames, index_frames, missing_index = [], [], []
    while day <= last_day:
        # Weekends are checked too: NSE sometimes holds Saturday or Sunday sessions.
        try:
            equity_frames.append(nse.fetch_bhavcopy(day, session))
            time.sleep(PAUSE_SECONDS)
            try:
                index_frames.append(nse.fetch_indices(day, session))
            except nse.NotTradingDay:
                missing_index.append(str(day))
        except nse.NotTradingDay:
            pass
        time.sleep(PAUSE_SECONDS)
        day += dt.timedelta(days=1)

    sessions = len(equity_frames)
    full_year_done = last_day == dt.date(year, 12, 31)
    if full_year_done and sessions < 230:
        raise RuntimeError(f"{year}: only {sessions} sessions found, expected ~245. "
                           "Not saving a year with gaps.")
    if sessions == 0:
        return {"year": year, "sessions": 0}

    equity = pd.concat(equity_frames, ignore_index=True)
    indices = pd.concat(index_frames, ignore_index=True) if index_frames else None
    size_eq = storage.save_parquet(equity, f"equity/{year}.parquet")
    size_ix = storage.save_parquet(indices, f"indices/{year}.parquet") if indices is not None else 0
    return {"year": year, "sessions": sessions, "equity_rows": len(equity),
            "stocks": int(equity["symbol"].nunique()), "missing_index_days": missing_index,
            "mb": round((size_eq + size_ix) / 1e6, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=ist_today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stored = set(storage.list_files("equity"))
    session = nse.new_session()
    results = []
    for year in range(args.start_year, args.end_year + 1):
        is_current = year == ist_today().year
        if f"{year}.parquet" in stored and not args.force and not is_current:
            print(f"{year}: already stored, skipping")
            continue
        print(f"{year}: downloading ...", flush=True)
        try:
            summary = backfill_year(year, session)
        except nse.BlockedError as error:
            storage.log_run("backfill", "blocked", {"year": year, "error": str(error)})
            print(f"STOPPED: NSE blocked the requests ({error}). Re-run later; "
                  "finished years are kept.")
            sys.exit(1)
        except Exception as error:
            storage.log_run("backfill", "failed", {"year": year, "error": str(error)})
            raise
        print(f"{year}: {summary}", flush=True)
        results.append(summary)
    storage.log_run("backfill", "ok", {"years": results})
    print("Backfill finished.")


if __name__ == "__main__":
    main()
