"""Merge the 2008-2012 NIFTY 500 history into Supabase and fill any remaining gaps.

Safe to run more than once: existing values are never overwritten.
Stops without writing anything if the 2012 overlap check fails.
"""
import datetime as dt
import sys
import time

import pandas as pd

from pipeline import index_repair, nse, storage

HISTORY_CSV = "data/nifty500_2008_2012.csv"


def main() -> None:
    history = index_repair.load_history_csv(HISTORY_CSV)

    stored_2012 = storage.load_parquet("indices/2012.parquet")
    check = index_repair.overlap_check(
        stored_2012.assign(date=pd.to_datetime(stored_2012["date"])), history)
    print(f"Overlap check 2012: {check}")
    if not check["passed"]:
        storage.log_run("import_nifty500", "failed", {"overlap": check})
        print("STOPPED: downloaded history does not match NSE's own files. Nothing written.")
        sys.exit(1)

    session = nse.new_session()

    def fetch_day(day: dt.date):
        try:
            time.sleep(0.5)
            return nse.fetch_indices(day, session)
        except Exception as error:  # missing file or refused: fall back to carry-forward
            print(f"  {day}: index file not available ({type(error).__name__})")
            return None

    years = sorted(int(name.split(".")[0]) for name in storage.list_files("equity"))
    last_close, summary = None, {"overlap": check, "years": {}}
    for year in years:
        equity = storage.load_parquet(f"equity/{year}.parquet")
        trading_days = sorted(pd.to_datetime(equity["date"]).unique())
        indices = storage.load_parquet(f"indices/{year}.parquet")
        repaired, report, last_close = index_repair.repair_year(
            indices, [pd.Timestamp(d) for d in trading_days], history, fetch_day, last_close)
        storage.save_parquet(repaired, f"indices/{year}.parquet")
        n500_days = int((repaired["index_name"] == index_repair.INDEX).sum())
        report.update({"trading_days": len(trading_days), "nifty500_days": n500_days})
        summary["years"][year] = report
        print(f"{year}: {report}", flush=True)
        if n500_days < len(trading_days):
            raise RuntimeError(f"{year}: still missing NIFTY 500 values")

    storage.log_run("import_nifty500", "ok", summary)
    print("Done: every trading day 2008-today now has a NIFTY 500 value.")


if __name__ == "__main__":
    main()
