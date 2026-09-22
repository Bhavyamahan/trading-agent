"""Nightly: add every NSE session since the last stored day to Supabase.

* equity/<year>.parquet and indices/<year>.parquet get the new days appended;
* a missing NIFTY 500 value on a trading day carries forward the previous close;
* NSE's corporate-action list for the last 120 days is merged into
  reference/corporate_actions_nse.parquet (skipped with a warning if NSE blocks it).
Safe to run more than once a day: days already stored are not downloaded again.
"""
import datetime as dt
import sys
import time

import pandas as pd

from pipeline import nse, storage

INDEX = "NIFTY 500"


def ist_now() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)


def last_stored_day() -> dt.date:
    years = sorted(int(n.split(".")[0]) for n in storage.list_files("equity"))
    for year in reversed(years):
        frame = storage.load_parquet(f"equity/{year}.parquet")
        if frame is not None and len(frame):
            return pd.to_datetime(frame["date"]).max().date()
    raise RuntimeError("No stored equity data found. Run the backfill first.")


def merge_year(kind: str, year: int, new: pd.DataFrame, keys: list[str]) -> int:
    old = storage.load_parquet(f"{kind}/{year}.parquet")
    frame = new if old is None else pd.concat([old, new], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    storage.save_parquet(frame, f"{kind}/{year}.parquet")
    return len(frame)


def fetch_new_sessions(start: dt.date, end: dt.date, session, fetch_bhav=None, fetch_idx=None):
    fetch_bhav = fetch_bhav or nse.fetch_bhavcopy
    fetch_idx = fetch_idx or nse.fetch_indices
    equity, indices, days = [], [], []
    day = start
    while day <= end:
        try:
            bhav = fetch_bhav(day, session)
            equity.append(bhav)
            days.append(day)
            time.sleep(0.3)
            try:
                idx = fetch_idx(day, session)
                idx["index_name"] = idx["index_name"].replace({"CNX 500": INDEX, "S&P CNX 500": INDEX})
                indices.append(idx.assign(source="nse_daily"))
            except nse.NotTradingDay:
                pass
        except nse.NotTradingDay:
            pass
        time.sleep(0.3)
        day += dt.timedelta(days=1)
    return equity, indices, days


def fill_index_gaps(indices: pd.DataFrame, days: list[dt.date], previous_close: float | None) -> tuple[pd.DataFrame, list]:
    have = indices[indices["index_name"] == INDEX].set_index("date")["close"] if len(indices) else pd.Series(dtype=float)
    added, carried, last = [], [], previous_close
    for day in sorted(days):
        ts = pd.Timestamp(day)
        if ts in have.index:
            last = float(have[ts])
            continue
        if last is None:
            continue
        added.append({"date": ts, "index_name": INDEX, "open": last, "high": last, "low": last,
                      "close": last, "source": "carried_forward"})
        carried.append(str(day))
    if added:
        indices = pd.concat([indices, pd.DataFrame(added)], ignore_index=True)
    return indices, carried


def previous_index_close(before: dt.date) -> float | None:
    for year in (before.year, before.year - 1):
        frame = storage.load_parquet(f"indices/{year}.parquet")
        if frame is None:
            continue
        n500 = frame[(frame["index_name"].astype(str).str.upper() == INDEX)
                     & (pd.to_datetime(frame["date"]) < pd.Timestamp(before))]
        if len(n500):
            return float(n500.sort_values("date")["close"].iloc[-1])
    return None


def refresh_corporate_actions() -> str:
    try:
        from scripts.fetch_corporate_actions import fetch_range, new_session
        today = ist_now().date()
        data = fetch_range(new_session(), today - dt.timedelta(days=120), today + dt.timedelta(days=30))
        if not data:
            return "NSE corporate-action refresh blocked or empty (fallback detection still active)"
        fresh = pd.DataFrame(data).astype(str)
        old = storage.load_parquet("reference/corporate_actions_nse.parquet")
        merged = fresh if old is None else pd.concat([old.astype(str), fresh], ignore_index=True).drop_duplicates()
        storage.save_parquet(merged, "reference/corporate_actions_nse.parquet")
        return f"corporate actions refreshed ({len(fresh)} recent rows)"
    except Exception as error:
        return f"corporate-action refresh failed: {type(error).__name__}: {error}"


def main() -> None:
    today = ist_now().date()
    last = last_stored_day()
    print(f"Last stored session: {last}; checking up to {today}")
    session = nse.new_session()
    try:
        equity, indices, days = fetch_new_sessions(last + dt.timedelta(days=1), today, session)
    except nse.BlockedError as error:
        storage.log_run("daily_update", "blocked", {"error": str(error)})
        print(f"STOPPED: NSE blocked the download ({error}).")
        sys.exit(1)
    summary = {"new_sessions": [str(d) for d in days]}
    if days:
        eq = pd.concat(equity, ignore_index=True)
        ix = pd.concat(indices, ignore_index=True) if indices else pd.DataFrame(
            columns=["date", "index_name", "open", "high", "low", "close", "source"])
        ix, carried = fill_index_gaps(ix, days, previous_index_close(min(days)))
        summary["index_carried_forward"] = carried
        for year in sorted({d.year for d in days}):
            merge_year("equity", year, eq[pd.to_datetime(eq["date"]).dt.year == year], ["symbol", "date"])
            merge_year("indices", year, ix[pd.to_datetime(ix["date"]).dt.year == year], ["index_name", "date"])
        rows = eq.groupby("date").size().to_dict()
        summary["stocks_per_session"] = {str(pd.Timestamp(k).date()): int(v) for k, v in rows.items()}
        if min(rows.values()) < 1000:
            summary["warning"] = "a session has fewer than 1,000 stocks; check the data"
    summary["corporate_actions"] = refresh_corporate_actions()
    storage.log_run("daily_update", "ok", summary)
    print(summary)


if __name__ == "__main__":
    main()
