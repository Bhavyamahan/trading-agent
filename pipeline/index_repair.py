"""Fill gaps in the NIFTY 500 index series so every stock-trading day has a value.

Order of preference for a trading day with no NIFTY 500 value:
  1. NSE's daily index file we already stored            (source = "nse_daily")
  2. The niftyindices.com history in data/*.csv (2008-12) (source = "niftyindices")
  3. Re-downloading NSE's daily index file for that day  (source = "nse_daily")
  4. Carrying forward the previous close                 (source = "carried_forward")
"""
import datetime as dt
from typing import Callable

import pandas as pd

INDEX = "NIFTY 500"
NSE_FILES_START = pd.Timestamp("2012-02-17")
MAX_OVERLAP_DIFF_PCT = 0.1

# Before NSE's 2015 rebranding, several indices had older names in the daily files.
NAME_ALIASES = {
    "CNX 500": "NIFTY 500", "S&P CNX 500": "NIFTY 500",
    "CNX NIFTY": "NIFTY 50", "S&P CNX NIFTY": "NIFTY 50",
}


def normalise(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Current index names and one date format, whatever year the file is from."""
    if frame is None:
        return None
    frame = frame.copy()
    frame["index_name"] = (frame["index_name"].astype(str).str.strip().str.upper()
                           .replace(NAME_ALIASES))
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame


def names_like(frame: pd.DataFrame | None, text: str = "500") -> list[str]:
    if frame is None or "index_name" not in frame:
        return []
    names = frame["index_name"].astype(str).unique()
    return sorted(n for n in names if text in n)[:20]


def load_history_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    prices = raw[["open", "high", "low", "close"]].astype(float)
    frame = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "index_name": INDEX,
        "open": prices["open"],
        # Fix rows where NSE's file had high/low swapped or close outside the range.
        "high": prices.max(axis=1),
        "low": prices.min(axis=1),
        "close": prices["close"],
        "source": "niftyindices",
    })
    return frame.sort_values("date").reset_index(drop=True)


def overlap_check(stored: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Compare closes on dates present in both sources."""
    stored = normalise(stored)
    ours = stored[stored["index_name"] == INDEX][["date", "close"]]
    both = ours.merge(history[["date", "close"]], on="date", suffixes=("_nse", "_hist"))
    if both.empty:
        return {"days": 0, "passed": False, "reason": "no overlapping dates",
                "index_names_with_500": names_like(stored)}
    diff_pct = ((both["close_nse"] - both["close_hist"]).abs() / both["close_hist"] * 100)
    return {"days": int(len(both)), "max_diff_pct": round(float(diff_pct.max()), 4),
            "mean_diff_pct": round(float(diff_pct.mean()), 5),
            "passed": bool(diff_pct.max() <= MAX_OVERLAP_DIFF_PCT)}


def repair_year(indices: pd.DataFrame | None, trading_days: list[pd.Timestamp],
                history: pd.DataFrame, fetch_day: Callable[[dt.date], pd.DataFrame | None],
                last_close: float | None) -> tuple[pd.DataFrame, dict, float | None]:
    frame = (normalise(indices) if indices is not None and len(indices)
             else pd.DataFrame(columns=["date", "index_name", "open", "high", "low", "close"]))
    if "source" not in frame.columns:
        frame["source"] = "nse_daily"
    frame["source"] = frame["source"].fillna("nse_daily")
    frame["date"] = pd.to_datetime(frame["date"])

    have = dict(zip(frame.loc[frame["index_name"] == INDEX, "date"],
                    frame.loc[frame["index_name"] == INDEX, "close"]))
    hist = history.set_index("date")
    added, report = [], {"from_history": 0, "fetched": [], "carried_forward": []}

    for day in sorted(trading_days):
        if day in have:
            last_close = float(have[day])
            continue
        if day in hist.index:
            row = hist.loc[day]
            added.append(pd.DataFrame([{"date": day, "index_name": INDEX, "open": row["open"],
                                        "high": row["high"], "low": row["low"],
                                        "close": row["close"], "source": "niftyindices"}]))
            report["from_history"] += 1
            last_close = float(row["close"])
            continue
        fetched = fetch_day(day.date()) if day >= NSE_FILES_START else None
        fetched = normalise(fetched) if fetched is not None and len(fetched) else None
        if fetched is not None and (fetched["index_name"] == INDEX).any():
            fetched = fetched.assign(source="nse_daily", date=pd.Timestamp(day))
            added.append(fetched)
            report["fetched"].append(str(day.date()))
            last_close = float(fetched.loc[fetched["index_name"] == INDEX, "close"].iloc[0])
            continue
        if last_close is None:
            raise RuntimeError(f"No NIFTY 500 value available on or before {day.date()}")
        added.append(pd.DataFrame([{"date": day, "index_name": INDEX, "open": last_close,
                                    "high": last_close, "low": last_close, "close": last_close,
                                    "source": "carried_forward"}]))
        report["carried_forward"].append(str(day.date()))

    if added:
        frame = pd.concat([frame] + added, ignore_index=True)
    frame = (frame.drop_duplicates(subset=["date", "index_name"], keep="first")
             .sort_values(["date", "index_name"]).reset_index(drop=True))
    return frame, report, last_close
