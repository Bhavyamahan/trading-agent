"""Load everything the backtest needs into memory.

prices : long table, one row per stock per trading day, adjusted for splits/bonuses
nifty  : NIFTY 500 close per trading day
industry: symbol -> NSE industry label ("UNCLASSIFIED" when unknown)
"""
import io
import os
import re

import pandas as pd
import requests

from pipeline import adjust, corporate_actions, nse, storage

INDUSTRY_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
]
LOCAL_INDUSTRY_CSV = "data/industry.csv"  # optional manual override: Symbol,Industry
UNCLASSIFIED = "UNCLASSIFIED"
LAST_EVENTS = {"events": None, "official_audit": [], "official_sources": [], "excluded_funds": []}


def fund_symbols(frame: pd.DataFrame) -> set[str]:
    """Symbols that are funds, not companies: ISIN starting INF, or a fund-like name."""
    symbols = frame["symbol"].astype(str)
    if "isin" in frame.columns:
        isin = frame["isin"].astype(str)
        by_isin = set(symbols[isin.str.startswith("INF")])
        with_isin = set(symbols[isin.str.startswith("IN")])
    else:
        by_isin, with_isin = set(), set()
    no_isin = set(symbols.unique()) - with_isin
    by_name = {s for s in no_isin if FUND_NAME.search(s)}
    return by_isin | by_name
PRICE_COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
                 "prev_close", "volume", "value"]
# Funds that trade like shares (ETFs, liquid/gold/index funds) are not companies.
# Their ISIN starts with INF (company shares start with INE). For old rows without
# an ISIN, names are checked as a backup.
FUND_NAME = re.compile(r"BEES$|ETF|^LIQUID|GOLDSHARE|^SETF|^NIFTY|SENSEX|^M50$|^M100$|^MON100$|"
                       r"^MAFANG$|^HNGSNGBEES|GILT|^CPSE|^BHARAT22|^N100$|^INFRABEES", re.I)


def load_prices(years: list[int] | None = None) -> pd.DataFrame:
    if years is None:
        years = sorted(int(n.split(".")[0]) for n in storage.list_files("equity"))
    frames = []
    for year in years:
        frame = storage.load_parquet(f"equity/{year}.parquet")
        if frame is not None:
            columns = PRICE_COLUMNS + (["isin"] if "isin" in frame.columns else [])
            frame = frame[columns]
            frame = frame[frame["series"].isin(["EQ", "BE", "BZ"])].copy()
            if "isin" in frame.columns:
                frame["isin"] = frame["isin"].astype("category")
            frame["symbol"] = frame["symbol"].astype("category")
            frame["series"] = frame["series"].astype("category")
            frames.append(frame)
            print(f"  loaded equity {year}: {len(frame):,} rows", flush=True)
    # Give every year the same category list so concatenation stays compact.
    for fr in frames:
        if "isin" not in fr.columns:
            fr["isin"] = pd.Categorical([None] * len(fr))
    for column in ["symbol", "series", "isin"]:
        union = sorted(set().union(*[set(fr[column].cat.categories) for fr in frames]))
        for fr in frames:
            fr[column] = fr[column].cat.set_categories(union)
    prices = pd.concat(frames, ignore_index=True)
    funds = fund_symbols(prices)
    LAST_EVENTS["excluded_funds"] = sorted(funds)
    prices = prices[~prices["symbol"].astype(str).isin(funds)].drop(columns="isin")
    prices["symbol"] = prices["symbol"].cat.remove_unused_categories()
    print(f"  excluded {len(funds)} ETFs/funds (ISIN INF or fund-like name)", flush=True)
    official, sources = corporate_actions.load_official_events()
    LAST_EVENTS["official_sources"] = sources
    print(f"  official corporate actions: {len(official):,} events from {sources}", flush=True)
    return prepare_prices(prices, official)


def prepare_prices(raw: pd.DataFrame, official: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = raw.drop(columns=["isin"], errors="ignore").copy()
    raw["date"] = pd.to_datetime(raw["date"]).astype("datetime64[ns]")
    # A symbol can appear twice on one day only through data errors; keep EQ first.
    raw["series_rank"] = raw["series"].astype(str).map({"EQ": 0, "BE": 1, "BZ": 2}).fillna(3)
    raw = (raw.sort_values(["symbol", "date", "series_rank"])
           .drop_duplicates(["symbol", "date"]).drop(columns="series_rank"))
    frame = adjust.add_adjusted_prices(raw, official)
    LAST_EVENTS["official_audit"] = frame.attrs.get("official_audit", [])
    flagged = frame["ca_event"] | frame["unexplained_gap"]
    LAST_EVENTS["events"] = frame.loc[flagged, ["symbol", "date", "close", "prev_close", "ca_method",
                                                "ca_factor", "unexplained_gap"]].astype(
        {"symbol": str}).reset_index(drop=True)
    frame = frame.drop(columns=["open", "low", "volume", "adj_factor", "ca_event", "ca_method",
                                "ca_factor", "unexplained_gap"])
    for column in ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "value",
                   "high", "close", "prev_close"]:
        frame[column] = frame[column].astype("float64")
    frame["symbol"] = frame["symbol"].astype("category")
    frame["series"] = frame["series"].astype("category")
    return frame.reset_index(drop=True)


def load_index(name: str, years: list[int] | None = None) -> pd.Series | None:
    """Close series for any NSE index stored in indices/<year>.parquet (None if absent)."""
    if years is None:
        years = sorted(int(n.split(".")[0]) for n in storage.list_files("indices"))
    frames = []
    for year in years:
        frame = storage.load_parquet(f"indices/{year}.parquet")
        if frame is not None:
            frame = frame[frame["index_name"].astype(str).str.upper() == name.upper()]
            if len(frame):
                frames.append(frame[["date", "close"]])
    if not frames:
        return None
    series = pd.concat(frames)
    series["date"] = pd.to_datetime(series["date"]).astype("datetime64[ns]")
    return series.drop_duplicates("date").set_index("date")["close"].sort_index().astype(float)


def load_nifty(years: list[int] | None = None) -> pd.Series:
    if years is None:
        years = sorted(int(n.split(".")[0]) for n in storage.list_files("indices"))
    frames = []
    for year in years:
        frame = storage.load_parquet(f"indices/{year}.parquet")
        if frame is not None:
            frame = frame[frame["index_name"].astype(str).str.upper() == "NIFTY 500"]
            frames.append(frame[["date", "close"]])
    series = pd.concat(frames)
    series["date"] = pd.to_datetime(series["date"]).astype("datetime64[ns]")
    return series.drop_duplicates("date").set_index("date")["close"].sort_index().astype(float)


def _parse_industry_csv(content: bytes) -> dict[str, str]:
    table = pd.read_csv(io.BytesIO(content), dtype=str)
    table.columns = [c.strip().lower() for c in table.columns]
    if "symbol" not in table.columns or "industry" not in table.columns:
        return {}
    table = table.dropna(subset=["symbol", "industry"])
    return dict(zip(table["symbol"].str.strip(), table["industry"].str.strip().str.upper()))


def load_industry() -> tuple[dict[str, str], list[str]]:
    """Merge every industry list we can reach. Returns (mapping, sources used)."""
    mapping, used = {}, []
    session = nse.new_session()
    for url in INDUSTRY_URLS:
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200 and response.content:
                found = _parse_industry_csv(response.content)
                if found:
                    for symbol, label in found.items():
                        mapping.setdefault(symbol, label)
                    used.append(f"{url} ({len(found)})")
        except requests.RequestException:
            continue
    if os.path.exists(LOCAL_INDUSTRY_CSV):
        with open(LOCAL_INDUSTRY_CSV, "rb") as handle:
            found = _parse_industry_csv(handle.read())
        mapping.update(found)
        used.append(f"{LOCAL_INDUSTRY_CSV} ({len(found)})")
    return mapping, used
