"""Download and parse official NSE end-of-day files (bhavcopy + index closes).

Two bhavcopy formats exist:
  * Legacy format (until early July 2024): cm01JAN2024bhav.csv.zip
  * UDiFF format (from 8 July 2024):       BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip
Both are normalised to the same columns so the rest of the system never cares.
"""
import datetime as dt
import io
import time
import zipfile

import pandas as pd
import requests

ARCHIVE = "https://nsearchives.nseindia.com"
UDIFF_START = dt.date(2024, 7, 8)
KEEP_SERIES = {"EQ", "BE", "BZ"}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

EQUITY_COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low",
                  "close", "prev_close", "volume", "value", "trades"]
INDEX_COLUMNS = ["date", "index_name", "open", "high", "low", "close"]


class NotTradingDay(Exception):
    """File does not exist for this date (holiday, weekend or not yet published)."""


class BlockedError(Exception):
    """NSE refused the request (403/401). Stop the run instead of saving gaps."""


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _get(session: requests.Session, url: str, retries: int = 3, timeout: int = 30) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
            continue
        if response.status_code == 200 and response.content:
            return response.content
        if response.status_code == 404:
            raise NotTradingDay(url)
        if response.status_code in (401, 403):
            if attempt == retries:
                raise BlockedError(f"HTTP {response.status_code} for {url}")
            time.sleep(10 * attempt)
            continue
        if attempt == retries:
            response.raise_for_status()
            raise NotTradingDay(url)  # 200 with empty body
        time.sleep(3 * attempt)
    raise NotTradingDay(url)


def _legacy_url(day: dt.date) -> str:
    mon = MONTHS[day.month - 1]
    return (f"{ARCHIVE}/content/historical/EQUITIES/{day.year}/{mon}/"
            f"cm{day.day:02d}{mon}{day.year}bhav.csv.zip")


def _udiff_url(day: dt.date) -> str:
    return f"{ARCHIVE}/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def _read_zipped_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, dtype=str)
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def _to_numbers(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.strip()
                                      .str.replace(",", ""), errors="coerce")
    return frame


def parse_legacy(raw: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": raw["SYMBOL"].str.strip(),
        "series": raw["SERIES"].str.strip(),
        "isin": raw["ISIN"].str.strip(),
        "open": raw["OPEN"], "high": raw["HIGH"], "low": raw["LOW"],
        "close": raw["CLOSE"], "prev_close": raw["PREVCLOSE"],
        "volume": raw["TOTTRDQTY"], "value": raw["TOTTRDVAL"],
        "trades": raw["TOTALTRADES"],
    })
    return _finish(frame)


def parse_udiff(raw: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    raw = raw[raw["FinInstrmTp"].str.strip() == "STK"]
    frame = pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": raw["TckrSymb"].str.strip(),
        "series": raw["SctySrs"].str.strip(),
        "isin": raw["ISIN"].str.strip(),
        "open": raw["OpnPric"], "high": raw["HghPric"], "low": raw["LwPric"],
        "close": raw["ClsPric"], "prev_close": raw["PrvsClsgPric"],
        "volume": raw["TtlTradgVol"], "value": raw["TtlTrfVal"],
        "trades": raw["TtlNbOfTxsExctd"],
    })
    return _finish(frame)


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["series"].isin(KEEP_SERIES)].copy()
    frame = _to_numbers(frame, ["open", "high", "low", "close", "prev_close",
                                "volume", "value", "trades"])
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    return frame[EQUITY_COLUMNS].reset_index(drop=True)


def fetch_bhavcopy(day: dt.date, session: requests.Session) -> pd.DataFrame:
    """Return the day's equity bhavcopy, trying the likely format first."""
    attempts = [(_udiff_url, parse_udiff), (_legacy_url, parse_legacy)]
    if day < UDIFF_START:
        attempts.reverse()
    for build_url, parser in attempts:
        try:
            content = _get(session, build_url(day))
        except NotTradingDay:
            continue
        return parser(_read_zipped_csv(content), day)
    raise NotTradingDay(f"No bhavcopy for {day}")


def fetch_indices(day: dt.date, session: requests.Session) -> pd.DataFrame:
    """Return closing values of every NSE index (Nifty 500, sector indices, ...)."""
    url = f"{ARCHIVE}/content/indices/ind_close_all_{day:%d%m%Y}.csv"
    raw = pd.read_csv(io.BytesIO(_get(session, url)), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    frame = pd.DataFrame({
        "date": pd.Timestamp(day),
        "index_name": raw["Index Name"].str.strip().str.upper(),
        "open": raw["Open Index Value"], "high": raw["High Index Value"],
        "low": raw["Low Index Value"], "close": raw["Closing Index Value"],
    })
    frame = _to_numbers(frame, ["open", "high", "low", "close"])
    return frame.dropna(subset=["close"])[INDEX_COLUMNS].reset_index(drop=True)
