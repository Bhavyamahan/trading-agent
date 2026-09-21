"""NSE's official corporate-action list: exact split, bonus and demerger ex-dates.

Sources (merged):
  * Supabase Storage reference/corporate_actions_nse.parquet, written by
    scripts/fetch_corporate_actions.py from NSE's website API.
  * Any CSV in data/corporate_actions/ in the repo: manual downloads from
    nseindia.com > Corporate Actions (columns SYMBOL, PURPOSE, EX-DATE).

Factors (price multiplier for all days before the ex-date):
  Bonus a:b              -> b / (a + b)      e.g. 1:1 -> 0.5, 1:2 -> 0.667, 4:1 -> 0.2
  Face value split X->Y  -> Y / X            e.g. Rs 10 -> Rs 2 -> 0.2
  Consolidation X->Y     -> Y / X            (a reverse split, factor above 1)
  Demerger / spin-off    -> measured from prices on the ex-date (no fixed ratio)
Several actions on one ex-date (e.g. bonus + split) multiply.
Rights issues and dividends are ignored (small effects; the index is price-only).
"""
import glob
import io
import re

import pandas as pd

BONUS = re.compile(r"bonus[^0-9]{0,20}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
FROM_TO = re.compile(r"from\s*(?:rs\.?|re\.?|inr|₹)?\s*([\d.]+)\s*/?-?.{0,40}?\bto\s*(?:rs\.?|re\.?|inr|₹)?\s*([\d.]+)", re.I)
SPLIT_WORDS = re.compile(r"split|sub-?division|consolidat", re.I)
DEMERGER_WORDS = re.compile(r"demerger|de-merger|spin[\s-]?off|scheme of arrangement|capital reduction", re.I)


def parse_purpose(text: str) -> tuple[float | None, str]:
    """Return (factor, kind). factor None with kind 'demerger' = measure from prices."""
    text = str(text or "")
    factor, kinds = 1.0, []
    for a, b in BONUS.findall(text):
        a, b = float(a), float(b)
        if a > 0 and b > 0:
            factor *= b / (a + b)
            kinds.append("bonus")
    if SPLIT_WORDS.search(text):
        match = FROM_TO.search(text)
        if match:
            old, new = float(match.group(1).rstrip(".")), float(match.group(2).rstrip("."))
            if old > 0 and new > 0 and old != new:
                factor *= new / old
                kinds.append("split" if new < old else "consolidation")
    if kinds:
        return factor, "+".join(kinds)
    if DEMERGER_WORDS.search(text):
        return None, "demerger"
    return None, ""


def _normalise(table: pd.DataFrame) -> pd.DataFrame:
    table = table.rename(columns=lambda c: str(c).strip().lower().replace("-", "").replace(" ", ""))
    symbol = table.get("symbol")
    purpose = table.get("purpose", table.get("subject"))
    ex = table.get("exdate")
    if symbol is None or purpose is None or ex is None:
        return pd.DataFrame(columns=["symbol", "ex_date", "purpose"])
    out = pd.DataFrame({"symbol": symbol.astype(str).str.strip(), "purpose": purpose.astype(str),
                        "ex_date": pd.to_datetime(ex.astype(str).str.strip(), format="%d-%b-%Y",
                                                  errors="coerce")})
    missing = out["ex_date"].isna()
    if missing.any():
        out.loc[missing, "ex_date"] = pd.to_datetime(ex[missing], dayfirst=True, errors="coerce")
    return out.dropna(subset=["ex_date"])


def events_from_table(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that change the share count or the business (not dividends)."""
    rows = _normalise(table)
    parsed = rows["purpose"].map(parse_purpose)
    rows["factor"] = [f for f, _ in parsed]
    rows["kind"] = [k for _, k in parsed]
    rows = rows[rows["kind"] != ""]
    return (rows.drop_duplicates(["symbol", "ex_date", "kind"])
            .sort_values(["symbol", "ex_date"]).reset_index(drop=True))


def load_official_events() -> tuple[pd.DataFrame, list[str]]:
    tables, sources = [], []
    try:
        from pipeline import storage
        stored = storage.load_parquet("reference/corporate_actions_nse.parquet")
        if stored is not None and len(stored):
            tables.append(stored)
            sources.append(f"NSE API download ({len(stored):,} rows)")
    except Exception as error:  # storage not configured (tests) or unreachable
        sources.append(f"storage unavailable: {type(error).__name__}")
    for path in sorted(glob.glob("data/corporate_actions/*.csv")):
        with open(path, "rb") as handle:
            table = pd.read_csv(io.BytesIO(handle.read()), dtype=str)
        tables.append(table)
        sources.append(f"{path} ({len(table):,} rows)")
    if not tables:
        return pd.DataFrame(columns=["symbol", "ex_date", "purpose", "factor", "kind"]), sources
    events = pd.concat([events_from_table(t) for t in tables], ignore_index=True)
    return events.drop_duplicates(["symbol", "ex_date", "kind"]).reset_index(drop=True), sources
