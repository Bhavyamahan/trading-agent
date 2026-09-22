"""'Why this stock': a plain explanation for each ranked or held stock.

Everything here is INFORMATION. Trades use only the momentum rank (Rulebook v2.0);
the 7-layer checklist from the retired breakout rulebook is shown for context and
never changes what is bought or sold.
"""
import numpy as np
import pandas as pd

from strategy import vcp
from strategy.config import Params

CUT_OFF = 40
HOLDINGS = 20


def _f(x, digits=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), digits)


def rank_history(month_end_tables: dict[pd.Timestamp, pd.DataFrame], today_table: pd.DataFrame,
                 symbols: set[str], months: int = 6) -> dict[str, list[dict]]:
    history = {s: [] for s in symbols}
    dates = sorted(month_end_tables)[-months:]
    for d in dates:
        ranks = dict(zip(month_end_tables[d]["symbol"], range(1, len(month_end_tables[d]) + 1)))
        for s in symbols:
            history[s].append({"date": str(d.date()), "rank": ranks.get(s)})
    ranks = dict(zip(today_table["symbol"], today_table["rank"]))
    for s in symbols:
        history[s].append({"date": "today", "rank": ranks.get(s)})
    return history


def sell_line_note(rank: int | None, held: bool) -> str:
    if rank is None:
        return ("Not eligible or not ranked today. If this is still true at the month-end check, "
                "it would be sold." if held else "Not eligible or not ranked today.")
    if rank <= HOLDINGS:
        base = f"Rank {rank}: inside the top {HOLDINGS}"
    elif rank <= CUT_OFF:
        base = f"Rank {rank}: outside the top {HOLDINGS} but inside the top {CUT_OFF}"
    else:
        base = f"Rank {rank}: outside the top {CUT_OFF}"
    if held:
        if rank <= 30:
            return base + ". Safe: it is kept at the month-end check."
        if rank <= CUT_OFF:
            return base + f". Close to the sell line: {CUT_OFF - rank + 1} more places down and it is sold at the month-end."
        return base + ". It would be sold at the month-end check if it stays there."
    if rank <= HOLDINGS:
        return base + ". It would be bought at the month-end if a slot is free."
    return base + ". Not a buy candidate right now."


def layer_checklist(row: pd.Series, regime_state: str, setup) -> list[dict]:
    """The 7 layers of the retired breakout rulebook, evaluated on today's data."""
    c = row["adj_close"]
    tt = {
        "Price above the 150- and 200-day averages": bool(c > row["dma150"] and c > row["dma200"]),
        "150-day average above the 200-day": bool(row["dma150"] > row["dma200"]),
        "200-day average rising for a month": bool(row["dma200"] > row["dma200_prev21"]),
        "50-day average above the 150- and 200-day": bool(row["dma50"] > row["dma150"] and row["dma50"] > row["dma200"]),
        "Price above the 50-day average": bool(c > row["dma50"]),
        "At least 30% above the 52-week low": bool(c >= 1.30 * row["lo252"]),
        "Within 25% of the 52-week high": bool(c >= 0.75 * row["hi252"]),
        "Relative-strength percentile 70 or more": bool(row["rs_pct"] >= 70),
    }
    rs = {
        "Relative-strength percentile 85 or more": bool(row["rs4a"]),
        "Relative-strength line near its 1-year high": bool(row["rs4b"]),
        "Its industry is outperforming the market": bool(row["rs4c"]),
    }
    return [
        {"layer": 1, "name": "Market regime", "status": "pass" if regime_state == "ON" else "partial" if regime_state == "CAUTION" else "fail",
         "detail": f"Market regime is {regime_state} (NIFTY 500 vs its 50- and 200-day averages and breadth)."},
        {"layer": 2, "name": "Universe and liquidity", "status": "pass" if bool(row["layer2"]) else "fail",
         "detail": "Among the 500 most-traded stocks, liquid enough, price and listing age OK."},
        {"layer": 3, "name": "Trend template", "status": "pass" if all(tt.values()) else "fail",
         "detail": f"{sum(tt.values())} of 8 trend conditions met.", "checks": tt},
        {"layer": 4, "name": "Relative strength", "status": "pass" if int(row["rs_points"]) >= 2 else "partial" if int(row["rs_points"]) == 1 else "fail",
         "detail": f"{int(row['rs_points'])} of 3 relative-strength points.", "checks": rs},
        {"layer": 5, "name": "Fundamentals", "status": "no_data",
         "detail": "Needs quarterly results data, which is not connected yet."},
        {"layer": 6, "name": "Earnings catalyst", "status": "no_data",
         "detail": "Needs results announcement data, which is not connected yet."},
        {"layer": 7, "name": "Base / setup (VCP)", "status": "pass" if setup else "fail",
         "detail": (f"A tightening base with pivot {setup.pivot:,.2f} and contractions "
                    f"{', '.join(f'{c * 100:.0f}%' for c in setup.contractions)}." if setup
                    else "No tightening base right now (common for stocks already moving strongly).")},
    ]


def explain_stock(g: pd.DataFrame, rank_row: dict | None, regime_state: str, p: Params,
                  history: list[dict], held: bool) -> dict:
    """g = one stock's rows sorted by date (full feature frame); explanation for its last row."""
    g = g.reset_index(drop=True)
    row = g.iloc[-1]
    s = len(g) - 1
    setup = None
    if not (np.isnan(row["avgvol10"]) or np.isnan(row["avgvol50"])):
        setup = vcp.find_setup(g["adj_high"].to_numpy(), g["adj_low"].to_numpy(),
                               float(row["avgvol10"]), float(row["avgvol50"]), s, p)
    vol = row.get("vol252", np.nan)
    rank = int(rank_row["rank"]) if rank_row else None
    eligibility = {
        "Among the 500 most-traded NSE stocks": bool(row["in_universe"]),
        "Normal equity series (EQ)": str(row["series"]) == "EQ",
        f"Share price at least Rs {p.min_price:.0f}": bool(row["close"] >= p.min_price),
        f"Listed for {p.min_listing_sessions} sessions or more": bool(row["pos"] + 1 >= p.min_listing_sessions),
        f"Average daily trading at least Rs {p.min_adv_crore:g} crore": bool(row["adv_cr"] >= p.min_adv_crore),
        "Not stuck at the upper circuit today": not bool(row["upper_circuit"]),
    }
    checklist = layer_checklist(row, regime_state, setup)
    checkable = [c for c in checklist if c["status"] != "no_data"]
    ret6, ret12 = _f(row["ret126"]), _f(row["ret252"])
    why = None
    if rank_row and ret6 is not None and ret12 is not None:
        why = (f"Up {ret6 * 100:.0f}% in 6 months and {ret12 * 100:.0f}% in 12 months, with "
               f"{'low' if vol and vol < 0.35 else 'moderate' if vol and vol < 0.55 else 'high'} volatility "
               f"({vol * 100:.0f}% a year). Its risk-adjusted momentum score of {rank_row['score']:.2f} "
               f"ranks {rank} among {rank_row['of']} eligible stocks.")
    return {
        "symbol": str(row["symbol"]), "date": str(pd.Timestamp(row["date"]).date()), "industry": str(row["industry"]),
        "held": held, "rank": rank, "score": _f(rank_row["score"]) if rank_row else None,
        "ret_6m": ret6, "ret_12m": ret12, "volatility": _f(vol),
        "why": why, "sell_line": sell_line_note(rank, held), "rank_history": history,
        "eligibility": eligibility, "eligible": all(eligibility.values()),
        "layers": checklist, "layers_passed": sum(c["status"] == "pass" for c in checkable),
        "layers_checkable": len(checkable),
        "context": {"close": _f(row["adj_close"], 2),
                    "from_52w_high": _f(row["adj_close"] / row["hi252"] - 1) if row["hi252"] else None,
                    "above_200_day_average": bool(row["adj_close"] > row["dma200"]) if not np.isnan(row["dma200"]) else None},
    }


def build_all(f: pd.DataFrame, today_table: pd.DataFrame, month_end_tables: dict, regime_state: str,
              p: Params, held: set[str], extra: set[str], top_n: int = 60) -> dict[str, dict]:
    symbols = set(today_table["symbol"].head(top_n)) | held | extra
    history = rank_history(month_end_tables, today_table, symbols)
    n = len(today_table)
    rows = {r["symbol"]: {"rank": int(r["rank"]), "score": float(r["score"]), "of": n}
            for r in today_table.to_dict("records")}
    out = {}
    subset = f[f["symbol"].astype(str).isin(symbols)]
    for sym, g in subset.groupby(subset["symbol"].astype(str)):
        out[sym] = explain_stock(g.sort_values("date"), rows.get(sym), regime_state, p, history[sym], sym in held)
    return out
