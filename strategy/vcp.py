"""Layer 7: find valid bases (volatility contraction) and the breakout day.

A signal on day T means: all gates except the market regime pass on T, and a
setup confirmed on some day S in [T-10, T-1] has its pivot broken on T.
The regime and sizing are applied later by the portfolio simulator.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.config import Params


LAST_STATS = {"candidates": 0, "setups": 0}


@dataclass
class Setup:
    pivot: float
    final_low: float
    base_high: float
    contractions: list


def find_setup(high: np.ndarray, low: np.ndarray, avgvol10: float, avgvol50: float,
               s: int, p: Params) -> Setup | None:
    """Check whether a valid base exists as of index s (uses data up to s only)."""
    start = max(0, s - p.base_max_len + 1)
    if s - start + 1 < p.base_min_len:
        return None
    window_high = high[start:s + 1]
    i0 = start + int(np.argmax(window_high))
    if s - i0 < p.base_min_len:                       # base must be 15+ sessions old
        return None
    h0 = high[i0]
    depth = (h0 - low[i0:s + 1].min()) / h0
    if not (p.base_depth_min <= depth <= p.base_depth_max):
        return None
    if not (avgvol50 > 0 and avgvol10 <= p.dryup_ratio * avgvol50):
        return None

    k = p.swing_side
    swing_highs = [i0]
    for j in range(i0 + 1, s - k + 1):
        if high[j] > high[j - k:j].max() and high[j] > high[j + 1:j + k + 1].max():
            swing_highs.append(j)
    contractions, lows = [], []
    for n, sh in enumerate(swing_highs):
        end = swing_highs[n + 1] if n + 1 < len(swing_highs) else s + 1
        seg_low = low[sh:end].min()
        contractions.append((high[sh] - seg_low) / high[sh])
        lows.append(seg_low)
    if len(contractions) < 2:
        return None
    for prev, cur in zip(contractions, contractions[1:]):
        if not (0 < cur <= p.contraction_ratio * prev):
            return None
    if contractions[-1] > p.final_contraction_max:
        return None
    pivot = high[swing_highs[-1]:s + 1].max()
    if pivot < p.upper_base * h0:
        return None
    return Setup(float(pivot), float(lows[-1]), float(h0), [round(c, 4) for c in contractions])


def stock_signals(g: pd.DataFrame, p: Params) -> list[dict]:
    """All breakout signals for one stock (rows sorted by date)."""
    high, low = g["adj_high"].to_numpy(), g["adj_low"].to_numpy()
    close, vol = g["adj_close"].to_numpy(), g["adj_volume"].to_numpy()
    av10, av50 = g["avgvol10"].to_numpy(), g["avgvol50"].to_numpy()
    gates = (g["layer2"] & g["layer3"]).to_numpy()
    rng = high - low
    upper_half = np.where(rng > 0, (close - low) / np.where(rng > 0, rng, 1), 1.0) >= p.upper_half
    volume_ok = np.where(np.isnan(av50), False, vol >= p.breakout_volume * np.nan_to_num(av50))
    candidates = np.where(gates & volume_ok & upper_half)[0]
    cache, out = {}, []
    LAST_STATS["candidates"] += len(candidates)
    for t in candidates:
        for s in range(t - 1, max(t - p.setup_valid_sessions, 0) - 1, -1):
            if s not in cache:
                cache[s] = (find_setup(high, low, av10[s], av50[s], s, p)
                            if not (np.isnan(av10[s]) or np.isnan(av50[s])) else None)
                if cache[s] is not None:
                    LAST_STATS["setups"] += 1
            setup = cache[s]
            if setup is None:
                continue
            if close[t] > setup.pivot and close[t] <= p.chase_limit * setup.pivot:
                row = g.iloc[t]
                out.append({"date": row["date"], "symbol": str(row["symbol"]),
                            "pivot": setup.pivot, "final_low": setup.final_low,
                            "setup_day": g.iloc[s]["date"], "contractions": setup.contractions,
                            "rs_pct": float(row["rs_pct"]), "rs_points": int(row["rs_points"]),
                            "industry": str(row["industry"])})
            break  # only the most recent valid setup counts
    return out


def all_signals(f: pd.DataFrame, p: Params) -> pd.DataFrame:
    LAST_STATS.update(candidates=0, setups=0)
    rows = []
    for _, g in f.groupby("symbol", sort=False, observed=True):
        if g["layer3"].any():
            rows.extend(stock_signals(g.sort_values("date").reset_index(drop=True), p))
    return pd.DataFrame(rows)
