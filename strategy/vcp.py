"""Layer 7: find valid bases (volatility contraction) and turn them into entries.

Swing methods
  bars   (v1.1): a swing high is a high above the 3 highs on each side.
  zigzag (v1.2): a swing is confirmed only after a 3% reversal, so small wiggles
                 inside a pullback are not counted as separate contractions.

Entry modes
  next_open (v1.1): signal on the breakout CLOSE (volume + upper-half checks),
                    buy at the next open.
  buy_stop  (v1.2): each evening a valid, unbroken setup becomes a stop-limit buy
                    order for the next session: triggered at the pivot, never filled
                    above 1.02 x pivot. Gates are checked on the evening the order
                    is placed.
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


def _bar_contractions(high, low, i0, s, k):
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
    pivot = high[swing_highs[-1]:s + 1].max()
    return contractions, lows, pivot


def _zigzag_contractions(high, low, i0, s, pct):
    """Contractions between swing highs and lows that reverse by at least pct."""
    hi_val = high[i0]
    points = []                      # confirmed (swing high, swing low) pairs
    mode, cand_low, cand_high = "down", low[i0], None
    for j in range(i0 + 1, s + 1):
        if mode == "down":
            if high[j] > hi_val:                     # higher high before a real pullback
                hi_val, cand_low = high[j], low[j]
                continue
            cand_low = min(cand_low, low[j])
            if high[j] >= cand_low * (1 + pct):      # 3% bounce confirms the swing low
                points.append((hi_val, cand_low))
                mode, cand_high = "up", high[j]
        else:
            if low[j] < points[-1][1]:               # lower low: the pullback continues
                points[-1] = (points[-1][0], low[j])
                cand_high = high[j]
                continue
            cand_high = max(cand_high, high[j])
            if low[j] <= cand_high * (1 - pct):      # 3% drop confirms a new swing high
                hi_val, cand_low, mode = cand_high, low[j], "down"
    if mode == "down":
        points.append((hi_val, cand_low))            # the pullback still in progress
        pivot = hi_val
    else:
        if cand_high > points[-1][0]:                # already above the last swing high
            return [], [], np.nan
        pivot = points[-1][0]
    contractions = [(h - l) / h for h, l in points]
    return contractions, [l for _, l in points], pivot


def find_setup(high: np.ndarray, low: np.ndarray, avgvol10: float, avgvol50: float,
               s: int, p: Params) -> Setup | None:
    """Check whether a valid base exists as of index s (uses data up to s only)."""
    start = max(0, s - p.base_max_len + 1)
    if s - start + 1 < p.base_min_len:
        return None
    i0 = start + int(np.argmax(high[start:s + 1]))
    if s - i0 < p.base_min_len:
        return None
    h0 = high[i0]
    depth = (h0 - low[i0:s + 1].min()) / h0
    if not (p.base_depth_min <= depth <= p.base_depth_max):
        return None
    if not (avgvol50 > 0 and avgvol10 <= p.dryup_ratio * avgvol50):
        return None
    if p.swing_method == "zigzag":
        contractions, lows, pivot = _zigzag_contractions(high, low, i0, s, p.zigzag_pct)
    else:
        contractions, lows, pivot = _bar_contractions(high, low, i0, s, p.swing_side)
    if len(contractions) < 2:
        return None
    for prev, cur in zip(contractions, contractions[1:]):
        if not (0 < cur <= p.contraction_ratio * prev):
            return None
    if contractions[-1] > p.final_contraction_max or pivot < p.upper_base * h0:
        return None
    return Setup(float(pivot), float(lows[-1]), float(h0), [round(c, 4) for c in contractions])


def _arrays(g):
    return (g["adj_high"].to_numpy(), g["adj_low"].to_numpy(), g["adj_close"].to_numpy(),
            g["adj_volume"].to_numpy(), g["avgvol10"].to_numpy(), g["avgvol50"].to_numpy(),
            (g["layer2"] & g["layer3"]).to_numpy())


def _setup_at(cache, s, high, low, av10, av50, p):
    if s not in cache:
        ok = not (np.isnan(av10[s]) or np.isnan(av50[s]))
        cache[s] = find_setup(high, low, av10[s], av50[s], s, p) if ok else None
        if cache[s] is not None:
            LAST_STATS["setups"] += 1
    return cache[s]


def _row(g, i, setup, s):
    row = g.iloc[i]
    return {"date": row["date"], "symbol": str(row["symbol"]), "pivot": setup.pivot,
            "final_low": setup.final_low, "setup_day": g.iloc[s]["date"],
            "rs_pct": float(row["rs_pct"]), "rs_points": int(row["rs_points"]),
            "industry": str(row["industry"])}


def stock_signals_next_open(g: pd.DataFrame, p: Params) -> list[dict]:
    """v1.1: breakout close on day T -> buy at T+1 open."""
    high, low, close, vol, av10, av50, gates = _arrays(g)
    rng = high - low
    upper = np.where(rng > 0, (close - low) / np.where(rng > 0, rng, 1), 1.0) >= p.upper_half
    volume_ok = np.where(np.isnan(av50), False, vol >= p.breakout_volume * np.nan_to_num(av50))
    candidates = np.where(gates & volume_ok & upper)[0]
    LAST_STATS["candidates"] += len(candidates)
    cache, out = {}, []
    for t in candidates:
        for s in range(t - 1, max(t - p.setup_valid_sessions, 0) - 1, -1):
            setup = _setup_at(cache, s, high, low, av10, av50, p)
            if setup is None:
                continue
            if setup.pivot < close[t] <= p.chase_limit * setup.pivot:
                out.append(_row(g, t, setup, s))
            break
    return out


def stock_orders_buy_stop(g: pd.DataFrame, p: Params) -> list[dict]:
    """v1.2: every evening d with an active setup -> stop-limit buy order for d+1."""
    high, low, close, vol, av10, av50, gates = _arrays(g)
    candidates = np.where(gates)[0]
    LAST_STATS["candidates"] += len(candidates)
    cache, out = {}, []
    for d in candidates:
        for s in range(d, max(d - p.setup_valid_sessions + 1, 0) - 1, -1):
            setup = _setup_at(cache, s, high, low, av10, av50, p)
            if setup is None:
                continue
            broken = s < d and high[s + 1:d + 1].max() > setup.pivot
            planned_stop = setup.final_low * (1 - p.stop_buffer)
            too_wide = (setup.pivot - planned_stop) / setup.pivot > p.stop_max
            if not broken and close[d] >= setup.final_low and not too_wide:
                out.append(_row(g, d, setup, s))
            break  # only the most recent valid setup counts
    return out


def all_signals(f: pd.DataFrame, p: Params) -> pd.DataFrame:
    LAST_STATS.update(candidates=0, setups=0)
    make = stock_orders_buy_stop if p.entry_mode == "buy_stop" else stock_signals_next_open
    rows = []
    for _, g in f.groupby("symbol", sort=False, observed=True):
        if g["layer3"].any():
            rows.extend(make(g.sort_values("date").reset_index(drop=True), p))
    columns = ["date", "symbol", "pivot", "final_low", "setup_day", "rs_pct", "rs_points", "industry"]
    return pd.DataFrame(rows, columns=columns)
