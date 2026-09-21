"""Layer calculations that work on the whole market at once (Layers 1 to 4).

Everything is computed on each stock's own trading days (a suspended day does not
break its moving averages), using split/bonus-adjusted prices.
"""
import numpy as np
import pandas as pd

from strategy.config import Params
from strategy.data import UNCLASSIFIED

CIRCUIT_BANDS = (0.02, 0.05, 0.10, 0.20)


def _within_stock(frame: pd.DataFrame, values: pd.Series, window: int, how: str) -> pd.Series:
    """Rolling stat over the column, blanked where the window would cross stocks."""
    roller = values.rolling(window)
    result = getattr(roller, how)()
    return result.where(frame["pos"] >= window - 1)


def _shift_within(frame: pd.DataFrame, values: pd.Series, n: int) -> pd.Series:
    return values.shift(n).where(frame["pos"] >= n)


FLOAT32 = ["dma10", "dma20", "dma50", "dma150", "dma200", "dma200_prev21", "hi252", "lo252",
           "avgvol10", "avgvol50", "adv_cr", "ret63", "ret126", "ret189", "ret252",
           "close_prev15", "day_ret", "rs_line", "rs_line_hi252", "rs_raw", "nifty"]


def add_stock_indicators(prices: pd.DataFrame, nifty: pd.Series) -> pd.DataFrame:
    f = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    f["pos"] = f.groupby("symbol", observed=True).cumcount().astype("int32")
    close, high, low, vol = f["adj_close"], f["adj_high"], f["adj_low"], f["adj_volume"]
    for n in (10, 20, 50, 150, 200):
        f[f"dma{n}"] = _within_stock(f, close, n, "mean")
    f["dma200_prev21"] = _shift_within(f, f["dma200"], 21)
    f["hi252"] = _within_stock(f, high, 252, "max")
    f["lo252"] = _within_stock(f, low, 252, "min")
    f["avgvol10"] = _within_stock(f, vol, 10, "mean")
    f["avgvol50"] = _within_stock(f, vol, 50, "mean")
    f["adv_cr"] = _within_stock(f, f["value"].fillna(0.0), 50, "mean") / 1e7
    for n in (63, 126, 189, 252):
        f[f"ret{n}"] = close / _shift_within(f, close, n) - 1
    f["close_prev15"] = _shift_within(f, close, 15)
    f["day_ret"] = close / _shift_within(f, close, 1) - 1

    # Upper circuit proxy: closed at the day's high, up by (almost exactly) a band limit.
    change = f["close"] / f["prev_close"] - 1
    near_band = np.zeros(len(f), dtype=bool)
    for band in CIRCUIT_BANDS:
        near_band |= (change - band).abs().to_numpy() <= 0.001
    f["upper_circuit"] = near_band & (f["close"] >= f["high"])

    f["nifty"] = f["date"].map(nifty)
    f["rs_line"] = f["adj_close"] / f["nifty"]
    f["rs_line_hi252"] = _within_stock(f, f["rs_line"], 252, "max")
    f["rs_raw"] = 0.4 * f["ret63"] + 0.2 * f["ret126"] + 0.2 * f["ret189"] + 0.2 * f["ret252"]
    f = f.drop(columns=["high", "prev_close"])
    for column in FLOAT32:
        f[column] = f[column].astype("float32")
    return f


def build_universe(f: pd.DataFrame, p: Params) -> tuple[pd.Series, pd.DataFrame]:
    """Top N by median traded value over the previous 126 sessions, reset each Jan and Jul."""
    wide = (f[["date", "symbol", "value"]].assign(symbol=lambda x: x["symbol"].astype(str))
            .pivot(index="date", columns="symbol", values="value").fillna(0.0).astype("float32"))
    dates = wide.index
    halves = pd.Series(dates.year * 10 + (dates.month > 6), index=dates)
    rebalance = [d for d, prev in zip(dates[1:], halves.shift(1).iloc[1:]) if halves[d] != prev]
    members, rows = {}, []
    for i, day in enumerate(rebalance):
        loc = dates.get_loc(day)
        if loc < p.universe_lookback:
            continue
        median = wide.iloc[loc - p.universe_lookback:loc].median()
        top = median[median > 0].nlargest(p.universe_size)
        end = rebalance[i + 1] if i + 1 < len(rebalance) else dates[-1] + pd.Timedelta(days=1)
        members[(day, end)] = set(top.index)
        rows.append({"rebalance": day.date(), "members": len(top),
                     "median_value_cr_of_500th": round(float(top.iloc[-1]) / 1e7, 2)})
    del wide
    in_universe = np.zeros(len(f), dtype=bool)
    dates_np, symbols = f["date"].to_numpy(), f["symbol"].astype(str).to_numpy()
    for (start, end), names in members.items():
        rows_in = (dates_np >= np.datetime64(start)) & (dates_np < np.datetime64(end))
        idx = np.where(rows_in)[0]
        in_universe[idx[np.isin(symbols[idx], list(names))]] = True
    return pd.Series(in_universe, index=f.index), pd.DataFrame(rows)


def add_layers(f: pd.DataFrame, nifty: pd.Series, industry: dict[str, str],
               p: Params) -> tuple[pd.DataFrame, pd.Series, dict]:
    f["in_universe"], universe_log = build_universe(f, p)

    # Layer 2: tradable today
    f["layer2"] = (f["in_universe"] & (f["series"] == "EQ") & (f["close"] >= p.min_price)
                   & (f["adv_cr"] >= p.min_adv_crore) & (f["pos"] + 1 >= p.min_listing_sessions)
                   & ~f["upper_circuit"])

    # Layer 4 inputs: RS percentile across today's universe
    ranked = f[f["in_universe"] & f["rs_raw"].notna()]
    pct = ranked.groupby("date")["rs_raw"].rank(pct=True) * 100
    f["rs_pct"] = pct.clip(1, 99).reindex(f.index).astype("float32")
    del ranked

    # Layer 3: trend template (T8 uses the RS percentile)
    c = f["adj_close"]
    f["layer3"] = ((c > f["dma150"]) & (c > f["dma200"]) & (f["dma150"] > f["dma200"])
                   & (f["dma200"] > f["dma200_prev21"]) & (f["dma50"] > f["dma150"])
                   & (f["dma50"] > f["dma200"]) & (c > f["dma50"])
                   & (c >= p.above_52w_low * f["lo252"]) & (c >= p.within_52w_high * f["hi252"])
                   & (f["rs_pct"] >= p.rs_gate))

    # Layer 4 points
    nifty_ret63 = nifty / nifty.shift(63) - 1
    f["industry"] = (f["symbol"].astype(str).map(industry).fillna(UNCLASSIFIED)
                     .astype("category"))
    group = _industry_groups(f, p)
    joined = f[["date", "industry"]].astype({"industry": str}).merge(
        group.astype({"industry": str}), on=["date", "industry"], how="left")
    for column in ["group_line", "group_dma50", "group_ret63", "group_ok"]:
        f[column] = joined[column].to_numpy()
    del joined
    f["rs4a"] = f["rs_pct"] >= p.rs_bonus
    f["rs4b"] = f["rs_line"] >= p.rs_line_near_high * f["rs_line_hi252"]
    f["rs4c"] = ((f["group_line"] > f["group_dma50"])
                 & (f["group_ret63"] > f["date"].map(nifty_ret63)) & f["group_ok"].fillna(False))
    f["rs_points"] = f[["rs4a", "rs4b", "rs4c"]].fillna(False).astype(int).sum(axis=1)

    regime = market_regime(f, nifty, p)
    classified = f.loc[f["in_universe"], "industry"] != UNCLASSIFIED
    diagnostics = {"universe_log": universe_log,
                   "industry_coverage_pct": round(float(classified.mean() * 100), 1),
                   "median_universe_adv_cr": round(float(f.loc[f["in_universe"], "adv_cr"].median()), 1)}
    return f, regime, diagnostics


def _industry_groups(f: pd.DataFrame, p: Params) -> pd.DataFrame:
    members = f[f["in_universe"] & (f["industry"] != UNCLASSIFIED) & f["day_ret"].notna()]
    daily = (members.assign(industry=members["industry"].astype(str))
             .groupby(["industry", "date"])["day_ret"].agg(["mean", "count"]).reset_index())
    out = []
    for name, g in daily.groupby("industry"):
        g = g.sort_values("date").copy()
        g["ok"] = g["count"] >= p.industry_min_members
        g["line"] = (1 + g["mean"].where(g["ok"], 0.0)).cumprod()
        g["dma50"] = g["line"].rolling(50).mean()
        g["ret63"] = g["line"] / g["line"].shift(63) - 1
        out.append(g.assign(industry=name))
    if not out:
        return pd.DataFrame(columns=["date", "industry", "group_line", "group_dma50",
                                     "group_ret63", "group_ok"])
    groups = pd.concat(out)
    return groups.rename(columns={"line": "group_line", "dma50": "group_dma50",
                                  "ret63": "group_ret63", "ok": "group_ok"})[
        ["date", "industry", "group_line", "group_dma50", "group_ret63", "group_ok"]]


def market_regime(f: pd.DataFrame, nifty: pd.Series, p: Params) -> pd.Series:
    """ON / CAUTION / OFF per trading day, changing only after 2 confirming closes."""
    above = f[f["in_universe"] & f["dma200"].notna()].assign(
        up=lambda x: x["adj_close"] > x["dma200"]).groupby("date")["up"].mean()
    dma50, dma200 = nifty.rolling(50).mean(), nifty.rolling(200).mean()
    raw = pd.Series("OFF", index=nifty.index)
    caution = nifty > dma200
    on = caution & (dma50 > dma200) & (above.reindex(nifty.index) >= p.breadth_min)
    raw[caution] = "CAUTION"
    raw[on] = "ON"
    raw[dma200.isna()] = "OFF"
    state, pending, count, out = "OFF", None, 0, []
    for value in raw:
        if value == state:
            pending, count = None, 0
        elif value == pending:
            count += 1
        else:
            pending, count = value, 1
        if pending is not None and count >= p.regime_confirm_days:
            state, pending, count = pending, None, 0
        out.append(state)
    return pd.Series(out, index=nifty.index, name="regime")
