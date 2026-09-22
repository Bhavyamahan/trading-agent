"""Rulebook v2.0: monthly momentum rotation.

Ranking (last trading day of each month, using that day's close):
  score = average of the cross-sectional z-scores of
          ret(126) / vol(252)   and   ret(252) / vol(252)
  where vol(252) is the annualised standard deviation of daily returns
  (the risk-adjusted method NSE's momentum indices use).
Eligible: passes Layer 2 (v1.2 liquidity) that day.   M2 also needs close > DMA(200).

Portfolio (orders execute at the next session's open):
  * hold 20 stocks; a holding is sold only when it is no longer ranked in the top 40
    among eligible stocks (the buffer cuts needless turnover);
  * empty slots are filled from the top of the ranking, each bought at equity / 20;
  * weights are not re-balanced between months (winners are allowed to grow).
Regime switch (M1, M2): when the market regime turns OFF on any day, sell everything
at the next open; re-enter only at a month-end when the regime is not OFF.

Tax at each financial-year end, current rules: 20% on short-term gains (held under
12 months), 12.5% on long-term gains above Rs 1.25 lakh a year; short-term losses
offset either kind, long-term losses only long-term gains; losses carry forward.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from strategy.backtest import Book, order_cost
from strategy.config import Params


@dataclass(frozen=True)
class MomentumParams:
    holdings: int = 20
    buffer_rank: int = 40
    ltcg_tax: float = 0.125
    ltcg_exemption: float = 125_000.0
    long_term_days: int = 365


@dataclass(frozen=True)
class Variant:
    name: str
    regime_switch: bool
    stock_trend: bool
    description: str
    holdings: int = 20
    buffer_rank: int = 40
    schedule: str = "month_end"      # "month_end" or "mid_month"
    cost_multiplier: float = 1.0


VARIANTS = [
    Variant("M0", False, False, "Base: top 20, sell outside top 40, month-end"),
    Variant("M0-15", False, False, "15 stocks", holdings=15, buffer_rank=30),
    Variant("M0-25", False, False, "25 stocks", holdings=25, buffer_rank=50),
    Variant("M0-30", False, False, "30 stocks", holdings=30, buffer_rank=60),
    Variant("M0-buf30", False, False, "Sell outside top 30", buffer_rank=30),
    Variant("M0-buf60", False, False, "Sell outside top 60", buffer_rank=60),
    Variant("M0-mid", False, False, "Rebalance mid-month", schedule="mid_month"),
    Variant("M0-2xcost", False, False, "Double slippage and brokerage", cost_multiplier=2.0),
    Variant("M1", True, False, "Reference: cash when regime OFF (failed v2.0 test)"),
]


def mid_month_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day on or before the 15th of each month."""
    s = pd.Series(calendar, index=calendar)
    early = s[s.dt.day <= 15]
    picks = early.groupby(early.index.to_period("M")).max()
    return pd.DatetimeIndex(picks.to_numpy())[:-1] if len(picks) else pd.DatetimeIndex([])


def rebalance_dates(calendar: pd.DatetimeIndex, schedule: str) -> pd.DatetimeIndex:
    return mid_month_dates(calendar) if schedule == "mid_month" else month_end_dates(calendar)


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    months = pd.Series(calendar.to_period("M"), index=calendar)
    return calendar[(months != months.shift(-1)).to_numpy()][:-1]  # the last partial month has no next open


def add_volatility(f: pd.DataFrame) -> pd.DataFrame:
    returns = f["day_ret"].astype("float64")
    vol = returns.rolling(252).std() * math.sqrt(252)
    f["vol252"] = vol.where(f["pos"] >= 252).astype("float32")
    return f


def rank_table(f: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    snap = f[f["date"].isin(dates) & f["layer2"]].copy()
    snap = snap[snap["vol252"].notna() & (snap["vol252"] > 0) & snap["ret252"].notna()]
    snap["m6"] = snap["ret126"] / snap["vol252"]
    snap["m12"] = snap["ret252"] / snap["vol252"]

    def zscore(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd and sd > 0 else x * 0

    snap["score"] = (snap.groupby("date")["m6"].transform(zscore)
                     + snap.groupby("date")["m12"].transform(zscore)) / 2
    snap["above_dma200"] = snap["adj_close"] > snap["dma200"]
    return snap[["date", "symbol", "score", "above_dma200", "industry"]].assign(
        symbol=lambda x: x["symbol"].astype(str)).sort_values(["date", "score"], ascending=[True, False])


def simulate_momentum(ranks: pd.DataFrame, book: Book, calendar: pd.DatetimeIndex, regime: pd.Series,
                      variant: Variant, p: Params, mp: MomentumParams | None = None) -> dict:
    from dataclasses import replace as _replace
    if mp is None:
        mp = MomentumParams(holdings=variant.holdings, buffer_rank=variant.buffer_rank)
    if variant.cost_multiplier != 1.0:
        m = variant.cost_multiplier
        p = _replace(p, slippage=p.slippage * m, brokerage_per_order=p.brokerage_per_order * m)
    cash = p.starting_capital
    holdings: dict[str, dict] = {}
    pending_sells: dict[str, str] = {}
    pending_buys: list[tuple[str, float]] = []
    trades, rows = [], []
    fy_st, fy_lt, carry_st, carry_lt, taxes_paid = 0.0, 0.0, 0.0, 0.0, 0.0
    bought_value = 0.0
    rebalance_days = set(rebalance_dates(calendar, variant.schedule))
    by_date = {d: g for d, g in ranks.groupby("date")}
    in_market = True

    def fy(day):
        return day.year if day.month >= 4 else day.year - 1

    def price_on(sym, d64, field):
        i = book.index_on(sym, d64)
        return None if i is None else book.data[sym][field][i]

    def sell(sym, price, reason, day):
        nonlocal cash, fy_st, fy_lt
        h = holdings.pop(sym)
        value = h["shares"] * price
        proceeds = value - order_cost(value, "sell", p)
        gain = proceeds - h["cost"]
        days_held = (day - h["entry_date"]).days
        if days_held >= mp.long_term_days:
            fy_lt += gain
        else:
            fy_st += gain
        cash += proceeds
        trades.append({"variant": variant.name, "symbol": sym, "entry_date": h["entry_date"].date(),
                       "exit_date": day.date(), "days_held": days_held,
                       "return_pct": round((proceeds / h["cost"] - 1) * 100, 2),
                       "net_pnl": round(gain, 2), "exit_reason": reason})

    def charge_tax():
        nonlocal cash, fy_st, fy_lt, carry_st, carry_lt, taxes_paid
        st, lt = fy_st + carry_st, fy_lt + carry_lt
        carry_st = carry_lt = 0.0
        if st < 0 < lt:
            lt, st = lt + st, 0.0
            if lt < 0:
                st, lt = lt, 0.0
        if st < 0:
            carry_st = st
        if lt < 0:
            carry_lt = lt
        tax = p.stcg_tax * max(st, 0.0) + mp.ltcg_tax * max(lt - mp.ltcg_exemption, 0.0)
        cash -= tax
        taxes_paid += tax
        fy_st = fy_lt = 0.0

    prev_day = None
    for k, day in enumerate(calendar):
        d64 = np.datetime64(day, "ns")
        if prev_day is not None and fy(day) != fy(prev_day):
            charge_tax()
        state = regime.get(day, "OFF")

        # Open: sells first, then buys
        for sym, reason in list(pending_sells.items()):
            o = price_on(sym, d64, "adj_open")
            if o is not None and sym in holdings:
                sell(sym, o * (1 - p.slippage), reason, day)
                del pending_sells[sym]
        still_pending = []
        for sym, target in pending_buys:
            o = price_on(sym, d64, "adj_open")
            if o is None:
                still_pending.append((sym, target))
                continue
            price = o * (1 + p.slippage)
            budget = min(target, cash)
            shares = math.floor(budget / price)
            while shares > 0 and shares * price + order_cost(shares * price, "buy", p) > cash:
                shares -= max(1, shares // 50)
            if shares >= 1:
                value = shares * price
                cost = value + order_cost(value, "buy", p)
                cash -= cost
                bought_value += value
                holdings[sym] = {"shares": shares, "cost": cost, "entry_date": day, "last": o}
        pending_buys = still_pending if k < len(calendar) - 1 else []

        # Close: mark to market, handle stocks whose data ends
        for sym in list(holdings):
            i = book.index_on(sym, d64)
            if i is None:
                continue
            h = holdings[sym]
            h["last"] = book.data[sym]["adj_close"][i]
            if i == len(book.data[sym]["dates"]) - 1 and k < len(calendar) - 1:
                sell(sym, h["last"] * (1 - p.slippage), "data_ended", day)
                pending_sells.pop(sym, None)
        invested = sum(h["shares"] * h["last"] for h in holdings.values())
        equity = cash + invested
        rows.append({"date": day, "equity": equity, "pre_tax_equity": equity + taxes_paid,
                     "invested": invested, "positions": len(holdings), "regime": state})
        prev_day = day
        if k == len(calendar) - 1:
            continue

        # Regime switch: exit everything when the regime turns OFF
        if variant.regime_switch and state == "OFF":
            for sym in holdings:
                pending_sells[sym] = "regime_off"
            pending_buys = []
            in_market = False
        elif variant.regime_switch and not in_market and day in rebalance_days:
            in_market = True

        # Month-end rebalance
        if day in rebalance_days and in_market:
            table = by_date.get(day)
            if table is None:
                continue
            if variant.stock_trend:
                table = table[table["above_dma200"]]
            order = list(table["symbol"])
            keep_zone = set(order[:mp.buffer_rank])
            for sym in holdings:
                if sym not in keep_zone and sym not in pending_sells:
                    pending_sells[sym] = "dropped_out_of_buffer"
            staying = [s for s in holdings if s not in pending_sells]
            slots = mp.holdings - len(staying)
            target = equity / mp.holdings
            pending_buys = []
            for sym in order:
                if slots <= 0:
                    break
                if sym in holdings or sym in pending_sells:
                    continue
                pending_buys.append((sym, target))
                slots -= 1

    last = calendar[-1]
    for sym in list(holdings):
        sell(sym, holdings[sym]["last"] * (1 - p.slippage), "end_of_test", last)
    charge_tax()
    curve = pd.DataFrame(rows).set_index("date")
    if len(curve):
        curve.iloc[-1, curve.columns.get_loc("equity")] = cash
        curve.iloc[-1, curve.columns.get_loc("pre_tax_equity")] = cash + taxes_paid
    years = max((calendar[-1] - calendar[0]).days / 365.25, 1e-9)
    turnover = bought_value / max(curve["equity"].mean(), 1) / years if len(curve) else 0.0
    return {"trades": pd.DataFrame(trades), "equity": curve, "taxes_paid": taxes_paid,
            "turnover_per_year": turnover}
