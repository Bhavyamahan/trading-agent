"""Day-by-day portfolio simulation of Rulebook v1.1 (price-based layers).

Timeline for each trading day d:
  1. Open:  pending exits sell at the open; pending entries buy at the open
            (skipped if the open is more than 5% above the pivot).
  2. Day:   stop-loss (E1) and partial-profit target (E3) are checked against
            the day's low / high.
  3. Close: positions marked to market; close-based exits (E2, E4-E7, E9 and the
            trailing rules) are queued for the next open; new signals are sized
            and queued, subject to the portfolio limits.
Indian income tax on short-term gains is charged at each financial-year end.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from strategy.config import Params, Run
from strategy.data import UNCLASSIFIED


def order_cost(value: float, side: str, p: Params) -> float:
    brokerage = p.brokerage_per_order
    exchange = value * p.exchange_fee
    sebi = value * p.sebi_fee
    cost = brokerage + exchange + sebi + p.gst * (brokerage + exchange + sebi) + value * p.stt
    if side == "buy":
        cost += value * p.stamp_duty_buy
    else:
        cost += p.dp_charge_sell
    return cost


@dataclass
class Position:
    symbol: str
    industry: str
    idx: int
    entry_date: pd.Timestamp
    entry_idx: int
    entry: float
    stop: float
    initial_stop: float
    shares: int
    shares_initial: int
    pivot: float
    rs_points: int
    rs_pct: float
    risk_pct: float
    cash_out: float                      # purchase value + buy costs
    cash_in: float = 0.0                 # sale proceeds - sell costs
    partial_done: bool = False
    climax_done: bool = False
    fast: bool = False
    regime_trail: bool = False
    pending_exit: tuple | None = None    # (shares, reason)
    last_close: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def r_per_share(self) -> float:
        return self.entry - self.initial_stop


class Book:
    """Per-symbol price arrays for fast daily lookups."""

    def __init__(self, f: pd.DataFrame, symbols):
        self.data = {}
        cols = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
                "dma10", "dma20", "dma50", "avgvol50", "close_prev15"]
        subset = f[f["symbol"].astype(str).isin(set(symbols))]
        for symbol, g in subset.groupby(subset["symbol"].astype(str)):
            g = g.sort_values("date")
            arrays = {c: g[c].to_numpy(dtype=float) for c in cols}
            arrays["dates"] = g["date"].to_numpy(dtype="datetime64[ns]")
            self.data[symbol] = arrays

    def index_on(self, symbol: str, day: np.datetime64) -> int | None:
        dates = self.data[symbol]["dates"]
        i = int(np.searchsorted(dates, day))
        return i if i < len(dates) and dates[i] == day else None


def simulate(signals: pd.DataFrame, book: Book, calendar: pd.DatetimeIndex,
             regime: pd.Series, run: Run, p: Params) -> dict:
    cash = p.starting_capital
    positions: list[Position] = []
    pending_entries: list[dict] = []
    trades, equity_rows, skips = [], [], {}
    by_day = {d: g for d, g in signals.groupby("date")} if len(signals) else {}
    fy_realised, loss_carry, taxes_paid = 0.0, 0.0, 0.0
    peak, halved, pause_until, pause_armed = p.starting_capital, False, -1, True
    prev_equity = p.starting_capital

    def skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    def fy(day):
        return day.year if day.month >= 4 else day.year - 1

    def sell(pos: Position, shares: int, price: float, reason: str, day) -> None:
        nonlocal cash, fy_realised
        shares = min(shares, pos.shares)
        if shares <= 0:
            return
        value = shares * price
        proceeds = value - order_cost(value, "sell", p)
        cost_basis = pos.cash_out * shares / pos.shares_initial
        fy_realised += proceeds - cost_basis
        cash += proceeds
        pos.cash_in += proceeds
        pos.shares -= shares
        pos.reasons.append(reason)
        if pos.shares == 0:
            net = pos.cash_in - pos.cash_out
            risk_rupees = pos.shares_initial * pos.r_per_share
            trades.append({"run": run.name, "symbol": pos.symbol, "industry": pos.industry,
                           "entry_date": pos.entry_date.date(), "exit_date": day.date(),
                           "entry": round(pos.entry, 2), "initial_stop": round(pos.initial_stop, 2),
                           "shares": pos.shares_initial, "net_pnl": round(net, 2),
                           "r_multiple": round(net / risk_rupees, 3) if risk_rupees > 0 else 0.0,
                           "return_pct": round(net / pos.cash_out * 100, 2),
                           "sessions": pos.idx - pos.entry_idx, "exit_reason": reason,
                           "path": "+".join(pos.reasons), "rs_points": pos.rs_points,
                           "risk_pct": pos.risk_pct})

    def charge_tax() -> None:
        nonlocal cash, fy_realised, loss_carry, taxes_paid
        taxable = fy_realised + loss_carry
        if taxable > 0:
            tax = p.stcg_tax * taxable
            cash -= tax
            taxes_paid += tax
            loss_carry = 0.0
        else:
            loss_carry = taxable
        fy_realised = 0.0

    prev_day = None
    for k, day in enumerate(calendar):
        d64 = np.datetime64(day, "ns")
        if prev_day is not None and fy(day) != fy(prev_day):
            charge_tax()
        state = regime.get(day, "OFF")

        # 1. Open: queued exits, then queued entries
        for pos in positions:
            i = book.index_on(pos.symbol, d64)
            if i is None:
                continue
            pos.idx = i
            if pos.pending_exit:
                shares, reason = pos.pending_exit
                pos.pending_exit = None
                sell(pos, shares, book.data[pos.symbol]["adj_open"][i] * (1 - p.slippage), reason, day)
        positions = [q for q in positions if q.shares > 0]

        for order in pending_entries:
            sym = order["symbol"]
            i = book.index_on(sym, d64)
            if i is None:
                skip("no_trade_next_day")
                continue
            px = book.data[sym]
            if px["adj_open"][i] > p.chase_limit * order["pivot"]:
                skip("gap_above_chase_limit")
                continue
            entry = px["adj_open"][i] * (1 + p.slippage)
            stop = order["final_low"] * (1 - p.stop_buffer)
            if (entry - stop) / entry > p.stop_max:
                skip("stop_too_wide")
                continue
            if (entry - stop) / entry < p.stop_min:
                stop = entry * (1 - p.stop_min)
            shares = math.floor(prev_equity * order["risk_pct"] / (entry - stop))
            shares = min(shares, math.floor(p.max_position_pct * prev_equity / entry))
            while shares > 0 and shares * entry + order_cost(shares * entry, "buy", p) > cash:
                shares = math.floor(shares * 0.95) if shares > 20 else shares - 1
            if shares < 1:
                skip("not_enough_cash")
                continue
            value = shares * entry
            outlay = value + order_cost(value, "buy", p)
            cash -= outlay
            positions.append(Position(sym, order["industry"], i, day, i, entry, stop, stop, shares,
                                      shares, order["pivot"], order["rs_points"], order["rs_pct"],
                                      order["risk_pct"], outlay, last_close=px["adj_close"][i]))
        pending_entries = []

        # 2. During the day: stop-loss first (conservative), then partial target
        for pos in positions:
            i = book.index_on(pos.symbol, d64)
            if i is None:
                continue
            pos.idx = i
            px = book.data[pos.symbol]
            o, h, l = px["adj_open"][i], px["adj_high"][i], px["adj_low"][i]
            if l <= pos.stop:
                fill = (o if o < pos.stop else pos.stop) * (1 - p.slippage)
                sell(pos, pos.shares, fill, "stop" if pos.stop < pos.entry else "breakeven_stop", day)
                continue
            target = pos.entry + p.partial_r * pos.r_per_share
            if not pos.partial_done and h >= target:
                part = max(1, math.floor(pos.shares_initial * p.partial_fraction))
                if part < pos.shares:
                    sell(pos, part, max(o, target), "partial_2.5R", day)
                    pos.partial_done = True
                    pos.stop = max(pos.stop, pos.entry)
        positions = [q for q in positions if q.shares > 0]

        # 3. Close: mark to market, close-based exits
        for pos in positions:
            px = book.data[pos.symbol]
            i = book.index_on(pos.symbol, d64)
            if i is None:
                continue
            c = px["adj_close"][i]
            pos.last_close = c
            if i == len(px["dates"]) - 1 and k < len(calendar) - 1:
                sell(pos, pos.shares, c * (1 - p.slippage), "data_ended", day)
                continue
            held = i - pos.entry_idx
            if held <= p.failed_breakout_sessions and c < p.failed_breakout_level * pos.pivot:
                pos.pending_exit = (pos.shares, "failed_breakout")
                continue
            prev15, dma50 = px["close_prev15"][i], px["dma50"][i]
            if (not pos.climax_done and not np.isnan(prev15) and not np.isnan(dma50)
                    and c >= p.climax_gain_15 * prev15 and c >= p.climax_above_dma50 * dma50):
                pos.climax_done = True
                half = max(1, pos.shares // 2)
                pos.pending_exit = (half if half < pos.shares else pos.shares, "climax_half")
                continue
            if not pos.fast and held <= p.fast_window and c / pos.entry - 1 >= p.fast_gain:
                pos.fast = True
            if run.use_regime and state == "OFF":
                pos.regime_trail = True
            if (held == p.time_stop_sessions and not pos.partial_done
                    and c < pos.entry + p.time_stop_min_r * pos.r_per_share):
                pos.pending_exit = (pos.shares, "time_stop")
                continue
            dma10, dma20, vol, av50 = px["dma10"][i], px["dma20"][i], px["adj_volume"][i], px["avgvol50"][i]
            if pos.climax_done:
                if not np.isnan(dma10) and c < dma10:
                    pos.pending_exit = (pos.shares, "trail_dma10")
            elif pos.fast or pos.regime_trail:
                if not np.isnan(dma20) and c < dma20:
                    pos.pending_exit = (pos.shares, "trail_dma20")
            elif not np.isnan(dma50) and ((c < dma50 and vol >= p.dma50_break_volume * av50)
                                          or c < p.dma50_hard_break * dma50):
                pos.pending_exit = (pos.shares, "trail_dma50")
        positions = [q for q in positions if q.shares > 0]

        invested = sum(q.shares * q.last_close for q in positions)
        equity = cash + invested
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        halved = drawdown >= p.dd_halve or (halved and drawdown > p.dd_halve_recover)
        if drawdown >= p.dd_pause and pause_armed:
            pause_until, pause_armed = k + p.dd_pause_sessions, False
        if drawdown < p.dd_halve:
            pause_armed = True
        equity_rows.append({"date": day, "equity": equity, "pre_tax_equity": equity + taxes_paid,
                            "invested": invested, "positions": len(positions), "regime": state})
        prev_equity = equity
        prev_day = day

        # New entries for tomorrow's open
        todays = by_day.get(day)
        if todays is None or k == len(calendar) - 1:
            continue
        if k < pause_until:
            skip("drawdown_pause")
            continue
        if run.use_regime and state == "OFF":
            skip("regime_off")
            continue
        ranked = todays.sort_values(["rs_points", "rs_pct"] if run.use_rs_score else ["rs_pct"],
                                    ascending=False)
        held_syms = {q.symbol for q in positions}
        heat = sum(max(0.0, q.last_close - q.stop) * q.shares
                   for q in positions if q.stop < q.entry and q.pending_exit is None)
        live = [q for q in positions if not (q.pending_exit and q.pending_exit[0] >= q.shares)]
        by_industry = {}
        for q in live:
            by_industry[q.industry] = by_industry.get(q.industry, 0) + 1
        for _, sig in ranked.iterrows():
            if len(pending_entries) >= p.max_new_per_day:
                skip("max_new_per_day")
                continue
            if len(live) + len(pending_entries) >= p.max_positions:
                skip("max_positions")
                continue
            if sig["symbol"] in held_syms:
                skip("already_held")
                continue
            if run.use_rs_score:
                pts = sig["rs_points"]
                if state == "CAUTION":
                    risk = p.risk_half if pts == 3 else None
                else:
                    risk = p.risk_full if pts == 3 else p.risk_half if pts == 2 else None
            else:
                risk = p.risk_half if (run.use_regime and state == "CAUTION") else p.risk_full
            if risk is None:
                skip("score_too_low")
                continue
            if halved:
                risk /= 2
            ind = sig["industry"]
            if ind != UNCLASSIFIED and by_industry.get(ind, 0) >= p.max_per_industry:
                skip("industry_cap")
                continue
            if heat + risk * equity > p.max_heat * equity:
                skip("portfolio_heat")
                continue
            heat += risk * equity
            by_industry[ind] = by_industry.get(ind, 0) + 1
            held_syms.add(sig["symbol"])
            pending_entries.append({"symbol": sig["symbol"], "industry": ind, "pivot": sig["pivot"],
                                    "final_low": sig["final_low"], "rs_points": int(sig["rs_points"]),
                                    "rs_pct": float(sig["rs_pct"]), "risk_pct": risk})

    # End of test: close everything at the last close, then settle tax
    last = calendar[-1]
    for pos in positions:
        sell(pos, pos.shares, pos.last_close * (1 - p.slippage), "end_of_test", last)
    charge_tax()
    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    if len(equity_curve):
        equity_curve.iloc[-1, equity_curve.columns.get_loc("equity")] = cash
        equity_curve.iloc[-1, equity_curve.columns.get_loc("pre_tax_equity")] = cash + taxes_paid
    return {"trades": pd.DataFrame(trades), "equity": equity_curve, "skips": skips,
            "taxes_paid": taxes_paid}
