"""Live forward test: a paper portfolio that follows Rulebook v2.0 exactly.

Each evening after the session's data arrives (day d):
  1. splits / bonuses with ex-date d adjust the share count; a demerger credits the
     value of the spun-off business as cash (as if those shares were sold at fair value);
  2. orders planned on the previous evening fill at d's actual opening prices
     (sells first, then buys; same slippage and costs as the backtest);
  3. positions are marked at d's close and a snapshot is stored;
  4. if d is the last trading day of the month, next session's orders are planned:
     sell holdings ranked outside the top 40, buy from the top of the ranking to
     hold 20 stocks, each bought for equity / 20.
The whole state is one JSON document, so it is easy to inspect and back up.
"""
import datetime as dt
import math

import pandas as pd

from strategy.backtest import order_cost
from strategy.config import Params


def new_state(capital: float, created: dt.date) -> dict:
    return {"starting_capital": capital, "cash": capital, "created": str(created), "start_date": None,
            "nifty_at_start": None, "last_processed": None, "last_rebalance": None,
            "positions": {}, "pending": [], "closed": [], "snapshots": [], "log": []}


def is_month_end(day: dt.date, holidays: set[dt.date]) -> bool:
    nxt = day + dt.timedelta(days=1)
    while nxt.weekday() >= 5 or nxt in holidays:
        nxt += dt.timedelta(days=1)
    return nxt.month != day.month


def apply_corporate_actions(state: dict, events: dict[str, tuple[float, str]], day: dt.date,
                            prev_closes: dict[str, float]) -> list[str]:
    notes = []
    for sym, (factor, kind) in events.items():
        pos = state["positions"].get(sym)
        if not pos or not factor or factor <= 0 or str(day) in pos.get("ca_applied", []):
            continue
        if "demerger" in kind:
            credit = pos["shares"] * prev_closes.get(sym, pos["last_price"]) * (1 - factor)
            state["cash"] += credit
            notes.append(f"{sym}: demerger, Rs {credit:,.0f} credited for the spun-off business")
        else:
            before = pos["shares"]
            pos["shares"] = int(round(before / factor))
            pos["entry_price"] *= factor
            notes.append(f"{sym}: {kind or 'split/bonus'} x{1 / factor:g}, shares {before} -> {pos['shares']}")
        pos.setdefault("ca_applied", []).append(str(day))
    state["log"].extend(f"{day} {n}" for n in notes)
    return notes


def fill_orders(state: dict, day: dt.date, opens: dict[str, float], p: Params) -> list[str]:
    notes, still = [], []
    for order in sorted(state["pending"], key=lambda o: o["side"] != "sell"):   # sells first
        sym, price = order["symbol"], opens.get(order["symbol"])
        if price is None or not price > 0:
            still.append(order)
            notes.append(f"{sym}: no trade today, order kept for the next session")
            continue
        if order["side"] == "sell":
            pos = state["positions"].pop(sym, None)
            if pos is None:
                continue
            fill = price * (1 - p.slippage)
            value = pos["shares"] * fill
            proceeds = value - order_cost(value, "sell", p)
            state["cash"] += proceeds
            state["closed"].append({"symbol": sym, "entry_date": pos["entry_date"], "exit_date": str(day),
                                    "shares": pos["shares"], "net_pnl": round(proceeds - pos["cost"], 2),
                                    "return_pct": round((proceeds / pos["cost"] - 1) * 100, 2),
                                    "reason": order.get("reason", "")})
            notes.append(f"SOLD {sym}: {pos['shares']} @ Rs {fill:,.2f} ({order.get('reason', '')})")
        else:
            fill = price * (1 + p.slippage)
            budget = min(order["target_value"], state["cash"])
            shares = math.floor(budget / fill)
            while shares > 0 and shares * fill + order_cost(shares * fill, "buy", p) > state["cash"]:
                shares -= max(1, shares // 50)
            if shares < 1:
                notes.append(f"{sym}: not enough cash to buy")
                continue
            value = shares * fill
            cost = value + order_cost(value, "buy", p)
            state["cash"] -= cost
            state["positions"][sym] = {"shares": shares, "entry_date": str(day), "entry_price": fill,
                                       "cost": cost, "last_price": price, "ca_applied": []}
            if state["start_date"] is None:
                state["start_date"] = str(day)
            notes.append(f"BOUGHT {sym}: {shares} @ Rs {fill:,.2f}")
    state["pending"] = still
    state["log"].extend(f"{day} {n}" for n in notes)
    return notes


def mark_to_market(state: dict, day: dt.date, closes: dict[str, float], nifty: float | None) -> dict:
    for sym, pos in state["positions"].items():
        if closes.get(sym):
            pos["last_price"] = closes[sym]
    invested = sum(p["shares"] * p["last_price"] for p in state["positions"].values())
    equity = state["cash"] + invested
    if state["start_date"] and state["nifty_at_start"] is None and nifty:
        state["nifty_at_start"] = nifty
    snap = {"date": str(day), "equity": round(equity, 2), "cash": round(state["cash"], 2),
            "invested": round(invested, 2), "positions": len(state["positions"]), "nifty500": nifty}
    state["snapshots"] = [s for s in state["snapshots"] if s["date"] != str(day)] + [snap]
    return snap


def plan_rebalance(state: dict, day: dt.date, ranking: list[str], equity: float,
                   holdings: int = 20, buffer_rank: int = 40) -> list[dict]:
    keep_zone = set(ranking[:buffer_rank])
    orders = [{"symbol": s, "side": "sell", "created": str(day), "reason": "outside top 40"}
              for s in state["positions"] if s not in keep_zone]
    selling = {o["symbol"] for o in orders}
    slots = holdings - (len(state["positions"]) - len(selling))
    target = equity / holdings
    for sym in ranking:
        if slots <= 0:
            break
        if sym in state["positions"]:
            continue
        orders.append({"symbol": sym, "side": "buy", "created": str(day), "target_value": round(target, 2),
                       "reason": f"rank {ranking.index(sym) + 1}"})
        slots -= 1
    state["pending"] = orders
    state["last_rebalance"] = str(day)
    return orders


def process_day(state: dict, day: dt.date, opens: dict, closes: dict, prev_closes: dict,
                events: dict, nifty: float | None, ranking: list[str], holidays: set, p: Params) -> dict:
    """Run one evening. Returns what happened (for the report)."""
    if state["last_processed"] and pd.Timestamp(state["last_processed"]).date() >= day:
        return {"skipped": True, "reason": f"{day} was already processed"}
    out = {"corporate_actions": apply_corporate_actions(state, events, day, prev_closes),
           "fills": fill_orders(state, day, opens, p)}
    out["snapshot"] = mark_to_market(state, day, closes, nifty)
    out["month_end"] = is_month_end(day, holidays)
    if out["month_end"] and ranking:
        out["orders"] = plan_rebalance(state, day, ranking, out["snapshot"]["equity"])
    state["last_processed"] = str(day)
    return out
