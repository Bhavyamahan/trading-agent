"""Nightly: momentum ranking + paper portfolio + report (Rulebook v2.0).

Reads the last three calendar years of data (enough for every indicator), ranks the
market on the latest session, runs one evening of the paper portfolio and publishes:
  * the GitHub job summary (your private dashboard v1),
  * live/latest_report.md, live/paper_state.json and live/rankings/<date>.csv in Supabase,
  * a Telegram message if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets exist.
Environment: PAPER_CAPITAL (default 1000000).
"""
import datetime as dt
import json
import os
import time

import pandas as pd
import requests

from pipeline import storage
from strategy import data, explain, features, live, momentum
from strategy.config import V12
from scripts.run_backtest import ever_in_universe

HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def nse_holidays() -> set[dt.date]:
    try:
        from scripts.fetch_corporate_actions import new_session
        response = new_session().get(HOLIDAY_API, timeout=30)
        rows = response.json().get("CM", []) if response.status_code == 200 else []
        return {pd.to_datetime(r["tradingDate"], format="%d-%b-%Y").date() for r in rows if r.get("tradingDate")}
    except Exception:
        return set()


def load_state(capital: float, today: dt.date) -> dict:
    raw = storage.download_bytes("live/paper_state.json")
    return json.loads(raw) if raw else live.new_state(capital, today)


def build_ranking(f: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    table = momentum.rank_table(f, pd.DatetimeIndex([day]))
    return table.reset_index(drop=True).assign(rank=lambda x: range(1, len(x) + 1))


def buy_reason(info: dict | None, with_rank: bool = True) -> str:
    if not info or info.get("ret_6m") is None:
        return ""
    lead = f"rank {info['rank']}: " if with_rank else ""
    return (f" ({lead}{info['ret_6m'] * 100:+.0f}% in 6m, {info['ret_12m'] * 100:+.0f}% in 12m, "
            f"layers {info['layers_passed']}/{info['layers_checkable']})")


def report(state: dict, day: dt.date, result: dict, ranking: pd.DataFrame, holidays_ok: bool,
           nifty: float | None, explanations: dict | None = None) -> tuple[str, str]:
    explanations = explanations or {}
    snap = state["snapshots"][-1] if state["snapshots"] else None
    lines = [f"# Momentum agent: {day}", ""]
    if not state["start_date"] and not state["pending"]:
        lines += ["**Paper portfolio waiting for its first month-end rebalance.** "
                  "It will buy the top 20 at the open after the last trading day of this month.", ""]
    if snap:
        ret = snap["equity"] / state["starting_capital"] - 1
        bench = (nifty / state["nifty_at_start"] - 1) if (nifty and state["nifty_at_start"]) else None
        lines += ["## Paper portfolio", "",
                  f"- Equity: **Rs {snap['equity']:,.0f}** (started Rs {state['starting_capital']:,.0f}"
                  f"{', on ' + state['start_date'] if state['start_date'] else ''})",
                  f"- Return: **{ret * 100:+.2f}%**" + (f" vs Nifty 500 {bench * 100:+.2f}%" if bench is not None else ""),
                  f"- Cash: Rs {snap['cash']:,.0f}; positions: {snap['positions']}", ""]
    ranks = dict(zip(ranking["symbol"], ranking["rank"]))
    if state["positions"]:
        lines += ["| Holding | Shares | Bought | Last | P&L % | Rank now | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for sym, pos in sorted(state["positions"].items()):
            pnl = pos["shares"] * pos["last_price"] / pos["cost"] - 1
            rank = ranks.get(sym)
            status = "keep" if rank and rank <= 30 else "at risk (rank 31-40)" if rank and rank <= 40 else "would be sold at month-end"
            lines.append(f"| {sym} | {pos['shares']} | {pos['entry_date']} | {pos['last_price']:,.2f} | "
                         f"{pnl * 100:+.1f} | {rank or '-'} | {status} |")
        lines.append("")
    if result.get("fills"):
        lines += ["## Filled today", ""] + [f"- {n}" for n in result["fills"]] + [""]
    if result.get("corporate_actions"):
        lines += ["## Corporate actions today", ""] + [f"- {n}" for n in result["corporate_actions"]] + [""]
    if result.get("orders"):
        lines += ["## REBALANCE: orders for the next session's open", "",
                  "Do these in your real account at tomorrow's open (market orders, or limit near the open).", "",
                  "| Action | Stock | Amount | Why |", "| --- | --- | --- | --- |"]
        for o in result["orders"]:
            amount = "all shares" if o["side"] == "sell" else f"Rs {o['target_value']:,.0f}"
            why = (o["reason"] + buy_reason(explanations.get(o["symbol"]), with_rank=False)
                   if o["side"] == "buy" else o["reason"])
            lines.append(f"| {o['side'].upper()} | {o['symbol']} | {amount} | {why} |")
        lines.append("")
    else:
        lines += [f"No rebalance tonight. Next rebalance: the last trading day of this month"
                  f"{'' if holidays_ok else ' (holiday list unavailable; weekends only assumed)'}.", ""]
    lines += ["## Top 20 right now", "", "| Rank | Stock | Industry | Score |", "| --- | --- | --- | --- |"]
    lines += [f"| {r.rank} | {r.symbol} | {r.industry} | {r.score:.2f} |" for r in ranking.head(20).itertuples()]
    text = "\n".join(lines) + "\n"

    short = [f"Momentum agent {day}"]
    if snap:
        short.append(f"Paper equity Rs {snap['equity']:,.0f} ({snap['equity'] / state['starting_capital'] * 100 - 100:+.2f}%)")
    if result.get("orders"):
        short.append("REBALANCE tomorrow at the open:")
        short += [f"{o['side'].upper()} {o['symbol']}" + ("" if o["side"] == "sell" else
                  f" Rs {o['target_value']:,.0f}" + buy_reason(explanations.get(o["symbol"])))
                  for o in result["orders"]]
        short.append("Why each stock: open your dashboard and tap its name.")
    short.append("Top 5: " + ", ".join(ranking["symbol"].head(5)))
    return text, "\n".join(short)


def send_telegram(message: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": message[:4000]}, timeout=30).raise_for_status()
    except Exception as error:
        print(f"[warn] Telegram message failed: {error}")


def main():
    capital = float(os.environ.get("PAPER_CAPITAL", "1000000"))
    today = (dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).date()
    years = [today.year - 2, today.year - 1, today.year]
    log("Loading the last three years")
    prices = ever_in_universe(data.load_prices(years), V12)
    events = data.LAST_EVENTS["events"]
    audit = pd.DataFrame(data.LAST_EVENTS["official_audit"])
    nifty = data.load_nifty(years)
    industry, _ = data.load_industry()
    f = features.add_stock_indicators(prices, nifty)
    del prices
    f, regime, _ = features.add_layers(f, nifty, industry, V12)
    f = momentum.add_volatility(f)
    day_ts = f["date"].max()
    day = day_ts.date()
    ranking = build_ranking(f, day_ts)
    log(f"Latest session {day}: {len(ranking)} stocks ranked")

    today_rows = f[f["date"] == day_ts]
    opens = dict(zip(today_rows["symbol"].astype(str), today_rows["adj_open"].astype(float)))
    closes = dict(zip(today_rows["symbol"].astype(str), today_rows["adj_close"].astype(float)))
    prev_day = f.loc[f["date"] < day_ts, "date"].max()
    prev_rows = f[f["date"] == prev_day]
    # Previous close in today's units (after any split today), used for demerger credits
    prev_closes = dict(zip(prev_rows["symbol"].astype(str), prev_rows["adj_close"].astype(float)))

    kinds = {}
    if len(audit):
        applied = audit[audit["status"] == "applied"]
        kinds = {(r.symbol, str(r.date)): r.kind for r in applied.itertuples()}
    todays_events = {}
    if events is not None and len(events):
        ev = events[pd.to_datetime(events["date"]).dt.date == day]
        for r in ev.itertuples():
            if r.ca_method:
                todays_events[str(r.symbol)] = (float(r.ca_factor), kinds.get((str(r.symbol), str(day)), "split/bonus"))
    # prev_closes are split-adjusted already; convert back so the demerger credit uses the real price
    for sym, (factor, kind) in todays_events.items():
        if "demerger" in kind and sym in prev_closes:
            prev_closes[sym] = prev_closes[sym] / factor

    holidays = nse_holidays()
    state = load_state(capital, today)
    result = live.process_day(state, day, opens, closes, prev_closes, todays_events,
                              float(nifty.get(day_ts, float("nan"))), list(ranking["symbol"]), holidays, V12)
    if result.get("skipped"):
        log(result["reason"])

    # "Why this stock": momentum numbers, rank history, eligibility and the 7-layer checklist
    month_ends = [d for d in momentum.month_end_dates(nifty.index) if d < day_ts][-6:]
    history_tables = {d: g for d, g in momentum.rank_table(f, pd.DatetimeIndex(month_ends)).groupby("date")}
    held = set(state["positions"])
    pending = {o["symbol"] for o in state["pending"]}
    explanations = explain.build_all(f, ranking, history_tables, str(regime.get(day_ts, "OFF")), V12, held, pending)
    log(f"Explanations built for {len(explanations)} stocks")
    storage.upload_bytes("live/explain/latest.json", json.dumps(explanations, default=str).encode(), "application/json")
    storage.upload_bytes(f"live/explain/{day}.json", json.dumps(explanations, default=str).encode(), "application/json")

    text, short = report(state, day, result, ranking, bool(holidays), float(nifty.get(day_ts, float("nan"))),
                         explanations)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    storage.upload_bytes("live/paper_state.json", json.dumps(state, indent=1, default=str).encode(), "application/json")
    storage.upload_bytes("live/latest_report.md", text.encode(), "text/markdown")
    storage.upload_bytes(f"live/rankings/{day}.csv", ranking.head(60).to_csv(index=False).encode(), "text/csv")
    if not result.get("skipped"):
        send_telegram(short)
    storage.log_run("nightly", "ok", {"day": str(day), "equity": state["snapshots"][-1]["equity"] if state["snapshots"] else None,
                                      "orders": len(result.get("orders", []) or [])})


if __name__ == "__main__":
    main()
