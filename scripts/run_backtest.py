"""Backtest: Rulebook v1.1 (on corrected data) and v1.2, ablations A0-A2, in/out of sample.

The report appears on the GitHub Actions run page (job summary) and is saved to
Supabase Storage under results/backtest/<timestamp>/ with trade lists and the
corporate-action audit.
"""
import datetime as dt
import os
import time

import pandas as pd

from pipeline import storage
from strategy import data, features, report, vcp
from strategy.backtest import Book, simulate
from strategy.config import PERIODS, RUNS, VERSIONS, V11


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ever_in_universe(prices: pd.DataFrame, p) -> pd.DataFrame:
    log("Building universe on all stocks")
    universe, _ = features.build_universe(prices, p)
    ever = set(prices.loc[universe.to_numpy(), "symbol"].astype(str))
    kept = prices[prices["symbol"].astype(str).isin(ever)].reset_index(drop=True)
    kept["symbol"] = kept["symbol"].cat.remove_unused_categories()
    log(f"Stocks that were ever in the universe: {len(ever):,} ({len(kept):,} rows)")
    return kept


def run_version(name, vp, f, regime, nifty):
    f["layer2"] = f["layer2_base"] & (f["adv_cr"] >= vp.min_adv_crore)
    period = f["date"] >= "2010-01-01"
    funnel = {"pass_layer2": int((period & f["layer2"]).sum()),
              "pass_layer2_and_3": int((period & f["layer2"] & f["layer3"]).sum())}
    log(f"{name}: finding setups ({vp.swing_method} swings, {vp.entry_mode} entries)")
    signals = vcp.all_signals(f, vp)
    funnel.update(setups_confirmed=int(vcp.LAST_STATS["setups"]), signals_or_orders=len(signals))
    log(f"{name}: {funnel}")
    book = Book(f, signals["symbol"].unique() if len(signals) else [])
    out = {"summary": {}, "yearly": {}, "skips": {}, "exits": {}, "trades": {}, "funnel": funnel,
           "signals": signals}
    for run in RUNS:
        run_trades, run_years, run_skips = [], [], {}
        for period_name, (start, end) in PERIODS.items():
            cal = nifty.index[(nifty.index >= start) & (nifty.index <= end)]
            sig = signals[(signals["date"] >= cal[0]) & (signals["date"] <= cal[-1])]
            result = simulate(sig, book, cal, regime, run, vp)
            out["summary"][(name, run.name, period_name)] = report.metrics(result, nifty)
            log(f"{name} {run.name} {period_name}: {out['summary'][(name, run.name, period_name)]}")
            run_trades.append(result["trades"].assign(period=period_name, version=name))
            run_years.append(report.yearly_returns(result["equity"], nifty))
            for reason, count in result["skips"].items():
                run_skips[reason] = run_skips.get(reason, 0) + count
        trades = pd.concat(run_trades, ignore_index=True)
        key = f"{name} {run.name}"
        out["trades"][key] = trades
        out["yearly"][key] = pd.concat(run_years)
        out["skips"][key] = run_skips
        out["exits"][key] = trades["exit_reason"].value_counts().to_dict() if len(trades) else {}
    return out


def main() -> None:
    log("Loading prices from Supabase")
    prices = ever_in_universe(data.load_prices(), V11)
    events = data.LAST_EVENTS["events"]
    events = events[events["symbol"].isin(set(prices["symbol"].astype(str)))].reset_index(drop=True)
    nifty = data.load_nifty()
    industry, sources = data.load_industry()
    log(f"Industry labels: {len(industry):,} symbols")

    log("Computing indicators and layers 1 to 4")
    f = features.add_stock_indicators(prices, nifty)
    del prices
    f, regime, diagnostics = features.add_layers(f, nifty, industry, V11)
    diagnostics["industry_sources"] = "; ".join(sources) or "none reachable (all UNCLASSIFIED)"
    diagnostics["regime_days_since_2010"] = regime[regime.index >= "2010-01-01"].value_counts().to_dict()

    results = {name: run_version(name, vp, f, regime, nifty) for name, vp in VERSIONS.items()}
    audit = pd.DataFrame(data.LAST_EVENTS["official_audit"])
    diagnostics["official_corporate_action_sources"] = "; ".join(data.LAST_EVENTS["official_sources"]) or "none"
    text = report.markdown_report(results, diagnostics, events, audit)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    print(text)

    folder = f"results/backtest/{dt.datetime.utcnow():%Y%m%d-%H%M}"
    storage.upload_bytes(f"{folder}/report.md", text.encode(), "text/markdown")
    for name, res in results.items():
        for key, trades in res["trades"].items():
            storage.upload_bytes(f"{folder}/trades_{key.replace(' ', '_')}.csv",
                                 trades.to_csv(index=False).encode(), "text/csv")
        storage.upload_bytes(f"{folder}/signals_{name}.csv",
                             res["signals"].to_csv(index=False).encode(), "text/csv")
    storage.upload_bytes(f"{folder}/corporate_actions.csv", events.to_csv(index=False).encode(), "text/csv")
    storage.upload_bytes(f"{folder}/official_actions_audit.csv", audit.to_csv(index=False).encode(), "text/csv")
    storage.upload_bytes(f"{folder}/universe_log.csv",
                         diagnostics["universe_log"].to_csv(index=False).encode(), "text/csv")
    storage.log_run("backtest", "ok", {"folder": folder, "results": {
        "|".join(k): m for res in results.values() for k, m in res["summary"].items()}})
    log(f"Saved results to {folder}")


if __name__ == "__main__":
    main()
