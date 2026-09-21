"""Step 4a: run ablations A0, A1, A2 in-sample and out-of-sample and publish a report.

The report appears on the GitHub Actions run page (job summary) and is saved to
Supabase Storage under results/step4a/<timestamp>/ with the trade lists.
"""
import datetime as dt
import io
import os
import time

import pandas as pd

from pipeline import storage
from strategy import data, features, report, vcp
from strategy.backtest import Book, simulate
from strategy.config import PERIODS, RUNS, Params


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ever_in_universe(prices: pd.DataFrame, p: Params) -> pd.DataFrame:
    log("Building universe on all stocks")
    universe, _ = features.build_universe(prices, p)
    ever = set(prices.loc[universe.to_numpy(), "symbol"].astype(str))
    kept = prices[prices["symbol"].astype(str).isin(ever)].reset_index(drop=True)
    kept["symbol"] = kept["symbol"].cat.remove_unused_categories()
    log(f"Stocks that were ever in the universe: {len(ever):,} ({len(kept):,} rows)")
    return kept


def prepare(prices: pd.DataFrame, nifty: pd.Series, industry: dict, p: Params):
    log("Computing indicators")
    f = features.add_stock_indicators(prices, nifty)
    log("Computing layers 1 to 4")
    f, regime, diagnostics = features.add_layers(f, nifty, industry, p)
    period = f["date"] >= "2010-01-01"
    diagnostics["funnel_stock_days_since_2010"] = {
        "in_universe": int((period & f["in_universe"]).sum()),
        "pass_layer2": int((period & f["layer2"]).sum()),
        "pass_layer3": int((period & f["layer3"]).sum()),
        "pass_layer2_and_3": int((period & f["layer2"] & f["layer3"]).sum()),
    }
    log(f"Funnel: {diagnostics['funnel_stock_days_since_2010']}")
    log("Finding setups and breakout signals (layer 7)")
    signals = vcp.all_signals(f, p)
    diagnostics["signals_found"] = len(signals)
    diagnostics["setups_confirmed_stock_days"] = int(vcp.LAST_STATS.get("setups", 0))
    diagnostics["breakout_candidates_checked"] = int(vcp.LAST_STATS.get("candidates", 0))
    return f, regime, signals, diagnostics


def run_all(f, regime, signals, nifty, p: Params):
    book = Book(f, signals["symbol"].unique() if len(signals) else [])
    summary, yearly, skips, exit_mix, trade_lists = {}, {}, {}, {}, {}
    for run in RUNS:
        run_trades, run_years, run_skips = [], [], {}
        for period, (start, end) in PERIODS.items():
            calendar = nifty.index[(nifty.index >= start) & (nifty.index <= end)]
            period_signals = signals[(signals["date"] >= calendar[0]) & (signals["date"] <= calendar[-1])]
            result = simulate(period_signals, book, calendar, regime, run, p)
            summary[(run.name, period)] = report.metrics(result, nifty)
            log(f"{run.name} {period}: {summary[(run.name, period)]}")
            run_trades.append(result["trades"].assign(period=period))
            run_years.append(report.yearly_returns(result["equity"], nifty))
            for reason, count in result["skips"].items():
                run_skips[reason] = run_skips.get(reason, 0) + count
        trades = pd.concat(run_trades, ignore_index=True)
        trade_lists[run.name] = trades
        yearly[run.name] = pd.concat(run_years)
        skips[run.name] = run_skips
        exit_mix[run.name] = trades["exit_reason"].value_counts().to_dict() if len(trades) else {}
    return summary, yearly, skips, exit_mix, trade_lists


def main() -> None:
    p = Params()
    log("Loading prices from Supabase")
    prices = ever_in_universe(data.load_prices(), p)
    nifty = data.load_nifty()
    industry, sources = data.load_industry()
    log(f"Industry labels: {len(industry):,} symbols from {sources or 'no source reachable'}")

    f, regime, signals, diagnostics = prepare(prices, nifty, industry, p)
    del prices
    diagnostics["industry_sources"] = "; ".join(sources) or "none reachable (all UNCLASSIFIED)"
    diagnostics["regime_days"] = regime[regime.index >= "2010-01-01"].value_counts().to_dict()
    summary, yearly, skips, exit_mix, trade_lists = run_all(f, regime, signals, nifty, p)

    text = report.markdown_report(summary, diagnostics, yearly, skips, exit_mix)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    print(text)

    folder = f"results/step4a/{dt.datetime.utcnow():%Y%m%d-%H%M}"
    storage.upload_bytes(f"{folder}/report.md", text.encode(), "text/markdown")
    for name, trades in trade_lists.items():
        storage.upload_bytes(f"{folder}/trades_{name}.csv", trades.to_csv(index=False).encode(), "text/csv")
    signals_csv = signals.drop(columns=["contractions"], errors="ignore").to_csv(index=False).encode()
    storage.upload_bytes(f"{folder}/signals.csv", signals_csv, "text/csv")
    universe_csv = diagnostics["universe_log"].to_csv(index=False).encode()
    storage.upload_bytes(f"{folder}/universe_log.csv", universe_csv, "text/csv")
    storage.log_run("backtest_step4a", "ok", {
        "folder": folder, "signals": len(signals),
        "results": {f"{r}|{per}": m for (r, per), m in summary.items()}})
    log(f"Saved results to {folder}")


if __name__ == "__main__":
    main()
