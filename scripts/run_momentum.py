"""Backtest Rulebook v2.0 (momentum rotation): variants M0-M2, in-sample, out-of-sample, full."""
import datetime as dt
import os
import time

import pandas as pd

from pipeline import storage
from strategy import data, features, momentum, momentum_report as mr
from strategy.backtest import Book
from strategy.config import V12
from scripts.run_backtest import ever_in_universe

PERIODS = {"in_sample": ("2010-01-01", "2019-12-31"), "out_of_sample": ("2020-01-01", "2099-12-31"),
           "full": ("2010-01-01", "2099-12-31")}
EXTRA_INDICES = {"nifty200_momentum30": "NIFTY200 MOMENTUM 30", "nifty50": "NIFTY 50"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    p = V12
    log("Loading prices")
    prices = ever_in_universe(data.load_prices(), p)
    nifty = data.load_nifty()
    extra = {k: s for k, name in EXTRA_INDICES.items() if (s := data.load_index(name)) is not None}
    industry, _ = data.load_industry()
    log("Indicators and layers")
    f = features.add_stock_indicators(prices, nifty)
    del prices
    f, regime, diag = features.add_layers(f, nifty, industry, p)
    f = momentum.add_volatility(f)
    month_ends = momentum.month_end_dates(nifty.index)
    ranks = momentum.rank_table(f, month_ends)
    log(f"Ranked {ranks['date'].nunique()} month-ends, {len(ranks):,} stock-months")
    top = ranks.groupby("date").head(60)["symbol"].unique()
    book = Book(f, top)
    summary, years, exits, all_trades = {}, {}, {}, []
    for v in momentum.VARIANTS:
        for period, (start, end) in PERIODS.items():
            cal = nifty.index[(nifty.index >= start) & (nifty.index <= end)]
            res = momentum.simulate_momentum(ranks[ranks["date"].between(cal[0], cal[-1])], book, cal, regime, v, p)
            summary[(v.name, period)] = mr.metrics(res, nifty, extra)
            log(f"{v.name} {period}: {summary[(v.name, period)]}")
            if period == "full":
                years[v.name] = mr.yearly(res["equity"], nifty)
                exits[v.name] = res["trades"]["exit_reason"].value_counts().to_dict() if len(res["trades"]) else {}
                all_trades.append(res["trades"])
    diagnostics = {"stocks_ranked_per_month_median": int(ranks.groupby("date").size().median()),
                   "benchmarks_found": ", ".join(extra) or "none besides NIFTY 500",
                   "regime_days_since_2010": regime[regime.index >= "2010-01-01"].value_counts().to_dict(),
                   "liquidity_threshold_cr": p.min_adv_crore}
    text = mr.markdown(summary, years, exits, diagnostics)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    folder = f"results/momentum/{dt.datetime.utcnow():%Y%m%d-%H%M}"
    storage.upload_bytes(f"{folder}/report.md", text.encode(), "text/markdown")
    storage.upload_bytes(f"{folder}/positions.csv", pd.concat(all_trades).to_csv(index=False).encode(), "text/csv")
    latest = ranks[ranks["date"] == ranks["date"].max()].head(40)
    storage.upload_bytes(f"{folder}/latest_ranking.csv", latest.to_csv(index=False).encode(), "text/csv")
    storage.log_run("momentum_backtest", "ok", {"folder": folder, "results": {
        "|".join(k): m for k, m in summary.items()}})
    log(f"Saved results to {folder}")


if __name__ == "__main__":
    main()
