"""Turn simulation output into the numbers the rulebook's acceptance criteria use."""
import numpy as np
import pandas as pd


def cagr(series: pd.Series) -> float:
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")


def max_drawdown(series: pd.Series) -> float:
    return float((1 - series / series.cummax()).max()) if len(series) else float("nan")


def metrics(result: dict, nifty: pd.Series) -> dict:
    trades, eq = result["trades"], result["equity"]
    out = {"trades": int(len(trades))}
    if len(trades):
        wins, losses = trades[trades["net_pnl"] > 0], trades[trades["net_pnl"] <= 0]
        gross_loss = -losses["net_pnl"].sum()
        out.update({
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "avg_win_r": round(wins["r_multiple"].mean(), 2) if len(wins) else 0.0,
            "avg_loss_r": round(losses["r_multiple"].mean(), 2) if len(losses) else 0.0,
            "expectancy_r": round(trades["r_multiple"].mean(), 3),
            "profit_factor": round(wins["net_pnl"].sum() / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "avg_sessions": round(trades["sessions"].mean(), 1),
        })
    if len(eq):
        bench = nifty.reindex(eq.index).ffill()
        out.update({
            "cagr_pre_tax_pct": round(cagr(eq["pre_tax_equity"]) * 100, 2),
            "cagr_after_tax_pct": round(cagr(eq["equity"]) * 100, 2),
            "max_drawdown_pct": round(max_drawdown(eq["pre_tax_equity"]) * 100, 1),
            "exposure_pct": round(float((eq["invested"] / eq["equity"]).mean() * 100), 1),
            "nifty500_pr_cagr_pct": round(cagr(bench) * 100, 2),
            "nifty500_max_drawdown_pct": round(max_drawdown(bench) * 100, 1),
            "final_equity": round(float(eq["equity"].iloc[-1])),
        })
    return out


def yearly_returns(eq: pd.DataFrame, nifty: pd.Series) -> pd.DataFrame:
    year_end = eq["pre_tax_equity"].groupby(eq.index.year).last()
    start = eq["pre_tax_equity"].iloc[0]
    strat = year_end / year_end.shift(1).fillna(start) - 1
    bench = nifty.reindex(eq.index).ffill()
    b_end = bench.groupby(bench.index.year).last()
    b = b_end / b_end.shift(1).fillna(bench.iloc[0]) - 1
    return pd.DataFrame({"strategy_pct": (strat * 100).round(1), "nifty500_pct": (b * 100).round(1)})


COLUMNS = [("trades", "Trades"), ("win_rate_pct", "Win %"), ("avg_win_r", "Avg win (R)"),
           ("avg_loss_r", "Avg loss (R)"), ("expectancy_r", "Expectancy (R)"),
           ("profit_factor", "Profit factor"), ("cagr_pre_tax_pct", "CAGR pre-tax %"),
           ("cagr_after_tax_pct", "CAGR after-tax %"), ("max_drawdown_pct", "Max DD %"),
           ("exposure_pct", "Invested %"), ("nifty500_pr_cagr_pct", "Nifty 500 CAGR %")]

CRITERIA = [("trades", ">=", 40, "Trades (out-of-sample)"),
            ("expectancy_r", ">=", 0.25, "Expectancy after costs (R)"),
            ("profit_factor", ">=", 1.4, "Profit factor"),
            ("max_drawdown_pct", "<=", 25.0, "Max drawdown %")]


def _fmt(value):
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,}" if isinstance(value, int) else str(value)


def markdown_report(results: dict, diagnostics: dict, events: pd.DataFrame,
                    audit: pd.DataFrame | None = None) -> str:
    lines = ["# Backtest report: Rulebook v1.1 (corrected data) vs v1.2", "",
             "v1.1 = original rules, now on split/bonus-corrected prices. "
             "v1.2 = corrected data + buy-stop entry at the pivot, liquidity sized to our "
             "positions, and swings counted only after a 3% reversal.", ""]
    lines += ["## Results by version, run and period", "",
              "| Version | Run | Period | " + " | ".join(label for _, label in COLUMNS) + " |",
              "|" + " --- |" * (len(COLUMNS) + 3)]
    for res in results.values():
        for (version, run, period), m in res["summary"].items():
            lines.append(f"| {version} | {run} | {period} | "
                         + " | ".join(_fmt(m.get(k, "n/a")) for k, _ in COLUMNS) + " |")
    lines += ["", "R = profit or loss divided by the rupee risk taken at entry. "
              "Nifty 500 CAGR is the price index (excludes dividends, about 1 to 1.5% a year).", ""]

    lines += ["## Acceptance check, v1.2 (partial: fundamental layers not yet included)", "",
              "Judge mainly on in-sample: the out-of-sample period has now been seen once.", "",
              "| Run | Period | Criterion | Threshold | Value | Pass |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (version, run, period), m in results.get("v1.2", {"summary": {}})["summary"].items():
        for key, op, limit, label in CRITERIA:
            limit_used = 100 if (key == "trades" and period == "in_sample") else limit
            value = m.get(key, float("nan"))
            ok = (value >= limit_used) if op == ">=" else (value <= limit_used)
            lines.append(f"| {run} | {period} | {label.replace(' (out-of-sample)', '')} | "
                         f"{op} {limit_used} | {_fmt(value)} | {'yes' if ok else 'no'} |")
    lines.append("")

    lines += ["## Layer funnel (stock-days since 2010)", "", "| Version | Step | Count |", "| --- | --- | --- |"]
    for name, res in results.items():
        lines += [f"| {name} | {k} | {v:,} |" for k, v in res["funnel"].items()]
    lines.append("")

    for key, table in results.get("v1.2", {"yearly": {}})["yearly"].items():
        lines += [f"## Year-by-year returns, {key} (pre-tax)", "", "| Year | Strategy % | Nifty 500 % |",
                  "| --- | --- | --- |"]
        lines += [f"| {y} | {_fmt(r.strategy_pct)} | {_fmt(r.nifty500_pct)} |" for y, r in table.iterrows()]
        lines.append("")

    lines += ["## How trades ended (all periods)", "", "| Version and run | Exit reason | Trades |",
              "| --- | --- | --- |"]
    for res in results.values():
        for key, mix in res["exits"].items():
            lines += [f"| {key} | {reason} | {count} |" for reason, count in mix.items()]
    lines += ["", "## Signals or orders not taken (all periods)", "", "| Version and run | Reason | Count |",
              "| --- | --- | --- |"]
    for res in results.values():
        for key, reasons in res["skips"].items():
            lines += [f"| {key} | {reason} | {count} |" for reason, count in sorted(reasons.items())]

    lines += ["", "## Official corporate actions (NSE list)", ""]
    if audit is not None and len(audit):
        table = audit.groupby(["kind", "status"]).size().unstack(fill_value=0)
        lines += ["| Kind | " + " | ".join(table.columns) + " |", "|" + " --- |" * (len(table.columns) + 1)]
        lines += [f"| {k} | " + " | ".join(str(int(v)) for v in row) + " |" for k, row in table.iterrows()]
        missing = audit[audit["status"] != "applied"].sort_values("ex_date", ascending=False).head(15)
        if len(missing):
            lines += ["", "Listed events not applied (no matching price move within 3 sessions):", "",
                      "| Symbol | Ex-date | Kind | Status |", "| --- | --- | --- | --- |"]
            lines += [f"| {r.symbol} | {r.ex_date} | {r.kind} | {r.status} |" for r in missing.itertuples()]
    else:
        lines.append("No official list loaded: run workflow '5 - Fetch corporate actions' first.")
    lines += ["", "## Corporate-action audit (stocks that were ever in the universe)", ""]
    if events is not None and len(events):
        ev = events.copy()
        ev["year"] = pd.to_datetime(ev["date"]).dt.year
        ev["kind"] = np.where(ev["unexplained_gap"], "unexplained_gap", ev["ca_method"])
        counts = ev.pivot_table(index="year", columns="kind", values="symbol", aggfunc="count", fill_value=0)
        lines += ["| Year | " + " | ".join(counts.columns) + " |", "|" + " --- |" * (len(counts.columns) + 1)]
        lines += [f"| {y} | " + " | ".join(str(int(v)) for v in row) + " |" for y, row in counts.iterrows()]
        lines += ["", "official = from NSE's list. prev_close = NSE's adjusted previous close. gap_ratio = "
                  "the overnight-gap fallback. unexplained_gap = a move beyond -40%/+80% explained by "
                  "neither, left as a real move.", "", "Fallback gap-ratio events (not in the official list):",
                  "", "| Symbol | Date | Factor |", "| --- | --- | --- |"]
        gap = ev[ev["kind"] == "gap_ratio"].sort_values("date", ascending=False).head(40)
        lines += [f"| {r.symbol} | {pd.Timestamp(r.date).date()} | {r.ca_factor:.4g} |" for r in gap.itertuples()]
        odd = ev[ev["kind"] == "unexplained_gap"].sort_values("date", ascending=False).head(20)
        if len(odd):
            lines += ["", "Unexplained large moves (left as real price moves):", "",
                      "| Symbol | Date | Close | Previous close |", "| --- | --- | --- | --- |"]
            lines += [f"| {r.symbol} | {pd.Timestamp(r.date).date()} | {r.close:g} | {r.prev_close:g} |"
                      for r in odd.itertuples()]
    else:
        lines.append("No corporate actions detected.")

    lines += ["", "## Data checks", ""]
    for key, value in diagnostics.items():
        if key != "universe_log":
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines) + "\n"
