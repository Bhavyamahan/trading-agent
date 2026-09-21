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


def markdown_report(summary: dict, diagnostics: dict, yearly: dict, skips: dict,
                    exit_mix: dict) -> str:
    lines = ["# Backtest report: Rulebook v1.1, Step 4a (price-based layers)", ""]
    lines += ["## Results by run and period", "",
              "| Run | Period | " + " | ".join(label for _, label in COLUMNS) + " |",
              "|" + " --- |" * (len(COLUMNS) + 2)]
    for (run, period), m in summary.items():
        lines.append(f"| {run} | {period} | " + " | ".join(_fmt(m.get(k, "n/a")) for k, _ in COLUMNS) + " |")
    lines += ["", "R = profit or loss divided by the rupee risk taken at entry. "
              "Nifty 500 CAGR is the price index (excludes dividends, about 1 to 1.5% a year).", ""]

    lines += ["## Acceptance check (partial: fundamental layers not yet included)", "",
              "| Run | Criterion | Threshold | Out-of-sample value | Pass |", "| --- | --- | --- | --- | --- |"]
    for (run, period), m in summary.items():
        if period != "out_of_sample":
            continue
        for key, op, limit, label in CRITERIA:
            value = m.get(key, float("nan"))
            ok = (value >= limit) if op == ">=" else (value <= limit)
            lines.append(f"| {run} | {label} | {op} {limit} | {_fmt(value)} | {'yes' if ok else 'no'} |")
        beat = m.get("cagr_after_tax_pct", float("nan")) >= m.get("nifty500_pr_cagr_pct", float("nan")) + 3
        lines.append(f"| {run} | After-tax CAGR vs Nifty 500 + 3 pts | "
                     f"{_fmt(m.get('nifty500_pr_cagr_pct', float('nan')) + 3)} | "
                     f"{_fmt(m.get('cagr_after_tax_pct', float('nan')))} | {'yes' if beat else 'no'} |")
    lines.append("")

    for run, table in yearly.items():
        lines += [f"## Year-by-year returns, {run} (pre-tax)", "", "| Year | Strategy % | Nifty 500 % |",
                  "| --- | --- | --- |"]
        lines += [f"| {y} | {_fmt(r.strategy_pct)} | {_fmt(r.nifty500_pct)} |" for y, r in table.iterrows()]
        lines.append("")

    lines += ["## How trades ended (all periods)", "", "| Run | Exit reason | Trades |", "| --- | --- | --- |"]
    for run, mix in exit_mix.items():
        lines += [f"| {run} | {reason} | {count} |" for reason, count in mix.items()]
    lines += ["", "## Signals not taken (all periods)", "", "| Run | Reason | Count |", "| --- | --- | --- |"]
    for run, reasons in skips.items():
        lines += [f"| {run} | {reason} | {count} |" for reason, count in sorted(reasons.items())]

    lines += ["", "## Data checks", ""]
    for key, value in diagnostics.items():
        if key != "universe_log":
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines) + "\n"
