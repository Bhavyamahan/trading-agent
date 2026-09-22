"""Metrics and markdown report for the momentum strategy (Rulebook v2.0)."""
import numpy as np
import pandas as pd

from strategy.report import _fmt, cagr, max_drawdown


def rolling_beat(equity: pd.Series, bench: pd.Series, months: int = 36) -> tuple[float, int]:
    """Share of rolling 3-year windows (monthly steps) where the strategy beat the index."""
    s = equity.resample("ME").last().dropna()
    b = bench.reindex(equity.index).ffill().resample("ME").last().dropna()
    s, b = s.align(b, join="inner")
    if len(s) <= months:
        return float("nan"), 0
    wins = (s / s.shift(months) > b / b.shift(months)).iloc[months:]
    return float(wins.mean() * 100), int(len(wins))


def metrics(result: dict, nifty: pd.Series, extra: dict[str, pd.Series]) -> dict:
    eq, trades = result["equity"], result["trades"]
    bench = nifty.reindex(eq.index).ffill()
    beat, windows = rolling_beat(eq["equity"], nifty)
    out = {
        "cagr_pre_tax_pct": round(cagr(eq["pre_tax_equity"]) * 100, 2),
        "cagr_after_tax_pct": round(cagr(eq["equity"]) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(eq["pre_tax_equity"]) * 100, 1),
        "nifty500_cagr_pct": round(cagr(bench) * 100, 2),
        "nifty500_max_dd_pct": round(max_drawdown(bench) * 100, 1),
        "beat_3y_windows_pct": round(beat, 1) if not np.isnan(beat) else float("nan"),
        "windows": windows,
        "invested_pct": round(float((eq["invested"] / eq["equity"]).mean() * 100), 1),
        "turnover_x_per_year": round(result["turnover_per_year"], 2),
        "closed_positions": int(len(trades)),
        "win_rate_pct": round(float((trades["net_pnl"] > 0).mean() * 100), 1) if len(trades) else float("nan"),
        "avg_days_held": round(float(trades["days_held"].mean()), 0) if len(trades) else float("nan"),
        "taxes_paid": round(result["taxes_paid"]),
        "final_equity": round(float(eq["equity"].iloc[-1])),
    }
    for name, series in extra.items():
        s = series.reindex(eq.index).ffill().dropna()
        if len(s) > 250 and s.index[0] <= eq.index[0] + pd.Timedelta(days=10):
            out[f"{name}_cagr_pct"] = round(cagr(s) * 100, 2)
    return out


def yearly(eq: pd.DataFrame, nifty: pd.Series) -> pd.DataFrame:
    s = eq["equity"]
    s_end = s.groupby(s.index.year).last()
    strat = s_end / s_end.shift(1).fillna(s.iloc[0]) - 1
    b = nifty.reindex(eq.index).ffill()
    b_end = b.groupby(b.index.year).last()
    bench = b_end / b_end.shift(1).fillna(b.iloc[0]) - 1
    return pd.DataFrame({"strategy": (strat * 100).round(1), "nifty500": (bench * 100).round(1)})


COLUMNS = [("cagr_after_tax_pct", "CAGR after tax %"), ("cagr_pre_tax_pct", "CAGR pre-tax %"),
           ("nifty500_cagr_pct", "Nifty 500 CAGR %"), ("max_drawdown_pct", "Max DD %"),
           ("nifty500_max_dd_pct", "Nifty 500 max DD %"), ("beat_3y_windows_pct", "Beat index, 3y windows %"),
           ("invested_pct", "Invested %"), ("turnover_x_per_year", "Turnover x/yr"),
           ("closed_positions", "Positions closed"), ("win_rate_pct", "Win %"), ("avg_days_held", "Avg days held")]


def compare_with_fund(curve: pd.Series | None, fund: pd.Series | None, nifty: pd.Series) -> dict | None:
    """Strategy vs the Nifty200 Momentum 30 index over the dates both exist."""
    if curve is None or fund is None or len(fund) < 250:
        return None
    start = max(curve.index[0], fund.index[0])
    s, f = curve[curve.index >= start], fund.reindex(curve.index).ffill()
    f = f[f.index >= start].dropna()
    s = s.reindex(f.index)
    n = nifty.reindex(f.index).ffill()
    return {"from": str(f.index[0].date()), "to": str(f.index[-1].date()),
            "strategy_after_tax_cagr_pct": round(cagr(s) * 100, 2),
            "momentum30_index_cagr_pct": round(cagr(f) * 100, 2),
            "nifty500_cagr_pct": round(cagr(n) * 100, 2),
            "strategy_max_dd_pct": round(max_drawdown(s) * 100, 1),
            "momentum30_max_dd_pct": round(max_drawdown(f) * 100, 1)}


def passes(m: dict) -> dict:
    return {"CAGR after tax >= Nifty 500 + 3 pts": m["cagr_after_tax_pct"] >= m["nifty500_cagr_pct"] + 3,
            "Max drawdown no worse than Nifty 500": m["max_drawdown_pct"] <= m["nifty500_max_dd_pct"],
            "Beats index in most 3-year windows": (m["beat_3y_windows_pct"] > 50)
            if not np.isnan(m["beat_3y_windows_pct"]) else False}


def markdown(summary: dict, years: dict, exits: dict, diagnostics: dict, fund: dict | None = None) -> str:
    lines = ["# Backtest report: Rulebook v2.0, momentum rotation", "",
             "Top 20 by volatility-adjusted 6- and 12-month momentum, sold when outside the top 40, "
             "rebalanced monthly at the next open. Costs, slippage and current Indian capital-gains tax "
             "included. Starting capital Rs 10 lakh per test.", "",
             "## Results", "", "| Variant | Period | " + " | ".join(l for _, l in COLUMNS) + " |",
             "|" + " --- |" * (len(COLUMNS) + 2)]
    for (variant, period), m in summary.items():
        lines.append(f"| {variant} | {period} | " + " | ".join(_fmt(m.get(k, float('nan'))) for k, _ in COLUMNS) + " |")
    extra = sorted({k for m in summary.values() for k in m if k.endswith("_cagr_pct")
                    and k not in ("cagr_pre_tax_pct", "cagr_after_tax_pct", "nifty500_cagr_pct")})
    if extra:
        lines += ["", "Other benchmarks over the same windows (where data exists): " + "; ".join(
            f"{v} {p}: {k.replace('_cagr_pct', '')} {m[k]}%" for (v, p), m in summary.items()
            for k in extra if k in m and v == "M0")]
    lines += ["", "Index figures are price-only (no dividends, worth about 1 to 1.5% a year); the strategy's "
              "stock prices are price-only too, so the comparison is like for like.", "",
              "## Pass criteria for the base strategy M0 (fixed before testing)", "",
              "| Variant | Period | Criterion | Pass |", "| --- | --- | --- | --- |"]
    for (variant, period), m in summary.items():
        if period == "full" or variant != "M0":
            continue
        for label, ok in passes(m).items():
            lines.append(f"| {variant} | {period} | {label} | {'yes' if ok else 'no'} |")
    lines += ["", "## Robustness: does every variant beat the Nifty 500 after tax?", "",
              "Robust = every M0 variant beats the index after tax in both periods (not just the base case).", "",
              "| Variant | In-sample excess (pts) | Out-of-sample excess (pts) | Full-period max DD % | Beats both |",
              "| --- | --- | --- | --- | --- |"]
    names = list(dict.fromkeys(v for v, _ in summary))
    robust = True
    for name in names:
        ins, oos, full = summary.get((name, "in_sample")), summary.get((name, "out_of_sample")), summary.get((name, "full"))
        if not (ins and oos and full):
            continue
        e1 = ins["cagr_after_tax_pct"] - ins["nifty500_cagr_pct"]
        e2 = oos["cagr_after_tax_pct"] - oos["nifty500_cagr_pct"]
        ok = e1 > 0 and e2 > 0
        if name.startswith("M0"):
            robust &= ok
        lines.append(f"| {name} | {_fmt(round(e1, 2))} | {_fmt(round(e2, 2))} | {_fmt(full['max_drawdown_pct'])} | "
                     f"{'yes' if ok else 'no'} |")
    lines += ["", f"**Robustness verdict: {'PASS' if robust else 'FAIL'}** (M1 is shown for reference only).", ""]
    if fund:
        lines += ["## Versus simply buying a Nifty200 Momentum 30 fund", "",
                  f"Over {fund['from']} to {fund['to']} (the dates the index data covers):", "",
                  "| | CAGR % | Max DD % |", "| --- | --- | --- |",
                  f"| Our strategy (M0, after tax) | {_fmt(fund['strategy_after_tax_cagr_pct'])} | {_fmt(fund['strategy_max_dd_pct'])} |",
                  f"| Nifty200 Momentum 30 index (before fund fees and tax) | {_fmt(fund['momentum30_index_cagr_pct'])} | "
                  f"{_fmt(fund['momentum30_max_dd_pct'])} |",
                  f"| Nifty 500 | {_fmt(fund['nifty500_cagr_pct'])} | |", ""]
    for variant, table in years.items():
        lines += ["", f"## Year by year, {variant} (after tax, full period)", "", "| Year | Strategy % | Nifty 500 % |",
                  "| --- | --- | --- |"]
        lines += [f"| {y} | {_fmt(r.strategy)} | {_fmt(r.nifty500)} |" for y, r in table.iterrows()]
    lines += ["", "## How positions ended (full period)", "", "| Variant | Reason | Count |", "| --- | --- | --- |"]
    for variant, mix in exits.items():
        lines += [f"| {variant} | {k} | {v} |" for k, v in mix.items()]
    lines += ["", "## Data checks", ""]
    lines += [f"- **{k}:** {v}" for k, v in diagnostics.items()]
    return "\n".join(lines) + "\n"
