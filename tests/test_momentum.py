"""Offline tests for the momentum strategy (Rulebook v2.0)."""
from dataclasses import replace

import numpy as np
import pandas as pd

from strategy import features, momentum
from strategy.backtest import Book
from strategy.config import V12
from strategy.momentum import MomentumParams, Variant, simulate_momentum

ZERO = replace(V12, slippage=0.0, brokerage_per_order=0.0, stt=0.0, exchange_fee=0.0, sebi_fee=0.0,
               stamp_duty_buy=0.0, dp_charge_sell=0.0)


def flat_book(dates, paths):
    rows = []
    for sym, closes in paths.items():
        rows.append(pd.DataFrame({"date": dates, "symbol": sym, "adj_open": closes, "adj_high": closes,
                                  "adj_low": closes, "adj_close": closes, "adj_volume": 1e5,
                                  "dma10": 1.0, "dma20": 1.0, "dma50": 1.0, "avgvol50": 1e5,
                                  "close_prev15": np.nan}))
    return Book(pd.concat(rows, ignore_index=True), list(paths))


def test_rotation_and_buffer():
    dates = pd.bdate_range("2015-01-01", "2015-04-30")
    n = len(dates)
    book = flat_book(dates, {"A": np.linspace(100, 120, n), "B": np.full(n, 50.0), "C": np.full(n, 80.0)})
    me = momentum.month_end_dates(dates)
    # Month 1: A top. Month 2: C top, A 2nd (inside buffer 2 -> kept). Month 3: A falls to 3rd -> sold, C bought.
    ranks = pd.DataFrame([(me[0], "A", 2.0), (me[0], "B", 1.0), (me[0], "C", 0.0),
                          (me[1], "C", 2.0), (me[1], "A", 1.0), (me[1], "B", 0.0),
                          (me[2], "C", 2.0), (me[2], "B", 1.0), (me[2], "A", 0.0)],
                         columns=["date", "symbol", "score"]).assign(above_dma200=True, industry="X")
    res = simulate_momentum(ranks, book, dates, pd.Series("ON", index=dates),
                            Variant("T", False, False, ""), ZERO, MomentumParams(holdings=1, buffer_rank=2))
    t = res["trades"]
    assert list(t["symbol"]) == ["A", "C"], t
    assert t.iloc[0]["exit_reason"] == "dropped_out_of_buffer" and t.iloc[1]["exit_reason"] == "end_of_test"
    eq = res["equity"]
    assert abs(eq["equity"].iloc[-1] - (ZERO.starting_capital + t["net_pnl"].sum() - res["taxes_paid"])) < 1


def test_regime_exit_and_reentry():
    dates = pd.bdate_range("2016-01-01", "2016-06-30")
    book = flat_book(dates, {"A": np.linspace(100, 130, len(dates))})
    me = momentum.month_end_dates(dates)
    ranks = pd.DataFrame({"date": me, "symbol": "A", "score": 1.0, "above_dma200": True, "industry": "X"})
    regime = pd.Series("ON", index=dates)
    regime[(dates >= "2016-03-10") & (dates < "2016-04-15")] = "OFF"
    res = simulate_momentum(ranks, book, dates, regime, Variant("T", True, False, ""), ZERO,
                            MomentumParams(holdings=1, buffer_rank=1))
    t = res["trades"]
    assert t.iloc[0]["exit_reason"] == "regime_off" and pd.Timestamp(t.iloc[0]["exit_date"]) == pd.Timestamp("2016-03-11")
    assert pd.Timestamp(t.iloc[1]["entry_date"]) > pd.Timestamp("2016-04-15")   # back in at a month-end only


def test_tax_short_vs_long_term():
    dates = pd.bdate_range("2017-01-02", "2019-06-28")
    n = len(dates)
    book = flat_book(dates, {"A": np.linspace(100, 300, n)})
    me = momentum.month_end_dates(dates)
    ranks = pd.DataFrame({"date": me, "symbol": "A", "score": 1.0, "above_dma200": True, "industry": "X"})
    res = simulate_momentum(ranks, book, dates, pd.Series("ON", index=dates), Variant("T", False, False, ""),
                            ZERO, MomentumParams(holdings=1, buffer_rank=1))
    gain = res["trades"]["net_pnl"].sum()
    # Held > 1 year -> 12.5% on the gain above Rs 1.25 lakh
    assert abs(res["taxes_paid"] - 0.125 * (gain - 125_000)) < 1, (res["taxes_paid"], gain)


def test_pipeline_accounting():
    import tests.test_strategy as T
    from strategy.data import prepare_prices
    raw, nifty = T.synthetic_market(n_stocks=50, seed=4)
    f = features.add_stock_indicators(prepare_prices(raw), nifty)
    f, regime, _ = features.add_layers(f, nifty, {}, V12)
    f = momentum.add_volatility(f)
    ranks = momentum.rank_table(f, momentum.month_end_dates(nifty.index))
    book = Book(f, ranks["symbol"].unique())
    cal = nifty.index[nifty.index >= "2010-01-01"]
    mid = momentum.rank_table(f, momentum.mid_month_dates(nifty.index))
    for v in momentum.VARIANTS:
        table = mid if v.schedule == "mid_month" else ranks
        res = simulate_momentum(table, book, cal, regime, v, V12)
        t, eq = res["trades"], res["equity"]
        assert len(t) > 0 and (eq["positions"] <= v.holdings).all()
        assert abs(eq["equity"].iloc[-1] - (V12.starting_capital + t["net_pnl"].sum() - res["taxes_paid"])) < 1.0
        assert eq["equity"].min() > 0
        print(v.name, "positions closed:", len(t), "final equity:", round(eq["equity"].iloc[-1]),
              "invested %:", round(float((eq["invested"] / eq["equity"]).mean() * 100), 1))


def test_mid_month_dates():
    cal = pd.bdate_range("2020-01-01", "2020-04-30")
    mids = momentum.mid_month_dates(cal)
    assert [d.day for d in mids] == [15, 14, 13]  # Jan 15 (Wed), Feb 14 (Fri), Mar 13 (Fri); April is the last month


def test_fund_exclusion_in_loader():
    from strategy.data import fund_symbols
    df = pd.DataFrame({"symbol": ["LIQUID1", "GOLDIAM", "GOLDBEES"], "isin": ["INF1", "INE1", None]})
    assert fund_symbols(df) == {"LIQUID1", "GOLDBEES"}


if __name__ == "__main__":
    test_rotation_and_buffer(); test_regime_exit_and_reentry(); test_tax_short_vs_long_term()
    test_mid_month_dates(); test_fund_exclusion_in_loader(); test_pipeline_accounting()
    print("All momentum tests passed")
