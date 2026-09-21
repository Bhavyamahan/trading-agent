"""Offline tests for the strategy engine (no network)."""
import numpy as np
import pandas as pd

from strategy import features, vcp
from strategy.backtest import Book, order_cost, simulate
from strategy.config import Params, Run

P = Params()


def vcp_path():
    """Uptrend to 100, then pullbacks of ~20%, ~12%, ~6%, then a tight range near 99."""
    closes = list(np.linspace(60, 100, 40))
    for top, bottom, n in [(100, 80, 10), (97, 85.5, 10), (96, 90.5, 8), (95.5, 92, 6)]:
        closes += list(np.linspace(top, bottom, n // 2)) + list(np.linspace(bottom, top * 0.99, n - n // 2))
    closes += [97.0, 97.5, 97.2, 97.8]
    c = np.array(closes)
    return c * 1.004, c * 0.996, c


def test_find_setup():
    high, low, close = vcp_path()
    s = len(close) - 1
    setup = vcp.find_setup(high, low, avgvol10=60.0, avgvol50=100.0, s=s, p=P)
    assert setup is not None, "expected a valid VCP"
    assert len(setup.contractions) >= 2
    assert all(b <= 0.7 * a for a, b in zip(setup.contractions, setup.contractions[1:]))
    assert setup.pivot >= 0.9 * setup.base_high
    # No dry-up in volume -> no setup
    assert vcp.find_setup(high, low, 90.0, 100.0, s, P) is None


def test_costs():
    buy = order_cost(100_000, "buy", P)
    sell = order_cost(100_000, "sell", P)
    # STT 100 + stamp 15 + brokerage 20 + GST + small fees ~ 139
    assert 135 < buy < 145 and 130 < sell < 145


def synthetic_market(n_stocks=30, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", "2021-12-31")
    rows, nifty = [], 1000 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates)))
    for k in range(n_stocks):
        drift = rng.uniform(-0.0002, 0.0012)
        c = 100 * np.cumprod(1 + rng.normal(drift, 0.018, len(dates)))
        o = c * (1 + rng.normal(0, 0.004, len(dates)))
        h = np.maximum(c, o) * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
        l = np.minimum(c, o) * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
        v = rng.lognormal(12, 0.5, len(dates))
        frame = pd.DataFrame({"date": dates, "symbol": f"S{k:02d}", "series": "EQ", "open": o,
                              "high": h, "low": l, "close": c, "volume": v})
        frame["prev_close"] = frame["close"].shift(1).fillna(frame["close"])
        frame["value"] = 2e8 * (1 + k)
        rows.append(frame)
    prices = pd.concat(rows, ignore_index=True)
    return prices, pd.Series(nifty, index=dates)


def test_pipeline_and_simulation():
    from strategy.data import prepare_prices
    raw, nifty = synthetic_market()
    prices = prepare_prices(raw)
    f = features.add_stock_indicators(prices, nifty)
    f, regime, diag = features.add_layers(f, nifty, {f"S{k:02d}": f"IND{k % 5}" for k in range(30)}, P)
    assert set(regime.unique()) <= {"ON", "CAUTION", "OFF"}
    assert f["in_universe"].any() and f["layer3"].any()
    assert diag["industry_coverage_pct"] == 100.0

    # Relax the pattern rules so random data produces signals, to exercise the simulator.
    loose = Params(contraction_ratio=1.5, final_contraction_max=0.25, dryup_ratio=5.0,
                   upper_base=0.5, breakout_volume=0.0, base_depth_min=0.0)
    signals = vcp.all_signals(f, loose)
    assert len(signals) > 20, f"only {len(signals)} signals"
    book = Book(f, signals["symbol"].unique())
    cal = nifty.index[(nifty.index >= "2010-01-01") & (nifty.index <= "2019-12-31")]
    for run in [Run("A0", False, False, ""), Run("A2", True, True, "")]:
        res = simulate(signals[signals["date"].between(cal[0], cal[-1])], book, cal, regime, run, loose)
        t, eq = res["trades"], res["equity"]
        assert len(t) > 0
        # Accounting: final cash equals start + all trade P&L - taxes
        assert abs(eq["equity"].iloc[-1] - (P.starting_capital + t["net_pnl"].sum() - res["taxes_paid"])) < 1.0
        assert (eq["positions"] <= P.max_positions).all()
        assert t["risk_pct"].max() <= P.risk_full + 1e-12
        assert (t.groupby("entry_date").size() <= P.max_new_per_day).all()
        print(run.name, len(t), "trades, exits:", t["exit_reason"].value_counts().to_dict())




def test_exit_path():
    """Entry 100 (stop 95, R=5) -> high hits 112.5 (2.5R): sell 1/3, stop to breakeven -> falls to 99: breakeven stop."""
    dates = pd.bdate_range("2015-01-01", periods=12)
    o = [100, 101, 104, 108, 111, 110, 106, 102, 100.5, 99, 98, 98]
    h = [101, 104, 108, 111, 113, 111, 107, 103, 101, 100, 99, 99]
    l = [99.5, 100.5, 103, 107, 110, 105, 102, 100.5, 99.8, 98.5, 97, 97]
    c = [100.5, 103, 107, 110, 111, 106, 103, 101, 100, 99, 98, 98]
    f = pd.DataFrame({"date": dates, "symbol": "X", "adj_open": o, "adj_high": h, "adj_low": l,
                      "adj_close": c, "adj_volume": 1e5, "dma10": 90.0, "dma20": 90.0, "dma50": 80.0,
                      "avgvol50": 1e5, "close_prev15": np.nan})
    signal = pd.DataFrame([{"date": dates[0] - pd.Timedelta(days=3), "symbol": "X", "pivot": 99.0,
                            "final_low": 95 / 0.995 / (1 - 0), "rs_points": 3, "rs_pct": 90.0,
                            "industry": "I"}])
    cal = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=3)]).append(dates)
    book = Book(f, ["X"])
    zero_cost = Params(slippage=0.0, brokerage_per_order=0.0, stt=0.0, exchange_fee=0.0, sebi_fee=0.0,
                       stamp_duty_buy=0.0, dp_charge_sell=0.0, stcg_tax=0.0)
    res = simulate(signal, book, cal, pd.Series("ON", index=cal), Run("A0", False, False, ""), zero_cost)
    t = res["trades"].iloc[0]
    assert t["path"] == "partial_2.5R+breakeven_stop", t["path"]
    # 1/3 sold at 112.5 (+12.5 each), 2/3 at 100 (0): net = shares/3 * 12.5
    assert abs(t["net_pnl"] - (t["shares"] // 3) * 12.5) < 1e-6
    assert t["shares"] == 2000  # 1% of 10 lakh / R of 5


def test_zigzag_setup():
    from dataclasses import replace
    high, low, close = vcp_path()
    s = len(close) - 5                      # the last day of the tight range, before the breakout
    zz = replace(P, swing_method="zigzag")
    setup = vcp.find_setup(high, low, 60.0, 100.0, s, zz)
    assert setup is not None, "zigzag should see the contracting base"
    assert all(b <= 0.7 * a for a, b in zip(setup.contractions, setup.contractions[1:]))
    # A wiggle inside the first pullback (+1.5%) must not create an extra contraction.
    h2, l2 = high.copy(), low.copy()
    j = 42
    h2[j] *= 1.015
    assert vcp.find_setup(h2, l2, 60.0, 100.0, s, zz) is not None
    # After the breakout above the last swing high there is no pre-breakout setup.
    assert vcp.find_setup(high, low, 60.0, 100.0, len(close) - 1, zz) is None


def test_buy_stop_fills():
    from dataclasses import replace
    dates = pd.bdate_range("2016-01-01", periods=6)
    def run(o, h, l):
        f = pd.DataFrame({"date": dates, "symbol": "Y", "adj_open": o, "adj_high": h, "adj_low": l,
                          "adj_close": [x * 0.999 for x in h], "adj_volume": 1e5, "dma10": 50.0,
                          "dma20": 50.0, "dma50": 50.0, "avgvol50": 1e5, "close_prev15": np.nan})
        order = pd.DataFrame([{"date": dates[0], "symbol": "Y", "pivot": 100.0, "final_low": 95.0,
                               "rs_points": 3, "rs_pct": 90.0, "industry": "I"}])
        zp = replace(P, entry_mode="buy_stop", slippage=0.0)
        res = simulate(order, Book(f, ["Y"]), pd.DatetimeIndex(dates), pd.Series("ON", index=dates),
                       Run("A0", False, False, ""), zp)
        return res["trades"]
    # Opens below the pivot, trades through it: filled exactly at the pivot.
    t = run([99, 99, 100, 101, 102, 103], [99.5, 101, 102, 103, 104, 105], [98, 98.5, 99.5, 100.5, 101, 102])
    assert len(t) == 1 and abs(t.iloc[0]["entry"] - 100.0) < 1e-9
    # Gaps to 103 (above the 102 limit) and never trades back to 102: not filled.
    t = run([99, 103, 104, 105, 106, 107], [99.5, 104, 105, 106, 107, 108], [98, 102.5, 103, 104, 105, 106])
    assert len(t) == 0
    # Gaps to 103 but dips to 101.5 later that day: filled at the 102 limit.
    t = run([99, 103, 103, 104, 105, 106], [99.5, 104, 104, 105, 106, 107], [98, 101.5, 102.5, 103, 104, 105])
    assert len(t) == 1 and abs(t.iloc[0]["entry"] - 102.0) < 1e-9


def test_v12_pipeline_accounting():
    from dataclasses import replace
    from strategy.config import V12
    from strategy.data import prepare_prices
    raw, nifty = synthetic_market(seed=5)
    f = features.add_stock_indicators(prepare_prices(raw), nifty)
    f, regime, _ = features.add_layers(f, nifty, {}, V12)
    loose = replace(V12, contraction_ratio=1.5, final_contraction_max=0.25, dryup_ratio=5.0,
                    upper_base=0.5, base_depth_min=0.0)
    orders = vcp.all_signals(f, loose)
    assert len(orders) > 20
    cal = nifty.index[(nifty.index >= "2010-01-01") & (nifty.index <= "2019-12-31")]
    res = simulate(orders[orders["date"].between(cal[0], cal[-1])], Book(f, orders["symbol"].unique()),
                   cal, regime, Run("A1", True, False, ""), loose)
    t, eq = res["trades"], res["equity"]
    assert len(t) > 0
    assert abs(eq["equity"].iloc[-1] - (P.starting_capital + t["net_pnl"].sum() - res["taxes_paid"])) < 1.0
    assert (t.groupby("entry_date").size() <= P.max_new_per_day).all()
    assert (t["entry"] >= 0).all()
    print("v1.2 synthetic:", len(t), "trades,", t["exit_reason"].value_counts().to_dict())


if __name__ == "__main__":
    test_find_setup(); test_costs(); test_pipeline_and_simulation(); test_exit_path()
    test_zigzag_setup(); test_buy_stop_fills(); test_v12_pipeline_accounting()
    print("All strategy tests passed")
