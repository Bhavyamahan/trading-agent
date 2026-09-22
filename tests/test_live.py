"""Offline tests for the live paper portfolio and the nightly data updater."""
import datetime as dt
from dataclasses import replace

import pandas as pd

from strategy import live
from strategy.config import V12

ZERO = replace(V12, slippage=0.0, brokerage_per_order=0.0, stt=0.0, exchange_fee=0.0, sebi_fee=0.0,
               stamp_duty_buy=0.0, dp_charge_sell=0.0)
D = dt.date


def test_month_end_with_holiday():
    assert live.is_month_end(D(2026, 9, 30), set())                      # Wed 30 Sep
    assert not live.is_month_end(D(2026, 9, 29), set())
    assert live.is_month_end(D(2026, 10, 29), {D(2026, 10, 30)})         # Fri 30 Oct is a holiday
    assert live.is_month_end(D(2026, 1, 30), set())                      # Fri 30 Jan, weekend follows


def test_full_cycle():
    s = live.new_state(1_000_000, D(2026, 9, 1))
    ranking = [f"S{i}" for i in range(60)]
    # Month-end evening: plan 20 buys
    r1 = live.process_day(s, D(2026, 9, 30), {}, {}, {}, {}, 1000.0, ranking, set(), ZERO)
    assert r1["month_end"] and len(r1["orders"]) == 20 and all(o["side"] == "buy" for o in r1["orders"])
    # Next evening: buys fill at the open (S5 did not trade: stays pending)
    opens = {f"S{i}": 100.0 for i in range(60) if i != 5}
    r2 = live.process_day(s, D(2026, 10, 1), opens, {k: 110.0 for k in opens}, {}, {}, 1010.0, ranking, set(), ZERO)
    assert len(s["positions"]) == 19 and [o["symbol"] for o in s["pending"]] == ["S5"]
    assert s["positions"]["S0"]["shares"] == 500                       # Rs 50,000 / Rs 100
    assert abs(r2["snapshot"]["equity"] - (1_000_000 + 19 * 500 * 10)) < 1
    # Same evening again: nothing happens twice
    assert live.process_day(s, D(2026, 10, 1), opens, {}, {}, {}, 1010.0, ranking, set(), ZERO)["skipped"]
    # S5 fills the next day
    live.process_day(s, D(2026, 10, 5), {"S5": 100.0}, {"S5": 101.0}, {}, {}, 1011.0, ranking, set(), ZERO)
    assert len(s["positions"]) == 20 and not s["pending"]
    # Next month-end: S0-S9 fall out of the top 40 -> 10 sells, 10 buys from the new top
    new_rank = [f"S{i}" for i in range(10, 60)] + [f"S{i}" for i in range(10)]
    r3 = live.process_day(s, D(2026, 10, 30), {}, {}, {}, {}, 1020.0, new_rank, set(), ZERO)
    sells = [o["symbol"] for o in r3["orders"] if o["side"] == "sell"]
    buys = [o["symbol"] for o in r3["orders"] if o["side"] == "buy"]
    assert sorted(sells) == sorted(f"S{i}" for i in range(10)) and buys == [f"S{i}" for i in range(20, 30)]
    opens = {f"S{i}": 120.0 for i in range(60)}
    live.process_day(s, D(2026, 11, 2), opens, opens, {}, {}, 1030.0, new_rank, set(), ZERO)
    assert len(s["positions"]) == 20 and len(s["closed"]) == 10
    assert all(abs(c["return_pct"] - 20.0) < 1e-6 for c in s["closed"])


def test_split_and_demerger_while_holding():
    s = live.new_state(1_000_000, D(2026, 1, 1))
    s["positions"] = {"SPL": {"shares": 100, "entry_date": "2026-01-02", "entry_price": 500.0, "cost": 50_000.0,
                              "last_price": 520.0, "ca_applied": []},
                      "DEM": {"shares": 100, "entry_date": "2026-01-02", "entry_price": 200.0, "cost": 20_000.0,
                              "last_price": 210.0, "ca_applied": []}}
    cash0 = s["cash"]
    events = {"SPL": (0.2, "split"), "DEM": (0.7, "demerger")}
    live.process_day(s, D(2026, 2, 2), {}, {"SPL": 105.0, "DEM": 148.0}, {"DEM": 210.0}, events, 1.0, [], set(), ZERO)
    assert s["positions"]["SPL"]["shares"] == 500 and abs(s["positions"]["SPL"]["entry_price"] - 100.0) < 1e-9
    assert abs(s["cash"] - (cash0 + 100 * 210.0 * 0.3)) < 1e-6      # spun-off value credited
    # Re-running the same corporate action must not double it
    live.apply_corporate_actions(s, events, D(2026, 2, 2), {"DEM": 210.0})
    assert s["positions"]["SPL"]["shares"] == 500


def test_daily_update_helpers():
    from scripts import daily_update as du
    from pipeline import nse
    days = [D(2026, 9, 21), D(2026, 9, 22)]
    def bhav(day, session):
        if day.weekday() >= 5 or day not in days:
            raise nse.NotTradingDay(str(day))
        return pd.DataFrame({"date": [pd.Timestamp(day)], "symbol": ["X"], "series": ["EQ"], "close": [1.0]})
    def idx(day, session):
        if day == D(2026, 9, 22):
            raise nse.NotTradingDay("index file missing")
        return pd.DataFrame({"date": [pd.Timestamp(day)], "index_name": ["NIFTY 500"], "open": [1.0],
                             "high": [1.0], "low": [1.0], "close": [22000.0]})
    eq, ix, got = du.fetch_new_sessions(D(2026, 9, 19), D(2026, 9, 22), None, bhav, idx)
    assert got == days and len(eq) == 2
    filled, carried = du.fill_index_gaps(pd.concat(ix), got, 21900.0)
    assert carried == ["2026-09-22"]
    assert filled.set_index("date").loc[pd.Timestamp("2026-09-22"), "close"] == 22000.0


if __name__ == "__main__":
    test_month_end_with_holiday(); test_full_cycle(); test_split_and_demerger_while_holding()
    test_daily_update_helpers()
    print("All live tests passed")
