"""Offline tests for the 'why this stock' explanations."""
import pandas as pd

from strategy import explain, features, momentum
from strategy.config import V12
from strategy.data import prepare_prices


def test_sell_line_notes():
    assert "Safe" in explain.sell_line_note(12, held=True)
    assert "3 more places" in explain.sell_line_note(38, held=True)
    assert "would be sold" in explain.sell_line_note(45, held=True)
    assert "bought at the month-end" in explain.sell_line_note(5, held=False)
    assert "Not eligible" in explain.sell_line_note(None, held=False)


def test_build_all_on_synthetic_market():
    import tests.test_strategy as T
    raw, nifty = T.synthetic_market(n_stocks=50, seed=4)
    f = features.add_stock_indicators(prepare_prices(raw), nifty)
    f, regime, _ = features.add_layers(f, nifty, {f"S{k:02d}": f"IND{k % 5}" for k in range(50)}, V12)
    f = momentum.add_volatility(f)
    day = f["date"].max()
    today = momentum.rank_table(f, pd.DatetimeIndex([day])).reset_index(drop=True)
    today["rank"] = range(1, len(today) + 1)
    month_ends = [d for d in momentum.month_end_dates(nifty.index) if d < day][-6:]
    tables = {d: g for d, g in momentum.rank_table(f, pd.DatetimeIndex(month_ends)).groupby("date")}
    held = {today["symbol"].iloc[0], "S49"}
    out = explain.build_all(f, today, tables, str(regime.get(day)), V12, held, set(), top_n=10)
    top = out[today["symbol"].iloc[0]]
    assert top["rank"] == 1 and top["held"] and "ranks 1 among" in top["why"]
    assert len(top["rank_history"]) == 7 and top["rank_history"][-1] == {"date": "today", "rank": 1}
    assert [l["layer"] for l in top["layers"]] == [1, 2, 3, 4, 5, 6, 7]
    assert top["layers_checkable"] == 5 and 0 <= top["layers_passed"] <= 5
    assert {l["status"] for l in top["layers"] if l["layer"] in (5, 6)} == {"no_data"}
    assert top["eligible"] and all(top["eligibility"].values())
    assert len(top["layers"][2]["checks"]) == 8          # the trend template's 8 conditions
    print("example:", top["why"], "|", top["sell_line"], "| layers", top["layers_passed"], "/", top["layers_checkable"])


if __name__ == "__main__":
    test_sell_line_notes(); test_build_all_on_synthetic_market()
    print("All explanation tests passed")
