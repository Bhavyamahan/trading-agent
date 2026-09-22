"""Offline tests for the dashboard (no network)."""
import datetime as dt
import os

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "x")
os.environ["DASHBOARD_PASSWORD"] = "secret"

from dashboard import app as dash  # noqa: E402

RANKING = [{"symbol": f"S{i}", "rank": str(i + 1), "score": str(3 - i * 0.1), "industry": "CAPITAL GOODS"} for i in range(45)]


def state(with_positions=True, pending=False):
    s = {"starting_capital": 1_000_000, "cash": 20_000.0, "start_date": "2026-10-01" if with_positions else None,
         "nifty_at_start": 22000.0 if with_positions else None, "positions": {}, "pending": [], "closed": [],
         "snapshots": [{"date": "2026-10-01", "equity": 1_000_000, "nifty500": 22000.0},
                       {"date": "2026-10-02", "equity": 1_031_500, "nifty500": 22220.0}] if with_positions else []}
    if with_positions:
        s["positions"] = {"S0": {"shares": 100, "entry_date": "2026-10-01", "last_price": 520.0, "cost": 50_000.0},
                          "S35": {"shares": 50, "entry_date": "2026-10-01", "last_price": 900.0, "cost": 50_000.0},
                          "OLD": {"shares": 10, "entry_date": "2026-10-01", "last_price": 4_000.0, "cost": 50_000.0}}
        s["closed"] = [{"symbol": "X", "entry_date": "2026-09-01", "exit_date": "2026-10-01", "net_pnl": -500.0, "return_pct": -1.0}]
    if pending:
        s["pending"] = [{"symbol": "OLD", "side": "sell", "reason": "outside top 40"},
                        {"symbol": "S1", "side": "buy", "target_value": 51575.0, "reason": "rank 2"}]
    return s


def client_with(st):
    dash.load_state = lambda: st
    dash.load_ranking = lambda: ("2026-10-02", RANKING)
    dash.last_run = lambda: {"run_at": "2026-10-02T14:40:00", "status": "ok"}
    dash._cache.clear()
    c = dash.app.test_client()
    return c


def test_inr():
    assert dash.inr(1_000_000) == "₹10,00,000"
    assert dash.inr(51575) == "₹51,575" and dash.inr(999) == "₹999" and dash.inr(-1234567) == "-₹12,34,567"
    assert dash.inr(123456789) == "₹12,34,56,789"


def test_login_required():
    c = client_with(state())
    r = c.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert "not right" in c.post("/login", data={"password": "wrong"}).get_data(as_text=True)
    assert c.post("/login", data={"password": "secret"}).status_code == 302
    assert c.get("/").status_code == 200


def render(st):
    c = client_with(st)
    c.post("/login", data={"password": "secret"})
    return c.get("/").get_data(as_text=True)


def test_pages():
    waiting = render(state(with_positions=False))
    assert "starts at the first month-end" in waiting and "S0" in waiting        # ranking still shown
    holding = render(state())
    assert "₹10,31,500" in holding and "+3.2%" in holding and "+1.0%" in holding  # value, return, Nifty
    assert "Near the cut-off" in holding and "Would be sold" in holding     # S35 rank 36, OLD unranked
    assert "perf" in holding                                                      # chart present
    orders = render(state(pending=True))
    assert "Rebalance orders are ready" in orders and "₹51,575" in orders and "All shares" in orders


def test_month_strip():
    s = dash.month_strip(dt.date(2026, 10, 12))
    assert s["rebalance"] == dt.date(2026, 10, 30) and s["left"] == 14 and len(s["days"]) == 22


if __name__ == "__main__":
    test_inr(); test_login_required(); test_pages(); test_month_strip()
    print("All dashboard tests passed")
