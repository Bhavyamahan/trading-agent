"""Private dashboard for the momentum agent (Rulebook v2.0).

Reads what the nightly job writes to Supabase Storage (live/paper_state.json and
live/rankings/<date>.csv) plus the run log. Read-only; protected by one password.

Environment (set in Render): SUPABASE_URL, SUPABASE_SERVICE_KEY,
DASHBOARD_PASSWORD, SECRET_KEY.
"""
import calendar
import csv
import datetime as dt
import io
import json
import os
import time
from functools import wraps

import requests
from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("RENDER") is not None,
                  PERMANENT_SESSION_LIFETIME=dt.timedelta(days=30))
BUCKET = "market-data"
_cache: dict = {}


# ---------- Supabase access (plain HTTP, cached for a minute) ----------
def _headers():
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return {"Authorization": f"Bearer {key}", "apikey": key}


def _base():
    return os.environ["SUPABASE_URL"].rstrip("/")


def cached(key, loader, seconds=60):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < seconds:
        return hit[1]
    value = loader()
    _cache[key] = (time.time(), value)
    return value


def storage_get(path):
    r = requests.get(f"{_base()}/storage/v1/object/{BUCKET}/{path}", headers=_headers(), timeout=20)
    return r.content if r.status_code == 200 else None


def storage_list(prefix):
    r = requests.post(f"{_base()}/storage/v1/object/list/{BUCKET}", headers=_headers(),
                      json={"prefix": prefix, "limit": 1000, "offset": 0}, timeout=20)
    return [i["name"] for i in r.json() if i.get("id")] if r.status_code == 200 else []


def last_run():
    r = requests.get(f"{_base()}/rest/v1/pipeline_runs",
                     params={"job": "eq.nightly", "order": "run_at.desc", "limit": "1"},
                     headers=_headers(), timeout=20)
    rows = r.json() if r.status_code == 200 else []
    return rows[0] if rows else None


def load_state():
    raw = cached("state", lambda: storage_get("live/paper_state.json"))
    return json.loads(raw) if raw else None


def load_ranking():
    def loader():
        names = sorted(n for n in storage_list("live/rankings") if n.endswith(".csv"))
        if not names:
            return None, []
        raw = storage_get(f"live/rankings/{names[-1]}")
        rows = list(csv.DictReader(io.StringIO(raw.decode()))) if raw else []
        return names[-1][:-4], rows
    return cached("ranking", loader)


# ---------- formatting ----------
def inr(value, decimals=0):
    """Indian digit grouping: 1000000 -> 10,00,000."""
    if value is None:
        return "-"
    negative, value = value < 0, abs(value)
    whole, frac = f"{value:.{decimals}f}".split(".") if decimals else (f"{value:.0f}", "")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail]) if groups else tail
    return ("-" if negative else "") + "₹" + text + (f".{frac}" if frac else "")


def pct(value, signed=True):
    return "-" if value is None else (f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%")


def short_date(text):
    try:
        d = dt.date.fromisoformat(str(text)[:10])
        return f"{d.day} {d.strftime('%b')}" + ("" if d.year == dt.date.today().year else f" {d.year}")
    except ValueError:
        return text


app.jinja_env.filters.update(inr=inr, pct=pct, short=short_date)


# ---------- the month rhythm ----------
def month_strip(today: dt.date):
    """Weekdays of the current month; the last one is the (approximate) rebalance day."""
    _, days = calendar.monthrange(today.year, today.month)
    weekdays = [dt.date(today.year, today.month, d) for d in range(1, days + 1)
                if dt.date(today.year, today.month, d).weekday() < 5]
    rebalance = weekdays[-1]
    left = sum(1 for d in weekdays if today < d <= rebalance)
    return {"days": weekdays, "today": today, "rebalance": rebalance, "left": left,
            "month": today.strftime("%B")}


def build_view(state, ranking_date, ranking, today):
    ranks = {r["symbol"]: int(r["rank"]) for r in ranking}
    view = {"strip": month_strip(today), "ranking_date": ranking_date, "top": ranking[:20],
            "state": state, "holdings": [], "orders": [], "closed": [], "chart": None,
            "equity": None, "ret": None, "bench": None, "drawdown": None}
    if not state:
        return view
    snaps = state.get("snapshots", [])
    if snaps:
        last = snaps[-1]
        view["equity"] = last["equity"]
        view["ret"] = last["equity"] / state["starting_capital"] - 1
        start_nifty = state.get("nifty_at_start")
        if start_nifty and last.get("nifty500"):
            view["bench"] = last["nifty500"] / start_nifty - 1
        peak = max(s["equity"] for s in snaps)
        view["drawdown"] = last["equity"] / peak - 1 if peak else None
        started = [s for s in snaps if state.get("start_date") and s["date"] >= state["start_date"]]
        if len(started) >= 2 and start_nifty:
            view["chart"] = {"labels": [s["date"] for s in started],
                             "strategy": [round((s["equity"] / state["starting_capital"] - 1) * 100, 2) for s in started],
                             "nifty": [round((s["nifty500"] / start_nifty - 1) * 100, 2) if s.get("nifty500") else None
                                       for s in started]}
    for sym, pos in sorted(state.get("positions", {}).items()):
        rank = ranks.get(sym)
        status = ("keep" if rank and rank <= 30 else "watch" if rank and rank <= 40 else "sell")
        view["holdings"].append({"symbol": sym, "shares": pos["shares"], "since": pos["entry_date"],
                                 "value": pos["shares"] * pos["last_price"],
                                 "pnl": pos["shares"] * pos["last_price"] / pos["cost"] - 1,
                                 "rank": rank, "status": status})
    view["orders"] = state.get("pending", [])
    view["closed"] = list(reversed(state.get("closed", [])))[:30]
    return view


# ---------- routes ----------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("ok"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        expected = os.environ.get("DASHBOARD_PASSWORD")
        if expected and request.form.get("password") == expected:
            session.permanent = True
            session["ok"] = True
            return redirect(request.args.get("next") or url_for("home"))
        error = "That password is not right. Check DASHBOARD_PASSWORD in Render."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    today = (dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).date()
    problems = []
    try:
        state = load_state()
        ranking_date, ranking = load_ranking()
        run = cached("run", last_run)
    except Exception as error:  # show the problem instead of a blank page
        state, ranking_date, ranking, run = None, None, [], None
        problems.append(f"Could not reach Supabase ({type(error).__name__}). Check SUPABASE_URL and SUPABASE_SERVICE_KEY in Render.")
    view = build_view(state, ranking_date, ranking, today)
    return render_template("index.html", v=view, run=run, problems=problems)


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
