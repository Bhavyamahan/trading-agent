"""Check that GitHub Actions can reach NSE and Supabase before the big backfill.

Run it from the Actions tab ("Probe data sources"). Every line should say OK.
"""
import datetime as dt
import sys

import pandas as pd

from pipeline import nse, storage

failures = []


def check(label, func):
    try:
        result = func()
        print(f"OK    {label}: {result}")
    except Exception as error:
        print(f"FAIL  {label}: {type(error).__name__}: {error}")
        failures.append(label)


def recent_weekday_rows():
    session = nse.new_session()
    day = dt.date.today()
    for _ in range(10):
        day -= dt.timedelta(days=1)
        try:
            frame = nse.fetch_bhavcopy(day, session)
            return f"{day} -> {len(frame)} stocks (new UDiFF format)"
        except nse.NotTradingDay:
            continue
    raise RuntimeError("no bhavcopy found in the last 10 days")


def legacy_rows():
    frame = nse.fetch_bhavcopy(dt.date(2019, 1, 2), nse.new_session())
    return f"2019-01-02 -> {len(frame)} stocks (legacy format)"


def index_rows():
    frame = nse.fetch_indices(dt.date(2019, 1, 2), nse.new_session())
    has_500 = "NIFTY 500" in set(frame["index_name"])
    return f"2019-01-02 -> {len(frame)} indices, NIFTY 500 present: {has_500}"


def supabase_roundtrip():
    test = pd.DataFrame({"x": [1, 2, 3]})
    storage.save_parquet(test, "probe/test.parquet")
    back = storage.load_parquet("probe/test.parquet")
    assert back is not None and back["x"].tolist() == [1, 2, 3]
    return "upload + download worked"


check("NSE recent bhavcopy", recent_weekday_rows)
check("NSE legacy bhavcopy", legacy_rows)
check("NSE index closes", index_rows)
check("Supabase storage", supabase_roundtrip)
storage.log_run("probe", "failed" if failures else "ok", {"failures": failures})

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nAll checks passed. You can run the backfill now.")
