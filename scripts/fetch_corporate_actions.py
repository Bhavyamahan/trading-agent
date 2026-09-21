"""Download NSE's official corporate-action list (2008 to today) into Supabase.

Saves reference/corporate_actions_nse.parquet (raw NSE fields) and prints how many
splits, bonuses and demergers were found per year.

If NSE blocks the requests, the job fails with instructions for the manual route:
nseindia.com > Companies > Corporate Actions > Equity > date range > Download (CSV),
then upload the CSVs to data/corporate_actions/ in the repo.
"""
import datetime as dt
import sys
import time

import pandas as pd
import requests

from pipeline import nse, storage
from pipeline.corporate_actions import events_from_table

HOME = "https://www.nseindia.com/"
PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
API = ("https://www.nseindia.com/api/corporates-corporateActions?index=equities"
       "&from_date={start}&to_date={end}")


def new_session() -> requests.Session:
    session = nse.new_session()
    session.headers.update({"Accept": "application/json, text/plain, */*", "Referer": PAGE})
    for url in (HOME, PAGE):  # these visits set the cookies the API expects
        try:
            session.get(url, timeout=30)
        except requests.RequestException:
            pass
        time.sleep(1)
    return session


def fetch_range(session, start: dt.date, end: dt.date) -> list | None:
    url = API.format(start=start.strftime("%d-%m-%Y"), end=end.strftime("%d-%m-%Y"))
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else data.get("data", [])
            print(f"  {start}..{end}: HTTP {response.status_code}, retrying with fresh cookies")
        except (requests.RequestException, ValueError) as error:
            print(f"  {start}..{end}: {type(error).__name__}, retrying")
        time.sleep(5 * (attempt + 1))
        session = new_session()
    return None


def main() -> None:
    session = new_session()
    today = dt.date.today()
    rows, failed = [], []
    for year in range(2008, today.year + 1):
        chunks = []
        for m in (1, 4, 7, 10):
            start = dt.date(year, m, 1)
            if start > today:
                break
            next_quarter = dt.date(year + 1, 1, 1) if m == 10 else dt.date(year, m + 3, 1)
            chunks.append((start, min(next_quarter - dt.timedelta(days=1), today)))
        for start, end in chunks:
            data = fetch_range(session, start, end)
            if data is None:
                failed.append(f"{start}..{end}")
                continue
            rows.extend(data)
            time.sleep(1.0)
        print(f"{year}: {len(rows):,} rows so far", flush=True)

    if not rows:
        storage.log_run("fetch_corporate_actions", "blocked", {"failed": failed[:10]})
        print("\nSTOPPED: NSE did not return any data (probably blocking automated requests).\n"
              "Manual route: nseindia.com > Companies > Corporate Actions > Equity, choose a date\n"
              "range, click Download (CSV), and upload the files to data/corporate_actions/.")
        sys.exit(1)

    table = pd.DataFrame(rows).astype(str)
    table = table.drop_duplicates()
    storage.save_parquet(table, "reference/corporate_actions_nse.parquet")
    events = events_from_table(table)
    per_year = events.groupby([events["ex_date"].dt.year, "kind"]).size().unstack(fill_value=0)
    print("\nEvents that change prices, by year:\n" + per_year.to_string())
    storage.log_run("fetch_corporate_actions", "ok" if not failed else "partial",
                    {"rows": len(table), "events": len(events), "failed_chunks": failed})
    if failed:
        print(f"\nWARNING: {len(failed)} date ranges failed: {failed}. Re-run to fill them in.")
    print(f"\nSaved {len(table):,} rows; {len(events):,} split/bonus/demerger events.")


if __name__ == "__main__":
    main()
