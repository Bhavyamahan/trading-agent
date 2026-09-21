# Trading Agent — Step 3a: Data pipeline

Free, official NSE end-of-day data (bhavcopy + index closes), stored in Supabase.
Implements the data rules of *Multi-Layer Swing Strategy — Rulebook v1.0*.

## What's here

| File | Purpose |
| --- | --- |
| `pipeline/nse.py` | Downloads and parses NSE bhavcopy (legacy and UDiFF formats) and index closes |
| `pipeline/adjust.py` | Adjusts prices for splits/bonuses using NSE's official previous close |
| `pipeline/storage.py` | Saves/loads parquet files in Supabase Storage, writes the run log |
| `scripts/probe.py` | Checks that GitHub can reach NSE and Supabase |
| `scripts/backfill.py` | Downloads history year by year (resumable) |
| `supabase/setup.sql` | Creates the storage bucket and run-log table |
| `tests/test_offline.py` | Offline tests for parsing and adjustment |

## Setup (no terminal needed)

1. **Supabase:** create a new project. Open *SQL Editor → New query*, paste all of
   `supabase/setup.sql`, click *Run*.
2. **Keys:** in Supabase *Project Settings → API*, copy the *Project URL* and the
   *service_role* key. Never put the service_role key in any frontend code.
3. **GitHub:** create a **private** repo, then *Add file → Upload files* and drag in
   everything from this folder (keep the folder structure, including `.github`).
4. **Secrets:** repo *Settings → Secrets and variables → Actions → New repository secret*:
   - `SUPABASE_URL` = the Project URL
   - `SUPABASE_SERVICE_KEY` = the service_role key
5. **Probe:** *Actions → "1 - Probe data sources" → Run workflow*. All four lines must say OK.
6. **Backfill:** *Actions → "2 - Backfill history" → Run workflow* (defaults 2008–2026).
   Expect roughly 2–3 hours. If it stops, just run it again: finished years are skipped.

## Files produced in the `market-data` bucket

- `equity/<year>.parquet` — raw daily prices of every EQ/BE/BZ stock
- `indices/<year>.parquet` — daily closes of every NSE index, including NIFTY 500 and sector indices

Prices are stored raw; `adjust.add_adjusted_prices()` is applied when data is loaded.
History starts in 2008 so the indicators (200-DMA, 52-week range) are ready by 2010.
