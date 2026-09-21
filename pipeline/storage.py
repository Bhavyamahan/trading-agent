"""Supabase Storage (parquet files) and run-log helpers, using plain HTTP calls.

Needs two environment variables (set as GitHub Actions secrets):
  SUPABASE_URL          e.g. https://abcdxyz.supabase.co
  SUPABASE_SERVICE_KEY  the service_role key (keep secret, never put it in a frontend)
"""
import io
import os

import pandas as pd
import requests

BUCKET = os.environ.get("SUPABASE_BUCKET", "market-data")


def _base() -> str:
    return os.environ["SUPABASE_URL"].rstrip("/")


def _headers(extra: dict | None = None) -> dict:
    key = os.environ["SUPABASE_SERVICE_KEY"]
    headers = {"Authorization": f"Bearer {key}", "apikey": key}
    if extra:
        headers.update(extra)
    return headers


def upload_bytes(path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    response = requests.post(
        f"{_base()}/storage/v1/object/{BUCKET}/{path}",
        headers=_headers({"Content-Type": content_type, "x-upsert": "true"}),
        data=data, timeout=180)
    response.raise_for_status()


def download_bytes(path: str) -> bytes | None:
    response = requests.get(f"{_base()}/storage/v1/object/{BUCKET}/{path}",
                            headers=_headers(), timeout=180)
    if response.status_code in (400, 404):
        return None
    response.raise_for_status()
    return response.content


def list_files(prefix: str) -> list[str]:
    response = requests.post(
        f"{_base()}/storage/v1/object/list/{BUCKET}",
        headers=_headers({"Content-Type": "application/json"}),
        json={"prefix": prefix, "limit": 1000, "offset": 0}, timeout=60)
    response.raise_for_status()
    return [item["name"] for item in response.json() if item.get("id")]


def save_parquet(frame: pd.DataFrame, path: str) -> int:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    data = buffer.getvalue()
    upload_bytes(path, data)
    return len(data)


def load_parquet(path: str) -> pd.DataFrame | None:
    data = download_bytes(path)
    return None if data is None else pd.read_parquet(io.BytesIO(data))


def log_run(job: str, status: str, details: dict) -> None:
    """Write one row to public.pipeline_runs. Never crashes the pipeline."""
    try:
        requests.post(f"{_base()}/rest/v1/pipeline_runs",
                      headers=_headers({"Content-Type": "application/json",
                                        "Prefer": "return=minimal"}),
                      json={"job": job, "status": status, "details": details},
                      timeout=30).raise_for_status()
    except Exception as error:  # logging must not break data jobs
        print(f"[warn] could not write run log: {error}")
