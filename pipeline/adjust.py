"""Adjust raw bhavcopy prices for splits, bonuses and similar corporate actions.

Priority order:
0. official: NSE's corporate-action list (exact bonus/split ratios; demergers
   measured from the ex-date open). Each event is matched to the price data
   within +/-3 sessions of its ex-date, where the price gap confirms it.
Then, for days without an official event, two fallback methods:

1. prev_close: on an ex-date, NSE's legacy files report an already-adjusted
   "previous close", so  factor = prev_close(t) / close(t-1)  differs from 1.
2. gap_ratio: NSE's newer UDiFF files (from July 2024) do not always adjust the
   previous close. So an overnight move beyond -40% / +80% whose open-to-previous-
   close ratio matches a standard split/bonus ratio (1/2, 1/5, 1/10, ...) within
   3% is also treated as a corporate action. Genuine crashes that match no
   standard ratio are left untouched (never hidden).

Every price before an event is multiplied by its factor (volume divided by it).
"""
import numpy as np
import pandas as pd

TOLERANCE = 0.005          # prev_close method: ratios within 0.5% of 1.0 are normal days
GAP_DOWN, GAP_UP = 0.60, 1.80
SNAP_TOLERANCE = 0.03
STANDARD_RATIOS = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 2 / 5, 3 / 5, 1 / 6, 1 / 8,
                            1 / 10, 1 / 20, 1 / 25, 1 / 50, 2, 3, 4, 5, 10])


def snap_ratio(ratio: np.ndarray) -> np.ndarray:
    """Nearest standard ratio if within 3%, else NaN."""
    ratio = np.asarray(ratio, dtype=float)
    rel = np.abs(ratio[:, None] / STANDARD_RATIOS[None, :] - 1)
    best = rel.argmin(axis=1)
    snapped = STANDARD_RATIOS[best]
    return np.where(rel[np.arange(len(ratio)), best] <= SNAP_TOLERANCE, snapped, np.nan)


OFFICIAL_WINDOW = 3          # sessions either side of the listed ex-date
OFFICIAL_MATCH = 0.10        # price gap must be within 10% of the official ratio
DEMERGER_MAX_GAP = 0.97      # a demerger is applied only if the open drops 3%+


def match_official(frame: pd.DataFrame, gap_open: np.ndarray, gap_close: np.ndarray,
                   official: pd.DataFrame | None) -> tuple[np.ndarray, list[dict]]:
    """Place each official event on the right row. Returns (factor per row, audit list)."""
    factor = np.full(len(frame), np.nan)
    audit = []
    if official is None or not len(official):
        return factor, audit
    positions = frame.groupby(frame["symbol"].astype(str), observed=True).indices
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    grouped = official.groupby(["symbol", "ex_date"])
    for (symbol, ex_date), events in grouped:
        rows = positions.get(symbol)
        if rows is None:
            continue
        sym_dates = dates[rows]
        nominal = int(np.searchsorted(sym_dates, np.datetime64(ex_date, "ns")))
        if nominal >= len(rows):
            continue
        known = [f for f in events["factor"] if f is not None and not pd.isna(f)]
        kinds = "+".join(sorted(set(events["kind"])))
        record = {"symbol": symbol, "ex_date": pd.Timestamp(ex_date).date(), "kind": kinds}
        if known:
            target = float(np.prod(known))
            best, best_err = None, np.inf
            for k in range(max(1, nominal - OFFICIAL_WINDOW), min(len(rows), nominal + OFFICIAL_WINDOW + 1)):
                row = rows[k]
                for r in (gap_open[row], gap_close[row]):
                    if np.isfinite(r) and r > 0:
                        err = abs(r / target - 1)
                        if err < best_err:
                            best, best_err = row, err
            if best is not None and best_err <= OFFICIAL_MATCH:
                factor[best] = target
                record.update(status="applied", factor=round(target, 5), date=pd.Timestamp(dates[best]).date())
            else:
                record.update(status="no_matching_price_gap", factor=round(target, 5))
        else:
            row = rows[nominal]
            r = gap_open[row]
            if nominal > 0 and np.isfinite(r) and 0.05 < r < DEMERGER_MAX_GAP:
                factor[row] = r
                record.update(status="applied", factor=round(float(r), 5), date=pd.Timestamp(dates[row]).date())
            else:
                record.update(status="no_price_drop", factor=None)
        audit.append(record)
    return factor, audit


def add_adjusted_prices(raw: pd.DataFrame, official: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    last_close = frame.groupby("symbol", observed=True)["close"].shift(1)
    open_col = frame["open"] if "open" in frame.columns else frame["close"]
    gap_open = (open_col / last_close).to_numpy(dtype=float)
    gap_close = (frame["close"] / last_close).to_numpy(dtype=float)

    official_factor, audit = match_official(frame, gap_open, gap_close, official)
    by_official = ~np.isnan(official_factor)

    ratio = frame["prev_close"] / last_close
    by_prev = (last_close.notna() & (frame["prev_close"] > 0)
               & ((ratio - 1).abs() > TOLERANCE)).to_numpy() & ~by_official

    has_prev = last_close.notna().to_numpy()
    moved = lambda g: (g < GAP_DOWN) | (g > GAP_UP)
    big_gap = ~by_official & ~by_prev & has_prev & (moved(gap_open) | moved(gap_close))
    snapped = np.full(len(frame), np.nan)
    if big_gap.any():
        snap_open, snap_close = snap_ratio(gap_open[big_gap]), snap_ratio(gap_close[big_gap])
        err_open = np.abs(gap_open[big_gap] / np.where(np.isnan(snap_open), 1, snap_open) - 1)
        err_close = np.abs(gap_close[big_gap] / np.where(np.isnan(snap_close), 1, snap_close) - 1)
        use_open = ~np.isnan(snap_open) & (np.isnan(snap_close) | (err_open <= err_close))
        snapped[big_gap] = np.where(use_open, snap_open, snap_close)
    by_gap = big_gap & ~np.isnan(snapped)

    factor = np.where(by_prev, ratio.to_numpy(dtype=float), 1.0)
    factor = np.where(by_gap, snapped, factor)
    factor = np.where(by_official, official_factor, factor)
    factor = pd.Series(factor, index=frame.index)

    # Adjustment for a row = product of the factors of all LATER rows of that stock.
    reversed_symbols = frame["symbol"].iloc[::-1]
    later_product = factor.iloc[::-1].groupby(reversed_symbols, observed=True).cumprod().iloc[::-1]
    frame["adj_factor"] = later_product / factor
    frame["ca_event"] = by_official | by_prev | by_gap
    frame["ca_method"] = np.where(by_official, "official",
                                  np.where(by_gap, "gap_ratio", np.where(by_prev, "prev_close", "")))
    frame["ca_factor"] = factor.to_numpy()
    frame["unexplained_gap"] = big_gap & np.isnan(snapped)
    for column in ["open", "high", "low", "close"]:
        if column in frame.columns:
            frame[f"adj_{column}"] = frame[column] * frame["adj_factor"]
    frame["adj_volume"] = frame["volume"] / frame["adj_factor"]
    frame.attrs["official_audit"] = audit
    return frame
