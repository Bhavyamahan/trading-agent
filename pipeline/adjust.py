"""Adjust raw bhavcopy prices for splits, bonuses and similar corporate actions.

Two detection methods, applied per stock per day:

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


def add_adjusted_prices(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    last_close = frame.groupby("symbol", observed=True)["close"].shift(1)

    ratio = frame["prev_close"] / last_close
    by_prev = (last_close.notna() & (frame["prev_close"] > 0)
               & ((ratio - 1).abs() > TOLERANCE))

    open_col = frame["open"] if "open" in frame.columns else frame["close"]
    gap = (open_col / last_close).to_numpy(dtype=float)
    big_gap = (~by_prev.to_numpy()) & last_close.notna().to_numpy() & ((gap < GAP_DOWN) | (gap > GAP_UP))
    snapped = np.full(len(frame), np.nan)
    if big_gap.any():
        snapped[big_gap] = snap_ratio(gap[big_gap])
    by_gap = big_gap & ~np.isnan(snapped)

    factor = np.where(by_prev.to_numpy(), ratio.to_numpy(dtype=float), 1.0)
    factor = np.where(by_gap, snapped, factor)
    factor = pd.Series(factor, index=frame.index)

    # Adjustment for a row = product of the factors of all LATER rows of that stock.
    reversed_symbols = frame["symbol"].iloc[::-1]
    later_product = factor.iloc[::-1].groupby(reversed_symbols, observed=True).cumprod().iloc[::-1]
    frame["adj_factor"] = later_product / factor
    frame["ca_event"] = by_prev.to_numpy() | by_gap
    frame["ca_method"] = np.where(by_gap, "gap_ratio", np.where(by_prev, "prev_close", ""))
    frame["ca_factor"] = factor.to_numpy()
    frame["unexplained_gap"] = big_gap & np.isnan(snapped)
    for column in ["open", "high", "low", "close"]:
        if column in frame.columns:
            frame[f"adj_{column}"] = frame[column] * frame["adj_factor"]
    frame["adj_volume"] = frame["volume"] / frame["adj_factor"]
    return frame
