"""Adjust raw bhavcopy prices for splits, bonuses and similar corporate actions.

Trick: on an ex-date, NSE's official "previous close" is already adjusted for the
action. So for each stock:

    factor(t) = prev_close(t) / close(t-1)

is 1.0 on normal days and, e.g., 0.5 on the ex-date of a 1:2 split. Every price
before that date is multiplied by the factor (volume divided by it), so the chart
shows no fake crash. No separate corporate-actions feed is needed.
"""
import pandas as pd

TOLERANCE = 0.005  # factors within 0.5% of 1.0 are treated as normal days


def add_adjusted_prices(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    last_close = frame.groupby("symbol")["close"].shift(1)
    ratio = frame["prev_close"] / last_close
    is_event = (last_close.notna() & (frame["prev_close"] > 0)
                & ((ratio - 1).abs() > TOLERANCE))
    factor = ratio.where(is_event, 1.0)

    # Adjustment for a row = product of the factors of all LATER rows of that stock.
    later_product = (factor.iloc[::-1].groupby(frame["symbol"].iloc[::-1]).cumprod()
                     .iloc[::-1])
    frame["adj_factor"] = later_product / factor
    frame["ca_event"] = is_event
    for column in ["open", "high", "low", "close"]:
        frame[f"adj_{column}"] = frame[column] * frame["adj_factor"]
    frame["adj_volume"] = frame["volume"] / frame["adj_factor"]
    return frame
