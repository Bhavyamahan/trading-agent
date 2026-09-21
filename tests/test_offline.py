"""Offline tests: parsing both bhavcopy formats and split/bonus adjustment."""
import datetime as dt
import io
import zipfile

import pandas as pd

from pipeline import adjust, nse


def _zip(csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("file.csv", csv_text)
    return buffer.getvalue()


def test_legacy_parse():
    csv = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
           "ABC,EQ,100,110,95,105,105,99,1000,105000,02-JAN-2019,50,INE000A01011,\n"
           "ABC,N1,100,110,95,105,105,99,1000,105000,02-JAN-2019,50,INE000A01012,\n"
           "XYZ,BE,50,52,49,51,51,50,200,10200,02-JAN-2019,10,INE000B01011,\n")
    frame = nse.parse_legacy(nse._read_zipped_csv(_zip(csv)), dt.date(2019, 1, 2))
    assert list(frame["symbol"]) == ["ABC", "XYZ"]          # N1 (bond series) dropped
    assert frame.loc[0, "prev_close"] == 99 and frame.loc[0, "volume"] == 1000


def test_legacy_2008_without_isin():
    csv = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
           "ABC,EQ,100,110,95,105,105,99,1000,105000,01-JAN-2008,\n")
    frame = nse.parse_legacy(nse._read_zipped_csv(_zip(csv)), dt.date(2008, 1, 1))
    assert len(frame) == 1 and pd.isna(frame.loc[0, "isin"]) and pd.isna(frame.loc[0, "trades"])
    assert frame.loc[0, "close"] == 105


def test_udiff_parse():
    csv = ("TradDt,FinInstrmTp,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd\n"
           "2024-07-08,STK,INE000A01011,ABC,EQ,100,110,95,105,99,1000,105000,50\n"
           "2024-07-08,STK,INE000C01011,SME1,SM,10,11,9,10,10,10,100,1\n")
    frame = nse.parse_udiff(nse._read_zipped_csv(_zip(csv)), dt.date(2024, 7, 8))
    assert list(frame["symbol"]) == ["ABC"]                 # SME series dropped
    assert frame.loc[0, "close"] == 105


def test_split_adjustment():
    # 1:2 split on day 3: close 200 -> next day's official prev_close is 100.
    rows = [("ABC", "2024-01-01", 190, 190, 1000),
            ("ABC", "2024-01-02", 200, 190, 1000),
            ("ABC", "2024-01-03", 102, 100, 2000),
            ("ABC", "2024-01-04", 104, 102, 2000),
            ("XYZ", "2024-01-01", 50, 49, 10),
            ("XYZ", "2024-01-02", 51, 50, 10)]
    raw = pd.DataFrame(rows, columns=["symbol", "date", "close", "prev_close", "volume"])
    raw["date"] = pd.to_datetime(raw["date"])
    for column in ["open", "high", "low"]:
        raw[column] = raw["close"]
    out = adjust.add_adjusted_prices(raw).set_index(["symbol", "date"])
    abc = out.loc["ABC"]
    assert abc["adj_close"].round(2).tolist() == [95.0, 100.0, 102.0, 104.0]
    assert abc["adj_volume"].tolist() == [2000, 2000, 2000, 2000]
    assert abc["ca_event"].tolist() == [False, False, True, False]
    assert out.loc["XYZ"]["adj_factor"].tolist() == [1.0, 1.0]  # untouched




def test_index_repair():
    from pipeline import index_repair
    history = index_repair.load_history_csv("data/nifty500_2008_2012.csv")
    # Swapped high/low on 29 Apr 2009 is fixed on load.
    row = history[history["date"] == pd.Timestamp("2009-04-29")].iloc[0]
    assert row["high"] >= row["low"] and row["high"] >= row["close"]

    days = [pd.Timestamp(d) for d in ["2010-05-14", "2010-05-16", "2010-05-17"]]
    frame, report, last = index_repair.repair_year(None, days, history, lambda d: None, None)
    assert report["from_history"] == 2 and report["carried_forward"] == ["2010-05-16"]
    n500 = frame.set_index("date")["close"]
    assert n500[pd.Timestamp("2010-05-16")] == n500[pd.Timestamp("2010-05-14")]

    # After 2012-02-17 a missing day is re-fetched before carrying forward.
    stored = pd.DataFrame({"date": [pd.Timestamp("2013-10-08")], "index_name": ["NIFTY 500"],
                           "open": [1.0], "high": [1.0], "low": [1.0], "close": [100.0]})
    fetched = pd.DataFrame({"date": [pd.Timestamp("2013-10-09")], "index_name": ["NIFTY 500"],
                            "open": [2.0], "high": [2.0], "low": [2.0], "close": [101.0]})
    days = [pd.Timestamp("2013-10-08"), pd.Timestamp("2013-10-09"), pd.Timestamp("2013-10-10")]
    frame, report, _ = index_repair.repair_year(
        stored, days, history, lambda d: fetched if d == dt.date(2013, 10, 9) else None, None)
    assert report["fetched"] == ["2013-10-09"] and report["carried_forward"] == ["2013-10-10"]
    assert len(frame) == 3

    ok = index_repair.overlap_check(history[history["date"].dt.year == 2012], history)
    assert ok["passed"] and ok["max_diff_pct"] == 0


if __name__ == "__main__":
    test_legacy_parse(); test_legacy_2008_without_isin(); test_udiff_parse(); test_split_adjustment(); test_index_repair()
    print("All offline tests passed")
