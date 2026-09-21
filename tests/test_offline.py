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


if __name__ == "__main__":
    test_legacy_parse(); test_udiff_parse(); test_split_adjustment()
    print("All offline tests passed")
