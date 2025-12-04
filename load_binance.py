# load_binance.py

import os
import zipfile
import numpy as np
import pandas as pd

DATA_DIR = "data/binance_klines"
OUTPUT_CSV = "data/btc_15m_preprocessed.csv"

def load_binance_klines(
    folder=DATA_DIR,
    symbol="BTCUSDT",
    interval="15m"
):
    dfs = []

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".zip") and fname.startswith(f"{symbol}-{interval}"):
            full_path = os.path.join(folder, fname)
            print("Loading:", fname)

            with zipfile.ZipFile(full_path) as z:
                csv_name = fname.replace(".zip", ".csv")
                with z.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=[
                            "open_time","open","high","low","close","volume",
                            "close_time","quote_asset_volume","num_trades",
                            "taker_buy_base_vol","taker_buy_quote_vol","ignore"
                        ]
                    )
                    dfs.append(df)

    if not dfs:
        raise RuntimeError("No kline files found. Check folder and filenames.")

    df = pd.concat(dfs, ignore_index=True)

    # basic cleaning
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    numeric_cols = [
        "open","high","low","close","volume",
        "quote_asset_volume","num_trades",
        "taker_buy_base_vol","taker_buy_quote_vol"
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df.sort_index()
    return df

def realized_volatility(series, window):
    # realized vol proxy from intraday returns
    return series.rolling(window).std() * np.sqrt(window)

def build_preprocessed():
    df = load_binance_klines()

    # log returns
    df["log_ret"] = np.log(df["close"]).diff()
    df = df.dropna()

    # for 15m data: 1 day = 96 intervals
    day = 96
    week = day * 7
    month = day * 30
    two_month = day * 60

    # realized volatility horizons
    df["rv_1d"]  = realized_volatility(df["log_ret"], day)
    df["rv_7d"]  = realized_volatility(df["log_ret"], week)
    df["rv_30d"] = realized_volatility(df["log_ret"], month)
    df["rv_60d"] = realized_volatility(df["log_ret"], two_month)

    df = df.dropna()

    # forecast targets: future RV at each horizon
    df["target_1d"]  = df["rv_1d"].shift(-day)
    df["target_7d"]  = df["rv_7d"].shift(-week)
    df["target_30d"] = df["rv_30d"].shift(-month)
    df["target_60d"] = df["rv_60d"].shift(-two_month)

    df = df.dropna()

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print("Saved preprocessed data to", OUTPUT_CSV)
    print(df.head())
    print(df.tail())
    print("Rows:", len(df))

if __name__ == "__main__":
    build_preprocessed()
