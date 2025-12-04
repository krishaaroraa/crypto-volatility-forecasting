# models/har_model.py

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_CSV = "data/btc_15m_preprocessed.csv"

def build_har_features(df):
    """
    HAR on daily RV. We downsample to daily using rv_1d.
    """
    # keep one row per day (last 15m candle of each day)
    daily = df.resample("1D").last().dropna(subset=["rv_1d", "target_1d"])
    daily = daily.copy()

    daily["RV_d"] = daily["rv_1d"]
    daily["RV_w"] = daily["rv_1d"].rolling(7).mean()
    daily["RV_m"] = daily["rv_1d"].rolling(30).mean()
    daily = daily.dropna()

    X = daily[["RV_d", "RV_w", "RV_m"]]
    y = daily["target_1d"]

    return X, y, daily

def train_test_split_time(X, y, train_frac=0.7):
    n = len(X)
    split = int(n * train_frac)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test

def main():
    if not os.path.exists(DATA_CSV):
        raise RuntimeError("Run load_binance.py first to create the preprocessed CSV.")

    df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)

    X, y, daily = build_har_features(df)

    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split_time(X, y)

    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\nHAR 1-day RV forecast performance (test set):")
    print("RMSE:", rmse)
    print("MAE :", mae)

if __name__ == "__main__":
    main()
