# models/garch_model.py

import os
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_CSV = "data/btc_15m_preprocessed.csv"

def main():
    if not os.path.exists(DATA_CSV):
        raise RuntimeError("Run load_binance.py first to create the preprocessed CSV.")

    df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
    returns = df["log_ret"]

    # We'll use 80% train, 20% test in time order
    n = len(returns)
    split = int(n * 0.8)

    train_ret = returns.iloc[:split]
    test_ret = returns.iloc[split:]

    # Fit GARCH(1,1) on training returns
    am = arch_model(train_ret*100, p=1, q=1)  # scale returns to % for numerical stability
    res = am.fit(disp="off")
    print(res.summary())

    # Forecast 1-step ahead variance for test set by walking forward
    # (simplified: refit every K steps or use rolling updates; here we do a cheap pseudo-OOS)
    forecasts = []
    true_rv = []

    # We want to compare to 1-day RV, so map each test day to a GARCH prediction
    # Take the last return of each day as the forecast point
    df_test = df.iloc[split:].copy()
    df_test["date"] = df_test.index.date
    daily_groups = df_test.groupby("date")

    # Start from the last date of train as initial state
    last_train_index = df.index[split]

    for date, group in daily_groups:
        # Refit on all data up to the start of this day (simple but expensive; ok for project)
        sub_ret = df.loc[:group.index[0], "log_ret"] * 100
        am = arch_model(sub_ret, p=1, q=1)
        res = am.fit(disp="off")

        fcast = res.forecast(horizon=96)  # 96 steps in a day (15m)
        # Sum of next-day variances ≈ daily variance
        var_1d = fcast.variance.values[-1].sum()
        vol_1d = np.sqrt(var_1d) / 100.0  # back to original scale

        forecasts.append(vol_1d)

        # True realized 1-day vol from our precomputed rv_1d (use last row of that day)
        true_rv.append(group["rv_1d"].iloc[-1])

    forecasts = np.array(forecasts)
    true_rv = np.array(true_rv)

    mask = ~np.isnan(true_rv)
    forecasts = forecasts[mask]
    true_rv = true_rv[mask]

    rmse = np.sqrt(mean_squared_error(true_rv, forecasts))
    mae = mean_absolute_error(true_rv, forecasts)

    print("\nGARCH 1-day RV forecast performance (test daily):")
    print("RMSE:", rmse)
    print("MAE :", mae)

if __name__ == "__main__":
    main()
