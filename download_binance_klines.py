import os
import requests

def download_binance_klines(symbol="BTCUSDT", interval="15m",
                             start_year=2017, end_year=2025,
                             save_dir="data/binance_klines"):
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            mm = str(month).zfill(2)
            filename = f"{symbol}-{interval}-{year}-{mm}.zip"
            url = (
                f"https://data.binance.vision/data/spot/monthly/klines/"
                f"{symbol}/{interval}/{filename}"
            )
            local_path = os.path.join(save_dir, filename)

            # Skip if already downloaded
            if os.path.exists(local_path):
                print("Already exists:", filename)
                continue

            print("Downloading:", url)
            r = requests.get(url)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print("Saved:", local_path)
            else:
                print("Not found or error:", filename, "| status:", r.status_code)

if __name__ == "__main__":
    download_binance_klines(
        symbol="BTCUSDT",
        interval="15m",
        start_year=2018,
        end_year=2022,
    )
