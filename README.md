# Bitcoin Volatility Forecasting (HAR, GARCH, LSTM, CNN-LSTM + MTF)

This project builds a full volatility-forecasting pipeline for BTCUSDT 15-minute data (2018–2022), comparing classical econometric models with deep learning architectures.

## 📈 Models Implemented
- **HAR Model** – Long-memory econometric benchmark  
- **GARCH(1,1)** – Traditional volatility model  
- **LSTM** – Non-linear sequence model  
- **CNN-LSTM + Markov Transition Field (MTF)** – Image-based temporal representation for volatility structure  

## 🔍 Key Findings
| Model | Horizon | RMSE | MAE |
|------|---------|------|------|
| GARCH(1,1) | 1-day | unstable | unstable |
| HAR | 1-day | **0.01189** | 0.00811 |
| LSTM | 1-day | **0.01198** | 0.00820 |
| CNN-LSTM + MTF | 7-day | **0.04426** | 0.04226 |

- HAR ≈ LSTM for 1-day forecasting (as in literature)  
- GARCH struggles on high-frequency crypto data  
- **CNN-LSTM + MTF achieves the best multi-day performance**, capturing richer temporal structure

## 📁 Project Structure
data/ # processed + raw Binance data

models/ # HAR, GARCH, LSTM, CNN-LSTM code

download_binance.py # bulk Binance K-lines downloader

load_binance.py # preprocess + build RV features

## 🚀 Usage
```bash
python3 -m models.har_model
python3 -m models.garch_model
python3 -m models.lstm_model
python3 -m models.cnn_lstm_mtf
