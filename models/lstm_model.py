# models/lstm_model.py

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_CSV = "data/btc_15m_preprocessed.csv"
SEQ_LEN = 96 * 3  

class VolDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMVolModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       
        last = out[:, -1, :]      
        return self.fc(last).squeeze(1)

def build_sequences(df, feature_cols, target_col, seq_len):
    data = df[feature_cols].values
    target = df[target_col].values

    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])

    return np.array(X), np.array(y)

def train_test_split_time(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    if not os.path.exists(DATA_CSV):
        raise RuntimeError("Run load_binance.py first to create the preprocessed CSV.")

    df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)

    feature_cols = ["log_ret", "rv_1d", "rv_7d", "rv_30d"]
    target_col = "target_1d"  

    df = df.dropna(subset=feature_cols + [target_col])

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = build_sequences(df, feature_cols, target_col, SEQ_LEN)
    print("Sequences:", X.shape, y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_time(X, y)

    train_ds = VolDataset(X_train, y_train)
    val_ds   = VolDataset(X_val, y_val)
    test_ds  = VolDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = LSTMVolModel(input_dim=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    EPOCHS = 5

    for epoch in range(1, EPOCHS+1):
        # ---- Train ----
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch}/{EPOCHS} - train loss: {avg_train:.6f}, val loss: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    mae = mean_absolute_error(all_true, all_preds)

    print("\nLSTM 1-day RV forecast performance (test):")
    print("RMSE:", rmse)
    print("MAE :", mae)

if __name__ == "__main__":
    main()
