import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pyts.image import MarkovTransitionField



# Change path if your dataset name/path is different
DATA_CSV = "data/btc_15m_preprocessed.csv"

# Base series used to build MTF images
BASE_COL = "rv_1d"     # you could also try "log_ret" but volatility makes sense here
TARGET_COL = "target_7d"   # 7-day ahead realized volatility

# MTF + sequence config
SERIES_LEN = 256       # length of raw 1D series per sample
NUM_STEPS  = 4         # sequence length = number of images per sample
IMG_SIZE   = 32        # MTF image size (32x32)

# Training config
BATCH_SIZE = 64
EPOCHS     = 10      # can increase if training is fast
LR         = 1e-3
MAX_SAMPLES = 8000   # limit total samples to keep training reasonable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# DATA LOADING
# =========================

df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
print("Raw shape:", df.shape)

# Keep only necessary columns and recent data
df = df[[BASE_COL, TARGET_COL]].copy()
df = df.dropna()

# Optional: keep only recent chunk to make things manageable
df = df.iloc[-200_000:]
print("After truncation:", df.shape)

# Ensure no NaNs
df = df.dropna(subset=[BASE_COL, TARGET_COL])
base_series = df[BASE_COL].values
target_series = df[TARGET_COL].values


# =========================
# BUILD MTF SEQUENCE DATA
# =========================

def build_mtf_sequences(base, targets,
                        series_len=256,
                        num_steps=4,
                        img_size=32,
                        max_samples=None):
    """
    For each time index t, use the previous `series_len` points of `base`
    split into `num_steps` segments, each converted to an MTF image.
    Output shape: (N, num_steps, 1, img_size, img_size)
    """
    mtf = MarkovTransitionField(image_size=img_size)
    sub_len = series_len // num_steps

    X_list = []
    y_list = []

    start_idx = series_len - 1
    end_idx = len(base)  # we'll stop earlier if max_samples reached

    for idx in range(start_idx, end_idx):
        # target at time idx
        y_val = targets[idx]
        if np.isnan(y_val):
            continue

        segment = base[idx - series_len + 1 : idx + 1]
        if len(segment) != series_len:
            continue
        if np.isnan(segment).any():
            continue

        # split into num_steps segments
        imgs = []
        for s in range(num_steps):
            sub = segment[s * sub_len : (s + 1) * sub_len]
            if np.isnan(sub).any():
                imgs = []
                break
            # sub shape (sub_len,) -> MTF image (img_size, img_size)
            img = mtf.fit_transform(sub.reshape(1, -1))[0]  # (H, W)
            imgs.append(img)
        if not imgs:
            continue

        imgs = np.stack(imgs, axis=0)        # (num_steps, H, W)
        imgs = imgs[:, None, :, :]           # (num_steps, 1, H, W)
        X_list.append(imgs)
        y_list.append(y_val)

        if max_samples is not None and len(X_list) >= max_samples:
            break

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


print("Building MTF sequences...")
X, y = build_mtf_sequences(
    base_series,
    target_series,
    series_len=SERIES_LEN,
    num_steps=NUM_STEPS,
    img_size=IMG_SIZE,
    max_samples=MAX_SAMPLES,
)
print("MTF sequences shape:", X.shape, "Targets shape:", y.shape)

# Train/val/test split (time-based)
def train_val_test_split_time(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_time(X, y)
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# =========================
# DATASET & MODEL
# =========================

class MTFSequenceDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, T, 1, H, W)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = MTFSequenceDataset(X_train, y_train)
val_ds   = MTFSequenceDataset(X_val, y_val)
test_ds  = MTFSequenceDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class CNNLSTM_MTF(nn.Module):
    """
    CNN over each MTF image -> feature vector per timestep
    LSTM over sequence of feature vectors -> final volatility prediction
    """
    def __init__(self, img_size=32, lstm_hidden=64, num_layers=1, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # after 2x MaxPool2d(2), spatial dims -> img_size / 4
        feat_size = (img_size // 4) * (img_size // 4) * 32

        self.lstm = nn.LSTM(
            input_size=feat_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, T, 1, H, W)
        b, T, C, H, W = x.shape

        # Process each timestep's image with CNN
        # We'll reshape to (batch*T, C, H, W) for efficiency
        x_reshaped = x.view(b * T, C, H, W)
        cnn_out = self.cnn(x_reshaped)      # (b*T, 32, H', W')
        cnn_out = cnn_out.view(b * T, -1)   # (b*T, feat_size)

        # Reshape back to (batch, T, feat_size)
        feat_size = cnn_out.shape[-1]
        seq_feats = cnn_out.view(b, T, feat_size)

        # LSTM over sequence
        lstm_out, _ = self.lstm(seq_feats)  # (batch, T, hidden)
        last = lstm_out[:, -1, :]           # (batch, hidden)
        out = self.fc(last).squeeze(1)      # (batch,)
        return out


model = CNNLSTM_MTF(img_size=IMG_SIZE, lstm_hidden=64, num_layers=1, dropout=0.2)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)


# =========================
# TRAINING LOOP
# =========================

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
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

    # ---- Validation ----
    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())

    avg_train = float(np.mean(train_losses))
    avg_val   = float(np.mean(val_losses))

    print(f"Epoch {epoch}/{EPOCHS} - train loss: {avg_train:.6f} | val loss: {avg_val:.6f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state = model.state_dict()

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)


# =========================
# TEST EVALUATION
# =========================

model.eval()
all_preds = []
all_true  = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

rmse = float(np.sqrt(mean_squared_error(all_true, all_preds)))
mae  = float(mean_absolute_error(all_true, all_preds))

print("\nCNN-LSTM + MTF 7-day RV forecast performance (test):")
print("RMSE:", rmse)
print("MAE :", mae)
