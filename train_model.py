import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

DATA_PATH = "data/household_power_consumption.txt"
os.makedirs("model", exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";", na_values="?", low_memory=False)
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df[["datetime", "Global_active_power"]].dropna().set_index("datetime")
hourly = df.resample("h").mean().dropna()

values = hourly["Global_active_power"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

sequence_length = 24
X, y = [], []
for i in range(sequence_length, len(scaled)):
    X.append(scaled[i-sequence_length:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    Input(shape=(24, 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_split=0.1, shuffle=False)

pred = model.predict(X_test, verbose=0)
pred_original = scaler.inverse_transform(pred)
actual_original = scaler.inverse_transform(y_test)

mae = mean_absolute_error(actual_original, pred_original)
rmse = np.sqrt(mean_squared_error(actual_original, pred_original))

model.save("model/electricity_lstm.keras")
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
np.save("model/recent_sequence.npy", scaled[-24:])

with open("model/metrics.txt", "w") as f:
    f.write(f"MAE={mae}\nRMSE={rmse}\n")

print(f"Training complete. MAE={mae:.4f}, RMSE={rmse:.4f}")
