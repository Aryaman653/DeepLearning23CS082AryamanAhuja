import argparse
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class StockLSTM(nn.Module):
	def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return out


def set_seed(seed: int = 42) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def download_close_prices(symbol: str, start: str, end: str) -> np.ndarray:
	df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
	if df.empty:
		raise ValueError(f"No data downloaded for symbol '{symbol}' in range {start} to {end}.")

	close = df[["Close"]].dropna()
	if close.empty:
		raise ValueError("Downloaded data has no valid Close prices.")

	return close.values.astype(np.float32)


def make_sequences(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
	x, y = [], []
	for i in range(seq_len, len(series)):
		x.append(series[i - seq_len : i])
		y.append(series[i])
	return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(prices: np.ndarray, seq_len: int, train_ratio: float = 0.8):
	split_idx = int(len(prices) * train_ratio)
	train_prices = prices[:split_idx]
	test_prices = prices[split_idx - seq_len :]

	scaler = MinMaxScaler(feature_range=(0, 1))
	train_scaled = scaler.fit_transform(train_prices)
	test_scaled = scaler.transform(test_prices)

	x_train, y_train = make_sequences(train_scaled, seq_len)
	x_test, y_test = make_sequences(test_scaled, seq_len)

	return x_train, y_train, x_test, y_test, scaler


def train_model(
	model: nn.Module,
	x_train: torch.Tensor,
	y_train: torch.Tensor,
	epochs: int,
	batch_size: int,
	lr: float,
	device: torch.device,
) -> None:
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	model.train()
	n = x_train.size(0)
	for epoch in range(1, epochs + 1):
		perm = torch.randperm(n, device=device)
		x_shuffled = x_train[perm]
		y_shuffled = y_train[perm]
		epoch_loss = 0.0

		for i in range(0, n, batch_size):
			xb = x_shuffled[i : i + batch_size]
			yb = y_shuffled[i : i + batch_size]

			optimizer.zero_grad()
			preds = model(xb)
			loss = criterion(preds, yb)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() * xb.size(0)

		epoch_loss /= n
		print(f"Epoch {epoch:02d}/{epochs}, Loss: {epoch_loss:.6f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="LSTM Stock Price Prediction using Yahoo Finance data")
	parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol, e.g. AAPL, MSFT")
	parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
	parser.add_argument("--end", type=str, default="2026-01-01", help="End date (YYYY-MM-DD)")
	parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
	parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
	parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
	args = parser.parse_args()

	set_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	prices = download_close_prices(args.symbol, args.start, args.end)
	if len(prices) <= args.seq_len + 10:
		raise ValueError("Not enough data points for the selected sequence length.")

	x_train, y_train, x_test, y_test, scaler = prepare_data(prices, args.seq_len)

	x_train_t = torch.tensor(x_train, device=device)
	y_train_t = torch.tensor(y_train, device=device)
	x_test_t = torch.tensor(x_test, device=device)

	model = StockLSTM(
		input_size=1,
		hidden_size=args.hidden_size,
		num_layers=args.num_layers,
		dropout=0.2,
	).to(device)

	train_model(
		model=model,
		x_train=x_train_t,
		y_train=y_train_t,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		device=device,
	)

	model.eval()
	with torch.no_grad():
		test_preds = model(x_test_t).cpu().numpy()

	y_test_np = y_test.reshape(-1, 1)
	preds_inv = scaler.inverse_transform(test_preds)
	y_test_inv = scaler.inverse_transform(y_test_np)

	rmse = math.sqrt(mean_squared_error(y_test_inv, preds_inv))
	print(f"RMSE: {rmse:.4f}")

	os.makedirs("Results", exist_ok=True)

	model_path = f"Results/{args.symbol.lower()}_lstm.pth"
	torch.save(model.state_dict(), model_path)
	print(f"Model saved to: {model_path}")

	pred_df = pd.DataFrame({"Actual": y_test_inv.flatten(), "Predicted": preds_inv.flatten()})
	csv_path = f"Results/{args.symbol.lower()}_predictions.csv"
	pred_df.to_csv(csv_path, index=False)
	print(f"Predictions saved to: {csv_path}")

	plt.figure(figsize=(12, 6))
	plt.plot(pred_df["Actual"].values, label="Actual", linewidth=2)
	plt.plot(pred_df["Predicted"].values, label="Predicted", linewidth=2)
	plt.title(f"{args.symbol} Stock Price Prediction (LSTM)")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	plt.grid(alpha=0.3)
	plot_path = "Results/exp7_stock_prediction.png"
	plt.tight_layout()
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"Plot saved to: {plot_path}")
if __name__ == "__main__":
	main()

#Result

# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab %  /Users/aryaman/Desktop/De
# epLearningLab/.venv/bin/python Exp7.py --epochs 3
# Using device: cpu
# Epoch 01/3, Loss: 0.110347
# Epoch 02/3, Loss: 0.005578
# Epoch 03/3, Loss: 0.001864
# RMSE: 23.7477
# Model saved to: Results/aapl_lstm.pth
# Predictions saved to: Results/aapl_predictions.csv
# Plot saved to: Results/exp7_stock_prediction.png
# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % 

