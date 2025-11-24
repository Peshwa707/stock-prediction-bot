import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def prepare_sequences(series, look_back=60):
    values = series.values
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i+look_back])
        y.append(values[i+look_back])
    X = np.array(X)
    y = np.array(y)
    return X, y


def train_model(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    series = pd.Series(scaled.flatten())
    X, y = prepare_sequences(series, look_back=60)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    model.save('model.h5')
    print('Model trained and saved to model.h5')


if __name__ == '__main__':
    train_model('AAPL', '2023-01-01', '2024-01-01')
