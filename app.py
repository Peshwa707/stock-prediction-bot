from fastapi import FastAPI, HTTPException
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = FastAPI(title="Stock Prediction API")

# load model (you should train your model and place model.h5 in repository)
try:
    model = tf.keras.models.load_model("model.h5")
except Exception:
    model = None

def prepare_data(ticker: str):
    # fetch last 60 closing prices
    data = yf.download(ticker, period="2y")['Close']
    if len(data) < 60:
        raise ValueError("Not enough data to predict")
    close_prices = data[-60:].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    X = scaled.reshape((1, 60, 1))
    return X, scaler

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Prediction API"}

@app.get("/predict")
def predict(ticker: str):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        X, scaler = prepare_data(ticker)
        pred_scaled = model.predict(X)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return {"ticker": ticker, "predicted_price": float(pred_price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
