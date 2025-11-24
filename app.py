from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = FastAPI(title="Stock Prediction App")

# load model (train and place model.h5 in repository)
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

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction</title>
    </head>
    <body>
        <h1>Stock Prediction App</h1>
        <form id="form">
            <label for="ticker">Ticker:</label>
            <input type="text" id="ticker" name="ticker" required />
            <button type="submit">Predict</button>
        </form>
        <pre id="result"></pre>
        <script>
        document.getElementById('form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const ticker = document.getElementById('ticker').value;
            const res = await fetch('/predict?ticker=' + ticker);
            const data = await res.json();
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

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
