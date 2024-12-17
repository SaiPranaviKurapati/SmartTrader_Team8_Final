from flask import Flask, request, jsonify, render_template
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from xgb_wrapper import XGBRegressorWrapper
import sys
import os

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_PATH = os.path.join(os.path.dirname(__file__), "model")

try:
    model_path = os.path.join(BASE_PATH, "xgboost_model.pkl")
    scaler_features_path = os.path.join(BASE_PATH, "scaler_features.pkl")
    scaler_target_path = os.path.join(BASE_PATH, "scaler_target.pkl")

    with open(model_path, "rb") as model_file:
        xgboost_model = pickle.load(model_file)
        logging.info("XGBoost model loaded successfully.")

    with open(scaler_features_path, "rb") as scaler_features_file:
        scaler_features = pickle.load(scaler_features_file)
        logging.info("Feature scaler loaded successfully.")

    with open(scaler_target_path, "rb") as scaler_target_file:
        scaler_target = pickle.load(scaler_target_file)
        logging.info("Target scaler loaded successfully.")

except Exception as e:
    logging.error(f"Error loading model or scalers: {str(e)}")
    raise e

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period=10):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    return rolling_mean + (rolling_std * 2), rolling_mean, rolling_mean - (rolling_std * 2)

def fetch_and_preprocess_data(ticker, start_date, end_date):
    logging.info(f"Fetching data for ticker {ticker} from {start_date} to {end_date}.")
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    stock_data = yf.download(ticker, start=extended_start_date, end=end_date)

    if stock_data.empty:
        logging.error(f"No data found for ticker {ticker}. Check the date range or ticker symbol.")
        raise ValueError(f"No data found for ticker {ticker}. Please check the date range.")

    try:
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        stock_data['5-Day_MA'] = stock_data['Close'].rolling(5).mean()
        stock_data['10-Day_MA'] = stock_data['Close'].rolling(10).mean()
        stock_data['5-Day_Volatility'] = stock_data['Close'].rolling(5).std()
        stock_data['Lag_1'] = stock_data['Close'].shift(1)
        stock_data['Lag_2'] = stock_data['Close'].shift(2)
        stock_data['Lag_5'] = stock_data['Close'].shift(5)
        stock_data['RSI'] = calculate_rsi(stock_data)
        stock_data['EMA_10'] = calculate_ema(stock_data)
        stock_data['Bollinger_Upper'], stock_data['Bollinger_Middle'], stock_data['Bollinger_Lower'] = calculate_bollinger_bands(stock_data)

        stock_data.dropna(inplace=True)
        logging.info("Features generated successfully.")
        return stock_data.tail(5)  

    except Exception as e:
        logging.error(f"Error during feature generation: {str(e)}")
        raise e

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_prices():
    try:
        data = request.get_json()
        chosen_date = data.get("chosen_date")

        if not chosen_date:
            logging.warning("No chosen_date provided in the request.")
            return jsonify({"error": "chosen_date is required"}), 400

        chosen_datetime = datetime.strptime(chosen_date, "%Y-%m-%d")

        end_date = (chosen_datetime + timedelta(days=10)).strftime("%Y-%m-%d")  
        
        stock_data = fetch_and_preprocess_data(
            "NVDA",
            chosen_date,
            end_date
        )

        features = stock_data[
            [
                'Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility',
                'Lag_1', 'Lag_2', 'Lag_5', 'RSI', 'EMA_10',
                'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower'
            ]
        ].values

        normalized_features = scaler_features.transform(features)

        predictions = xgboost_model.predict(normalized_features)

        all_predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()

        dates = []
        current_date = chosen_datetime + timedelta(days=1) 
        while len(dates) < 5:  
            if current_date.weekday() < 5:  
                dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        denormalized_predictions = all_predictions[:5]  

        price_threshold_percent = 0.1  
        strategy = []
        
        for i in range(len(denormalized_predictions)-1):
            price_change_percent = ((denormalized_predictions[i+1] - denormalized_predictions[i]) 
                                  / denormalized_predictions[i]) * 100
            
            if abs(price_change_percent) <= price_threshold_percent:
                strategy.append("IDLE")
            elif price_change_percent > price_threshold_percent:
                strategy.append("BULLISH")
            else:
                strategy.append("BEARISH")

        last_change_percent = ((denormalized_predictions[-1] - denormalized_predictions[-2]) 
                             / denormalized_predictions[-2]) * 100
        if abs(last_change_percent) <= price_threshold_percent:
            strategy.append("IDLE")
        elif last_change_percent > price_threshold_percent:
            strategy.append("BULLISH")
        else:
            strategy.append("BEARISH")

        response = {
            "dates": dates,
            "predicted_prices": denormalized_predictions.tolist(),
            "strategy": strategy,
            "summary": {
                "highest_price": float(max(denormalized_predictions)),
                "lowest_price": float(min(denormalized_predictions)),
                "average_price": float(np.mean(denormalized_predictions))
            }
        }
        
        logging.info("Prediction successful.")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)