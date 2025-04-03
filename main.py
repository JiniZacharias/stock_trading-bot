'''from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# List of stock symbols
STOCK_SYMBOLS = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS',
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'XOM'
]

# Prepare the data for LSTM
def create_dataset(data, time_step=1, prediction_days=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_days):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step:i + time_step + prediction_days, 0])
    return np.array(X), np.array(Y)

def predict_future_prices(model, data, scaler, time_step, days=10):
    predictions = []
    last_data = data[-time_step:]
    for _ in range(days):
        last_data_scaled = scaler.transform(last_data)
        X_input = last_data_scaled.reshape(1, time_step, last_data_scaled.shape[1])
        pred_price = model.predict(X_input)
        predictions.append(pred_price[0][0])
        new_row = np.append(last_data[1:], [[pred_price[0][0], 0, 0, 0, 0]], axis=0)  # Replace NaN with 0
        last_data = new_row
    placeholder = np.zeros((len(predictions), data.shape[1]))
    placeholder[:, 0] = predictions
    real_prices = scaler.inverse_transform(placeholder)[:, 0]
    real_prices = [float(price) if not np.isnan(price) else None for price in real_prices]  # Handle NaN values
    return real_prices

def generate_trade_dates(start_date, days):
    trade_dates = []
    current_date = start_date
    while len(trade_dates) < days:
        current_date += pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday are trade days
            trade_dates.append(current_date)
    return trade_dates

def get_predictions_for_risk_level(symbol, risk_level):
    file_path = os.path.join('data', f'{symbol}.csv')
    if not os.path.exists(file_path):
        return {'error': 'Stock data not found'}, 404

    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)  # Backward fill NaN values
    data.dropna(inplace=True)  # Drop any remaining NaN values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])
    time_step = 100
    model_path = os.path.join('models', f'{symbol}_lstm_model_{risk_level}.h5')
    lstm_model = load_model(model_path)

    future_prices = predict_future_prices(lstm_model, data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']].values, scaler, time_step, days=10)
    next_day_prediction = future_prices[0]  # Get the first predicted value
    next_10_days_predictions = future_prices[:10]  # Get the next 10 days predictions

    # Generate trade dates for the next 10 trade days
    last_date = data.index[-1]
    next_10_days_dates = generate_trade_dates(last_date, 10)
    next_10_days_dates = [date.strftime('%Y-%m-%d') for date in next_10_days_dates]

    last_10_days = data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']].tail(10).reset_index().to_dict(orient='records')

    response = {
        'next_day_prediction': next_day_prediction,
        'next_10_days_predictions': list(zip(next_10_days_dates, next_10_days_predictions)),
        'last_10_days': last_10_days
    }
    return response, 200

@app.route('/predict_low_risk', methods=['GET'])
def get_predictions_low_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'low')
    return jsonify(response), status

@app.route('/predict_medium_risk', methods=['GET'])
def get_predictions_medium_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'medium')
    return jsonify(response), status

@app.route('/predict_high_risk', methods=['GET'])
def get_predictions_high_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'high')
    return jsonify(response), status

if __name__ == '__main__':
    app.run(debug=True)
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import torch
from ddpg_agent import DDPGAgent
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# List of stock symbols
STOCK_SYMBOLS = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS',
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'XOM'
]

def create_dataset(data, time_step=1, prediction_days=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_days):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step:i + time_step + prediction_days, 0])
    return np.array(X), np.array(Y)

def predict_future_prices(model, data, scaler, time_step, days=10):
    predictions = []
    last_data = data[-time_step:]
    for _ in range(days):
        last_data_scaled = scaler.transform(last_data)
        X_input = last_data_scaled.reshape(1, time_step, last_data_scaled.shape[1])
        pred_price = model.predict(X_input)
        predictions.append(pred_price[0][0])
        new_row = np.append(last_data[1:], [[pred_price[0][0], 0, 0, 0, 0]], axis=0)  # Replace NaN with 0
        last_data = new_row
    placeholder = np.zeros((len(predictions), data.shape[1]))
    placeholder[:, 0] = predictions
    real_prices = scaler.inverse_transform(placeholder)[:, 0]
    real_prices = [float(price) if not np.isnan(price) else None for price in real_prices]  # Handle NaN values
    return real_prices

def generate_trade_dates(start_date, days):
    trade_dates = []
    current_date = start_date
    while len(trade_dates) < days:
        current_date += pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday are trade days
            trade_dates.append(current_date)
    return trade_dates
def get_predictions_for_risk_level(symbol, risk_level):
    file_path = os.path.join('data', f'{symbol}.csv')
    if not os.path.exists(file_path):
        return {'error': 'Stock data not found'}, 404

    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)  # Backward fill NaN values
    data.dropna(inplace=True)  # Drop any remaining NaN values

    # Map risk levels to numeric values
    risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
    numeric_risk_level = risk_level_mapping[risk_level]

    # Get the current state and risk level
    current_state = data.iloc[-1][['Close']].values.astype(np.float32)  # Ensure numeric type
    current_risk_level = numeric_risk_level

    # Load the appropriate DDPG model for the stock and risk level
    model_path = os.path.join('models', f'{symbol}_ddpg_actor_{risk_level}.pth')
    if not os.path.exists(model_path):
        return {'error': 'Model not found for the specified stock and risk level'}, 404
    agent = DDPGAgent(state_dim=2, action_dim=1)
    agent.actor.load_state_dict(torch.load(model_path))

    # Get trading suggestion
    state_with_risk = np.append(current_state, [current_risk_level])
    action = agent.select_action(state_with_risk)
    if action > 0.5:
        suggestion = 'Buy'
    elif action < -0.5:
        suggestion = 'Sell'
    else:
        suggestion = 'Hold'

    response = {
        'suggestion': suggestion,
        'current_state': current_state.tolist(),
        'risk_level': risk_level
    }
    return response, 200
@app.route('/predict', methods=['GET'])
def get_predictions():
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        file_path = os.path.join('data', f'{symbol}.csv')
        if not os.path.exists(file_path):
            return jsonify({'error': 'Stock data not found'}), 404
        
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)  # Backward fill NaN values
        data.dropna(inplace=True)  # Drop any remaining NaN values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])
        time_step = 100
        model_path = os.path.join('models', f'{symbol}_lstm_model.h5')
        lstm_model = load_model(model_path)
        
        future_prices = predict_future_prices(lstm_model, data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']].values, scaler, time_step, days=10)
        next_day_prediction = future_prices[0]  # Get the first predicted value
        next_10_days_predictions = future_prices[:10]  # Get the next 10 days predictions
        
        # Generate trade dates for the next 10 trade days
        last_date = data.index[-1]
        next_10_days_dates = generate_trade_dates(last_date, 10)
        next_10_days_dates = [date.strftime('%Y-%m-%d') for date in next_10_days_dates]
        
        last_10_days = data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']].tail(10).reset_index().to_dict(orient='records')
        
        response = {
            'next_day_prediction': next_day_prediction,
            'next_10_days_predictions': list(zip(next_10_days_dates, next_10_days_predictions)),
            'last_10_days': last_10_days
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Try Another Stock.'}), 500

@app.route('/predict_low_risk', methods=['GET'])
def get_predictions_low_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'low')
    return jsonify(response), status

@app.route('/predict_medium_risk', methods=['GET'])
def get_predictions_medium_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'medium')
    return jsonify(response), status

@app.route('/predict_high_risk', methods=['GET'])
def get_predictions_high_risk():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    response, status = get_predictions_for_risk_level(symbol, 'high')
    return jsonify(response), status

if __name__ == '__main__':
    app.run(debug=True)
