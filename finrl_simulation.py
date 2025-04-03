from finrl.env.env_stocktrading import StockTradingEnv

import pandas as pd
import numpy as np
from stable_baselines3 import DDPG

# Define configuration manually
config = {
    "DATA_SAVE_DIR": "./data/",
    "TRAINED_MODEL_DIR": "./trained_models/",
    "TENSORBOARD_LOG_DIR": "./tensorboard_log/",
    "RESULTS_DIR": "./results/",
    "START_DATE": "2015-01-01",
    "END_DATE": "2025-03-20",
    "STOCK_DIM": 1,
    "HMAX": 100,
    "INITIAL_ACCOUNT_BALANCE": 1000000,
    "TRANSACTION_COST_PERCENT": 0.001,
    "REWARD_SCALING": 1e-4,
    "STATE_SPACE_DIM": 61,
    "ACTION_SPACE_DIM": 1,
    "TECHNICAL_INDICATORS_LIST": ["SMA_10", "SMA_50", "RSI"],
    "TURBULENCE_THRESHOLD": 300,
    "TURBULENCE_INDICATOR": "turbulence",
    "RISK_INDICATORS_LIST": ["vix"],
}

# Load the data
data = pd.read_csv('data/AAPL.csv', parse_dates=['Date'], index_col='Date')

# Create the FinRL environment
env = StockTradingEnv(df=data, **config)

# Load the trained DDPG model
model = DDPG.load('ddpg_model')

# Simulate trading
obs = env.reset()
for _ in range(len(data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

# Print final portfolio value
print(f"Final portfolio value: {env.portfolio_value}")