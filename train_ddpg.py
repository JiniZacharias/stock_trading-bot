import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ddpg_agent import DDPGAgent

# Load data with LSTM predictions
#"AMZN", "AAPL","IBM", "NFLX", "NVDA", "ORCL"
data = pd.read_csv('predictions/ORCL_future_predictions.csv', parse_dates=['Date'], index_col='Date')

def categorize_risk_levels(data):
    data['Risk_Level'] = 'low'
    data.loc[data['Volatility'] > 0.02, 'Risk_Level'] = 'medium'
    data.loc[data['Volatility'] > 0.05, 'Risk_Level'] = 'high'
    return data

data['Volatility'] = data['LSTM_Prediction'].pct_change().rolling(window=30).std()
data = categorize_risk_levels(data)

def calculate_reward(state, action, next_state):
    # Define your reward function here
    return next_state[0] - state[0]

# Convert risk levels to numeric values
risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['Risk_Level'] = data['Risk_Level'].map(risk_level_mapping)

# Initialize DDPG agents for each risk level
agents = {
    'low': DDPGAgent(state_dim=2, action_dim=1),  # Adjust state_dim to include risk level
    'medium': DDPGAgent(state_dim=2, action_dim=1),
    'high': DDPGAgent(state_dim=2, action_dim=1)
}

# Train the agents
for risk_level in ['low', 'medium', 'high']:
    risk_data = data[data['Risk_Level'] == risk_level_mapping[risk_level]]
    if risk_data.empty:
        continue  # Skip if there is no data for this risk level
    agent = agents[risk_level]
    
    for episode in range(1000):
        state = np.append(risk_data.iloc[0][['LSTM_Prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])  # Include risk level in state
        for t in range(1, len(risk_data)):
            action = agent.select_action(state)
            next_state = np.append(risk_data.iloc[t][['LSTM_Prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])  # Include risk level in next state
            reward = calculate_reward(state, action, next_state)
            agent.store_transition(state, action, reward, next_state)
            agent.update()
            state = next_state

# Save the model weights
torch.save(agents['low'].actor.state_dict(), 'models/ORCL_ddpg_actor_low.pth')
torch.save(agents['medium'].actor.state_dict(), 'models/ORCL_ddpg_actor_medium.pth')
torch.save(agents['high'].actor.state_dict(), 'models/ORCL_ddpg_actor_high.pth')

def get_trading_suggestions(agent, state, risk_level):
    state_with_risk = np.append(state, [risk_level_mapping[risk_level]])  # Include risk level in state
    action = agent.select_action(state_with_risk)
    if action > 0.5:
        return 'Buy'
    elif action < -0.5:
        return 'Sell'
    else:
        return 'Hold'

# Get the current state and risk level
current_state = data.iloc[-1][['LSTM_Prediction']].values.astype(np.float32)  # Ensure numeric type
current_risk_level = data.iloc[-1]['Risk_Level']

# Print trading suggestions for each risk level
for risk_level, agent in agents.items():
    suggestion = get_trading_suggestions(agent, current_state, risk_level)
    print(f"Trading Suggestion for {risk_level} risk level: {suggestion}")
