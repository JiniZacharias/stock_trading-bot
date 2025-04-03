def categorize_risk_levels(data):
    data['Risk_Level'] = 'low'
    data.loc[data['Volatility'] > 0.02, 'Risk_Level'] = 'medium'
    data.loc[data['Volatility'] > 0.05, 'Risk_Level'] = 'high'
    return data

data['Volatility'] = data['Close'].pct_change().rolling(window=30).std()
data = categorize_risk_levels(data)