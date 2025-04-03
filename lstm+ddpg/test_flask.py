import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'stock': 'AAPL',
    'period': 3,
    'risk_level': 'medium'
}
response = requests.post(url, json=data)
print(response.json())