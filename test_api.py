import requests

url = "http://127.0.0.1:8000/predict"
data = {"text": "not good this product"}

response = requests.post(url, params=data)
print(response.json())
