import requests

URL = "http://127.0.0.1:8000/predict"

test_sentences = [
    "it is not bad",
    "i am not disappointed",
    "not really happy",
    "absolutely loved it",
    "worst product ever"
]

for text in test_sentences:
    print(f"\nSending text: {text}")

    try:
        response = requests.post(
            URL,
            json={"text": text},
            timeout=5
        )

        print("Status code:", response.status_code)

        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error response:", response.text)

    except Exception as e:
        print("Request failed:", e)
