# import requests

# url = "http://127.0.0.1:5000/predict"  # Change if hosted elsewhere
# data = {"comments": ["This video is amazing!", "I didn't like this one."]}

# response = requests.post(url, json=data)
import requests

url = "http://127.0.0.1:5000/predict"
data = {"comments": ["Test comment"]}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.text)

