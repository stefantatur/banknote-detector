import requests

url = "http://localhost:8000/detect/"
files = {"file": open("sample_2.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())