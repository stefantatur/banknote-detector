import requests


def send_request_to_api(image_path, model):
    url = "http://localhost:8001/detect/svm"
    files = {'file': open(image_path, 'rb')}
    data = {'model': model}

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Вызывает исключение для 4xx или 5xx статусов
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None


# Пример использования
image_path = r"C:\Users\steph\PycharmProjects\detect_banknotes\sample_2.jpg"
model = "SVM"
result = send_request_to_api(image_path, model)
if result:
    print(f"Результат: {result['result']}")

