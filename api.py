from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import tempfile
from detection import detect_money
app = FastAPI()

@app.post("/detect") # когда пользователь отправляет POST-запрос вызови detect()
async def detect(file: UploadFile= File(...)):
    contents = await file.load()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np.arr, cv2.IMREAD_COLOR)

    total_sum = detect_money(img)

    return {"total_sum": total_sum}

@app.get("/")
def kek():
    return("kek")