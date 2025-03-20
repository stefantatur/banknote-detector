from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import tempfile
from detection import detect_banknote
import os
import shutil
from fastapi.responses import JSONResponse
from main import run

app = FastAPI()

# Загружаем изображение и отправляем на распознавание
@app.post("/detect/svm")
async def detect(file: UploadFile = File(...), model: str = 'SVM'):
    # Сохраняем временный файл
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run(model, file_path)

    # Удаляем временный файл
    os.remove(file_path)

    return JSONResponse(content={"result": result}, media_type="application/json")

@app.get("/kek")
def kek():
    return {"kek": "popa"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
