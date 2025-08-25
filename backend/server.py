import os, io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import gdown

IMG_SIZE = 224
MODEL_PATH = "age_gender_model_vgg16_balanced.h5"
DRIVE_ID = "1gpBr8_U9bW9ZWQwdUO0J_hIfJh4q8v41"  # thay bằng ID từ link Drive
URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

# tải model nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("⬇️ Đang tải model từ Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

print("✅ Model ready")
model = load_model(MODEL_PATH, compile=False)   # thêm compile=False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    arr = preprocess(img)
    age_pred, gender_pred = model.predict(arr, verbose=0)
    age = int(age_pred[0][0])
    gender = "Nam" if gender_pred[0][0] > 0.5 else "Nữ"
    return {"age": age, "gender": gender}
