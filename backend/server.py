from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np, cv2, io
from PIL import Image

IMG_SIZE = 224
model = load_model("age_gender_model_vgg16_balanced.h5")

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
