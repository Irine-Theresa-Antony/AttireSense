import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from utils import extract_feature, recommend
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import cv2
from fastapi.responses import FileResponse


app = FastAPI()

model = YOLO("models/best.pt")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# serve images
app.mount("/images", StaticFiles(directory="images"), name="images")

# load database
myntra_features = np.load("myntra_features.npy")
myntra_names = np.load("myntra_names.npy")

@app.get("/")
def home():
    return {"message": "Recommendation API running"}

@app.post("/recommend")
async def recommend_image(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_feat = extract_feature(path)
    results = recommend(query_feat, myntra_features, myntra_names)

    formatted = [
    {
        "image": f"http://127.0.0.1:8000/images/Recommending_Images_BG_Removed/images/{os.path.basename(r[0])}",
        "score": r[1]
    }
    for r in results
]

    return {"recommendations": formatted}


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, "temp_input.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(input_path)
    r = results[0]

    original = cv2.imread(input_path)

    if r.masks is not None:
        mask = r.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        segmented = cv2.bitwise_and(original, original, mask=mask)
        output_path = os.path.join(UPLOAD_DIR, "output.png")
        cv2.imwrite(output_path, segmented)
    else:
        output_path = input_path

    return FileResponse(output_path)

