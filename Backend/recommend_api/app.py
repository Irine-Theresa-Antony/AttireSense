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
from fastapi import HTTPException


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
    
    # ---------------- SAVE INPUT ----------------
    input_path = os.path.join(UPLOAD_DIR, "temp_input.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # ✅ Validate image BEFORE running YOLO
    image = cv2.imread(input_path)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    
    # ---------------- REMOVE BACKGROUND ----------------
    results = model(input_path)
    r = results[0]

    original = cv2.imread(input_path)

    if r.masks is not None:
        mask = r.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        segmented = cv2.bitwise_and(original, original, mask=mask)
        bg_removed_path = os.path.join(UPLOAD_DIR, "bg_removed.png")
        cv2.imwrite(bg_removed_path, segmented)
    else:
        bg_removed_path = input_path

    # ---------------- FEATURE EXTRACTION ----------------
    query_feat = extract_feature(bg_removed_path)

    # ---------------- RECOMMEND ----------------
    results = recommend(query_feat, myntra_features, myntra_names, top_k=20)

    formatted = []

    for path, score in results:
        filename = os.path.basename(path)
        full_path = os.path.join(
            "images/Recommending_Images_BG_Removed/images", filename
        )

        if os.path.exists(full_path):
            formatted.append({
                "image": f"http://127.0.0.1:8001/images/Recommending_Images_BG_Removed/images/{filename}",
                "score": float(score)
            })

        if len(formatted) == 5:
            break
    return {"recommendations": formatted}