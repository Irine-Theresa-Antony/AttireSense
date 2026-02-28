import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from utils import extract_feature, recommend
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
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