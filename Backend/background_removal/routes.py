import os
import shutil

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse

from service import remove_background_logic

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
def home():
    return {"message": "Background Removal API Running"}


@router.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):

    input_path = os.path.join(UPLOAD_DIR, "input.jpg")
    output_path = os.path.join(UPLOAD_DIR, "output.png")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    remove_background_logic(input_path, output_path)

    return FileResponse(output_path, media_type="image/png")
