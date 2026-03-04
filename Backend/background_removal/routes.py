import os
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from service import remove_background_logic
from fastapi import HTTPException
import cv2
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
def home():
    return {"message": "Background Removal API Running"}


@router.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    bg_color: str = Form(None),
    bg_scene: str = Form(None)
):

    input_path = os.path.join(UPLOAD_DIR, "input.jpg")
    output_path = os.path.join(UPLOAD_DIR, "output.png")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(input_path)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # ---------------- COLOR HANDLING ----------------

    bg_color_rgb = None
    bg_image_path = None

    if bg_color:
        # Convert HEX (#RRGGBB) → BGR
        hex_color = bg_color.lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        bg_color_rgb = (b, g, r)   # OpenCV uses BGR

    # ---------------- SCENE HANDLING ----------------

    if bg_scene:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        bg_image_path = os.path.join(BASE_DIR, "backgrounds", f"{bg_scene}.jpg")
        print("Background path:", bg_image_path)
    # ---------------- RUN LOGIC ----------------

    remove_background_logic(
        input_path,
        output_path,
        bg_color=bg_color_rgb,
        bg_image_path=bg_image_path
    )

    return FileResponse(output_path, media_type="image/png")