from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import os
from warping_server.dataset_loader import load_person_inputs
from warping_server.preprocessing import process_cloth
from warping_server.inference import run_inference
import torch
ROOT = "dataset"
app = FastAPI()
# Load background removal model
bg_model = YOLO("models/best.pt")   # adjust path if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/tryon")
async def tryon(
    person_id: str = Form(...),
    cloth: UploadFile = File(...)
):

    # Load dataset-based inputs
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

    person, agnostic, densepose, pose, parse_agnostic = \
    load_person_inputs(DATASET_ROOT, person_id)
    # Load uploaded cloth
    # ---------------- SAVE TEMP IMAGE ----------------
    import uuid

    unique_id = str(uuid.uuid4())
    temp_input_path = f"temp_{unique_id}.jpg"

    with open(temp_input_path, "wb") as buffer:
        buffer.write(await cloth.read())

    # ---------------- RUN YOLO SEGMENTATION ----------------
    results = bg_model(temp_input_path)
    r = results[0]

    original = cv2.imread(temp_input_path)

    if r.masks is not None:
        mask = r.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        # Cloth only
        cloth_part = cv2.bitwise_and(original, original, mask=mask)

        # White background
        white_bg = np.ones_like(original, dtype=np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask * 255)
        background = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

        segmented = cv2.add(cloth_part, background)

    else:
        segmented = original

    # Convert OpenCV → PIL
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    cloth_img = Image.fromarray(segmented_rgb)

    # ---------------- PROCESS FOR MODEL ----------------
    cloth_tensor, _ = process_cloth(cloth_img)

    # Run model
    output = run_inference(
        person,
        cloth_tensor,
        agnostic,
        densepose,
        pose,
        parse_agnostic
    )

    # Convert tensor → image
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")