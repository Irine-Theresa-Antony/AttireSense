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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "best.pt")

bg_model = YOLO(model_path)  # adjust path if needed
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
    # Read uploaded cloth image into memory
    cloth_bytes = await cloth.read()

    # Convert bytes -> numpy array
    np_arr = np.frombuffer(cloth_bytes, np.uint8)

    # Decode to OpenCV image
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO segmentation directly on the image
    results = bg_model(original)
    r = results[0]

    if r.masks is not None:
        largest = np.argmax([m.sum() for m in r.masks.data])
        mask = r.masks.data[largest].cpu().numpy()
        mask = cv2.resize(mask,(original.shape[1],original.shape[0]),interpolation=cv2.INTER_NEAREST)        
        mask = (mask > 0.5).astype(np.uint8)

        # Cloth only
        cloth_part = cv2.bitwise_and(original, original, mask=mask)

        # White background
        white_bg = np.ones_like(original, dtype=np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask * 255)
        background = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

        segmented = cv2.add(cloth_part, background)
        # ---------------- NORMALIZE CLOTH FOR ACGPN ----------------

        mask_indices = np.where(mask == 1)

        y_min = np.min(mask_indices[0])
        y_max = np.max(mask_indices[0])
        x_min = np.min(mask_indices[1])
        x_max = np.max(mask_indices[1])

        cloth_crop = segmented[y_min:y_max, x_min:x_max]

        # create white canvas (ACGPN resolution)
        canvas = np.ones((256,192,3), dtype=np.uint8) * 255

        h, w = cloth_crop.shape[:2]

        scale = min(192 / w, 256 / h, 1.0)  # prevent upscaling

        new_w = int(w * scale)
        new_h = int(h * scale)

        cloth_resized = cv2.resize(
            cloth_crop,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        x_offset = (192 - new_w) // 2
        y_offset = (256 - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = cloth_resized

        segmented = canvas
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