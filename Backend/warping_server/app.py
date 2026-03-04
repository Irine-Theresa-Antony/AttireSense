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
    cloth_img = Image.open(cloth.file).convert("RGB")
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