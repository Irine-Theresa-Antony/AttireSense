import torch
from warping_server.dataset_loader import load_person_inputs
from warping_server.app import app
from warping_server.inference import run_inference
from warping_server.preprocessing import process_cloth
from PIL import Image

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

def test_inference_output_shape():
    person_id = "00826_00"

    person, agnostic, densepose, pose, parse_agnostic = \
        load_person_inputs(DATASET_ROOT, person_id)

    cloth_img = Image.new("RGB", (256, 256), "blue")
    cloth_tensor, cloth_mask = process_cloth(cloth_img)

    output = run_inference(
        person,
        cloth_tensor,
        agnostic,
        densepose,
        pose,
        cloth_mask
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape[1:] == (256, 192)