import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app import app
import numpy as np
import cv2

client = TestClient(app)


# -------------------------
# 1️⃣ Valid Image Test
# -------------------------
def test_remove_bg_valid_image():

    test_img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")

    # Create dummy white image
    dummy = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(test_img_path, dummy)

    with open(test_img_path, "rb") as f:
        response = client.post(
            "/remove-bg",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

    os.remove(test_img_path)


# -------------------------
# 2️⃣ Missing File Test
# -------------------------
def test_remove_bg_no_file():

    response = client.post("/remove-bg")

    # FastAPI returns 422 if required field missing
    assert response.status_code == 422


# -------------------------
# 3️⃣ Invalid File Type Test
# -------------------------
def test_remove_bg_invalid_file():

    response = client.post(
        "/remove-bg",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )

    # Could be 400 or 500 depending on implementation
    assert response.status_code in [400, 422, 500]