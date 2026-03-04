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
# 1️⃣ Valid Recommendation Test
# -------------------------
def test_recommend_valid():

    test_img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")

    dummy = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.imwrite(test_img_path, dummy)

    with open(test_img_path, "rb") as f:
        response = client.post(
            "/recommend",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()

    # Structure validation
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

    recommendations = data["recommendations"]

    # Check result count
    assert len(recommendations) == 5

    # Check uniqueness
    image_urls = [r["image"] for r in recommendations]
    assert len(image_urls) == len(set(image_urls))

    # Check required fields
    for r in recommendations:
        assert "image" in r
        assert "score" in r
        assert r["image"].startswith("http")

    os.remove(test_img_path)


# -------------------------
# 2️⃣ Missing File Test
# -------------------------
def test_recommend_no_file():

    response = client.post("/recommend")

    assert response.status_code == 422


# -------------------------
# 3️⃣ Invalid File Type Test
# -------------------------
def test_recommend_invalid_file():

    response = client.post(
        "/recommend",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )

    assert response.status_code in [400, 422, 500]