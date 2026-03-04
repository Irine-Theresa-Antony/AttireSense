from fastapi.testclient import TestClient
from warping_server.app import app
import os
client = TestClient(app)

def test_tryon_missing_fields():
    response = client.post("/tryon")
    assert response.status_code == 422

def test_tryon_valid():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    cloth_path = os.path.join(BASE_DIR, "dataset", "cloth", "08350_00.jpg")

    with open(cloth_path, "rb") as f:
        response = client.post(
            "/tryon",
            data={"person_id": "00826_00"},
            files={"cloth": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200