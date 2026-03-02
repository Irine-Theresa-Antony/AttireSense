import os
import gdown

MODEL_DIR = "models"

MODELS = {
    "gmm_latest.pth": "https://drive.google.com/file/d/18lKtQLBbigfUI1zHEpQHabPCpvmvJGWq/view?usp=sharing",
    "acgpn_latest.pth": "PASTE_ACGPN_FILE_ID"
}

def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for model_name, file_id in MODELS.items():
        model_path = os.path.join(MODEL_DIR, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            print(f"{model_name} downloaded.")
        else:
            print(f"{model_name} already exists.")