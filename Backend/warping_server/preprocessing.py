import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

H, W = 256, 192

transform_rgb = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

def generate_cloth_mask(cloth_pil):
    cloth = np.array(cloth_pil)
    cloth = cv2.resize(cloth, (W, H))

    gray = cv2.cvtColor(cloth, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    mask = mask.astype(np.float32) / 255.0

    return mask

def process_cloth(cloth_pil):

    cloth_tensor = transform_rgb(cloth_pil)

    cloth_mask_np = generate_cloth_mask(cloth_pil)
    cloth_mask = torch.from_numpy(cloth_mask_np).unsqueeze(0)

    return cloth_tensor, cloth_mask