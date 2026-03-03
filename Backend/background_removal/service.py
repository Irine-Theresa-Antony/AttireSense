import cv2
import numpy as np
from model_loader import model


def remove_background_logic(input_path: str, output_path: str):

    results = model(input_path)
    r = results[0]

    original = cv2.imread(input_path)

    if r.masks is not None:
        mask = r.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        # Create cloth-only image
        cloth = cv2.bitwise_and(original, original, mask=mask)

        # Create white background
        white_bg = np.ones_like(original, dtype=np.uint8) * 255

        # Invert mask
        mask_inv = cv2.bitwise_not(mask * 255)

        # Extract background region
        background = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

        # Combine cloth + white background
        segmented = cv2.add(cloth, background)
        cv2.imwrite(output_path, segmented)
    else:
        cv2.imwrite(output_path, original)

    return output_path
