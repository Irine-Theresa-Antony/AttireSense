import cv2
import numpy as np
from model_loader import model


def remove_background_logic(
    input_path: str,
    output_path: str,
    bg_color=None,
    bg_image_path=None
):

    results = model(input_path)
    r = results[0]

    original = cv2.imread(input_path)

    if r.masks is None:
        cv2.imwrite(output_path, original)
        return output_path

    mask = r.masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255

    cloth = cv2.bitwise_and(original, original, mask=mask)

    mask_inv = cv2.bitwise_not(mask)

    # -------- IMAGE BACKGROUND --------
    if bg_image_path is not None:

        background = cv2.imread(bg_image_path)

        if background is None:
            raise ValueError(f"Background image not found: {bg_image_path}")

        background = cv2.resize(background, (original.shape[1], original.shape[0]))

    # -------- COLOR BACKGROUND --------
    elif bg_color is not None:

        background = np.zeros_like(original, dtype=np.uint8)
        background[:] = bg_color

    # -------- DEFAULT WHITE BACKGROUND --------
    else:

        background = np.ones_like(original, dtype=np.uint8) * 255

    bg_part = cv2.bitwise_and(background, background, mask=mask_inv)

    final = cv2.add(cloth, bg_part)

    cv2.imwrite(output_path, final)

    return output_path