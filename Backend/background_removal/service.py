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

        segmented = cv2.bitwise_and(original, original, mask=mask)
        cv2.imwrite(output_path, segmented)
    else:
        cv2.imwrite(output_path, original)

    return output_path
