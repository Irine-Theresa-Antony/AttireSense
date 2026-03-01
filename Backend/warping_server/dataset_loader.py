import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

H, W = 256, 192

transform_rgb = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

def load_person_inputs(root, person_id):
    """
    root = path to warping_backend/dataset
    """

    image_dir = os.path.join(root, "image")
    agnostic_dir = os.path.join(root, "agnostic-v3.2")
    densepose_dir = os.path.join(root, "image-densepose")
    parse_agnostic_dir = os.path.join(root, "image-parse-agnostic-v3.2")
    pose_dir = os.path.join(root, "openpose_heatmaps")

    person = transform_rgb(
        Image.open(os.path.join(image_dir, person_id + ".jpg")).convert("RGB")
    )

    agnostic = transform_rgb(
        Image.open(os.path.join(agnostic_dir, person_id + ".jpg")).convert("RGB")
    )

    densepose = transform_rgb(
        Image.open(os.path.join(densepose_dir, person_id + ".jpg")).convert("RGB")
    )

    parse_agnostic = Image.open(
        os.path.join(parse_agnostic_dir, person_id + ".png")
    ).convert("L")

    parse_agnostic = transforms.Resize((H, W))(parse_agnostic)
    parse_agnostic = transforms.ToTensor()(parse_agnostic)

    pose = np.load(os.path.join(pose_dir, person_id + ".npy")).astype(np.float32) / 255.0
    pose = torch.from_numpy(pose)

    if pose.shape[0] != 18 and pose.shape[0] != 20:
        pose = pose.permute(2, 0, 1)

    return person, agnostic, densepose, pose, parse_agnostic