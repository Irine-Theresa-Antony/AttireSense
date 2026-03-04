from warping_server.preprocessing import process_cloth
from PIL import Image
import torch

def test_cloth_tensor_shape():
    img = Image.new("RGB", (512, 512), "red")
    cloth_tensor, cloth_mask = process_cloth(img)

    assert isinstance(cloth_tensor, torch.Tensor)
    assert cloth_tensor.shape[1:] == (256, 192)
    assert cloth_mask.shape[1:] == (256, 192)