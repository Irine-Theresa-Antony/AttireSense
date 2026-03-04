import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np

from warping_server.models.cfm_final import ACGPN_FullGenerator as ACGPN
from warping_server.models.gmm_final import ACGPN_GMM as GMM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

gmm_path = os.path.join(BASE_DIR, "models", "gmm_latest.pth")
acgpn_path = os.path.join(BASE_DIR, "models", "acgpn_latest_24.pth")
from PIL import Image
import numpy as np

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=2
)

esrgan_path = os.path.join(BASE_DIR, "models", "RealESRGAN_x2plus.pth")

sr_model = RealESRGANer(
    scale=2,
    model_path=esrgan_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

esrgan_path = os.path.join(BASE_DIR, "models", "RealESRGAN_x2.pth")

print("Real-ESRGAN loaded successfully")
# ---- Create model instances ----
gmm = GMM().to(device)
acgpn = ACGPN().to(device)

# ---- Load GMM weights ----
gmm_checkpoint = torch.load(gmm_path, map_location=device)
gmm.load_state_dict(gmm_checkpoint['model'])

# ---- Load ACGPN Generator weights ----
acgpn_checkpoint = torch.load(acgpn_path, map_location=device)
acgpn.load_state_dict(acgpn_checkpoint['G'])

gmm.eval()
acgpn.eval()

print("Models loaded successfully")

def run_inference(person, cloth, agnostic, densepose, pose, parse_agnostic):

    # Add batch dimension
    person = person.unsqueeze(0).to(device)
    cloth = cloth.unsqueeze(0).to(device)
    agnostic = agnostic.unsqueeze(0).to(device)
    densepose = densepose.unsqueeze(0).to(device)
    pose = pose.unsqueeze(0).to(device)
    parse_agnostic = parse_agnostic.unsqueeze(0).to(device)

    # Build EXACT training condition (25 channels)
    cond = torch.cat(
        [agnostic, densepose, pose, parse_agnostic],
        dim=1
    )

    with torch.no_grad():

        # 1️⃣ GMM warp
        warped_cloth, theta, source_pts = gmm(cloth, cond)

        # 2️⃣ ACGPN render
        output, seg_logits, cloth_mask = acgpn(
            agnostic,
            densepose,
            pose,
            warped_cloth
        )

    # ---- Convert tensor [-1,1] → uint8 image ----
        output = output.squeeze(0).cpu()
        output = (output * 0.5 + 0.5).clamp(0,1)
        output_np = output.permute(1,2,0).numpy()
        output_np = (output_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(output_np)

        # ---- Apply Real-ESRGAN ----
        img_np = np.array(pil_image)

        enhanced_img, _ = sr_model.enhance(img_np, outscale=2)

        enhanced = Image.fromarray(enhanced_img)

    return enhanced