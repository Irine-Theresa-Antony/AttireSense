import os
import torch

from models.cfm_final import ACGPN_FullGenerator as ACGPN
from models.gmm_final import ACGPN_GMM as GMM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

gmm_path = os.path.join(BASE_DIR, "models", "gmm_latest.pth")
acgpn_path = os.path.join(BASE_DIR, "models", "acgpn_latest_24.pth")

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

    return output.squeeze(0)