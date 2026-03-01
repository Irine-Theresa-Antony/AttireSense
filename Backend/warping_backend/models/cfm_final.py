
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class SegGenerator(nn.Module):
    def __init__(self, input_nc=3+3+18+3, output_nc=20):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, output_nc, 7, 1, 3)
        )

    def forward(self, x):
        return self.model(x)
class TryOnGenerator(nn.Module):
    def __init__(self, input_nc=3+3+20+3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.resblocks = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.resblocks(x)
        x = self.decoder(x)
        return x

class ACGPN_FullGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_gen = SegGenerator()
        self.tryon_gen = TryOnGenerator()

    def forward(self, agnostic, densepose, pose, warped_cloth):

        # --- G1 ---
        seg_input = torch.cat([agnostic, densepose, pose, warped_cloth], dim=1)
        seg_logits = self.seg_gen(seg_input)
        seg_soft = torch.softmax(seg_logits, dim=1)

        # extract clothing mask channel (class index 5 for upper cloth in VITON)
        cloth_mask = seg_soft[:, 5:6]

        # refine warped cloth
        refined_cloth = warped_cloth * cloth_mask

        # --- G2 ---
        tryon_input = torch.cat(
            [agnostic, densepose, seg_soft, refined_cloth],
            dim=1
        )

        rendered = self.tryon_gen(tryon_input)

        # composite final image
        output = refined_cloth + rendered * (1 - cloth_mask)

        return output, seg_logits, cloth_mask
