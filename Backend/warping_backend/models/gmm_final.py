import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureExtraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        layers = []
        ch = in_channels

        # 5 downsamples
        for out_ch in [64, 128, 256, 512]:
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True)
            ]
            ch = out_ch

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FeatureRegression(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(192, 512, 4, 2, 1),  # 16x12 → 8x6
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 256, 4, 2, 1),  # 8x6 → 4x3
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.fc = nn.Linear(256 * 4 * 3, num_points * 2)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
def feature_correlation(f1,f2):
    b,c,h,w = f1.shape
    f1 = f1.view(b,c,h*w)
    f2 = f2.view(b,c,h*w)
    corr = torch.bmm(f1.transpose(1,2),f2)
    return corr.view(b,h*w,h,w)


class TPSGridGen(nn.Module):
    """Thin-Plate Spline grid generator solving full TPS linear system."""
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        # target_control_points: (N,2) tensor of normalized [0,1] coordinates
        assert target_control_points.dim() == 2 and target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        # Build the forward kernel (L matrix) for TPS (N+3 x N+3)
        # following classic TPS: U(r)=r^2*log(r^2)
        def compute_U_matrix(points):
            # points: (N,2), returns (N,N) pairwise distance matrix in TPS space
            XX = points.unsqueeze(1)   # (N,1,2)
            YY = points.unsqueeze(0)   # (1,N,2)
            dist2 = torch.sum((XX - YY)**2, dim=2)  # (N,N)
            # Avoid log(0) by replacing zeros with epsilon
            dist2[dist2 == 0] = 1
            # U = r^2 * log(r^2)
            U = dist2 * torch.log(dist2)
            return U
        # Construct L = [[K, 1, P], [1^T, 0,0],[P^T,0,0]]
        K = compute_U_matrix(target_control_points)  # (N,N)
        forward_kernel = torch.zeros(N+3, N+3)
        # Top-left: U matrix
        forward_kernel[:N, :N] = K
        # Top-right and bottom-left blocks for affine terms
        forward_kernel[:N, N]   = 1
        forward_kernel[N, :N]   = 1
        forward_kernel[:N, N+1:] = target_control_points
        forward_kernel[N+1:, :N] = target_control_points.transpose(0,1)
        # Inverse kernel matrix (L^-1), stored as buffer
        inverse_kernel = torch.inverse(forward_kernel).float().contiguous()
        self.register_buffer('inverse_kernel', inverse_kernel)  # (N+3,N+3)
        # Prepare target coordinate grid (flattened) in [0,1]
        # and compute partial representation phi([x,y], ctrl_pts) = U for each pixel vs control point
        GH = target_height * target_width
        # Create meshgrid of normalized coordinates (x,y) of output image
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=target_height),
            torch.linspace(-1, 1, steps=target_width),
            indexing='ij')
        tgt_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (GH,2)
        # Compute U between every output pixel and control point (GH x N)
        XX = tgt_coords.unsqueeze(1)   # (GH,1,2)
        YY = target_control_points.unsqueeze(0)  # (1,N,2)
        d2 = torch.sum((XX - YY)**2, dim=2)  # (GH,N)
        d2[d2 == 0] = 1
        U_part = d2 * torch.log(d2)   # (GH,N)
        # Append ones and coords for affine part: [U | 1 | [x y]]
        target_coordinate_repr = torch.cat([
            U_part,
            torch.ones(GH,1),
            tgt_coords], dim=1)  # (GH, N+3)
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)  # (GH,N+3)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input, source_control_points):
        """
        Apply TPS to warp `input` cloth image.
        - input: (B, C, H, W) cloth image tensor.
        - source_control_points: (B, N, 2) predicted control points (normalized [0,1]).
        """
        B, C, H, W = input.shape
        # source_control_points should align with target ctrl points shape
        assert source_control_points.dim() == 3 and source_control_points.size(2) == 2
        # Augment source points with zeros for affine weights (3x2 padding)
        # Y = [source_pts; zeros(3,2)]
        padding = torch.zeros(B, 3, 2, device=source_control_points.device)
        Y = torch.cat([source_control_points, padding], dim=1)  # (B, N+3, 2)
        # Compute mapping matrix: (inverse_kernel) x Y => (N+3)x2 for each batch
        # inverse_kernel is (N+3,N+3); use batch matmul
        mapping = torch.matmul(self.inverse_kernel, Y)  # (B, N+3, 2)
        # Compute warped grid: target_coordinate_repr (GH,N+3) @ mapping (N+3,2) => (GH,2)
        # for each batch, then reshape to (H, W, 2)
        warped_coords = torch.matmul(self.target_coordinate_repr, mapping)  # (B, GH, 2)
        grid = warped_coords.view(B, H, W, 2)
        # Grid values are normalized [0,1]; convert to [-1,1] for grid_sample
        # Sample the input cloth at warped grid positions
        warped_image = F.grid_sample(input, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_image

class ACGPN_GMM(nn.Module):
    def __init__(self, grid_size=5):
        super().__init__()

        self.grid_size = grid_size
        self.num_points = grid_size * grid_size

        # ---- Feature Extractors ----
        self.cloth_extractor = FeatureExtraction(3)
        self.person_extractor = FeatureExtraction(25)

        # ---- Correlation + Regression ----
        self.regression = FeatureRegression(self.num_points)

        # ---- Build TPS control points in [0,1] ----
        axis = torch.linspace(-1, 1, steps=grid_size)
        P_Y, P_X = torch.meshgrid(axis, axis, indexing='ij')
        control_points = torch.stack(
            [P_X.reshape(-1), P_Y.reshape(-1)],
            dim=1
        )

        self.tps = TPSGridGen(
            target_height=256,
            target_width=192,
            target_control_points=control_points
        )

    def forward(self, cloth, cond):

        # ---- Extract features ----
        f_cloth = self.cloth_extractor(cloth)
        f_person = self.person_extractor(cond)

        f_cloth = F.normalize(f_cloth, dim=1)
        f_person = F.normalize(f_person, dim=1)

        # ---- Correlation ----
        corr = feature_correlation(f_person, f_cloth)

        # ---- Predict control point offsets ----
        theta = self.regression(corr)              # (B, 2*N)
        B = theta.size(0)
        theta = theta.view(B, self.num_points, 2)  # (B, N, 2)

        # ---- Convert offsets to absolute control points ----
        base_pts = self.tps.target_control_points.unsqueeze(0)  # (1, N, 2)
        #source_pts = base_pts + theta                            # (B, N, 2)
        source_pts = torch.clamp(base_pts + theta, -1, 1)

        # ---- Warp cloth ----
        warped = self.tps(cloth, source_pts)

        return warped, theta, source_pts