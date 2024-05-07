import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from typing import Tuple


class SSLHead(nn.Module):
    def __init__(self, spatial_dimensions: int,
                 in_channels: int,
                 feature_size: int,
                 stochastic_depth_rate: float,
                 dropout_rate: float,
                 use_checkpoint: bool,
                 depths: list,
                 heads: list,
                 upsample: str = "vae", dim: int = 768) -> None:
        super().__init__()
        patch_size = ensure_tuple_rep(2, spatial_dimensions)
        window_size = ensure_tuple_rep(7, spatial_dimensions)
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=dropout_rate,
            attn_drop_rate=0.0,
            drop_path_rate=stochastic_depth_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dimensions,
        )
        self.rotation_pre = nn.Linear(feature_size * 16, dim)
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Linear(feature_size * 16, dim)
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(feature_size * 16, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x_out = self.swinViT(x.contiguous())[4]  # Deepest embedding
        _, c, h, w, d = x_out.shape  # 12096 x 3 x 3 x 3
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])  # 2 x 12096
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])  # 2 x 12096
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec
