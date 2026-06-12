# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from timm.models.swin_transformer import SwinTransformerBlock

# ==========================================================
# FFT FEATURE EXTRACTION MODULE
# ==========================================================

class FFTModule(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()

        self.conv1x1 = nn.Conv2d(3, out_channels, kernel_size=1)

    def forward(self, x):

        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)

        B, C, H, W = fft.shape

        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )

        center_y = H // 2
        center_x = W // 2

        radius = min(H, W) // 8

        mask = (
            ((yy - center_y) ** 2 +
             (xx - center_x) ** 2)
            > radius ** 2
        ).float()

        mask = mask.unsqueeze(0).unsqueeze(0)

        fft_filtered = fft * mask

        #==================================
        # INVERSE FFT RECONSTRUCTION
        #==================================

        ifft = torch.fft.ifftshift(
            fft_filtered
        )

        reconstructed = torch.fft.ifft2(
            ifft
        )

        reconstructed = torch.abs(
            reconstructed
        )

        freq_features = self.conv1x1(
            reconstructed
        )

        return freq_features

# ==========================================================
# CA-SWIN TRANSFORMER
# ==========================================================

class CASwin(nn.Module):

    def __init__(self):

        super().__init__()

        self.window1 = SwinTransformerBlock(
            dim=512,
            input_resolution=(7,7),
            num_heads=8,
            window_size=7,
            shift_size=0
        )

        self.window2 = SwinTransformerBlock(
            dim=512,
            input_resolution=(7,7),
            num_heads=8,
            window_size=7,
            shift_size=3
        )

    def forward(self, x):

        B,C,H,W = x.shape

        # BCHW → BHWC
        x = x.permute(
            0,
            2,
            3,
            1
        )

        x = self.window1(x)

        x = self.window2(x)

        # BHWC → BCHW
        x = x.permute(
            0,
            3,
            1,
            2
        )

        return x

# ==========================================================
# CROSS ATTENTION
# ==========================================================

class CrossAttention(nn.Module):

    def __init__(
            self,
            dim=512,
            num_heads=8):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(
            self,
            spatial,
            frequency):

        B, C, H, W = spatial.shape

        q = spatial.flatten(2).transpose(1, 2)

        k = frequency.flatten(2).transpose(1, 2)

        v = frequency.flatten(2).transpose(1, 2)

        out, _ = self.attn(q, k, v)

        out = out.transpose(1, 2)

        out = out.reshape(B, C, H, W)

        return out


# ==========================================================
# FEATURE FUSION
# ==========================================================

class FusionModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.fusion_conv = nn.Sequential(

            nn.Conv2d(
                1536,
                512,
                kernel_size=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True)
        )

    def forward(
            self,
            spatial,
            attention,
            frequency):

        fused = torch.cat(
            [spatial,
             attention,
             frequency],
            dim=1
        )

        fused = self.fusion_conv(fused)

        return fused


# ==========================================================
# MAIN MODEL
# ==========================================================

class FFT_CASwin_Classifier(nn.Module):

    def __init__(
            self,
            num_classes=10):

        super().__init__()

        #=====================================
        # RESNET18 BACKBONE
        #=====================================

        resnet = models.resnet18(
            weights='DEFAULT'
        )

        self.encoder = nn.Sequential(

            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        #=====================================
        # FEATURE ALIGNMENT
        #=====================================

        self.feature_projection = nn.Sequential(

            nn.Conv2d(
                512,
                512,
                kernel_size=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU()
        )

        #=====================================
        # ENABLE CA-SWIN
        #=====================================

        self.ca_swin = CASwin()

        #=====================================

        self.fft_module = FFTModule()

        self.cross_attention = CrossAttention()

        self.fusion = FusionModule()

        self.decoder = nn.Sequential(

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.Conv2d(
                256,
                512,
                kernel_size=1
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(
            512,
            256
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

    def forward(self,x):

        #=========================
        # RESNET
        #=========================

        spatial = self.encoder(x)

        spatial = self.feature_projection(
            spatial
        )

        #=========================
        # CA-SWIN
        #=========================

        spatial = self.ca_swin(
            spatial
        )

        #=========================
        # FFT
        #=========================

        freq = self.fft_module(
            x
        )

        freq = F.interpolate(
            freq,
            size=spatial.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        #=========================
        # CROSS ATTENTION
        #=========================

        attention = self.cross_attention(
            spatial,
            freq
        )

        fused = self.fusion(
            spatial,
            attention,
            freq
        )

        fused = self.decoder(
            fused
        )

        pooled = self.pool(
            fused
        )

        pooled = pooled.view(
            pooled.size(0),
            -1
        )

        x = self.fc1(
            pooled
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        logits = self.fc2(
            x
        )

        return logits

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model = FFT_CASwin_Classifier(
        num_classes=10
    )

    x = torch.randn(
        2,
        3,
        224,
        224
    )

    y = model(x)

    print(y.shape)