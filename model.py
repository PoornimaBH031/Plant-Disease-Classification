# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import timm

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

        magnitude = torch.abs(fft_filtered)

        freq_features = self.conv1x1(magnitude)

        return freq_features

# ==========================================================
# CA-SWIN TRANSFORMER
# ==========================================================

class CASwin(nn.Module):

    def __init__(
            self,
            embed_dim=512,
            num_heads=8):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.refine = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=1
        )

    def forward(self, x):

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(
            tokens,
            tokens,
            tokens
        )

        attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(
            B, C, H, W
        )

        attn_out = self.refine(attn_out)

        return attn_out


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

        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True
        )

        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                768,
                512,
                kernel_size=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        #self.ca_swin = CASwin()

        self.fft_module = FFTModule()

        self.cross_attention = CrossAttention()

        self.fusion = FusionModule()

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

    def forward(self, x):

        features = self.encoder(x)

        spatial = features[-1]

        spatial = spatial.permute(0, 3, 1, 2)

        spatial = self.feature_projection(
            spatial
        )

        #spatial = self.ca_swin(spatial)

        freq = self.fft_module(x)

        freq = F.interpolate(
            freq,
            size=spatial.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        attention = self.cross_attention(
            spatial,
            freq
        )

        fused = self.fusion(
            spatial,
            attention,
            freq
        )

        pooled = self.pool(fused)

        pooled = pooled.view(
            pooled.size(0),
            -1
        )

        x = self.fc1(pooled)

        x = F.relu(x)

        x = self.dropout(x)

        logits = self.fc2(x)

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