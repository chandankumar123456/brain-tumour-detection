"""
Multi-Path Fusion Network with Global Attention for Brain Tumor Segmentation
Based on: Wu, D., Qiu, S., Qin, J., & Zhao, P. (2023).
Multi-Path Fusion Network Based Global Attention for Brain Tumor Segmentation.
Proceedings of ISAIMS 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """SE-style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention using avg+max channel pooling."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class GlobalAttentionModule(nn.Module):
    """CBAM-style: Channel Attention → Spatial Attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ─────────────────────────────────────────────────────────────
# Three Parallel Encoder Paths
# ─────────────────────────────────────────────────────────────

class LowLevelPath(nn.Module):
    """Path 1: 3 conv layers – captures fine/edge features."""
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_ch, base),
            ConvBNReLU(base, base),
            ConvBNReLU(base, base * 2),
        )

    def forward(self, x):
        return self.layers(x)


class MidLevelPath(nn.Module):
    """Path 2: 5 conv layers – captures intermediate features."""
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_ch, base),
            ConvBNReLU(base, base),
            ConvBNReLU(base, base * 2),
            ConvBNReLU(base * 2, base * 2),
            ConvBNReLU(base * 2, base * 2),
        )

    def forward(self, x):
        return self.layers(x)


class HighLevelPath(nn.Module):
    """Path 3: 7 conv layers – captures global/semantic features."""
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_ch, base),
            ConvBNReLU(base, base),
            ConvBNReLU(base, base * 2),
            ConvBNReLU(base * 2, base * 2),
            ConvBNReLU(base * 2, base * 4),
            ConvBNReLU(base * 4, base * 4),
            ConvBNReLU(base * 4, base * 2),
        )

    def forward(self, x):
        return self.layers(x)


# ─────────────────────────────────────────────────────────────
# Encoder (with down-sampling for skip connections)
# ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes differ
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# Main Model: Multi-Path Fusion Network + Global Attention
# ─────────────────────────────────────────────────────────────

class MultiPathFusionNet(nn.Module):
    """
    Multi-Path Fusion Network with Global Attention for Brain Tumor Segmentation.
    Input : (B, in_channels, 256, 256)  – MRI image
    Output: (B, num_classes, 256, 256)  – Per-pixel class logits
    """

    def __init__(self, in_channels=4, num_classes=4, base=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Stem (shared input projection) ─────────────────
        self.stem = ConvBNReLU(in_channels, base)

        # ── Three parallel paths ────────────────────────────
        self.path_low  = LowLevelPath(base, base)         # → base*2 ch
        self.path_mid  = MidLevelPath(base, base)         # → base*2 ch
        self.path_high = HighLevelPath(base, base)        # → base*2 ch

        fused_ch = base * 2 * 3   # 192  (concat of 3 paths)

        # ── Feature Fusion (1×1 conv to reduce channels) ────
        self.fusion = ConvBNReLU(fused_ch, base * 4, kernel_size=1, padding=0)

        # ── Global Attention (CBAM-style) ───────────────────
        self.attention = GlobalAttentionModule(base * 4)

        # ── U-Net Encoder for skip connections ─────────────
        enc_in = base * 4
        self.enc1 = EncoderBlock(enc_in, base * 4)    # 128 → 64
        self.enc2 = EncoderBlock(base * 4, base * 8)  # 64  → 32
        self.enc3 = EncoderBlock(base * 8, base * 16) # 32  → 16

        # ── Bottleneck ──────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBNReLU(base * 16, base * 32),
            ConvBNReLU(base * 32, base * 16),
        )

        # ── Decoder (U-Net style skip connections) ──────────
        self.dec3 = DecoderBlock(base * 16, base * 16, base * 8)
        self.dec2 = DecoderBlock(base * 8,  base * 8,  base * 4)
        self.dec1 = DecoderBlock(base * 4,  base * 4,  base * 2)

        # ── Final 1×1 classifier ────────────────────────────
        self.head = nn.Conv2d(base * 2, num_classes, kernel_size=1)

    def forward(self, x):
        # Shared stem
        s = self.stem(x)                    # (B, base, H, W)

        # Multi-path feature extraction
        f_low  = self.path_low(s)           # (B, base*2, H, W)
        f_mid  = self.path_mid(s)           # (B, base*2, H, W)
        f_high = self.path_high(s)          # (B, base*2, H, W)

        # Feature fusion via concatenation + 1×1 conv
        fused = torch.cat([f_low, f_mid, f_high], dim=1)   # (B, base*6, H, W)
        fused = self.fusion(fused)          # (B, base*4, H, W)

        # Global Attention
        fused = self.attention(fused)       # (B, base*4, H, W)

        # Encoder + skip connections
        d1, skip1 = self.enc1(fused)
        d2, skip2 = self.enc2(d1)
        d3, skip3 = self.enc3(d2)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder
        up3 = self.dec3(b,  skip3)
        up2 = self.dec2(up3, skip2)
        up1 = self.dec1(up2, skip1)

        # Resize to input dimensions
        logits = self.head(up1)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


# ─────────────────────────────────────────────────────────────
# Dice Loss + Cross-Entropy (for training)
# ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        loss = 0.0
        for c in range(num_classes):
            p = pred[:, c]
            t = (target == c).float()
            intersection = (p * t).sum()
            loss += 1.0 - (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return loss / num_classes


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.alpha * self.dice(pred, target) + (1 - self.alpha) * self.ce(pred, target)


# ─────────────────────────────────────────────────────────────
# Dice Score Metric
# ─────────────────────────────────────────────────────────────

def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, class_idx: int, smooth=1e-6) -> float:
    """Compute per-class Dice score."""
    p = (pred_mask == class_idx).float()
    t = (true_mask == class_idx).float()
    intersection = (p * t).sum().item()
    return (2 * intersection + smooth) / (p.sum().item() + t.sum().item() + smooth)


def compute_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, class_idx: int = 1) -> float:
    """Compute Dice score (as percentage) for a given class."""
    return round(dice_score(pred_mask, true_mask, class_idx) * 100, 1)


if __name__ == "__main__":
    # Quick sanity check
    model = MultiPathFusionNet(in_channels=1, num_classes=2)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (2, 2, 256, 256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
