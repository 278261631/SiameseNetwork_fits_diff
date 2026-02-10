from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese UNet（变化检测/差分分割）：
    - 共享 ResNet18 编码器，提取多尺度特征
    - 深层特征做 abs 差：fused = |f1_5 - f2_5|
    - 跳连使用 reference 分支的浅层特征（也可扩展为拼接差分）
    - 输出为 num_classes logits（未softmax）
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        enc = models.resnet18(weights=weights)

        old = enc.conv1
        enc.conv1 = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        if pretrained and input_channels == 1:
            with torch.no_grad():
                enc.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        self.enc = enc

        # Decoder: ResNet18 channels: layer1=64, layer2=128, layer3=256, layer4=512
        # We fuse at layer4 output (512, /32)
        self.up4 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)  # /16
        self.up3 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)  # /8
        self.up2 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)  # /4

        # Early skip: after conv1+bn1+relu is 64 channels at /2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # /1
            nn.ReLU(inplace=True),
            ConvBlock(32, 32),
        )

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _encode(self, x: torch.Tensor):
        # follow torchvision resnet forward to collect skips
        x0 = self.enc.conv1(x)
        x0 = self.enc.bn1(x0)
        x0 = self.enc.relu(x0)  # c=64, /2
        x1 = self.enc.maxpool(x0)  # /4
        x1 = self.enc.layer1(x1)  # c=64, /4
        x2 = self.enc.layer2(x1)  # c=128, /8
        x3 = self.enc.layer3(x2)  # c=256, /16
        x4 = self.enc.layer4(x3)  # c=512, /32
        return x0, x1, x2, x3, x4

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        r0, r1, r2, r3, r4 = self._encode(x1)
        _a0, _a1, _a2, _a3, a4 = self._encode(x2)

        fused = torch.abs(r4 - a4)

        d4 = self.up4(fused, r3)
        d3 = self.up3(d4, r2)
        d2 = self.up2(d3, r1)

        # bring to /2 then /1 using r0 resolution
        if d2.shape[-2:] != r0.shape[-2:]:
            d2 = F.interpolate(d2, size=r0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d2)

        logits = self.final(d1)  # [B,C,H,W]
        return logits

