from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseResNet18(nn.Module):
    """
    共享编码器的孪生网络（二分类）：
    - backbone: ResNet18，第一层改为单通道输入
    - fusion: abs(f1-f2)（也可扩展为 concat）
    - head: MLP 输出 logits（shape [B]）
    """

    def __init__(self, input_channels: int = 1, pretrained: bool = False, dropout: float = 0.1):
        super().__init__()

        if pretrained:
            # torchvision 0.15: weights 方式
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = models.resnet18(weights=weights)

        # 改第一层以接受单通道
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        # 若使用预训练，单通道可用 RGB 权重均值初始化
        if pretrained and input_channels == 1:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        # 拿掉 fc，保留到 avgpool
        self.backbone = backbone
        self.feature_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # resnet18 forward 到 avgpool + flatten 后的 embedding
        z = self.backbone(x)  # [B, feature_dim]
        return z

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        fused = torch.abs(z1 - z2)
        logits = self.head(fused).squeeze(1)  # [B]
        return logits

