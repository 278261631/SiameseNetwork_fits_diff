from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .data_pairs import PairSample


def _read_png_grayscale(path: Path) -> Image.Image:
    im = Image.open(path)
    # 强制为单通道
    if im.mode != "L":
        im = im.convert("L")
    return im


class PairedAugment:
    """
    对 (x1, x2) 同步做随机增强，保持两张图之间的几何一致性。
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.0,
        max_rotate_deg: float = 0.0,
    ) -> None:
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.max_rotate_deg = max_rotate_deg

    def __call__(self, x1: Image.Image, x2: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.p_hflip > 0 and random.random() < self.p_hflip:
            x1 = F.hflip(x1)
            x2 = F.hflip(x2)
        if self.p_vflip > 0 and random.random() < self.p_vflip:
            x1 = F.vflip(x1)
            x2 = F.vflip(x2)
        if self.max_rotate_deg and self.max_rotate_deg > 0:
            deg = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            # nearest 对天文图更“硬”，但不会引入灰度插值伪影太明显；这里用 bilinear 更平滑
            x1 = F.rotate(x1, deg, interpolation=F.InterpolationMode.BILINEAR)
            x2 = F.rotate(x2, deg, interpolation=F.InterpolationMode.BILINEAR)
        return x1, x2


@dataclass
class SiameseDatasetConfig:
    image_size: int = 100  # 数据本身就是 100x100，保留默认
    normalize: bool = True


class SiamesePairDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PairSample],
        *,
        config: SiameseDatasetConfig = SiameseDatasetConfig(),
        augment: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
    ) -> None:
        self.samples = list(samples)
        self.config = config
        self.augment = augment

        t: list[Callable] = []
        if config.image_size:
            t.append(T.Resize((config.image_size, config.image_size)))
        t.append(T.ToTensor())  # [0,1] float32, shape [1,H,W]
        if config.normalize:
            # 经验归一化：单通道灰度，映射到大致零均值范围
            t.append(T.Normalize(mean=[0.5], std=[0.5]))
        self.to_tensor = T.Compose(t)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x1 = _read_png_grayscale(s.x1_path)
        x2 = _read_png_grayscale(s.x2_path)

        if self.augment is not None:
            x1, x2 = self.augment(x1, x2)

        x1 = self.to_tensor(x1)
        x2 = self.to_tensor(x2)
        y = torch.tensor(s.y, dtype=torch.long)
        return x1, x2, y, s.key

