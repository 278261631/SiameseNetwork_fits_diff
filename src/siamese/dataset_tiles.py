from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .fits_io import read_fits_image, robust_normalize
from .tiles_pairs import TileTriplet


@dataclass
class TilesDatasetConfig:
    crop_size: int = 256  # 训练随机裁剪尺寸
    resize_to: Optional[int] = None  # 若设置，则先把整图缩放到该尺寸（方形），再裁剪
    normalize_to_minus1_1: bool = True


def _pil_resize(arr: np.ndarray, size: int, resample=Image.BILINEAR) -> np.ndarray:
    im = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    im = im.resize((size, size), resample=resample)
    a = np.array(im).astype(np.float32) / 255.0
    return a


def _random_crop_coords(h: int, w: int, cs: int) -> Tuple[int, int]:
    if h == cs and w == cs:
        return 0, 0
    top = random.randint(0, max(0, h - cs))
    left = random.randint(0, max(0, w - cs))
    return top, left


def _center_crop_coords(h: int, w: int, cs: int) -> Tuple[int, int]:
    top = max(0, (h - cs) // 2)
    left = max(0, (w - cs) // 2)
    return top, left


class SiameseTilesSegDataset(Dataset):
    """
    输出：
    - x1: [1,H,W] float32
    - x2: [1,H,W] float32
    - mask: [H,W] int64 (code 0/1/2)
    """

    def __init__(
        self,
        triplets: Sequence[TileTriplet],
        *,
        config: TilesDatasetConfig = TilesDatasetConfig(),
        train: bool = True,
    ) -> None:
        self.triplets = list(triplets)
        self.cfg = config
        self.train = train

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        t = self.triplets[idx]
        x1 = robust_normalize(read_fits_image(t.x1_path))
        x2 = robust_normalize(read_fits_image(t.x2_path))

        m = Image.open(t.mask_path).convert("L")
        mask = np.array(m).astype(np.int64)

        # 可选整体缩放到固定方形尺寸
        if self.cfg.resize_to is not None:
            s = int(self.cfg.resize_to)
            x1 = _pil_resize(x1, s, resample=Image.BILINEAR)
            x2 = _pil_resize(x2, s, resample=Image.BILINEAR)
            m_im = Image.fromarray(mask.astype(np.uint8), mode="L").resize((s, s), resample=Image.NEAREST)
            mask = np.array(m_im).astype(np.int64)

        h, w = x1.shape
        cs = int(self.cfg.crop_size)
        if cs > 0 and (h >= cs and w >= cs):
            if self.train:
                top, left = _random_crop_coords(h, w, cs)
            else:
                top, left = _center_crop_coords(h, w, cs)
            x1 = x1[top : top + cs, left : left + cs]
            x2 = x2[top : top + cs, left : left + cs]
            mask = mask[top : top + cs, left : left + cs]

        # 同步随机翻转（增强）
        if self.train:
            if random.random() < 0.5:
                x1 = np.fliplr(x1).copy()
                x2 = np.fliplr(x2).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                x1 = np.flipud(x1).copy()
                x2 = np.flipud(x2).copy()
                mask = np.flipud(mask).copy()

        x1_t = torch.from_numpy(x1).unsqueeze(0)  # [1,H,W]
        x2_t = torch.from_numpy(x2).unsqueeze(0)
        if self.cfg.normalize_to_minus1_1:
            x1_t = x1_t * 2.0 - 1.0
            x2_t = x2_t * 2.0 - 1.0

        mask_t = torch.from_numpy(mask).long()
        return x1_t, x2_t, mask_t, t.key

