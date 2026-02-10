from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits


def read_fits_image(path: str | Path) -> np.ndarray:
    """
    读取FITS为二维float32数组，处理NaN/Inf。
    """
    path = Path(path)
    with fits.open(path) as hdul:
        arr = hdul[0].data
    if arr is None:
        raise ValueError(f"FITS中没有数据: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        # 有的FITS可能带额外维度，取第一帧
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"期望2D图像，得到 shape={arr.shape} for {path}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def robust_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    按分位数做鲁棒归一化到 [0,1]，再可在后续转成 [-1,1]。
    """
    x = img.astype(np.float32, copy=False)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return x

