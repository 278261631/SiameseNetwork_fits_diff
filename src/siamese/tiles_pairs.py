from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_REF_RE = re.compile(r"(.*)_1_reference\.(fits|png)$", re.IGNORECASE)
_ALI_RE = re.compile(r"(.*)_2_aligned\.(fits|png)$", re.IGNORECASE)
_MSK_RE = re.compile(r"(.*)_mask\.(png|tif|tiff)$", re.IGNORECASE)


@dataclass(frozen=True)
class TileTriplet:
    key: str
    x1_path: Path  # reference
    x2_path: Path  # aligned
    mask_path: Path


def _key_from(name: str) -> Optional[str]:
    for r in (_REF_RE, _ALI_RE, _MSK_RE):
        m = r.match(name)
        if m:
            return m.group(1)
    return None


def build_tile_triplets(tiles_dir: str | os.PathLike) -> List[TileTriplet]:
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        raise FileNotFoundError(f"找不到 tiles 目录：{tiles_dir}")

    ref_by_key: dict[str, Path] = {}
    ali_by_key: dict[str, Path] = {}
    msk_by_key: dict[str, Path] = {}

    for p in sorted(tiles_dir.iterdir()):
        if not p.is_file():
            continue
        k = _key_from(p.name)
        if not k:
            continue
        lower = p.name.lower()
        if "_1_reference" in lower:
            ref_by_key[k] = p
        elif "_2_aligned" in lower:
            ali_by_key[k] = p
        elif "_mask" in lower:
            msk_by_key[k] = p

    keys = sorted(set(ref_by_key) & set(ali_by_key) & set(msk_by_key))
    if not keys:
        raise RuntimeError(f"在 {tiles_dir} 中没有找到可用的 triplet（reference/aligned/mask）")

    return [
        TileTriplet(key=k, x1_path=ref_by_key[k], x2_path=ali_by_key[k], mask_path=msk_by_key[k]) for k in keys
    ]

