from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_REF_RE = re.compile(r"(.*)_1_reference\.(png|fits)$", re.IGNORECASE)
_ALI_RE = re.compile(r"(.*)_2_aligned\.(png|fits)$", re.IGNORECASE)


@dataclass(frozen=True)
class PairSample:
    key: str
    x1_path: Path  # reference
    x2_path: Path  # aligned
    y: int  # 0=good, 1=bad by default


def _label_from_parent_dir(p: Path, good_name: str = "good", bad_name: str = "bad") -> int:
    parent = p.parent.name.lower()
    if parent == good_name.lower():
        return 0
    if parent == bad_name.lower():
        return 1
    raise ValueError(f"无法从父目录推断标签：{p}（期望目录名为 {good_name}/{bad_name}）")


def _pair_key_from_filename(name: str) -> Optional[str]:
    m = _REF_RE.match(name)
    if m:
        return m.group(1)
    m = _ALI_RE.match(name)
    if m:
        return m.group(1)
    return None


def build_pairs(
    data_dir: str | os.PathLike,
    ext: str = "png",
    good_name: str = "good",
    bad_name: str = "bad",
) -> List[PairSample]:
    """
    从 data_dir/{good,bad} 下扫描成对文件：
    - *_1_reference.{ext}
    - *_2_aligned.{ext}
    返回配对后的样本列表。
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录：{data_dir}")

    ext = ext.lower().lstrip(".")
    files: List[Path] = []
    for sub in [good_name, bad_name]:
        d = data_dir / sub
        if not d.exists():
            raise FileNotFoundError(f"找不到子目录：{d}")
        files.extend(sorted(d.glob(f"*.{ext}")))

    # group by key
    ref_by_key: dict[Tuple[str, int], Path] = {}
    ali_by_key: dict[Tuple[str, int], Path] = {}
    for p in files:
        key = _pair_key_from_filename(p.name)
        if not key:
            continue
        y = _label_from_parent_dir(p, good_name=good_name, bad_name=bad_name)
        k = (key, y)
        if "_1_reference" in p.name:
            ref_by_key[k] = p
        elif "_2_aligned" in p.name:
            ali_by_key[k] = p

    samples: List[PairSample] = []
    for k, ref in ref_by_key.items():
        if k not in ali_by_key:
            continue
        key, y = k
        samples.append(PairSample(key=key, x1_path=ref, x2_path=ali_by_key[k], y=y))

    if not samples:
        raise RuntimeError(
            f"没有在 {data_dir} 中找到可配对的数据。请确认文件命名符合 *_1_reference.{ext} 与 *_2_aligned.{ext}"
        )
    return sorted(samples, key=lambda s: (s.y, s.key))

