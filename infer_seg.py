from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.siamese.dataset_tiles import SiameseTilesSegDataset, TilesDatasetConfig
from src.siamese.model_siamese_unet import SiameseUNet
from src.siamese.tiles_pairs import build_tile_pairs, build_tile_triplets
from src.siamese.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_dir", type=str, default="data/tiles")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出目录；留空时默认输出到 tiles_dir（与数据同目录）",
    )
    p.add_argument("--crop_size", type=int, default=0, help="0不裁剪（全图推理）")
    p.add_argument("--resize_to", type=int, default=0, help="0不缩放（保持原始位宽/动态范围，避免8-bit量化）")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer_seg] device={device}")

    # 默认把输出写到数据目录，便于后续分析时直接与原始样本对应
    out_root = args.out_dir.strip() if args.out_dir else ""
    out_dir = ensure_dir(out_root or args.tiles_dir)
    resize_to = None if args.resize_to <= 0 else int(args.resize_to)
    cfg = TilesDatasetConfig(crop_size=args.crop_size, resize_to=resize_to)

    # 优先使用带mask的triplet；若测试目录没有mask，则fallback为pair推理模式
    try:
        triplets = build_tile_triplets(args.tiles_dir)
        ds = SiameseTilesSegDataset(triplets, config=cfg, train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        has_mask = True
    except RuntimeError:
        pairs = build_tile_pairs(args.tiles_dir)
        from src.siamese.dataset_tiles import SiameseTilesInferDataset

        ds = SiameseTilesInferDataset(pairs, config=cfg)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        has_mask = False

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SiameseUNet(input_channels=1, num_classes=args.num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for batch in loader:
        if has_mask:
            x1, x2, _mask, key = batch
        else:
            x1, x2, key = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        logits = model(x1, x2)  # [B,C,H,W]
        prob = torch.softmax(logits, dim=1)  # [B,C,H,W]
        pred = torch.argmax(prob, dim=1).cpu().numpy().astype(np.uint8)  # [B,H,W]

        for i, k in enumerate(key):
            # 预测mask默认按数据集标注命名规则输出为 *_mask.png。
            # 若同名文件已存在（常见于带GT的目录），避免覆盖真值，改存为 *_mask_pred.png。
            out_png = Path(out_dir) / f"{k}_mask.png"
            if out_png.exists():
                out_png = Path(out_dir) / f"{k}_mask_pred.png"
            Image.fromarray(pred[i], mode="L").save(out_png)

            # 同时保存每类概率（npz），便于后处理/可视化
            out_npz = Path(out_dir) / f"{k}_prob.npz"
            np.savez_compressed(out_npz, prob=prob[i].cpu().numpy().astype(np.float32))

    print(f"已输出到：{out_dir}")


if __name__ == "__main__":
    main()

