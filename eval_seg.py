from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.siamese.dataset_tiles import SiameseTilesSegDataset, TilesDatasetConfig
from src.siamese.model_siamese_unet import SiameseUNet
from src.siamese.tiles_pairs import build_tile_triplets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_dir", type=str, required=True, help="例如 test_data 或 test_data/tiles")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--resize_to", type=int, default=0, help="0不缩放")
    p.add_argument("--crop_size", type=int, default=0, help="0不裁剪（全图评估）")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_seg] device={device}")

    triplets = build_tile_triplets(args.tiles_dir)
    resize_to = None if args.resize_to <= 0 else int(args.resize_to)
    ds = SiameseTilesSegDataset(
        triplets,
        config=TilesDatasetConfig(crop_size=args.crop_size, resize_to=resize_to),
        train=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SiameseUNet(input_channels=1, num_classes=args.num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for x1, x2, mask, _key in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        mask_np = mask.numpy().astype(np.int64)  # [B,H,W]
        logits = model(x1, x2)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

        # accumulate confusion matrix
        for b in range(pred.shape[0]):
            p = pred[b].reshape(-1)
            g = mask_np[b].reshape(-1)
            for c in range(args.num_classes):
                for d in range(args.num_classes):
                    cm[c, d] += int(np.sum((g == c) & (p == d)))

    pixel_acc = float(np.trace(cm) / max(1, cm.sum()))

    ious = []
    per_class = {}
    for c in range(args.num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else float("nan")
        per_class[str(c)] = iou
        if denom > 0:
            ious.append(iou)
    miou = float(np.mean(ious)) if ious else float("nan")

    print(f"tiles_dir={args.tiles_dir}")
    print(f"pixel_acc={pixel_acc:.4f}  mean_iou={miou:.4f}")
    print("per_class_iou=", per_class)


if __name__ == "__main__":
    main()

