from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.siamese.model_siamese_unet import SiameseUNet
from src.siamese.tiles_pairs import build_tile_triplets
from src.siamese.dataset_tiles import SiameseTilesSegDataset, TilesDatasetConfig
from src.siamese.utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_dir", type=str, default="data/tiles", help="训练tiles目录")
    p.add_argument("--val_tiles_dir", type=str, default="", help="可选：独立验证tiles目录（如 test_data/tiles）。提供则不再从训练集随机划分val。")
    p.add_argument("--out_dir", type=str, default="runs/siamese_unet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1, help="全图(1024x1024)训练更稳妥，默认1")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.33)
    p.add_argument("--crop_size", type=int, default=0, help="0表示不裁剪（使用完整图像）")
    p.add_argument("--resize_to", type=int, default=0, help="0表示不缩放（保持原始位宽/动态范围，避免8-bit量化）")
    p.add_argument("--num_classes", type=int, default=3, help="来自mask codebook：0/1/2 -> 3类")
    p.add_argument("--pretrained", action="store_true", help="使用ResNet18预训练权重（可能需要联网下载）")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def fast_metrics(logits: torch.Tensor, mask: torch.Tensor, num_classes: int) -> dict:
    """
    简单像素级指标：
    - pixel_acc
    - mean_iou（对所有类求平均，忽略空类）
    """
    pred = torch.argmax(logits, dim=1)  # [B,H,W]
    correct = (pred == mask).float().mean().item()

    ious = []
    for c in range(num_classes):
        p = pred == c
        g = mask == c
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        if union > 0:
            ious.append(inter / union)
    miou = float(np.mean(ious)) if ious else float("nan")
    return {"pixel_acc": float(correct), "mean_iou": miou}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    out_dir = ensure_dir(args.out_dir)

    train_triplets = build_tile_triplets(args.tiles_dir)
    if args.val_tiles_dir:
        val_triplets = build_tile_triplets(args.val_tiles_dir)
        train_trip = train_triplets
        val_trip = val_triplets
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(train_triplets))
        rng.shuffle(idx)
        n_val = max(1, int(len(train_triplets) * args.val_ratio))
        val_idx = set(idx[:n_val].tolist())
        train_trip = [t for i, t in enumerate(train_triplets) if i not in val_idx]
        val_trip = [t for i, t in enumerate(train_triplets) if i in val_idx]

    resize_to = None if args.resize_to <= 0 else int(args.resize_to)
    cfg_train = TilesDatasetConfig(crop_size=args.crop_size, resize_to=resize_to)
    cfg_val = TilesDatasetConfig(crop_size=args.crop_size, resize_to=resize_to)
    train_ds = SiameseTilesSegDataset(train_trip, config=cfg_train, train=True)
    val_ds = SiameseTilesSegDataset(val_trip, config=cfg_val, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = SiameseUNet(input_channels=1, num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        "args": vars(args),
        "n_total_train": len(train_triplets),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    save_json(out_dir / "meta.json", meta)

    best_miou = -1.0
    best_path = Path(out_dir) / "best.pt"
    last_path = Path(out_dir) / "last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0

        for x1, x2, mask, _key in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()

            bs = x1.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            n += bs

        train_loss = total_loss / max(1, n)

        # val
        model.eval()
        mets = []
        for x1, x2, mask, _key in val_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            mask = mask.to(device)
            logits = model(x1, x2)
            mets.append(fast_metrics(logits, mask, args.num_classes))
        val_pixel = float(np.mean([m["pixel_acc"] for m in mets])) if mets else float("nan")
        val_miou = float(np.mean([m["mean_iou"] for m in mets])) if mets else float("nan")

        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
            f"val_pixel_acc={val_pixel:.3f} val_mIoU={val_miou:.3f} ({dt:.1f}s)"
        )

        torch.save({"model": model.state_dict(), "meta": meta}, last_path)
        score = val_miou
        if score != score:  # nan
            score = val_pixel
        if score > best_miou:
            best_miou = score
            torch.save({"model": model.state_dict(), "meta": meta}, best_path)

    print(f"训练完成。best={best_path} last={last_path}")


if __name__ == "__main__":
    main()

