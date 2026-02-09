from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.siamese.data_pairs import build_pairs
from src.siamese.dataset import PairedAugment, SiameseDatasetConfig, SiamesePairDataset
from src.siamese.model_siamese_resnet import SiameseResNet18
from src.siamese.utils import ensure_dir, save_json, set_seed, sigmoid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="数据目录（包含 good/ 和 bad/）")
    p.add_argument("--ext", type=str, default="png", choices=["png", "fits"], help="使用的文件扩展名")
    p.add_argument("--out_dir", type=str, default="runs/siamese_cls", help="输出目录")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--image_size", type=int, default=100)
    p.add_argument("--pretrained", action="store_true", help="使用ResNet18预训练权重（可能需要联网下载）")
    p.add_argument("--num_workers", type=int, default=0, help="Windows建议0或2")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys = []
    probs = []
    for x1, x2, y, _key in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        logits = model(x1, x2)
        p = sigmoid(logits)
        ys.append(y.detach().cpu())
        probs.append(p.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    y_pred = (y_prob >= 0.5).astype("int64")
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    out_dir = ensure_dir(args.out_dir)

    # 1) build pairs
    samples = build_pairs(args.data_dir, ext=args.ext)
    ys = torch.tensor([s.y for s in samples])

    # 2) stratified split
    idx_good = (ys == 0).nonzero(as_tuple=True)[0].tolist()
    idx_bad = (ys == 1).nonzero(as_tuple=True)[0].tolist()
    torch.random.manual_seed(args.seed)
    idx_good = torch.tensor(idx_good)[torch.randperm(len(idx_good))].tolist()
    idx_bad = torch.tensor(idx_bad)[torch.randperm(len(idx_bad))].tolist()

    nvg = max(1, int(len(idx_good) * args.val_ratio))
    nvb = max(1, int(len(idx_bad) * args.val_ratio))
    val_idx = set(idx_good[:nvg] + idx_bad[:nvb])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]

    cfg = SiameseDatasetConfig(image_size=args.image_size, normalize=True)
    train_ds = SiamesePairDataset(
        train_samples,
        config=cfg,
        augment=PairedAugment(p_hflip=0.5, p_vflip=0.0, max_rotate_deg=0.0),
    )
    val_ds = SiamesePairDataset(val_samples, config=cfg, augment=None)

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

    # 3) model
    model = SiameseResNet18(input_channels=1, pretrained=args.pretrained).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        "args": vars(args),
        "n_total": len(samples),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "label_map": {"0": "good", "1": "bad"},
    }
    save_json(out_dir / "meta.json", meta)

    best_auc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for x1, x2, y, _key in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            bs = x1.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            n += bs

        train_loss = total_loss / max(1, n)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f} val_acc={val_metrics['acc']:.3f} val_auc={val_metrics['auc']:.3f} "
            f"({dt:.1f}s)"
        )

        # save last
        torch.save({"model": model.state_dict(), "meta": meta}, last_path)

        # save best by auc (fallback to acc if auc is nan)
        score = val_metrics["auc"]
        if score != score:  # nan
            score = val_metrics["acc"]
        if score > best_auc:
            best_auc = score
            torch.save({"model": model.state_dict(), "meta": meta}, best_path)

    print(f"训练完成。best={best_path} last={last_path}")


if __name__ == "__main__":
    main()

