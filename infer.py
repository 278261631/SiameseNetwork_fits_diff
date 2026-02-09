from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.siamese.data_pairs import build_pairs
from src.siamese.dataset import SiameseDatasetConfig, SiamesePairDataset
from src.siamese.model_siamese_resnet import SiameseResNet18
from src.siamese.utils import ensure_dir, sigmoid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--ext", type=str, default="png", choices=["png", "fits"])
    p.add_argument("--ckpt", type=str, required=True, help="训练输出的 best.pt/last.pt")
    p.add_argument("--out_dir", type=str, default="runs/infer")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    out_dir = ensure_dir(args.out_dir)
    samples = build_pairs(args.data_dir, ext=args.ext)
    ds = SiamesePairDataset(samples, config=SiameseDatasetConfig(image_size=args.image_size, normalize=True))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SiameseResNet18(input_channels=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_csv = Path(out_dir) / "predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "y_true", "prob_bad", "pred"])
        for x1, x2, y, key in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            logits = model(x1, x2)
            prob = sigmoid(logits).cpu().numpy()
            y_np = y.cpu().numpy()
            pred = (prob >= 0.5).astype("int64")
            for k, yt, pb, pr in zip(key, y_np, prob, pred):
                w.writerow([k, int(yt), float(pb), int(pr)])

    print(f"已输出：{out_csv}")


if __name__ == "__main__":
    main()

