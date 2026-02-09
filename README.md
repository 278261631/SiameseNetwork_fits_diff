## 孪生网络（Siamese）对齐图像质量分析（good/bad）

这个仓库当前的数据组织为：

- `data/good/*.png` / `data/bad/*.png`
- 文件名成对出现：
  - `*_1_reference.png`
  - `*_2_aligned.png`

脚本会自动把同一 `key` 的 reference/aligned 配对，并以目录名 `good=0`、`bad=1` 作为标签，训练一个**共享编码器的孪生 ResNet18 二分类模型**。

## 环境

你当前环境已安装 `torch==2.0.1+cpu`、`torchvision==0.15.2+cpu`，依赖见 `requirements.txt`。

## 训练

在项目根目录运行：

```bash
python train.py --data_dir data --ext png --epochs 30 --batch_size 16 --out_dir runs/siamese_cls
```

- 默认在 CPU 上训练（数据量很小，能跑通）
- 若要用预训练权重（需要能下载权重文件）：

```bash
python train.py --pretrained
```

训练输出：

- `runs/siamese_cls/meta.json`
- `runs/siamese_cls/best.pt`
- `runs/siamese_cls/last.pt`

## 推理（导出 CSV）

```bash
python infer.py --data_dir data --ext png --ckpt runs/siamese_cls/best.pt --out_dir runs/infer
```

输出：

- `runs/infer/predictions.csv`：每对样本的 `prob_bad`（越大越像 bad）

## 说明

你给的参考伪代码是“孪生 UNet 输出概率图”的分割/变化检测结构；但当前数据只有 `good/bad` 的**图像级标签**，没有像素级 mask，因此这里先实现了最直接可监督的**孪生分类**流程。
如果你后续提供像素级标注（或希望做弱监督输出热力图/差分图），我可以把模型升级为孪生 UNet 并输出 2D 概率图。

