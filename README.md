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

## ✅ 新增：Siamese UNet 像素级分割（tiles + mask）

你更新后的数据结构：

- `data/tiles/*_1_reference.fits`
- `data/tiles/*_2_aligned.fits`
- `data/tiles/*_mask.png`
- `data/mask_codebook.json`：mask 像素值 code 映射

`mask_codebook.json` 当前为 3 类：

- 0: normal
- 1: good
- 2: bad

### 分割训练

```bash
python train_seg.py --tiles_dir data/tiles --out_dir runs/siamese_unet --epochs 30 --crop_size 256 --resize_to 512
```

说明：

- `resize_to` 会先把整图缩放到方形尺寸，再做 `crop_size` 的随机裁剪，CPU 也能跑通
- 模型输出为 3 类 logits，loss 使用 `CrossEntropyLoss`

### 分割推理（导出预测 mask + 每类概率）

```bash
python infer_seg.py --tiles_dir data/tiles --ckpt runs/siamese_unet/best.pt --out_dir runs/infer_unet --crop_size 512 --resize_to 512
```

输出：

- `runs/infer_unet/*_pred.png`：预测类别图（像素值为 0/1/2）
- `runs/infer_unet/*_prob.npz`：每类概率（shape `[C,H,W]`）

## 使用 test_data 做验证/测试

如果你有独立的测试集目录（例如 `test_data/tiles`），可以：

- **训练时指定验证集目录**（不从训练集随机划分）：

```bash
python train_seg.py --tiles_dir data/tiles --val_tiles_dir test_data/tiles --out_dir runs/siamese_unet
```

- **在 test_data 上评估指标**（像素精度 / mIoU / 每类 IoU）：

```bash
python eval_seg.py --tiles_dir test_data/tiles --ckpt runs/siamese_unet/best.pt --resize_to 512 --crop_size 512
```

也可以用一键脚本（Windows）：

- `setup_venv.bat`：创建 `.venv` 并安装依赖
- `run_train_test.bat`：训练 + 测试 + 导出预测
- `run_test_only.bat`：**只测试/导出预测**（不训练）
