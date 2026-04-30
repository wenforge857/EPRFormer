# EPRFormer

<p align="center">
  <b>A prompt-guided point segmentation network for X-ray battery plate endpoint detection.</b>
</p>

<p align="center">
  <a href="README_CN.md">Chinese</a> |
  <a href="https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD">PBD5K / X-ray-PBD</a> |
  <a href="CITATION.bib">Citation</a>
</p>

EPRFormer is a standalone PyTorch implementation for **point-level electrode endpoint detection** in X-ray power battery images. It follows the PBD5K task protocol, predicts positive and negative electrode endpoint response maps, and exports endpoint coordinates in the official `.npy` format.

The repository is self-contained: model code, training, inference, checkpoint evaluation, official-style metrics, reusable utilities, figures, and citation files are included. The upstream `X-ray-PBD` repository is acknowledged and referenced, but it is not required at runtime.

<p align="center">
  <img src="figures/Fig2.png" alt="EPRFormer overview" width="820">
</p>

## What Makes EPRFormer Different

EPRFormer is designed around the error sources of power battery endpoint detection rather than treating the task as generic detection or semantic segmentation.

- **Prompt-guided bi-level routing attention (PG-BRA)** selects relevant prompt regions before token-level cross attention, reducing interference from mismatched prompt areas.
- **Lightweight multi-scale linear attention (LMLA)** models long-range electrode endpoint ordering while keeping dense prediction efficient.
- **Content-aware detail decoder (CADD)** improves endpoint response recovery and reduces coordinate drift caused by upsampling.
- **Point, line, and counting supervision** preserve the PBD-style multi-clue training paradigm for number consistency and endpoint localization.

## Repository Layout

```text
EPRFormer/
  model/
    eprformer.py              # final EPRFormer architecture
    __init__.py
  utils/
    eval_utils.py             # official-style checkpoint evaluation helpers
    image_ops.py              # preprocessing and connected-component utilities
    pbd_metrics.py            # PBD endpoint metrics
    __init__.py
  figures/                    # manuscript figures
  train_eprformer.py          # the only training entrypoint
  infer_eprformer.py          # checkpoint inference and npy export
  eval_checkpoint.py          # checkpoint evaluation on PBD5K
  evaluate_predictions.py     # evaluation for exported predictions
  README_CN.md
  requirements.txt
  CITATION.bib
```

## Installation

Run all commands from the repository root.

Install PyTorch according to your CUDA or CPU environment:

```bash
# Example for CUDA 12.1. Choose the command that matches your machine.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`opencv-python` is recommended for fast connected-component post-processing. If it is unavailable, the repository falls back to a pure Python implementation.

## Dataset

Download PBD5K from the official X-ray-PBD release:

https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Dataset

Training data should be organized as:

```text
data/train_data/
  img_crop/
  neg_point_mask_crop/
  pos_point_mask_crop/
  neg_line_mask_crop/
  pos_line_mask_crop/
```

Official-style evaluation data should be organized as:

```text
data/PBD5K_test_data/
  img/
  mask/              # or crop_mask/, depending on your prepared copy
  neg_location/
    all/
    regular/
    difficult/
    tough/
  pos_location/
    all/
    regular/
    difficult/
    tough/
```

All paths can be overridden through command-line arguments.

## Quick Start

Run a smoke test first:

```bash
python train_eprformer.py \
  --device cuda \
  --image-size 64 \
  --batch-size 1 \
  --num-workers 0 \
  --smoke-test \
  --data-root data/train_data \
  --val-data-root data/PBD5K_test_data
```

Train the final model:

```bash
python train_eprformer.py \
  --device cuda \
  --image-size 512 \
  --backbone resnet50d \
  --batch-size 4 \
  --num-workers 2 \
  --amp \
  --amp-dtype bf16 \
  --epochs 150 \
  --data-root data/train_data \
  --val-data-root data/PBD5K_test_data
```

Training outputs are written to:

```text
runs/eprformer/
  train.log
  metrics.csv
  latest.pth
  best.pth
  top_epochXXX_pnaccXXXX_cntmaeXXXX.pth
```

## Evaluation

Evaluate a checkpoint:

```bash
python eval_checkpoint.py \
  --checkpoint runs/eprformer/best.pth \
  --val-data-root data/PBD5K_test_data \
  --official-eval-split all \
  --image-size 512 \
  --device cuda
```

The script reports:

- `AN-MAE`, `CN-MAE`: positive and negative endpoint count errors
- `AN-ACC`, `CN-ACC`, `PN-ACC`: count consistency accuracies
- `AL-MAE`, `CL-MAE`: endpoint localization errors
- `OH-MAE`: overhang error

## Inference

Export official-style endpoint coordinates:

```bash
python infer_eprformer.py \
  --checkpoint runs/eprformer/best.pth \
  --image-root data/PBD5K_test_data/img \
  --crop-mask-root data/PBD5K_test_data/mask \
  --output-root predictions \
  --image-size 512 \
  --device cuda
```

Output layout:

```text
predictions/
  neg_location/*.npy
  pos_location/*.npy
```

Use `--save-masks` to additionally save binary endpoint masks.

## Evaluate Exported Predictions

```bash
python evaluate_predictions.py \
  --prediction-root predictions \
  --gt-root data/PBD5K_test_data \
  --splits all regular difficult tough
```

## Acknowledgement And Notice

EPRFormer follows the PBD5K benchmark setting introduced by X-ray-PBD. The dataset format, endpoint detection task, and official-style metrics are based on the upstream project.

- X-ray-PBD repository: https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD
- PBD5K dataset release: https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Dataset
- X-ray-PBD model release: https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD/releases/tag/Model_pth
- Benchmark paper: https://arxiv.org/pdf/2312.02528v2.pdf

The reusable code required to run this repository is included locally under `utils/`. Please follow the upstream repository's dataset, model, and code usage terms when using PBD5K or comparing against X-ray-PBD baselines.

## Citation

If this repository is useful for your research, please cite the PBD5K/X-ray-PBD benchmark and this project. Benchmark BibTeX entries are provided in `CITATION.bib`.
