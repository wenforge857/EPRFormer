from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from utils.pbd_metrics import mean_or_nan, normalized_mae


def load_points(path: Path) -> List[List[float]]:
    if not path.exists():
        return []
    data = np.load(path)
    return sorted(data.reshape(-1, 2).astype(float).tolist(), key=lambda item: item[1])


def evaluate_split(pred_root: Path, gt_root: Path, split: str) -> Dict[str, float]:
    neg_pred_root = pred_root / "neg_location"
    pos_pred_root = pred_root / "pos_location"
    neg_gt_root = gt_root / "neg_location" / split
    pos_gt_root = gt_root / "pos_location" / split
    img_root = gt_root / "img"

    names = sorted(path.stem for path in neg_gt_root.glob("*.npy"))
    neg_num_mae, pos_num_mae = [], []
    neg_num_acc, pos_num_acc, pn_acc = [], [], []
    neg_location_mae, pos_location_mae, overhang_mae = [], [], []

    for name in names:
        neg_pred = load_points(neg_pred_root / f"{name}.npy")
        pos_pred = load_points(pos_pred_root / f"{name}.npy")
        neg_gt = load_points(neg_gt_root / f"{name}.npy")
        pos_gt = load_points(pos_gt_root / f"{name}.npy")

        image_path = next((p for p in img_root.glob(f"{name}.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]), None)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        neg_num_mae.append(abs(len(neg_pred) - len(neg_gt)))
        pos_num_mae.append(abs(len(pos_pred) - len(pos_gt)))
        neg_ok = int(len(neg_pred) == len(neg_gt))
        pos_ok = int(len(pos_pred) == len(pos_gt))
        neg_num_acc.append(neg_ok)
        pos_num_acc.append(pos_ok)
        pn_acc.append(int(neg_ok and pos_ok))

        if len(neg_pred) == len(neg_gt) and neg_gt:
            distances = [float(np.linalg.norm(np.asarray(neg_pred[i]) - np.asarray(neg_gt[i]))) for i in range(len(neg_gt))]
            neg_location_mae.append(normalized_mae(distances, height, width))
        if len(pos_pred) == len(pos_gt) and pos_gt:
            distances = [float(np.linalg.norm(np.asarray(pos_pred[i]) - np.asarray(pos_gt[i]))) for i in range(len(pos_gt))]
            pos_location_mae.append(normalized_mae(distances, height, width))
        if len(neg_pred) == len(neg_gt) and len(pos_pred) == len(pos_gt) and len(pos_pred) + 1 == len(neg_pred):
            sample = []
            for i in range(len(pos_pred)):
                pred_left = abs(neg_pred[i][0] - pos_pred[i][0])
                pred_right = abs(neg_pred[i + 1][0] - pos_pred[i][0])
                gt_left = abs(neg_gt[i][0] - pos_gt[i][0])
                gt_right = abs(neg_gt[i + 1][0] - pos_gt[i][0])
                sample.append(abs(pred_left - gt_left))
                sample.append(abs(pred_right - gt_right))
            overhang_mae.append(normalized_mae(sample, height, width))

    return {
        "split": split,
        "neg_num_MAE": mean_or_nan(neg_num_mae),
        "pos_num_MAE": mean_or_nan(pos_num_mae),
        "neg_num_Acc": mean_or_nan(neg_num_acc),
        "pos_num_Acc": mean_or_nan(pos_num_acc),
        "PN_Acc": mean_or_nan(pn_acc),
        "neg_location_MAE": mean_or_nan(neg_location_mae),
        "pos_location_MAE": mean_or_nan(pos_location_mae),
        "overhang_MAE": mean_or_nan(overhang_mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported EPRFormer predictions with official-style metrics.")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=Path("data/PBD5K_test_data"))
    parser.add_argument("--splits", nargs="+", default=["all", "regular", "difficult", "tough"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [evaluate_split(args.prediction_root, args.gt_root, split) for split in args.splits]
    lines = []
    for row in rows:
        line = (
            f"split: {row['split']} "
            f"neg_num_MAE: {row['neg_num_MAE']:.4f} "
            f"pos_num_MAE: {row['pos_num_MAE']:.4f} "
            f"neg_num_Acc: {row['neg_num_Acc']:.4f} "
            f"pos_num_Acc: {row['pos_num_Acc']:.4f} "
            f"PN_Acc: {row['PN_Acc']:.4f} "
            f"neg_location_MAE: {row['neg_location_MAE']:.4f} "
            f"pos_location_MAE: {row['pos_location_MAE']:.4f} "
            f"overhang_MAE: {row['overhang_MAE']:.4f}"
        )
        print(line)
        lines.append(line)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
