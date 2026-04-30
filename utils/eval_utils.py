from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .image_ops import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, connected_component_points, load_points_npy
from .pbd_metrics import mean_or_nan, normalized_mae


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


def find_image_paths(root: Path) -> List[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_image_by_stem(root: Path, stem: str) -> Optional[Path]:
    for suffix in IMAGE_EXTENSIONS:
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidate = root / f"{stem}{suffix.upper()}"
        if candidate.exists():
            return candidate
    matches = [path for path in root.iterdir() if path.is_file() and path.stem == stem]
    return matches[0] if matches else None


def is_official_eval_root(root: Optional[Path]) -> bool:
    if root is None:
        return False
    return (root / "img").is_dir() and (root / "neg_location").is_dir() and (root / "pos_location").is_dir()


def largest_bbox_from_mask(mask_path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not mask_path.exists():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if not mask.any():
        return None
    try:
        import cv2

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, -1]))
        x, y, w, h, _ = stats[largest]
        return int(x), int(y), int(w), int(h)
    except Exception:
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return x0, y0, x1 - x0, y1 - y0


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def points_from_logits_with_crop(
    logits: torch.Tensor,
    crop_h: float,
    crop_w: float,
    offset_x: float,
    offset_y: float,
    threshold: float,
) -> List[List[float]]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    points = connected_component_points(prob > threshold)
    if not points:
        return []
    h, w = prob.shape
    scale_x = crop_w / max(1.0, float(w))
    scale_y = crop_h / max(1.0, float(h))
    return [[offset_x + x * scale_x, offset_y + y * scale_y] for x, y in points]


class PBDOfficialEvalDataset:
    def __init__(
        self,
        data_root: Path,
        image_size: int = 512,
        prompt_root: Optional[Path] = None,
        prompt_image: Optional[Path] = None,
        split: str = "all",
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.img_root = data_root / "img"
        self.crop_mask_root = data_root / "crop_mask"
        self.neg_gt_root = data_root / "neg_location" / split
        self.pos_gt_root = data_root / "pos_location" / split
        if not self.neg_gt_root.is_dir() or not self.pos_gt_root.is_dir():
            raise RuntimeError(f"Official eval split '{split}' not found under {data_root}")
        self.names = sorted(path.stem for path in self.neg_gt_root.glob("*.npy"))
        if not self.names:
            raise RuntimeError(f"No official eval npy files found under {self.neg_gt_root}")

        if prompt_image is not None:
            self.prompt_path = prompt_image
        elif prompt_root is not None:
            prompts = find_image_paths(prompt_root)
            if not prompts:
                raise RuntimeError(f"No prompt images found under {prompt_root}")
            self.prompt_path = prompts[0]
        else:
            first_image = find_image_by_stem(self.img_root, self.names[0])
            if first_image is None:
                raise RuntimeError(f"Cannot find image for {self.names[0]} under {self.img_root}")
            self.prompt_path = first_image
        self.prompt_tensor = image_to_tensor(Image.open(self.prompt_path).convert("RGB"), self.image_size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image_path = find_image_by_stem(self.img_root, name)
        if image_path is None:
            raise RuntimeError(f"Cannot find official eval image for {name}")
        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size
        offset_x, offset_y = 0, 0
        crop_w, crop_h = original_w, original_h
        crop_box = largest_bbox_from_mask(self.crop_mask_root / f"{name}.png")
        if crop_box is not None:
            offset_x, offset_y, crop_w, crop_h = crop_box
            image = image.crop((offset_x, offset_y, offset_x + crop_w, offset_y + crop_h))
        return {
            "image": image_to_tensor(image, self.image_size).unsqueeze(0),
            "prompt": self.prompt_tensor.unsqueeze(0),
            "name": name,
            "crop_info": (float(original_h), float(original_w), float(offset_x), float(offset_y), float(crop_h), float(crop_w)),
            "gt_neg": load_points_npy(self.neg_gt_root / f"{name}.npy"),
            "gt_pos": load_points_npy(self.pos_gt_root / f"{name}.npy"),
        }


def update_official_metric_lists(
    pred_neg: List[List[float]],
    pred_pos: List[List[float]],
    gt_neg: List[List[float]],
    gt_pos: List[List[float]],
    original_h: float,
    original_w: float,
    metric_lists: Dict[str, List[float]],
) -> None:
    metric_lists["neg_num_mae"].append(abs(len(pred_neg) - len(gt_neg)))
    metric_lists["pos_num_mae"].append(abs(len(pred_pos) - len(gt_pos)))
    neg_ok = int(len(pred_neg) == len(gt_neg))
    pos_ok = int(len(pred_pos) == len(gt_pos))
    metric_lists["neg_num_acc"].append(neg_ok)
    metric_lists["pos_num_acc"].append(pos_ok)
    metric_lists["pn_acc"].append(int(neg_ok and pos_ok))

    if len(pred_neg) == len(gt_neg) and gt_neg:
        metric_lists["neg_location_mae"].append(
            normalized_mae([float(np.linalg.norm(np.asarray(pred_neg[i]) - np.asarray(gt_neg[i]))) for i in range(len(gt_neg))], original_h, original_w)
        )
    if len(pred_pos) == len(gt_pos) and gt_pos:
        metric_lists["pos_location_mae"].append(
            normalized_mae([float(np.linalg.norm(np.asarray(pred_pos[i]) - np.asarray(gt_pos[i]))) for i in range(len(gt_pos))], original_h, original_w)
        )
    if len(pred_neg) == len(gt_neg) and len(pred_pos) == len(gt_pos) and len(pred_pos) + 1 == len(pred_neg):
        sample_overhang = []
        for i in range(len(pred_pos)):
            pred_left = abs(pred_neg[i][0] - pred_pos[i][0])
            pred_right = abs(pred_neg[i + 1][0] - pred_pos[i][0])
            gt_left = abs(gt_neg[i][0] - gt_pos[i][0])
            gt_right = abs(gt_neg[i + 1][0] - gt_pos[i][0])
            sample_overhang.append(abs(pred_left - gt_left))
            sample_overhang.append(abs(pred_right - gt_right))
        if sample_overhang:
            metric_lists["overhang_mae"].append(normalized_mae(sample_overhang, original_h, original_w))


def evaluate_official(model: torch.nn.Module, dataset: PBDOfficialEvalDataset, device: torch.device, threshold: float) -> Dict[str, float]:
    model.eval()
    metric_lists = {
        "neg_num_mae": [],
        "pos_num_mae": [],
        "neg_num_acc": [],
        "pos_num_acc": [],
        "pn_acc": [],
        "neg_location_mae": [],
        "pos_location_mae": [],
        "overhang_mae": [],
    }
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].to(device, non_blocking=True)
            prompt = sample["prompt"].to(device, non_blocking=True)
            original_h, original_w, offset_x, offset_y, crop_h, crop_w = sample["crop_info"]
            logits = model(image, prompt)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_neg = points_from_logits_with_crop(logits[0, 0], crop_h, crop_w, offset_x, offset_y, threshold)
            pred_pos = points_from_logits_with_crop(logits[0, 1], crop_h, crop_w, offset_x, offset_y, threshold)
            update_official_metric_lists(
                pred_neg,
                pred_pos,
                sample["gt_neg"],
                sample["gt_pos"],
                original_h,
                original_w,
                metric_lists,
            )
    metrics = {key: mean_or_nan(value) for key, value in metric_lists.items()}
    metrics["score"] = metrics["pn_acc"]
    metrics["count_mae"] = metrics["neg_num_mae"] + metrics["pos_num_mae"]
    return metrics
