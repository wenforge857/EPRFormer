from __future__ import annotations

import argparse
import pathlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model.eprformer import EPRFormer
from utils.image_ops import IMAGENET_MEAN, IMAGENET_STD, connected_component_points


if pathlib.PosixPath is not pathlib.WindowsPath:
    pathlib.PosixPath = pathlib.WindowsPath


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(root: Path) -> List[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_model(checkpoint_path: Path, device: torch.device, backbone: str, pretrained: bool) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EPRFormer(backbone=backbone, pretrained=pretrained)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def logits_to_points(logits: torch.Tensor, original_h: int, original_w: int, threshold: float) -> List[List[float]]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    mask = prob > threshold
    points = connected_component_points(mask)
    h, w = mask.shape
    scale_x = original_w / max(1.0, float(w))
    scale_y = original_h / max(1.0, float(h))
    return [[x * scale_x, y * scale_y] for x, y in points]


def largest_bbox_from_mask(mask_path: Path) -> Optional[tuple[int, int, int, int]]:
    if not mask_path.exists():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    points = connected_component_points(mask)
    if not points:
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
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return x0, y0, x1 - x0, y1 - y0


def logits_to_original_points(
    logits: torch.Tensor,
    crop_h: int,
    crop_w: int,
    offset_x: int,
    offset_y: int,
    threshold: float,
) -> List[List[float]]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    points = connected_component_points(prob > threshold)
    h, w = prob.shape
    scale_x = crop_w / max(1.0, float(w))
    scale_y = crop_h / max(1.0, float(h))
    return [[offset_x + x * scale_x, offset_y + y * scale_y] for x, y in points]


def save_mask(logits: torch.Tensor, path: Path, original_h: int, original_w: int, threshold: float) -> None:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    mask = ((prob > threshold).astype(np.uint8) * 255)
    image = Image.fromarray(mask, mode="L").resize((original_w, original_h), Image.NEAREST)
    image.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EPRFormer inference and export official-style location files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prompt-image", type=Path, default=None)
    parser.add_argument("--prompt-root", type=Path, default=None)
    parser.add_argument("--crop-mask-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint, device, args.backbone, args.pretrained)

    images = find_images(args.image_root)
    if not images:
        raise RuntimeError(f"No images found under {args.image_root}")

    if args.prompt_image is not None:
        prompt_path = args.prompt_image
    elif args.prompt_root is not None:
        prompt_images = find_images(args.prompt_root)
        if not prompt_images:
            raise RuntimeError(f"No prompt images found under {args.prompt_root}")
        prompt_path = prompt_images[0]
    else:
        prompt_path = images[0]

    output_neg = args.output_root / "neg_location"
    output_pos = args.output_root / "pos_location"
    output_neg_mask = args.output_root / "neg_point_mask"
    output_pos_mask = args.output_root / "pos_point_mask"
    output_neg.mkdir(parents=True, exist_ok=True)
    output_pos.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        output_neg_mask.mkdir(parents=True, exist_ok=True)
        output_pos_mask.mkdir(parents=True, exist_ok=True)

    prompt = image_to_tensor(Image.open(prompt_path), args.image_size).unsqueeze(0).to(device)

    with torch.no_grad():
        for image_path in tqdm(images, desc="infer"):
            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            offset_x, offset_y = 0, 0
            crop_w, crop_h = original_w, original_h
            if args.crop_mask_root is not None:
                crop_box = largest_bbox_from_mask(args.crop_mask_root / f"{image_path.stem}.png")
                if crop_box is not None:
                    offset_x, offset_y, crop_w, crop_h = crop_box
                    image = image.crop((offset_x, offset_y, offset_x + crop_w, offset_y + crop_h))
            image_tensor = image_to_tensor(image, args.image_size).unsqueeze(0).to(device)
            logits = model(image_tensor, prompt)
            if logits.shape[-2:] != (args.image_size, args.image_size):
                logits = F.interpolate(logits, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            neg_points = logits_to_original_points(logits[0, 0], crop_h, crop_w, offset_x, offset_y, args.threshold)
            pos_points = logits_to_original_points(logits[0, 1], crop_h, crop_w, offset_x, offset_y, args.threshold)
            np.save(output_neg / f"{image_path.stem}.npy", np.asarray(neg_points, dtype=np.float32))
            np.save(output_pos / f"{image_path.stem}.npy", np.asarray(pos_points, dtype=np.float32))
            if args.save_masks:
                save_mask(logits[0, 0], output_neg_mask / f"{image_path.stem}.png", original_h, original_w, args.threshold)
                save_mask(logits[0, 1], output_pos_mask / f"{image_path.stem}.png", original_h, original_w, args.threshold)

    print(f"Saved predictions to {args.output_root}")


if __name__ == "__main__":
    main()
