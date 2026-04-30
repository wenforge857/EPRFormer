from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from model.eprformer import EPRFormer
from utils.eval_utils import PBDOfficialEvalDataset, evaluate_official, is_official_eval_root
from utils.image_ops import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, connected_component_points, count_components
from utils.pbd_metrics import mean_or_nan, normalized_mae


if pathlib.PosixPath is not pathlib.WindowsPath:
    pathlib.PosixPath = pathlib.WindowsPath


try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def find_images(root: Path) -> Dict[str, Path]:
    return {path.stem: path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


class PBDTrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        image_size: int = 512,
        prompt_root: Optional[Path] = None,
        augment: bool = True,
        names: Optional[Sequence[str]] = None,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.augment = augment

        self.img_paths = find_images(data_root / "img_crop")
        self.neg_point_paths = find_images(data_root / "neg_point_mask_crop")
        self.pos_point_paths = find_images(data_root / "pos_point_mask_crop")
        self.neg_line_paths = find_images(data_root / "neg_line_mask_crop")
        self.pos_line_paths = find_images(data_root / "pos_line_mask_crop")

        required = [
            set(self.img_paths),
            set(self.neg_point_paths),
            set(self.pos_point_paths),
            set(self.neg_line_paths),
            set(self.pos_line_paths),
        ]
        matched_names = sorted(set.intersection(*required))
        self.names = list(names) if names is not None else matched_names
        missing = sorted(set(self.names) - set(matched_names))
        if missing:
            raise RuntimeError(f"{len(missing)} requested samples are missing one or more masks. First missing: {missing[0]}")
        if not self.names:
            raise RuntimeError(f"No matched training samples found under {data_root}")

        if prompt_root is None:
            self.prompt_paths = self.img_paths
            self.prompt_names = self.names
        else:
            self.prompt_paths = find_images(prompt_root)
            self.prompt_names = sorted(self.prompt_paths)
            if not self.prompt_names:
                raise RuntimeError(f"No prompt images found under {prompt_root}")

        self._count_cache: Dict[Tuple[str, str], int] = {}

    def __len__(self) -> int:
        return len(self.names)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _mask_count(self, name: str, polarity: str, path: Path) -> int:
        key = (name, polarity)
        if key not in self._count_cache:
            mask = np.array(self._load_mask(path)) > 127
            self._count_cache[key] = count_components(mask)
        return self._count_cache[key]

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask = mask.resize((self.image_size, self.image_size), NEAREST)
        array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int):
        name = self.names[index]
        prompt_name = random.choice(self.prompt_names)

        image = self._load_rgb(self.img_paths[name])
        original_w, original_h = image.size
        prompt = self._load_rgb(self.prompt_paths[prompt_name])
        neg_point = self._load_mask(self.neg_point_paths[name])
        pos_point = self._load_mask(self.pos_point_paths[name])
        neg_line = self._load_mask(self.neg_line_paths[name])
        pos_line = self._load_mask(self.pos_line_paths[name])

        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            neg_point = ImageOps.mirror(neg_point)
            pos_point = ImageOps.mirror(pos_point)
            neg_line = ImageOps.mirror(neg_line)
            pos_line = ImageOps.mirror(pos_line)

        if self.augment:
            brightness = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            prompt = ImageEnhance.Brightness(prompt).enhance(random.uniform(0.9, 1.1))

        image_tensor = self._image_to_tensor(image)
        prompt_tensor = self._image_to_tensor(prompt)
        point_mask = torch.cat([self._mask_to_tensor(neg_point), self._mask_to_tensor(pos_point)], dim=0)
        line_mask = torch.cat([self._mask_to_tensor(neg_line), self._mask_to_tensor(pos_line)], dim=0)
        count = torch.tensor(
            [
                self._mask_count(name, "neg", self.neg_point_paths[name]),
                self._mask_count(name, "pos", self.pos_point_paths[name]),
            ],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "prompt": prompt_tensor,
            "point_mask": point_mask,
            "line_mask": line_mask,
            "count": count,
            "original_size": torch.tensor([original_h, original_w], dtype=torch.float32),
            "name": name,
        }


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    logits = logits.float()
    target = target.float()
    weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)

    pred = torch.sigmoid(logits)
    inter = ((pred * target) * weight).sum(dim=(2, 3))
    union = ((pred + target) * weight).sum(dim=(2, 3))
    iou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (bce + iou).mean()


def compute_loss(outputs, batch, weights):
    point_refine, point_course, reg_neg, reg_pos, line_neg, line_pos = outputs
    point_target = batch["point_mask"].float()
    line_target = batch["line_mask"].float()
    count_target = batch["count"].float()

    loss_refine = weighted_bce_iou_loss(point_refine, point_target)
    loss_course = weighted_bce_iou_loss(point_course, point_target)
    line_logits = torch.cat([line_neg, line_pos], dim=1)
    loss_line = weighted_bce_iou_loss(line_logits, line_target)
    loss_count = F.l1_loss(reg_neg.float().reshape(-1), count_target[:, 0]) + F.l1_loss(
        reg_pos.float().reshape(-1), count_target[:, 1]
    )

    total = (
        weights["point_refine"] * loss_refine
        + weights["point_course"] * loss_course
        + weights["line"] * loss_line
        + weights["count"] * loss_count
    )
    parts = {
        "total": float(total.detach().cpu()),
        "refine": float(loss_refine.detach().cpu()),
        "coarse": float(loss_course.detach().cpu()),
        "line": float(loss_line.detach().cpu()),
        "count": float(loss_count.detach().cpu()),
    }
    return total, parts


def split_names(names: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    names = list(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    val_count = int(round(len(names) * val_ratio))
    val_count = min(max(val_count, 1), max(1, len(names) - 1))
    return sorted(names[val_count:]), sorted(names[:val_count])


def points_from_tensor_mask(mask: torch.Tensor, original_h: float, original_w: float) -> List[List[float]]:
    array = mask.detach().cpu().numpy() > 0.5
    points = connected_component_points(array)
    if not points:
        return []
    h, w = array.shape
    scale_x = original_w / max(1.0, float(w))
    scale_y = original_h / max(1.0, float(h))
    return [[x * scale_x, y * scale_y] for x, y in points]


def points_from_logits(logits: torch.Tensor, original_h: float, original_w: float, threshold: float) -> List[List[float]]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    points = connected_component_points(prob > threshold)
    if not points:
        return []
    h, w = prob.shape
    scale_x = original_w / max(1.0, float(w))
    scale_y = original_h / max(1.0, float(h))
    return [[x * scale_x, y * scale_y] for x, y in points]


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Dict[str, float]:
    model.eval()
    neg_num_mae = []
    pos_num_mae = []
    neg_num_acc = []
    pos_num_acc = []
    pn_acc = []
    neg_loc = []
    pos_loc = []
    overhang = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            prompt = batch["prompt"].to(device, non_blocking=True)
            point_target = batch["point_mask"]
            original_size = batch["original_size"]
            logits = model(image, prompt)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.shape[-2:] != point_target.shape[-2:]:
                logits = F.interpolate(logits, size=point_target.shape[-2:], mode="bilinear", align_corners=False)

            for idx in range(logits.shape[0]):
                original_h = float(original_size[idx, 0].item())
                original_w = float(original_size[idx, 1].item())
                pred_neg = points_from_logits(logits[idx, 0], original_h, original_w, threshold)
                pred_pos = points_from_logits(logits[idx, 1], original_h, original_w, threshold)
                gt_neg = points_from_tensor_mask(point_target[idx, 0], original_h, original_w)
                gt_pos = points_from_tensor_mask(point_target[idx, 1], original_h, original_w)

                neg_num_mae.append(abs(len(pred_neg) - len(gt_neg)))
                pos_num_mae.append(abs(len(pred_pos) - len(gt_pos)))
                neg_ok = int(len(pred_neg) == len(gt_neg))
                pos_ok = int(len(pred_pos) == len(gt_pos))
                neg_num_acc.append(neg_ok)
                pos_num_acc.append(pos_ok)
                pn_acc.append(int(neg_ok and pos_ok))

                if len(pred_neg) == len(gt_neg) and gt_neg:
                    neg_dist = [
                        math.sqrt((pred_neg[i][0] - gt_neg[i][0]) ** 2 + (pred_neg[i][1] - gt_neg[i][1]) ** 2)
                        for i in range(len(gt_neg))
                    ]
                    neg_loc.append(normalized_mae(neg_dist, original_h, original_w))
                if len(pred_pos) == len(gt_pos) and gt_pos:
                    pos_dist = [
                        math.sqrt((pred_pos[i][0] - gt_pos[i][0]) ** 2 + (pred_pos[i][1] - gt_pos[i][1]) ** 2)
                        for i in range(len(gt_pos))
                    ]
                    pos_loc.append(normalized_mae(pos_dist, original_h, original_w))
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
                        overhang.append(normalized_mae(sample_overhang, original_h, original_w))

    metrics = {
        "neg_num_mae": mean_or_nan(neg_num_mae),
        "pos_num_mae": mean_or_nan(pos_num_mae),
        "neg_num_acc": mean_or_nan(neg_num_acc),
        "pos_num_acc": mean_or_nan(pos_num_acc),
        "pn_acc": mean_or_nan(pn_acc),
        "neg_location_mae": mean_or_nan(neg_loc),
        "pos_location_mae": mean_or_nan(pos_loc),
        "overhang_mae": mean_or_nan(overhang),
    }
    metrics["score"] = metrics["pn_acc"]
    metrics["count_mae"] = metrics["neg_num_mae"] + metrics["pos_num_mae"]
    return metrics


def metric_sort_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    overhang = metrics["overhang_mae"]
    if math.isnan(overhang):
        overhang = float("inf")
    return (metrics["score"], -metrics["count_mae"], -overhang)


def write_metrics_csv(path: Path, row: Dict[str, float]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "refine_loss",
        "coarse_loss",
        "line_loss",
        "count_loss",
        "AN-MAE",
        "CN-MAE",
        "AN-ACC",
        "CN-ACC",
        "PN-ACC",
        "AL-MAE",
        "CL-MAE",
        "OH-MAE",
        "neg_num_mae",
        "pos_num_mae",
        "neg_num_acc",
        "pos_num_acc",
        "pn_acc",
        "neg_location_mae",
        "pos_location_mae",
        "overhang_mae",
        "score",
        "count_mae",
    ]
    exists = path.exists()
    if exists:
        old_header = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        old_fields = old_header[0].split(",") if old_header else []
        if old_fields != fieldnames:
            path = path.with_name(f"{path.stem}_unified{path.suffix}")
            exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


class FileLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def save_topk_checkpoint(
    save_dir: Path,
    state: Dict,
    epoch: int,
    metrics: Dict[str, float],
    topk_records: List[Tuple[Tuple[float, float, float], Path]],
    keep: int,
) -> List[Tuple[Tuple[float, float, float], Path]]:
    key = metric_sort_key(metrics)
    ckpt_name = f"top_epoch{epoch:03d}_pnacc{metrics['pn_acc']:.4f}_cntmae{metrics['count_mae']:.4f}.pth"
    ckpt_path = save_dir / ckpt_name
    candidate = topk_records + [(key, ckpt_path)]
    candidate = sorted(candidate, key=lambda item: item[0], reverse=True)
    keep_records = candidate[:keep]
    keep_paths = {path for _, path in keep_records}
    if ckpt_path in keep_paths:
        torch.save(state, ckpt_path)
    for _, path in candidate[keep:]:
        if path.exists():
            path.unlink()
    return keep_records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_parser(experiment_name: str) -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=f"Train {experiment_name}")
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "train_data")
    parser.add_argument("--prompt-root", type=Path, default=None)
    parser.add_argument("--val-data-root", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-prompt-image", type=Path, default=None)
    parser.add_argument("--official-eval-split", type=str, default="all")
    parser.add_argument("--save-dir", type=Path, default=repo_root / "runs" / experiment_name)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=120)
    parser.add_argument("--lr-gamma", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("float16", "bf16"), default="float16")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on training steps per epoch.")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs.")
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=5, help="Number of metric-best checkpoints to keep.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one train step, write a smoke status file, and exit.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations.")
    return parser


def main(experiment_name: str = "eprformer") -> None:
    args = build_parser(experiment_name).parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log = FileLogger(args.save_dir / "train.log")

    official_eval = is_official_eval_root(args.val_data_root)
    full_dataset = PBDTrainDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        prompt_root=args.prompt_root,
        augment=False,
    )
    if args.val_data_root is None:
        train_names, val_names = split_names(full_dataset.names, args.val_ratio, args.seed)
        train_dataset = PBDTrainDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            augment=not args.no_augment,
            names=train_names,
        )
        val_dataset = PBDTrainDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            augment=False,
            names=val_names,
        )
        eval_dataset = val_dataset
    elif official_eval:
        train_dataset = PBDTrainDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            augment=not args.no_augment,
        )
        val_dataset = PBDOfficialEvalDataset(
            data_root=args.val_data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            prompt_image=args.eval_prompt_image,
            split=args.official_eval_split,
        )
        eval_dataset = val_dataset
    else:
        train_dataset = PBDTrainDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            augment=not args.no_augment,
        )
        val_dataset = PBDTrainDataset(
            data_root=args.val_data_root,
            image_size=args.image_size,
            prompt_root=args.prompt_root,
            augment=False,
        )
        eval_dataset = val_dataset
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=not args.smoke_test,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = None
    if not official_eval:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = EPRFormer(
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    requested_amp_dtype = args.amp_dtype
    if args.amp and device.type == "cuda" and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "float16"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    weights = {"point_refine": 1.0, "point_course": 1.0, "count": 0.05, "line": 0.5}

    start_epoch = 1
    best_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    log(f"Experiment: {experiment_name}")
    eval_name = f"official {args.official_eval_split}" if official_eval else "validation"
    log(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)} ({eval_name}) | Device: {device} | Save dir: {args.save_dir}")
    log("Model: EPRFormer")
    if requested_amp_dtype != args.amp_dtype:
        log(f"AMP: requested {requested_amp_dtype}, but this GPU does not support it; falling back to {args.amp_dtype}.")
    log(f"AMP: enabled={args.amp and device.type == 'cuda'} dtype={args.amp_dtype} grad_scaler={use_scaler}")
    if len(loader) == 0:
        raise RuntimeError("The DataLoader is empty. Reduce --batch-size or check the dataset.")

    topk_records: List[Tuple[Tuple[float, float, float], Path]] = []
    metrics_csv = args.save_dir / "metrics.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        running_parts = {"refine": 0.0, "coarse": 0.0, "line": 0.0, "count": 0.0}
        steps_this_epoch = 0
        for step, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda", dtype=amp_dtype):
                outputs = model(batch["image"], batch["prompt"])
                loss, parts = compute_loss(outputs, batch, weights)

            if not torch.isfinite(loss):
                log(f"WARNING: skipped non-finite loss at epoch {epoch} step {step}: {parts}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                log(f"WARNING: skipped non-finite gradients at epoch {epoch} step {step}: grad_norm={float(grad_norm)}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            running += parts["total"]
            for part_name in running_parts:
                running_parts[part_name] += parts[part_name]
            steps_this_epoch += 1
            if step == 1 or step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                log(
                    f"epoch {epoch:03d}/{args.epochs:03d} "
                    f"step {step:04d}/{len(loader):04d} "
                    f"lr {lr:.2e} "
                    f"loss {parts['total']:.4f} "
                    f"p_ref {parts['refine']:.4f} "
                    f"p_crs {parts['coarse']:.4f} "
                    f"line {parts['line']:.4f} "
                    f"count {parts['count']:.4f}"
                )
            if args.smoke_test:
                (args.save_dir / "smoke_test.txt").write_text(
                    (
                        f"Smoke test passed for {experiment_name}\n"
                        "model: EPRFormer\n"
                        f"loss: {parts}\n"
                    ),
                    encoding="utf-8",
                )
                log("Smoke test passed: one optimization step completed.")
                return
            if args.max_steps is not None and step >= args.max_steps:
                break

        scheduler.step()
        epoch_loss = running / max(1, steps_this_epoch)
        epoch_parts = {name: value / max(1, steps_this_epoch) for name, value in running_parts.items()}
        state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "experiment": experiment_name,
            "model_name": "EPRFormer",
        }
        torch.save(state, args.save_dir / "latest.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state["best_loss"] = best_loss
            torch.save(state, args.save_dir / "best.pth")
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "refine_loss": epoch_parts["refine"],
            "coarse_loss": epoch_parts["coarse"],
            "line_loss": epoch_parts["line"],
            "count_loss": epoch_parts["count"],
        }
        if epoch % args.eval_interval == 0:
            if official_eval:
                metrics = evaluate_official(model, val_dataset, device, args.eval_threshold)
            else:
                metrics = evaluate(model, val_loader, device, args.eval_threshold)
            row.update(metrics)
            row.update(
                {
                    "AN-MAE": metrics["neg_num_mae"],
                    "CN-MAE": metrics["pos_num_mae"],
                    "AN-ACC": metrics["neg_num_acc"],
                    "CN-ACC": metrics["pos_num_acc"],
                    "PN-ACC": metrics["pn_acc"],
                    "AL-MAE": metrics["neg_location_mae"],
                    "CL-MAE": metrics["pos_location_mae"],
                    "OH-MAE": metrics["overhang_mae"],
                }
            )
            state["metrics"] = metrics
            topk_records = save_topk_checkpoint(args.save_dir, state, epoch, metrics, topk_records, args.topk)
            metric_msg = (
                f"val AN-MAE {metrics['neg_num_mae']:.4f} "
                f"CN-MAE {metrics['pos_num_mae']:.4f} "
                f"AN-ACC {metrics['neg_num_acc']:.4f} "
                f"CN-ACC {metrics['pos_num_acc']:.4f} "
                f"PN-ACC {metrics['pn_acc']:.4f} "
                f"AL-MAE {metrics['neg_location_mae']:.4f} "
                f"CL-MAE {metrics['pos_location_mae']:.4f} "
                f"OH-MAE {metrics['overhang_mae']:.4f}"
            )
            log(metric_msg)
        write_metrics_csv(metrics_csv, row)
        log(f"epoch {epoch:03d} mean_loss {epoch_loss:.4f} best_loss {best_loss:.4f}")
