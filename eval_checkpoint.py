from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def mean_or_nan(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def normalized_mae(values: List[float], height: float, width: float) -> float:
    return float(np.mean(values)) / max(1.0, float(height * width))


def endpoint_metrics(
    pred_neg: List[List[float]],
    pred_pos: List[List[float]],
    gt_neg: List[List[float]],
    gt_pos: List[List[float]],
    height: float,
    width: float,
) -> Dict[str, float]:
    """PBD endpoint metrics following the X-ray-PBD official convention."""

    neg_ok = int(len(pred_neg) == len(gt_neg))
    pos_ok = int(len(pred_pos) == len(gt_pos))
    metrics = {
        "neg_num_mae": float(abs(len(pred_neg) - len(gt_neg))),
        "pos_num_mae": float(abs(len(pred_pos) - len(gt_pos))),
        "neg_num_acc": float(neg_ok),
        "pos_num_acc": float(pos_ok),
        "pn_acc": float(int(neg_ok and pos_ok)),
        "neg_location_mae": float("nan"),
        "pos_location_mae": float("nan"),
        "overhang_mae": float("nan"),
    }

    if len(pred_neg) == len(gt_neg) and gt_neg:
        distances = [math.dist(pred_neg[i], gt_neg[i]) for i in range(len(gt_neg))]
        metrics["neg_location_mae"] = normalized_mae(distances, height, width)
    if len(pred_pos) == len(gt_pos) and gt_pos:
        distances = [math.dist(pred_pos[i], gt_pos[i]) for i in range(len(gt_pos))]
        metrics["pos_location_mae"] = normalized_mae(distances, height, width)
    if len(pred_neg) == len(gt_neg) and len(pred_pos) == len(gt_pos) and len(pred_pos) + 1 == len(pred_neg):
        overhang = []
        for i in range(len(pred_pos)):
            pred_left = abs(pred_neg[i][0] - pred_pos[i][0])
            pred_right = abs(pred_neg[i + 1][0] - pred_pos[i][0])
            gt_left = abs(gt_neg[i][0] - gt_pos[i][0])
            gt_right = abs(gt_neg[i + 1][0] - gt_pos[i][0])
            overhang.append(abs(pred_left - gt_left))
            overhang.append(abs(pred_right - gt_right))
        if overhang:
            metrics["overhang_mae"] = normalized_mae(overhang, height, width)

    return metrics
