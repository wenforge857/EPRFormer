from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def count_components(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    if not mask.any():
        return 0

    try:
        import cv2

        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return int(num_labels - 1)
    except Exception:
        pass

    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    count = 0
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        count += 1
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return count


def connected_component_points(mask: np.ndarray) -> List[List[float]]:
    mask = mask.astype(bool)
    if not mask.any():
        return []

    try:
        import cv2

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        points = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area > 0:
                points.append([float(x + w / 2.0), float(y + h / 2.0)])
        return sorted(points, key=lambda item: item[1])
    except Exception:
        pass

    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    points = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        comp_x = []
        comp_y = []
        while queue:
            y, x = queue.popleft()
            comp_y.append(y)
            comp_x.append(x)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        points.append([float((min(comp_x) + max(comp_x) + 1) / 2.0), float((min(comp_y) + max(comp_y) + 1) / 2.0)])
    return sorted(points, key=lambda item: item[1])


def load_points_npy(path: Path) -> List[List[float]]:
    if not path.exists():
        return []
    data = np.load(path)
    return sorted(data.reshape(-1, 2).astype(float).tolist(), key=lambda item: item[1])
