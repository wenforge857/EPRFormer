from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

import torch

from model.eprformer import EPRFormer
from utils.eval_utils import PBDOfficialEvalDataset, evaluate_official


if pathlib.PosixPath is not pathlib.WindowsPath:
    pathlib.PosixPath = pathlib.WindowsPath


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EPRFormer checkpoint on PBD5K official test data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-data-root", type=Path, default=Path("data/PBD5K_test_data"))
    parser.add_argument("--prompt-root", type=Path, default=None)
    parser.add_argument("--eval-prompt-image", type=Path, default=None)
    parser.add_argument("--official-eval-split", type=str, default="all")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    backbone = args.backbone or saved_args.get("backbone", "resnet50d")

    model = EPRFormer(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset = PBDOfficialEvalDataset(
        data_root=args.val_data_root,
        image_size=args.image_size,
        prompt_root=args.prompt_root,
        prompt_image=args.eval_prompt_image,
        split=args.official_eval_split,
    )
    metrics = evaluate_official(model, dataset, device, args.threshold)
    print(
        " ".join(
            [
                f"AN-MAE {metrics['neg_num_mae']:.4f}",
                f"CN-MAE {metrics['pos_num_mae']:.4f}",
                f"AN-ACC {metrics['neg_num_acc']:.4f}",
                f"CN-ACC {metrics['pos_num_acc']:.4f}",
                f"PN-ACC {metrics['pn_acc']:.4f}",
                f"AL-MAE {metrics['neg_location_mae']:.4f}",
                f"CL-MAE {metrics['pos_location_mae']:.4f}",
                f"OH-MAE {metrics['overhang_mae']:.4f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
