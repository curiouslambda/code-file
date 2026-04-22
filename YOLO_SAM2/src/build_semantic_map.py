from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import argparse

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser("Resolve semantic segmentation map")
    parser.add_argument("--pred_json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--priority", type=str, default="area", choices=["area", "order"])
    args = parser.parse_args()

    pred_json_dir = Path(args.pred_json_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    json_files = sorted(pred_json_dir.glob("*.json"))
    assert json_files, f"No pred_json found in {pred_json_dir}"

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))

        W, H = int(data["width"]), int(data["height"])
        instances: List[Dict] = data["instances"]

        # semantic map 초기화 (background=0)
        semantic = np.zeros((H, W), dtype=np.uint8)

        # area 큰 것부터
        if args.priority == "area":
            instances = sorted(instances, key=lambda x: x.get("mask_area", 0), reverse=True)

        for inst in instances:
            cls_id = int(inst["cls_id"])
            mask_path = Path(inst["mask_path"])

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")

            mask_bool = mask > 0
            semantic[mask_bool] = cls_id

        out_path = out_dir / jf.name.replace(".json", ".png")
        cv2.imwrite(str(out_path), semantic)

        print(f"[semantic] {out_path.name}")

    print(f"[semantic] done. saved to {out_dir}")


if __name__ == "__main__":
    main()
