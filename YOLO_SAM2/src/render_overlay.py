from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_COLORS = {
    0: (0, 0, 0),          # background
    1: (255, 0, 0),        # person
    2: (0, 255, 0),        # car
    3: (0, 200, 255),      # motorcycle
    5: (255, 255, 0),      # bus
    7: (255, 0, 255),      # truck
    9: (255, 128, 0),      # traffic light
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser("Render semantic overlay")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--semantic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    semantic_dir = Path(args.semantic_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sem_maps = sorted(semantic_dir.glob("*.png"))
    assert sem_maps, f"No semantic maps in {semantic_dir}"

    for sem_path in sem_maps:
        frame_path = frames_dir / sem_path.name
        if not frame_path.exists():
            print(f"[warn] frame missing: {frame_path.name}")
            continue

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        semantic = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)

        if frame is None or semantic is None:
            raise RuntimeError(f"Failed to read frame or semantic map for {sem_path.name}")

        overlay = frame.copy()

        for cls_id, color in CLASS_COLORS.items():
            if cls_id == 0:
                continue
            mask = semantic == cls_id
            overlay[mask] = color

        out = cv2.addWeighted(frame, 1 - args.alpha, overlay, args.alpha, 0)
        cv2.imwrite(str(out_dir / sem_path.name), out)

        print(f"[overlay] {sem_path.name}")

    print(f"[overlay] done. saved to {out_dir}")


if __name__ == "__main__":
    main()
