# src/detector_infer.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
from ultralytics import YOLO


@dataclass
class DetBox:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: List[int]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def draw_boxes_bgr(img_bgr, boxes: List[DetBox]) -> Any:
    out = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = b.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{b.cls_name} {b.conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def run_yolo_on_images(
    model: YOLO,
    image_paths: List[Path],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[List[int]] = None,
) -> Dict[str, List[DetBox]]:
    """
    Return dict: image_path -> list of DetBox
    """
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        verbose=False,
        device=0,  # GPU 0 사용 (CUDA 환경)
    )

    out: Dict[str, List[DetBox]] = {}
    names = model.names  # class id -> name

    for p, r in zip(image_paths, results):
        boxes: List[DetBox] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                kid = int(k)
                boxes.append(
                    DetBox(
                        cls_id=kid,
                        cls_name=str(names.get(kid, kid)),
                        conf=float(c),
                        bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)],
                    )
                )

        out[str(p)] = boxes
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser("YOLOv8 detector test (Step 2)")
    parser.add_argument("--input_dir", type=str, required=True, help="raw image dir")
    parser.add_argument("--clip_id", type=str, required=True, help="target clip_id")
    parser.add_argument("--yolo_ckpt", type=str, default="checkpoints/yolov8m.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/_det_tmp")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_frames", type=int, default=3, help="number of frames to test")
    parser.add_argument("--classes", type=str, default="0,1,2,3,5,6,7,9,10,11,12,13,14,15,16", help="comma-separated class ids, e.g. '0,1'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "vis")
    ensure_dir(out_dir / "json")

    # clip 선택 : clip_id 일치하는 것만
    # <clip_id>_<8숫자>_F.png
    pats = sorted(input_dir.rglob(f"{args.clip_id}_????????_F.png"))

    if not pats:
        exts = {".png", ".jpg", ".jpeg"}
        pats = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    # max_frames 적용(0이면 전체)
    if args.max_frames and args.max_frames > 0:
        pats = pats[: args.max_frames]

    if not pats:
        raise FileNotFoundError(f"No frames found for clip_id={args.clip_id} under {input_dir}")


    model = YOLO(args.yolo_ckpt)

    classes = None
    if args.classes.strip():
        classes = [int(x) for x in args.classes.split(",") if x.strip()]

    det = run_yolo_on_images(
        model=model,
        image_paths=pats,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
    )

    for p in pats:
        p_str = str(p)
        boxes = det[p_str]

        payload = {
            "image_path": p_str,
            "clip_id": args.clip_id,
            "num_boxes": len(boxes),
            "boxes": [asdict(b) for b in boxes],
        }
        (out_dir / "json" / (p.stem + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        img = cv2.imread(p_str, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {p_str}")
        vis = draw_boxes_bgr(img, boxes)
        cv2.imwrite(str(out_dir / "vis" / (p.stem + ".png")), vis)

    print(f"[detector] done. saved to: {out_dir}")
    print(f"  - vis:  {out_dir / 'vis'}")
    print(f"  - json: {out_dir / 'json'}")


if __name__ == "__main__":
    main()
