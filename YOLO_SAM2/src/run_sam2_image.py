# src/sam2_image_infer.py
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch


# -------------------------
# IO / Util
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_det_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def bbox_xyxy_to_xyxy_np(b: List[int]) -> np.ndarray:
    # [x1,y1,x2,y2]
    return np.array(b, dtype=np.float32)


def mask_area(mask: np.ndarray) -> int:
    # mask: HxW boolean or {0,1}
    return int(mask.astype(np.uint8).sum())


def draw_overlay_bgr(
    img_bgr: np.ndarray,
    instances: List[Dict[str, Any]],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    instances: list of dict with keys:
      - mask_path
      - color (BGR)
      - label
      - bbox_xyxy
    """
    out = img_bgr.copy()
    overlay = img_bgr.copy()

    for ins in instances:
        mpath = ins["mask_path"]
        color = ins["color_bgr"]
        x1, y1, x2, y2 = ins["bbox_xyxy"]

        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127)

        overlay[m] = (overlay[m] * 0.0 + np.array(color, dtype=np.float32)).astype(np.uint8)

        # bbox + label
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            ins["label"],
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out


def stable_color(i: int) -> Tuple[int, int, int]:
    # deterministic "random-ish" BGR color
    rng = np.random.default_rng(seed=12345 + i * 17)
    c = rng.integers(40, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


# -------------------------
# SAM2 Loader (try a few APIs)
# -------------------------
def load_sam2_predictor(
    sam2_checkpoint: str,
    sam2_model_cfg: Optional[str],
    device: str,
):
    """
    Returns an object with:
      - set_image(image_rgb_uint8)
      - predict(box=..., multimask_output=...) -> masks, scores, logits (or similar)
    """

    # Candidate A: official-like "sam2" package layout
    try:
        # Example patterns seen in some SAM2 distributions
        # You may need to adjust these imports depending on your install.
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        model = build_sam2(
            config_file=sam2_model_cfg,
            ckpt_path=sam2_checkpoint,
            device=device,
        )
        predictor = SAM2ImagePredictor(model)
        return predictor
    except Exception:
        pass

    # Candidate B: alternative layout
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        model = build_sam2(
            sam2_model_cfg,
            sam2_checkpoint,
            device=device,
        )
        predictor = SAM2ImagePredictor(model)
        return predictor
    except Exception as e:
        raise ImportError(
            "SAM2 import/load failed. Your SAM2 package API may differ.\n"
            "Please paste the full ImportError / traceback and your SAM2 install method.\n"
            f"Last error: {repr(e)}"
        )


# -------------------------
# Main Inference
# -------------------------
@dataclass
class InstancePred:
    obj_idx: int
    cls_name: str
    det_conf: float
    bbox_xyxy: List[int]
    sam_score: float
    mask_path: str
    mask_area: int


def sam2_predict_one_box(
    predictor,
    image_rgb: np.ndarray,
    bbox_xyxy: List[int],
    multimask: bool,
    mask_threshold: float,
) -> Tuple[np.ndarray, float]:
    """
    Returns (best_mask_bool, best_score)
    """
    predictor.set_image(image_rgb)

    box = np.array(bbox_xyxy, dtype=np.float32)[None, :]  # (1,4)

    # Most SAM-like predictors return (masks, scores, logits) where:
    # masks: (N, H, W) boolean/0-1
    # scores: (N,) float
    out = predictor.predict(box=box, multimask_output=multimask)

    # normalize return shape
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        masks = out[0]
        scores = out[1]
    else:
        raise RuntimeError("Unexpected predictor.predict() output format")

    masks = np.asarray(masks)
    scores = np.asarray(scores)

    # masks could be (1, H, W) or (K, H, W)
    if masks.ndim == 4:
        masks = masks[:, 0]  # (K, H, W)

    best_i = int(np.argmax(scores))
    best_mask = masks[best_i]
    best_score = float(scores[best_i])

    # thresholding (in case mask is float)
    if best_mask.dtype != np.bool_:
        best_mask = best_mask > mask_threshold
    else:
        best_mask = best_mask.astype(bool)

    return best_mask, best_score


def main():
    ap = argparse.ArgumentParser("Step 3 (Option A): SAM2 image-mode inference using YOLO bbox prompts")

    ap.add_argument("--det_json_dir", type=str, required=True, help="YOLO outputs json dir (outputs/_det_tmp/json)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (e.g., outputs/_sam2_tmp)")
    ap.add_argument("--sam2_ckpt", type=str, required=True, help="SAM2 checkpoint path")
    ap.add_argument("--sam2_cfg", type=str, default="", help="SAM2 model config path (if required by your build)")
    ap.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="0=all jsons, else limit for quick test")
    ap.add_argument("--min_area", type=int, default=200, help="min mask area filter")
    ap.add_argument("--multimask", action="store_true", help="enable multimask output and choose best score")
    ap.add_argument("--mask_threshold", type=float, default=0.0, help="float mask threshold if mask is prob/logit")
    ap.add_argument("--amp", action="store_true", help="use autocast for speed (if supported)")

    args = ap.parse_args()

    det_dir = Path(args.det_json_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "masks")
    ensure_dir(out_dir / "overlay")
    ensure_dir(out_dir / "pred_json")

    det_jsons = sorted(det_dir.glob("*.json"))
    if args.max_frames and args.max_frames > 0:
        det_jsons = det_jsons[: args.max_frames]

    if not det_jsons:
        raise FileNotFoundError(f"No det json found in: {det_dir}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to cpu")
        device = "cpu"

    predictor = load_sam2_predictor(
        sam2_checkpoint=args.sam2_ckpt,
        sam2_model_cfg=(args.sam2_cfg if args.sam2_cfg.strip() else None),
        device=device,
    )

    print(f"[sam2] loaded predictor on device={device}")
    print(f"[sam2] frames={len(det_jsons)}")

    for frame_i, jpath in enumerate(det_jsons):
        det = load_det_json(jpath)

        image_path = Path(det["image_path"])
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"cv2.imread failed: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        boxes = det.get("boxes", [])
        clip_id = det.get("clip_id", "unknown")

        instances_for_overlay: List[Dict[str, Any]] = []
        pred_items: List[InstancePred] = []

        for obj_idx, b in enumerate(boxes):
            cls_name = b["cls_name"]
            det_conf = float(b["conf"])
            bbox = b["bbox_xyxy"]  # list[int]

            if args.amp and device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    m, s = sam2_predict_one_box(
                        predictor=predictor,
                        image_rgb=img_rgb,
                        bbox_xyxy=bbox,
                        multimask=args.multimask,
                        mask_threshold=args.mask_threshold,
                    )
            else:
                m, s = sam2_predict_one_box(
                    predictor=predictor,
                    image_rgb=img_rgb,
                    bbox_xyxy=bbox,
                    multimask=args.multimask,
                    mask_threshold=args.mask_threshold,
                )

            area = mask_area(m)
            if area < args.min_area:
                continue

            # 마스크 저장 (0/255)
            stem = image_path.stem
            mask_name = f"{stem}__obj{obj_idx:03d}.png"
            mask_path = out_dir / "masks" / mask_name
            cv2.imwrite(str(mask_path), (m.astype(np.uint8) * 255))

            pred_items.append(
                InstancePred(
                    obj_idx=obj_idx,
                    cls_name=cls_name,
                    det_conf=det_conf,
                    bbox_xyxy=[int(x) for x in bbox],
                    sam_score=float(s),
                    mask_path=str(mask_path),
                    mask_area=int(area),
                )
            )

            color = stable_color(obj_idx)
            label = f"{cls_name} d{det_conf:.2f} s{s:.2f}"
            instances_for_overlay.append(
                {
                    "mask_path": str(mask_path),
                    "color_bgr": color,
                    "label": label,
                    "bbox_xyxy": [int(x) for x in bbox],
                }
            )

        # 오버레이
        overlay = draw_overlay_bgr(img_bgr, instances_for_overlay, alpha=0.45)
        out_overlay_path = out_dir / "overlay" / f"{image_path.stem}.png"
        cv2.imwrite(str(out_overlay_path), overlay)

        # json
        out_pred = {
            "clip_id": clip_id,
            "image_path": str(image_path),
            "det_json": str(jpath),
            "num_instances": len(pred_items),
            "instances": [asdict(x) for x in pred_items],
            "width": img_rgb.shape[1],
            "height": img_rgb.shape[0],    
        }
        save_json(out_pred, out_dir / "pred_json" / f"{image_path.stem}.json")

        print(f"[sam2] ({frame_i+1}/{len(det_jsons)}) {image_path.name}: instances={len(pred_items)}")

    print(f"[sam2] done. out_dir={out_dir}")
    print(f"  - masks:    {out_dir / 'masks'}")
    print(f"  - overlay:  {out_dir / 'overlay'}")
    print(f"  - pred_json:{out_dir / 'pred_json'}")


if __name__ == "__main__":
    main()
