# src/sam2_video_infer.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_frames(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    frames.sort(key=lambda x: x.name)
    return frames


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def read_image_wh(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        w, h = im.size
    return w, h


def to_uint8_mask(mask_bool: np.ndarray) -> np.ndarray:
    return (mask_bool.astype(np.uint8) * 255)


def save_mask_png(mask_bool: np.ndarray, out_path: Path) -> int:
    m = np.asarray(mask_bool)

    # (1,H,W) or (H,W,1) 같은 형태를 (H,W)로 정리
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[:, :, 0]
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape={m.shape}")

    mask_u8 = (m.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(out_path)
    return int(m.sum())


# -----------------------------
# YOLO json
# -----------------------------
@dataclass
class Det:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: List[float]  # [x1,y1,x2,y2]


def load_det_json(det_json_path: Path, conf_thres: float, allowed_cls_ids: Optional[set]) -> List[Det]:
    """
    detector_infer.py가 저장한 json 스키마가 다를 경우 대비
    """
    if not det_json_path.exists():
        return []

    with open(det_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 후보 키
    cand_lists = []
    for k in ["boxes", "instances", "detections", "pred", "objects"]:
        if isinstance(data, dict) and k in data and isinstance(data[k], list):
            cand_lists = data[k]
            break

    if not cand_lists and isinstance(data, list):
        cand_lists = data

    dets: List[Det] = []
    for d in cand_lists:
        if not isinstance(d, dict):
            continue

        # 다양한 이름 대응
        cls_id = d.get("cls_id", d.get("class_id", d.get("cls", d.get("category_id", None))))
        cls_name = d.get("cls_name", d.get("name", d.get("class_name", "")))
        conf = d.get("conf", d.get("score", d.get("det_conf", 0.0)))
        bbox = d.get("bbox_xyxy", d.get("xyxy", d.get("bbox", None)))

        if cls_id is None or bbox is None:
            continue
        cls_id = int(cls_id)
        conf = float(conf)

        if conf < conf_thres:
            continue
        if allowed_cls_ids is not None and cls_id not in allowed_cls_ids:
            continue

        # bbox가 [x,y,w,h] 형태일 때 대비
        bbox = list(map(float, bbox))
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # [x,y,w,h] 일 때 x2<x1 일 경우
            if x2 < x1 or y2 < y1:
                # [x,y,w,h] 가정
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
            bbox_xyxy = [x1, y1, x2, y2]
        else:
            continue

        dets.append(Det(cls_id=cls_id, cls_name=str(cls_name), conf=conf, bbox_xyxy=bbox_xyxy))

    return dets


# -----------------------------
# SAM2 loader
# -----------------------------
def load_sam2_video_predictor(sam2_ckpt: str, sam2_cfg: str, device: str):
    from pathlib import Path
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore

    cfg = sam2_cfg
    p = Path(sam2_cfg)

    if p.exists() and p.is_file():
        txt = p.read_text(encoding="utf-8").strip()
        if txt.startswith("configs/"):
            cfg = txt
        else:
            # 로컬 파일 경로 지정 안됐을 때
            cfg = p.name

    predictor = build_sam2_video_predictor(
        config_file=cfg,
        ckpt_path=sam2_ckpt,
        device=device,
    )
    return predictor


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="dataset/clips/<clip_id>")
    ap.add_argument("--det_json_dir", required=True, help="outputs/_det_clip/<clip_id>/json")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--sam2_cfg", required=True)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--conf_thres", type=float, default=0.25)
    ap.add_argument("--allowed_classes", default="", help="comma-separated class ids. empty=all")

    ap.add_argument("--prompt_every", type=int, default=10,
                    help="N 프레임마다 YOLO bbox로 재프롬프트(추가/보정). 0이면 첫 프레임만.")
    ap.add_argument("--match_iou", type=float, default=0.5,
                    help="재프롬프트 시 기존 obj_id와 bbox IoU가 이 이상이면 같은 obj로 매칭")

    ap.add_argument("--max_frames", type=int, default=0, help="0=all")
    ap.add_argument("--amp", action="store_true")

    return ap.parse_args()


def stage_png_frames_to_jpg(frames: List[Path], stage_dir: Path) -> Path:
    """
    SAM2 video loader가 JPG만 읽어서, PNG 프레임을 stage_dir에 000000.jpg 형태로 저장.
    이미 있으면 재사용(개수 기준).
    """
    ensure_dir(stage_dir)

    # 이미 만들어둔 jpg가 충분하면 재생성 스킵
    existing = sorted([p for p in stage_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    if len(existing) == len(frames):
        return stage_dir

    # 새로 생성(덮어쓰기)
    for i, src in enumerate(frames):
        dst = stage_dir / f"{i:06d}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.save(dst, quality=95, subsampling=0, optimize=True)
    
    return stage_dir


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    det_json_dir = Path(args.det_json_dir)
    out_dir = Path(args.out_dir)

    clip_id = frames_dir.name
    frames = list_frames(frames_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: args.max_frames]

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    allowed = None
    if args.allowed_classes.strip():
        allowed = set(int(x.strip()) for x in args.allowed_classes.split(",") if x.strip())
    
    # PNG면 out_dir/_frames_jpg 로 스테이징
    video_dir_for_sam2 = frames_dir
    if any(p.suffix.lower() == ".png" for p in frames):
        stage_dir = out_dir / "_frames_jpg"
        video_dir_for_sam2 = stage_png_frames_to_jpg(frames, stage_dir)
        print(f"[sam2-video] staged frames to jpg: {video_dir_for_sam2}")

    ensure_dir(out_dir)
    masks_dir = out_dir / "masks"
    pred_dir = out_dir / "pred_json"
    ensure_dir(masks_dir)
    ensure_dir(pred_dir)

    # SAM2 video predictor
    predictor = load_sam2_video_predictor(
        sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        device=args.device,
    )
    print(f"[sam2-video] loaded predictor on device={args.device}")
    print(f"[sam2-video] clip={clip_id} frames={len(frames)}")

    # 배포판에 따라 arg 이름 다를 경우
    try:
        state = predictor.init_state(video_path=str(video_dir_for_sam2))  # type: ignore
    except TypeError:
        state = predictor.init_state(frames_dir=str(video_dir_for_sam2))  # type: ignore

    # obj tracking table
    next_obj_id = 1
    last_box: Dict[int, List[float]] = {}
    last_cls: Dict[int, int] = {}
    last_name: Dict[int, str] = {}
    last_conf: Dict[int, float] = {}

    def assign_obj_id(det: Det) -> int:
        nonlocal next_obj_id
        # 같은 class 중에서 IoU로 가장 잘 맞는 obj_id를 찾는다
        best_id = None
        best_iou = 0.0
        for oid, b in last_box.items():
            if last_cls.get(oid) != det.cls_id:
                continue
            iou = bbox_iou(b, det.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_id = oid

        if best_id is not None and best_iou >= args.match_iou:
            return best_id

        oid = next_obj_id
        next_obj_id += 1
        return oid

    def add_prompts(frame_idx: int):
        nonlocal next_obj_id
        det_json_path = det_json_dir / f"{frame_idx:06d}.json"
        dets = load_det_json(det_json_path, args.conf_thres, allowed)

        if len(dets) == 0:
            return 0

        added = 0
        for det in dets:
            oid = assign_obj_id(det)
            # SAM2 video prompt: box 기반 추가
            try:
                predictor.add_new_points_or_box(  # type: ignore
                    state=state,
                    frame_idx=frame_idx,
                    obj_id=oid,
                    box=np.array(det.bbox_xyxy, dtype=np.float32),
                )
            except TypeError:
                # 배포판 따라 파라미터 이름 다를 경우
                predictor.add_new_points_or_box(  # type: ignore
                    state,
                    frame_idx,
                    oid,
                    box=np.array(det.bbox_xyxy, dtype=np.float32),
                )

            last_box[oid] = det.bbox_xyxy
            last_cls[oid] = det.cls_id
            last_name[oid] = det.cls_name
            last_conf[oid] = det.conf
            added += 1
        return added

    # 1) 첫 프레임에서 seed
    seed_added = add_prompts(frame_idx=0)
    print(f"[sam2-video] seed prompts @0: {seed_added}")

    # 2) 필요 시 중간중간 재프롬프트
    if args.prompt_every and args.prompt_every > 0:
        for fi in range(args.prompt_every, len(frames), args.prompt_every):
            added = add_prompts(frame_idx=fi)
            if added:
                print(f"[sam2-video] refresh prompts @{fi}: {added}")

    # 3) propagate
    use_amp = bool(args.amp) and args.device.startswith("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda") if use_amp else torch.no_grad()

    results_by_frame: Dict[int, List[dict]] = {i: [] for i in range(len(frames))}
    w0, h0 = read_image_wh(frames[0])

    with torch.no_grad():
        if use_amp:
            ctx = torch.amp.autocast("cuda")
        else:
            class Dummy:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            ctx = Dummy()

        with ctx:
            for out in predictor.propagate_in_video(state):
                # 다양한 반환 형태 대응
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    frame_idx, obj_ids, masks = out[0], out[1], out[2]
                else:
                    continue

                frame_idx = int(frame_idx)
                if frame_idx >= len(frames):
                    continue

                # masks: (num_obj, H, W)
                masks_np = masks
                if torch.is_tensor(masks_np):
                    masks_np = masks_np.detach().float().cpu().numpy()

                # logits -> bool
                if masks_np.dtype != np.bool_:
                    masks_bool = masks_np > 0
                else:
                    masks_bool = masks_np

                # obj_ids
                if torch.is_tensor(obj_ids):
                    obj_ids = obj_ids.detach().cpu().tolist()
                obj_ids = [int(x) for x in obj_ids]

                # per-object mask 저장
                for k, oid in enumerate(obj_ids):
                    m = masks_bool[k]
                    if m.sum() == 0:
                        continue

                    mask_name = f"{clip_id}__f{frame_idx:06d}__obj{oid:04d}.png"
                    mask_path = masks_dir / mask_name
                    area = save_mask_png(m, mask_path)

                    results_by_frame[frame_idx].append({
                        "obj_id": oid,
                        "cls_id": int(last_cls.get(oid, -1)),
                        "cls_name": str(last_name.get(oid, "")),
                        "det_conf": float(last_conf.get(oid, 0.0)),
                        "mask_path": str(mask_path).replace("/", "\\"),
                        "mask_area": int(area),
                    })

    # 4) 프레임별 json 저장
    for fi, frame_path in enumerate(frames):
        w, h = read_image_wh(frame_path)
        rec = {
            "clip_id": clip_id,
            "frame_idx": fi,
            "image_path": str(frame_path).replace("/", "\\"),
            "width": int(w),
            "height": int(h),
            "num_instances": len(results_by_frame[fi]),
            "instances": results_by_frame[fi],
        }
        out_json = pred_dir / f"{fi:06d}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"[sam2-video] done. out_dir={out_dir}")
    print(f"  - masks:    {masks_dir}")
    print(f"  - pred_json:{pred_dir}")


if __name__ == "__main__":
    main()
