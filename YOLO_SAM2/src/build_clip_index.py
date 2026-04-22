# src/indexer.py
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# 파일명 패턴:
#   <clip_id>_<8digits>_F.png
# 예:
#   CK_A01_R01_day_clear_01008500_F.png
PATTERN = re.compile(r"^(?P<clip_id>.+)_(?P<frame>\d{6,12})_F\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class FrameItem:
    path: str         
    filename: str     
    frame_idx: int    
    ext: str          

def parse_name(fname: str) -> Optional[Tuple[str, int, str]]:
    m = PATTERN.match(fname)
    if not m:
        return None
    clip_id = m.group("clip_id")
    frame_idx = int(m.group("frame"))
    ext = m.group("ext").lower()
    return clip_id, frame_idx, ext


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    mode: copy | hardlink | symlink
    - Windows에서는 symlink가 권한/설정에 따라 실패할 수 있어 hardlink를 기본 추천.
    """
    if dst.exists():
        return

    if mode == "copy":
        safe_mkdir(dst.parent)
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        safe_mkdir(dst.parent)
        os.link(src, dst)
        return

    if mode == "symlink":
        safe_mkdir(dst.parent)
        os.symlink(src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def summarize_frames(frame_indices: List[int]) -> Tuple[int, int, int, int]:
    """
    returns:
      - n_frames
      - min_frame
      - max_frame
      - gaps_count, max_gap
    gaps:
      consecutive sorted frame_idx differences > 1 을 gap로 정의
    """
    if not frame_indices:
        return 0, -1, -1, 0, 0

    xs = sorted(frame_indices)
    min_f, max_f = xs[0], xs[-1]

    gaps = 0
    max_gap = 0
    for a, b in zip(xs, xs[1:]):
        d = b - a
        if d > 1:
            gaps += 1
            if d > max_gap:
                max_gap = d

    return len(xs), min_f, max_f, gaps, max_gap


def build_manifest(
    input_dir: Path,
    recursive: bool,
    exts: List[str],
    exclude_clips: List[str],
) -> Dict:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file()]

    exts_set = {e.lower().lstrip(".") for e in exts}

    clips: Dict[str, List[FrameItem]] = {}
    skipped = {"pattern_mismatch": 0, "ext_mismatch": 0, "excluded_clip": 0}

    for p in files:
        info = parse_name(p.name)
        if info is None:
            skipped["pattern_mismatch"] += 1
            continue

        clip_id, frame_idx, ext = info
        if ext not in exts_set:
            skipped["ext_mismatch"] += 1
            continue

        if clip_id in exclude_clips:
            skipped["excluded_clip"] += 1
            continue

        clips.setdefault(clip_id, []).append(
            FrameItem(path=str(p.resolve()), filename=p.name, frame_idx=frame_idx, ext=ext)
        )

    # 정렬 + manifest 형태로 변환
    clip_entries = []
    total_frames = 0

    for clip_id, items in clips.items():
        items.sort(key=lambda x: x.frame_idx)
        total_frames += len(items)

        clip_entries.append({
            "clip_id": clip_id,
            "num_frames": len(items),
            "frames": [
                {
                    "frame_idx": it.frame_idx,
                    "path": it.path,
                    "filename": it.filename,
                    "ext": it.ext,
                }
                for it in items
            ]
        })

    # clip_id 정렬(가독성)
    clip_entries.sort(key=lambda x: x["clip_id"])

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "recursive": recursive,
        "allowed_exts": sorted(list(exts_set)),
        "exclude_clips": exclude_clips,
        "stats": {
            "num_clips": len(clip_entries),
            "num_frames": total_frames,
            "skipped": skipped,
        },
        "clips": clip_entries,
    }
    return manifest


def make_clip_dirs(
    manifest: Dict,
    clips_root: Path,
    mode: str,
    renumber: bool,
    keep_ext: bool,
) -> None:
    """
    clips_root/
      <clip_id>/
        000000.png
        000001.png
    """
    safe_mkdir(clips_root)

    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        out_dir = clips_root / clip_id
        safe_mkdir(out_dir)

        frames = clip["frames"]
        for i, fr in enumerate(frames):
            src = Path(fr["path"])
            ext = fr["ext"]

            if renumber:
                # 000000.png
                new_name = f"{i:06d}.{ext if keep_ext else 'png'}"
            else:
                # 원본 이름 그대로
                new_name = fr["filename"]

            dst = out_dir / new_name
            link_or_copy(src, dst, mode)


def print_summary(manifest: Dict) -> None:
    clips = manifest["clips"]
    print(f"[indexer] clips={manifest['stats']['num_clips']} frames={manifest['stats']['num_frames']}")
    for clip in clips:
        clip_id = clip["clip_id"]
        frame_indices = [f["frame_idx"] for f in clip["frames"]]
        n, mn, mx, gaps, max_gap = summarize_frames(frame_indices)
        if n == 0:
            continue
        print(f"  - {clip_id}: {n} frames ({mn}..{mx}), gaps={gaps} max_gap={max_gap}")


def main():
    ap = argparse.ArgumentParser(description="Index mixed frame images into clip manifests and optional clip folders.")
    ap.add_argument("--input_dir", required=True, help="Directory containing mixed frames.")
    ap.add_argument("--out_manifest", default="outputs/_index/manifest.json", help="Where to save manifest JSON.")
    ap.add_argument("--recursive", action="store_true", help="Scan input_dir recursively.")
    ap.add_argument("--exts", default="png,jpg,jpeg", help="Allowed extensions, comma-separated.")
    ap.add_argument("--exclude_clip", action="append", default=[], help="Exclude exact clip_id (repeatable).")

    # 클립 폴더 생성 옵션
    ap.add_argument("--make_clip_dirs", action="store_true", help="Create per-clip folders under --clips_root.")
    ap.add_argument("--clips_root", default="dataset/clips", help="Root dir to create clip folders.")
    ap.add_argument("--mode", default="hardlink", choices=["copy", "hardlink", "symlink"],
                    help="How to populate clip folders. hardlink is fast if same drive.")
    ap.add_argument("--renumber", action="store_true",
                    help="Rename frames to 000000.png, 000001.png... inside each clip folder (recommended for video predictor).")
    ap.add_argument("--keep_ext", action="store_true",
                    help="When renumbering, keep original ext instead of forcing png name.")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_manifest = Path(args.out_manifest)

    exts = [x.strip() for x in args.exts.split(",") if x.strip()]
    exclude = args.exclude_clip or []

    manifest = build_manifest(
        input_dir=input_dir,
        recursive=args.recursive,
        exts=exts,
        exclude_clips=exclude,
    )

    safe_mkdir(out_manifest.parent)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(manifest)
    print(f"[indexer] manifest saved: {out_manifest}")

    if args.make_clip_dirs:
        clips_root = Path(args.clips_root)
        make_clip_dirs(
            manifest=manifest,
            clips_root=clips_root,
            mode=args.mode,
            renumber=args.renumber,
            keep_ext=args.keep_ext,
        )
        print(f"[indexer] clip dirs ready: {clips_root} (mode={args.mode}, renumber={args.renumber})")


if __name__ == "__main__":
    main()
