from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Any, Dict


@dataclass
class Tag:
    name: str
    t: float                 # seconds
    frame: Optional[int] = None  # frame index (0-based), optional for backward compat


@dataclass
class Annotation:
    video_path: str
    fps: float
    tags: List[Tag]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _fill_tag_time_and_frame(tag: Tag, fps: float) -> Tag:
    """
    Ensure tag has both t and frame if possible.
    - If frame missing but t exists -> derive frame = round(t*fps)
    - If t missing (or invalid) but frame exists -> derive t = frame/fps
    """
    if fps <= 0:
        fps = 30.0

    # frame missing -> from t
    if tag.frame is None:
        # t is always present in our Tag, but may be garbage; still try
        tag.frame = int(round(tag.t * fps))

    # t missing isn't possible in Tag, but could be invalid (NaN/None in JSON)
    # if JSON had only frame, we build t from it.
    if tag.t is None or not isinstance(tag.t, (int, float)):
        if tag.frame is not None:
            tag.t = float(tag.frame) / fps
        else:
            tag.t = 0.0

    # If frame exists and t seems inconsistent, we keep t as-is (human-edited),
    # but in old JSON conversion we want them consistent. A simple policy:
    # if tag came from frame-only, we already set t from frame.
    return tag


def load_annotation(path: str) -> Annotation:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_path = str(data.get("video_path", ""))
    fps = _safe_float(data.get("fps", 30.0), 30.0)

    tags_raw = data.get("tags", [])
    tags: List[Tag] = []

    if isinstance(tags_raw, list):
        for item in tags_raw:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", ""))

            # Backward/forward compat:
            # - old: {"name": "...", "t": 1.23}
            # - new: {"name": "...", "t": 1.23, "frame": 37}
            # - frame-only (if ever): {"name":"...", "frame": 37}
            t_val = item.get("t", None)
            frame_val = item.get("frame", None)

            frame = _safe_int(frame_val)

            if t_val is None:
                # frame-only
                if frame is None:
                    t = 0.0
                else:
                    t = float(frame) / fps
            else:
                t = _safe_float(t_val, 0.0)

            tag = Tag(name=name, t=t, frame=frame)
            tag = _fill_tag_time_and_frame(tag, fps)
            tags.append(tag)

    # sort by time (and frame as tie-break)
    tags.sort(key=lambda x: (x.t, x.frame if x.frame is not None else 10**18))

    return Annotation(video_path=video_path, fps=fps, tags=tags)


def save_annotation(ann: Annotation, path: str) -> None:
    fps = ann.fps if ann.fps > 0 else 30.0

    # Ensure every tag has both t and frame before saving
    tags_out: List[Dict[str, Any]] = []
    for tg in ann.tags:
        tg2 = Tag(name=tg.name, t=float(tg.t), frame=tg.frame)
        tg2 = _fill_tag_time_and_frame(tg2, fps)

        tags_out.append(
            {
                "name": tg2.name,
                "t": float(tg2.t),
                "frame": int(tg2.frame) if tg2.frame is not None else None,
            }
        )

    data = {
        "video_path": ann.video_path,
        "fps": float(fps),
        "tags": tags_out,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
