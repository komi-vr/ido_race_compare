from typing import List, Optional, Dict, Tuple
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, VideoClip
from alignment import AlignmentPlan, find_segment_index, map_out_time_to_local


def _format_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int]:
    # Pillow version compatibility
    try:
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return draw.textsize(text, font=font)


def render_comparison_video(
    video_paths: Dict[str, str],  # ← {"動画名": "/path/to/video.mp4"}
    plan: AlignmentPlan,
    out_path: str,
    *,
    out_w: int = 1280,
    bar_h: int = 140,
    margin: int = 8,
    font_path: Optional[str] = None,
    fps: int = 60,
    eps_hold: float = 1e-3,
) -> None:
    # keep stable order
    labels: List[str] = list(video_paths.keys())
    paths: List[str] = [video_paths[k] for k in labels]

    clips = [VideoFileClip(p) for p in paths]
    n = len(clips)
    if n == 0:
        raise ValueError("video_paths is empty")

    # layout: vertical stack (simple)
    available_h = 720
    vid_h_total = (available_h - margin * (n - 1))
    vid_h_each = max(1, vid_h_total // n)

    font_main = _load_font(font_path, 28)
    font_small = _load_font(font_path, 22)

    def make_frame(out_t: float) -> np.ndarray:
        seg_idx = find_segment_index(plan, out_t)
        seg = plan.segments[seg_idx]
        prev_seg = plan.segments[seg_idx - 1] if seg_idx - 1 >= 0 else None

        # time within segment on output timeline
        dt_in_seg = max(0.0, min(seg.out_dur, out_t - seg.out_start))

        # local times for each video (progress then hold)
        local_ts = [map_out_time_to_local(plan, seg_idx, v, out_t) for v in range(n)]

        # segment raw durations
        seg_raw = seg.per_video_dur
        ref_raw = seg_raw[0]
        seg_delta = [seg_raw[v] - ref_raw for v in range(n)]

        # cumulative raw time per video "so far" at current out_t
        cum_partial = []
        for v in range(n):
            partial_raw = min(dt_in_seg, seg_raw[v])  # progresses until raw_dur then holds
            prev_cum = plan.cum_raw_at_seg_end[seg_idx - 1][v] if seg_idx > 0 else 0.0
            cum_partial.append(prev_cum + partial_raw)

        # cumulative delta vs ref
        cum_ref = cum_partial[0]
        cum_delta = [cum_partial[v] - cum_ref for v in range(n)]

        # rank: smaller cumulative raw time => faster (at this moment)
        # tie-breaker: index
        rank_order = sorted(range(n), key=lambda v: (cum_partial[v], v))
        rank_of = {v: i + 1 for i, v in enumerate(rank_order)}

        # hold state: output time advanced beyond this video's raw duration in this segment
        # if dt_in_seg > raw_dur -> it is holding (waiting)
        hold_flags = [(dt_in_seg > (seg_raw[v] + eps_hold)) for v in range(n)]

        # build canvas
        canvas_h = bar_h + (vid_h_each * n) + margin * (n - 1)
        canvas = Image.new("RGB", (out_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # top bar background
        draw.rectangle([0, 0, out_w, bar_h], fill=(20, 20, 20))

        # top bar texts
        seg_elapsed = dt_in_seg
        seg_name = f"{seg.tag_a} → {seg.tag_b} (区間名: {seg.name})"
        draw.text((16, 10), seg_name, font=font_main, fill=(255, 255, 255))
        draw.text(
            (16, 52),
            f"区間経過(出力): {_format_time(seg_elapsed)} / {_format_time(seg.out_dur)}",
            font=font_small,
            fill=(220, 220, 220),
        )
        draw.text(
            (16, 84),
            f"全体経過(出力): {_format_time(out_t)} / {_format_time(plan.total_out_dur)}",
            font=font_small,
            fill=(220, 220, 220),
        )

        # per-video lines in top bar
        x0 = out_w // 2
        y0 = 52
        line_h = 26
        for v in range(n):
            hold_txt = " HOLD" if hold_flags[v] else ""
            txt = (
                f"#{rank_of[v]} {labels[v]}:{hold_txt} "
                f"区間={_format_time(seg_raw[v])}  Δ区間={seg_delta[v]:+0.3f}s   Δ累積={cum_delta[v]:+0.3f}s"
            )
            draw.text(
                (x0, y0 + v * line_h),
                txt,
                font=font_small,
                fill=(255, 255, 255) if v == 0 else (200, 200, 200),
            )

        # previous segment faint
        if prev_seg is not None:
            prev_raw = prev_seg.per_video_dur
            prev_ref = prev_raw[0]
            prev_delta = [prev_raw[v] - prev_ref for v in range(n)]
            base_y = bar_h - 26
            prev_title = f"前区間: {prev_seg.tag_a} → {prev_seg.tag_b}"
            draw.text((16, base_y), prev_title, font=font_small, fill=(140, 140, 140))
            for v in range(n):
                tprev = f"{labels[v]}: 区間={_format_time(prev_raw[v])}  Δ={prev_delta[v]:+0.3f}s"
                draw.text((x0, base_y + v * 20), tprev, font=font_small, fill=(120, 120, 120))

        # paste each video frame + label overlay
        y = bar_h
        for v, clip in enumerate(clips):
            lt = local_ts[v]
            frame = clip.get_frame(lt)  # numpy RGB
            img = Image.fromarray(frame)

            # resize to out_w width while keeping aspect, then center-crop to (out_w, vid_h_each)
            aspect = img.width / max(1, img.height)
            target_w = out_w
            target_h = int(target_w / max(1e-6, aspect))
            if target_h < vid_h_each:
                target_h = vid_h_each
                target_w = int(target_h * aspect)
            img = img.resize((target_w, target_h), Image.BICUBIC)

            left = max(0, (target_w - out_w) // 2)
            top = max(0, (target_h - vid_h_each) // 2)
            img = img.crop((left, top, left + out_w, top + vid_h_each))

            canvas.paste(img, (0, y))

            # label: rank + name + HOLD
            label_text = f"#{rank_of[v]} {labels[v]}"
            if hold_flags[v]:
                label_text += "  HOLD"

            pad = 8
            tw, th = _text_size(draw, font_small, label_text)
            box_w = tw + pad * 2
            box_h = th + pad * 2

            # semi-transparent box
            overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
            canvas.paste(overlay, (12, y + 12), overlay)

            # text
            draw.text((12 + pad, y + 12 + pad), label_text, font=font_small, fill=(255, 255, 255))

            y += vid_h_each + margin

        return np.asarray(canvas)

    from moviepy.editor import VideoClip
    out_clip = VideoClip(make_frame=make_frame, duration=plan.total_out_dur).set_fps(fps)

    out_clip.write_videofile(
        out_path,
        codec="libx264",
        audio=False,
        fps=fps,
        threads=4,
        preset="medium",
    )

    for c in clips:
        c.close()
