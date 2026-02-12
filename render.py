from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===== moviepy import (1.x / 2.x compatible) =====
MP_VER = "unknown"
try:
    # moviepy
    import moviepy
    MP_VER = getattr(moviepy, "__version__", "unknown")
except Exception:
    pass

# VideoFileClip / VideoClip の import 経路も違うので吸収
try:
    # moviepy 1.x
    from moviepy.editor import VideoFileClip, VideoClip
    MOVIEPY_API = "v1"
except Exception:
    # moviepy 2.x
    from moviepy import VideoFileClip, VideoClip
    MOVIEPY_API = "v2"


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
    try:
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return draw.textsize(text, font=font)


def _make_videoclip(make_frame_fn, duration: float, fps: int):
    """
    moviepy 1.x / 2.x でVideoClip生成APIが違うので吸収する。
    """
    # moviepy 1.x: VideoClip(make_frame=..., duration=...).set_fps(...)
    if MOVIEPY_API == "v1":
        clip = VideoClip(make_frame=make_frame_fn, duration=duration).set_fps(fps)
        return clip

    # moviepy 2.x: VideoClip(frame_function=...) -> with_duration/with_fps
    # 互換のため、存在するメソッドを順に試す
    clip = None

    # 2.x の代表: VideoClip(frame_function=...)
    try:
        clip = VideoClip(frame_function=make_frame_fn)
    except TypeError:
        # 環境差: VideoClip(make_frame_fn) 形式の可能性もある
        clip = VideoClip(make_frame_fn)

    # duration
    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration)
    elif hasattr(clip, "set_duration"):
        clip = clip.set_duration(duration)
    else:
        # 最悪: 属性直書き（古い互換用）
        try:
            clip.duration = duration
        except Exception:
            pass

    # fps
    if hasattr(clip, "with_fps"):
        clip = clip.with_fps(fps)
    elif hasattr(clip, "set_fps"):
        clip = clip.set_fps(fps)
    else:
        try:
            clip.fps = fps
        except Exception:
            pass

    return clip


def render_comparison_video(
    video_paths: Dict[str, str],  # {"動画名": "/path/to/video.mp4"}
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
    labels: List[str] = list(video_paths.keys())
    paths: List[str] = [video_paths[k] for k in labels]

    clips = [VideoFileClip(p) for p in paths]
    n = len(clips)
    if n == 0:
        raise ValueError("video_paths is empty")

    available_h = 720
    vid_h_total = (available_h - margin * (n - 1))
    vid_h_each = max(1, vid_h_total // n)

    font_main = _load_font(font_path, 28)
    font_small = _load_font(font_path, 22)

    def make_frame(out_t: float) -> np.ndarray:
        seg_idx = find_segment_index(plan, out_t)
        seg = plan.segments[seg_idx]
        prev_seg = plan.segments[seg_idx - 1] if seg_idx - 1 >= 0 else None

        dt_in_seg = max(0.0, min(seg.out_dur, out_t - seg.out_start))
        local_ts = [map_out_time_to_local(plan, seg_idx, v, out_t) for v in range(n)]

        seg_raw = seg.per_video_dur
        ref_raw = seg_raw[0]
        seg_delta = [seg_raw[v] - ref_raw for v in range(n)]

        cum_partial = []
        for v in range(n):
            partial_raw = min(dt_in_seg, seg_raw[v])
            prev_cum = plan.cum_raw_at_seg_end[seg_idx - 1][v] if seg_idx > 0 else 0.0
            cum_partial.append(prev_cum + partial_raw)

        cum_ref = cum_partial[0]
        cum_delta = [cum_partial[v] - cum_ref for v in range(n)]

        rank_order = sorted(range(n), key=lambda v: (cum_partial[v], v))
        rank_of = {v: i + 1 for i, v in enumerate(rank_order)}

        hold_flags = [(dt_in_seg > (seg_raw[v] + eps_hold)) for v in range(n)]

        canvas_h = bar_h + (vid_h_each * n) + margin * (n - 1)
        canvas = Image.new("RGB", (out_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, out_w, bar_h], fill=(20, 20, 20))
        seg_name = f"{seg.tag_a} → {seg.tag_b} (区間名: {seg.name})"
        draw.text((16, 10), seg_name, font=font_main, fill=(255, 255, 255))
        draw.text(
            (16, 52),
            f"区間経過(出力): {_format_time(dt_in_seg)} / {_format_time(seg.out_dur)}",
            font=font_small,
            fill=(220, 220, 220),
        )
        draw.text(
            (16, 84),
            f"全体経過(出力): {_format_time(out_t)} / {_format_time(plan.total_out_dur)}",
            font=font_small,
            fill=(220, 220, 220),
        )

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

        if prev_seg is not None:
            prev_raw = prev_seg.per_video_dur
            prev_ref = prev_raw[0]
            prev_delta = [prev_raw[v] - prev_ref for v in range(n)]
            base_y = bar_h - 26
            draw.text(
                (16, base_y),
                f"前区間: {prev_seg.tag_a} → {prev_seg.tag_b}",
                font=font_small,
                fill=(140, 140, 140),
            )
            for v in range(n):
                tprev = f"{labels[v]}: 区間={_format_time(prev_raw[v])}  Δ={prev_delta[v]:+0.3f}s"
                draw.text((x0, base_y + v * 20), tprev, font=font_small, fill=(120, 120, 120))

        y = bar_h
        for v, clip in enumerate(clips):
            lt = local_ts[v]
            frame = clip.get_frame(lt)
            img = Image.fromarray(frame)

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

            label_text = f"#{rank_of[v]} {labels[v]}" + ("  HOLD" if hold_flags[v] else "")
            pad = 8
            tw, th = _text_size(draw, font_small, label_text)
            box_w = tw + pad * 2
            box_h = th + pad * 2
            overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
            canvas.paste(overlay, (12, y + 12), overlay)
            draw.text((12 + pad, y + 12 + pad), label_text, font=font_small, fill=(255, 255, 255))

            y += vid_h_each + margin

        return np.asarray(canvas)

    # ★ ここが今回の本丸：moviepyの違いを吸収してVideoClip生成
    out_clip = _make_videoclip(make_frame, plan.total_out_dur, fps)

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
