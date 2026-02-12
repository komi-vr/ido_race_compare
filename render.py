from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from alignment import AlignmentPlan, find_segment_index, map_out_time_to_local


def _format_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def render_comparison_video(
    video_paths: List[str],
    plan: AlignmentPlan,
    out_path: str,
    *,
    out_w: int = 1280,
    bar_h: int = 140,
    margin: int = 8,
    font_path: Optional[str] = None,
    fps: int = 60,
) -> None:
    clips = [VideoFileClip(p) for p in video_paths]

    n = len(clips)

    # layout: vertical stack
    available_h = 720  # base; we'll scale via out_w and per-clip aspect
    # decide each video height by equal split (simple)
    vid_h = (available_h - margin * (n - 1))
    vid_h_each = max(1, vid_h // n)

    font_main = _load_font(font_path, 28)
    font_small = _load_font(font_path, 22)

    def make_frame(out_t: float) -> np.ndarray:
        seg_idx = find_segment_index(plan, out_t)
        seg = plan.segments[seg_idx]
        prev_seg = plan.segments[seg_idx - 1] if seg_idx - 1 >= 0 else None

        # compute per-video local times and raw durations
        local_ts = [map_out_time_to_local(plan, seg_idx, v, out_t) for v in range(n)]
        seg_raw = seg.per_video_dur
        ref_raw = seg_raw[0]
        # per-seg delta vs ref: (+ means slower than ref)
        seg_delta = [seg_raw[v] - ref_raw for v in range(n)]

        # cumulative raw delta vs ref at end of current segment boundary
        # use cum_raw_at_seg_end[seg_idx] for "so far including current segment fully"
        # for "so far at current out_t", approximate by partial in segment:
        dt_in_seg = max(0.0, min(seg.out_dur, out_t - seg.out_start))
        cum_partial = []
        for v in range(n):
            partial_raw = min(dt_in_seg, seg_raw[v])
            prev_cum = plan.cum_raw_at_seg_end[seg_idx - 1][v] if seg_idx > 0 else 0.0
            cum_partial.append(prev_cum + partial_raw)
        cum_ref = cum_partial[0]
        cum_delta = [cum_partial[v] - cum_ref for v in range(n)]

        # draw videos
        canvas = Image.new("RGB", (out_w, bar_h + (vid_h_each * n) + margin * (n - 1)), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # top bar background
        draw.rectangle([0, 0, out_w, bar_h], fill=(20, 20, 20))

        # text: current segment
        seg_elapsed = dt_in_seg
        seg_name = f"{seg.tag_a} → {seg.tag_b} (区間名: {seg.name})"
        draw.text((16, 10), seg_name, font=font_main, fill=(255, 255, 255))
        draw.text((16, 52), f"区間経過(出力): {_format_time(seg_elapsed)} / {_format_time(seg.out_dur)}", font=font_small, fill=(220, 220, 220))
        draw.text((16, 84), f"全体経過(出力): {_format_time(out_t)} / {_format_time(plan.total_out_dur)}", font=font_small, fill=(220, 220, 220))

        # per-video lines
        x0 = out_w // 2
        y0 = 52
        line_h = 26
        for v in range(n):
            # show seg raw time & delta vs ref, and cumulative delta
            txt = f"V{v+1}: 区間={_format_time(seg_raw[v])}  Δ区間={seg_delta[v]:+0.3f}s   Δ累積={cum_delta[v]:+0.3f}s"
            draw.text((x0, y0 + v * line_h), txt, font=font_small, fill=(255, 255, 255) if v == 0 else (200, 200, 200))

        # previous segment faint
        if prev_seg is not None:
            prev_raw = prev_seg.per_video_dur
            prev_ref = prev_raw[0]
            prev_delta = [prev_raw[v] - prev_ref for v in range(n)]
            base_y = bar_h - 26
            prev_title = f"前区間: {prev_seg.tag_a} → {prev_seg.tag_b}"
            draw.text((16, base_y), prev_title, font=font_small, fill=(140, 140, 140))
            for v in range(n):
                tprev = f"V{v+1}: 区間={_format_time(prev_raw[v])}  Δ={prev_delta[v]:+0.3f}s"
                draw.text((x0, base_y + v * 20), tprev, font=font_small, fill=(120, 120, 120))

        # paste each video frame
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
                # scale by height
                target_h = vid_h_each
                target_w = int(target_h * aspect)
            img = img.resize((target_w, target_h), Image.BICUBIC)

            # center crop
            left = max(0, (target_w - out_w) // 2)
            top = max(0, (target_h - vid_h_each) // 2)
            img = img.crop((left, top, left + out_w, top + vid_h_each))

            canvas.paste(img, (0, y))
            y += vid_h_each + margin

        return np.asarray(canvas)

    # build a moviepy clip from make_frame
    from moviepy.editor import VideoClip
    out_clip = VideoClip(make_frame=make_frame, duration=plan.total_out_dur)
    out_clip = out_clip.set_fps(fps)

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
