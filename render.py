from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# moviepy 1.x / 2.x import compatibility
try:
    from moviepy.editor import (
        VideoFileClip,
        VideoClip,
        CompositeAudioClip,
        AudioClip,
    )
    MOVIEPY_API = "v1"
except Exception:
    from moviepy import (
        VideoFileClip,
        VideoClip,
        CompositeAudioClip,
        AudioClip,
    )
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
    if MOVIEPY_API == "v1":
        return VideoClip(make_frame=make_frame_fn, duration=duration).set_fps(fps)

    try:
        clip = VideoClip(frame_function=make_frame_fn)
    except TypeError:
        clip = VideoClip(make_frame_fn)

    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration)
    else:
        clip = clip.set_duration(duration)

    if hasattr(clip, "with_fps"):
        clip = clip.with_fps(fps)
    else:
        clip = clip.set_fps(fps)

    return clip


def _set_audio_on_clip(clip, audio_clip):
    if MOVIEPY_API == "v1":
        return clip.set_audio(audio_clip)
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)


def _make_silence(duration: float, fps: int = 44100):
    def make_frame(t):
        if isinstance(t, np.ndarray):
            return np.zeros((len(t), 1), dtype=np.float32)
        return np.zeros((1,), dtype=np.float32)

    if MOVIEPY_API == "v1":
        return AudioClip(make_frame, duration=duration, fps=fps)

    try:
        a = AudioClip(frame_function=make_frame)
    except TypeError:
        a = AudioClip(make_frame)

    if hasattr(a, "with_duration"):
        a = a.with_duration(duration)
    else:
        a = a.set_duration(duration)

    if hasattr(a, "with_fps"):
        a = a.with_fps(fps)
    else:
        a = a.set_fps(fps)

    return a


def _audio_subclip(audio, t0: float, t1: float):
    if audio is None:
        return None
    try:
        return audio.subclip(t0, t1)
    except Exception:
        if hasattr(audio, "subclipped"):
            return audio.subclipped(t0, t1)
        raise


def _audio_set_start(audio_clip, start: float):
    if audio_clip is None:
        return None
    if hasattr(audio_clip, "set_start"):
        return audio_clip.set_start(start)
    if hasattr(audio_clip, "with_start"):
        return audio_clip.with_start(start)
    audio_clip.start = start
    return audio_clip


def _build_segment_fastest_audio(
    video_clips: List[VideoFileClip],
    plan: AlignmentPlan,
    total_out_dur: float,
    audio_fps: int = 44100,
):
    """
    区間ごと最速の動画の音声を採用（VideoFileClip.audio を使用）
    """
    base = _make_silence(total_out_dur, fps=audio_fps)
    pieces = [base]

    for seg_idx, seg in enumerate(plan.segments):
        durs = seg.per_video_dur
        fastest = min(range(len(durs)), key=lambda i: (durs[i], i))
        seg_raw = float(durs[fastest])
        if seg_raw <= 0:
            continue

        # 区間開始のローカル生時間
        prev_raw = plan.cum_raw_at_seg_end[seg_idx - 1][fastest] if seg_idx > 0 else 0.0
        local_a = prev_raw
        local_b = prev_raw + seg_raw

        src_audio = video_clips[fastest].audio
        sub = _audio_subclip(src_audio, local_a, local_b)
        if sub is None:
            continue

        sub = _audio_set_start(sub, seg.out_start)
        pieces.append(sub)

    return CompositeAudioClip(pieces)


def _make_frame_fn(
    clips: List[VideoFileClip],
    labels: List[str],
    plan: AlignmentPlan,
    *,
    out_w: int,
    out_h: int,
    bar_h: int,
    margin: int,
    font_path: Optional[str],
    start_times: Dict[str, float],
    end_times: Dict[str, float],
    fps: int,
    eps_hold: float,
):
    n = len(clips)
    available_w = out_w - margin * (n - 1)
    tile_w = max(1, available_w // n)
    tile_h = max(1, out_h - bar_h)

    font_main = _load_font(font_path, 30)
    font_small = _load_font(font_path, 22)
    font_tiny = _load_font(font_path, 18)

    total_times = {}
    for lab in labels:
        total = float(end_times[lab] - start_times[lab])
        if total < 1e-6:
            total = 1e-6
        total_times[lab] = total

    def make_frame(out_t: float) -> np.ndarray:
        seg_idx = find_segment_index(plan, out_t)
        seg = plan.segments[seg_idx]
        prev_seg = plan.segments[seg_idx - 1] if seg_idx - 1 >= 0 else None

        dt_in_seg = max(0.0, min(seg.out_dur, out_t - seg.out_start))
        local_ts = [map_out_time_to_local(plan, seg_idx, v, out_t) for v in range(n)]

        seg_raw = seg.per_video_dur
        ref_raw = seg_raw[0]

        if getattr(seg, "exclude_from_diff", False):
            seg_delta = [0.0 for _ in range(n)]
        else:
            seg_delta = [seg_raw[v] - ref_raw for v in range(n)]

        cum_excl_partial = []
        for v in range(n):
            prev_excl = plan.cum_excl_at_seg_end[seg_idx - 1][v] if seg_idx > 0 else 0.0
            partial_raw = min(dt_in_seg, seg_raw[v])
            if getattr(seg, "exclude_from_diff", False):
                cum_excl_partial.append(prev_excl)
            else:
                cum_excl_partial.append(prev_excl + partial_raw)

        cum_ref = cum_excl_partial[0]
        cum_delta = [cum_excl_partial[v] - cum_ref for v in range(n)]
        rank_order = sorted(range(n), key=lambda v: (cum_excl_partial[v], v))
        rank_of = {v: i + 1 for i, v in enumerate(rank_order)}
        hold_flags = [(dt_in_seg > (seg_raw[v] + eps_hold)) for v in range(n)]

        canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, out_w, bar_h], fill=(20, 20, 20))

        if seg.tag_a == "__BEGIN__":
            seg_title = f"pre-start → {seg.tag_b}（startまで：タイム差計算しない）"
        else:
            seg_title = f"{seg.tag_a} → {seg.tag_b} (区間名: {seg.name})"

        draw.text((16, 10), seg_title, font=font_main, fill=(255, 255, 255))
        draw.text((16, 54), f"区間経過(出力): {_format_time(dt_in_seg)} / {_format_time(seg.out_dur)}",
                  font=font_small, fill=(220, 220, 220))
        draw.text((16, 84), f"全体経過(出力): {_format_time(out_t)} / {_format_time(plan.total_out_dur)}",
                  font=font_small, fill=(220, 220, 220))

        base_y = 120
        line_h = 24
        for v in range(n):
            lab = labels[v]
            hold_txt = " HOLD" if hold_flags[v] else ""
            txt = (
                f"#{rank_of[v]} {lab}{hold_txt} | "
                f"区間={_format_time(seg_raw[v])}  Δ区間={seg_delta[v]:+0.3f}s  Δ累積={cum_delta[v]:+0.3f}s"
            )
            draw.text((16, base_y + v * line_h), txt, font=font_tiny, fill=(255, 255, 255))

        if prev_seg is not None:
            py = bar_h - 26
            if prev_seg.tag_a == "__BEGIN__":
                ptitle = f"前区間: pre-start → {prev_seg.tag_b}"
            else:
                ptitle = f"前区間: {prev_seg.tag_a} → {prev_seg.tag_b}"
            draw.text((16, py), ptitle, font=font_tiny, fill=(140, 140, 140))

        x = 0
        y = bar_h
        for v, clip in enumerate(clips):
            lab = labels[v]
            lt = float(local_ts[v])
            frame = clip.get_frame(lt)
            img = Image.fromarray(frame)

            aspect = img.width / max(1, img.height)
            target_w = tile_w
            target_h = int(target_w / max(1e-6, aspect))
            if target_h < tile_h:
                target_h = tile_h
                target_w = int(target_h * aspect)

            img = img.resize((target_w, target_h), Image.BICUBIC)
            left = max(0, (target_w - tile_w) // 2)
            top = max(0, (target_h - tile_h) // 2)
            img = img.crop((left, top, left + tile_w, top + tile_h))
            canvas.paste(img, (x, y))

            st = float(start_times.get(lab, 0.0))
            total = float(end_times[lab] - st)
            if total < 1e-6:
                total = 1e-6
            elapsed = lt - st
            elapsed = max(0.0, min(total, elapsed))
            progress = 100.0 * (elapsed / total)

            label_text = f"#{rank_of[v]} {lab}" + ("  HOLD" if hold_flags[v] else "")
            pad = 8
            tw, th = _text_size(draw, font_small, label_text)
            overlay = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 150))
            canvas.paste(overlay, (x + 10, y + 10), overlay)
            draw.text((x + 10 + pad, y + 10 + pad), label_text, font=font_small, fill=(255, 255, 255))

            info_text = f"TOTAL {_format_time(total)} | {_format_time(elapsed)} ({progress:5.1f}%)"
            tw2, th2 = _text_size(draw, font_small, info_text)
            overlay2 = Image.new("RGBA", (tw2 + pad * 2, th2 + pad * 2), (0, 0, 0, 150))
            canvas.paste(overlay2, (x + 10, y + tile_h - (th2 + pad * 2) - 10), overlay2)
            draw.text((x + 10 + pad, y + tile_h - (th2 + pad * 2) - 10 + pad), info_text, font=font_small, fill=(255, 255, 255))

            x += tile_w + margin

        return np.asarray(canvas)

    return make_frame


def preview_comparison_video(
    video_paths: Dict[str, str],
    plan: AlignmentPlan,
    *,
    out_w: int = 1280,
    out_h: int = 720,
    bar_h: int = 210,
    margin: int = 10,
    font_path: Optional[str] = None,
    fps: int = 60,
    start_times: Dict[str, float],
    end_times: Dict[str, float],
):
    """
    pygameでリアルタイム表示（音声なし）
    操作:
      SPACE: 再生/停止
      ←/→ : 1秒戻る/進む
      Shift+←/→ : 5秒戻る/進む
      Q or ESC: 終了
    """
    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame が必要です: uv pip install pygame")

    labels = list(video_paths.keys())
    paths = [video_paths[k] for k in labels]
    clips = [VideoFileClip(p) for p in paths]

    make_frame = _make_frame_fn(
        clips, labels, plan,
        out_w=out_w, out_h=out_h, bar_h=bar_h, margin=margin,
        font_path=font_path,
        start_times=start_times, end_times=end_times,
        fps=fps, eps_hold=1e-3,
    )

    pygame.init()
    screen = pygame.display.set_mode((out_w, out_h))
    pygame.display.set_caption("compare preview")
    clock = pygame.time.Clock()

    t = 0.0
    playing = True
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_SPACE:
                    playing = not playing
                elif e.key == pygame.K_LEFT:
                    dt = 5.0 if (mods & pygame.KMOD_SHIFT) else 1.0
                    t = max(0.0, t - dt)
                elif e.key == pygame.K_RIGHT:
                    dt = 5.0 if (mods & pygame.KMOD_SHIFT) else 1.0
                    t = min(plan.total_out_dur, t + dt)

        if playing:
            t += 1.0 / fps
            if t >= plan.total_out_dur:
                t = plan.total_out_dur
                playing = False

        frame = make_frame(t)  # HxWx3 uint8
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    for c in clips:
        c.close()
    pygame.quit()


def render_comparison_video(
    video_paths: Dict[str, str],
    plan: AlignmentPlan,
    out_path: str,
    *,
    out_w: int = 1280,
    out_h: int = 720,
    bar_h: int = 210,
    margin: int = 10,
    font_path: Optional[str] = None,
    fps: int = 60,
    start_times: Dict[str, float],
    end_times: Dict[str, float],
    audio_mode: str = "none",        # "none" / "seg_fastest"
    audio_fps: int = 44100,
    eps_hold: float = 1e-3,
) -> None:
    labels = list(video_paths.keys())
    paths = [video_paths[k] for k in labels]
    clips = [VideoFileClip(p) for p in paths]

    make_frame = _make_frame_fn(
        clips, labels, plan,
        out_w=out_w, out_h=out_h, bar_h=bar_h, margin=margin,
        font_path=font_path,
        start_times=start_times, end_times=end_times,
        fps=fps, eps_hold=eps_hold,
    )

    out_clip = _make_videoclip(make_frame, plan.total_out_dur, fps)

    # audio
    if audio_mode == "seg_fastest":
        audio = _build_segment_fastest_audio(clips, plan, plan.total_out_dur, audio_fps=audio_fps)
        out_clip = _set_audio_on_clip(out_clip, audio)

    # ★ 音声が無音になるケース対策：audio_codecを明示
    out_clip.write_videofile(
        out_path,
        codec="libx264",
        audio=(audio_mode != "none"),
        audio_codec="aac" if (audio_mode != "none") else None,
        fps=fps,
        threads=4,
        preset="medium",
    )

    for c in clips:
        c.close()
