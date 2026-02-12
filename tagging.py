import os
import cv2
from typing import List, Optional, Tuple

from rc_io import Annotation, Tag, load_annotation, save_annotation


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sec_to_str(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:06.3f}"


def annotate_video(
    video_path: str,
    out_json_path: str,
    *,
    initial_seek_sec: Optional[float] = None,
    window_name: str = "tagger",
    resume_if_exists: bool = True,
    # 追加: 表示サイズ（Noneなら動画サイズ）
    window_size: Optional[Tuple[int, int]] = (1280, 720),
) -> None:
    """
    pygame UIでタグ付け（画面移動なし）

    キー操作:
      SPACE : 再生/停止
      ←/→  : 1フレーム戻る/進む（停止中）
      Shift + ←/→ : 0.5秒戻る/進む
      T     : タグ入力モード開始（画面内で入力）
      Enter : タグ確定（入力モード中）
      Esc   : 入力キャンセル（入力モード中）
      Backspace : 文字削除（入力モード中）
      U     : 直近タグを削除
      ↑/↓  : タグ一覧の選択（停止中推奨）
      Delete: 選択中タグを削除
      R     : タグ一覧表示切替
      Q     : 終了して保存
    """
    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame が必要です: uv pip install pygame")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # ===== resume: load existing tags =====
    tags: List[Tag] = []
    if resume_if_exists and os.path.exists(out_json_path):
        try:
            prev = load_annotation(out_json_path)
            if prev.video_path != video_path:
                print(
                    f"[WARN] Existing JSON refers to a different video:\n"
                    f"  json video_path: {prev.video_path}\n"
                    f"  current video:   {video_path}\n"
                    f"  -> tags will be loaded anyway."
                )
            tags = prev.tags[:]
            tags.sort(key=lambda x: x.t)
            if initial_seek_sec is None and tags:
                initial_seek_sec = tags[-1].t
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON ({out_json_path}): {e}")
            print("[WARN] Start with empty tags.")

    if initial_seek_sec is None:
        initial_seek_sec = 0.0

    # initial frame
    cur_frame = int(_clamp(initial_seek_sec * fps, 0, max(0, total_frames - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    # ===== pygame init =====
    pygame.init()
    pygame.display.set_caption(window_name)

    if window_size is None:
        screen_w, screen_h = w, h
    else:
        screen_w, screen_h = window_size

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 28)
    font_small = pygame.font.SysFont(None, 22)

    playing = False
    show_list = True
    input_mode = False
    input_text = ""
    input_hint = "Type tag name, Enter=OK, Esc=Cancel"
    selected_idx = max(0, len(tags) - 1) if tags else 0

    # seek helpers
    def set_frame(idx: int) -> None:
        nonlocal cur_frame
        cur_frame = int(_clamp(idx, 0, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    def jump_seconds(dt: float) -> None:
        set_frame(cur_frame + int(dt * fps))

    def current_time_sec() -> float:
        return cur_frame / fps

    # render helpers
    def draw_text(s: str, x: int, y: int, color=(255, 255, 255), small=False) -> None:
        f = font_small if small else font
        surf = f.render(s, True, color)
        screen.blit(surf, (x, y))

    def draw_rect(x, y, ww, hh, color=(0, 0, 0), alpha=160) -> None:
        r = pygame.Surface((ww, hh), pygame.SRCALPHA)
        r.fill((color[0], color[1], color[2], alpha))
        screen.blit(r, (x, y))

    # main loop
    running = True
    frame_rgb = None

    while running:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                # input mode
                if input_mode:
                    if event.key == pygame.K_ESCAPE:
                        input_mode = False
                        input_text = ""
                    elif event.key == pygame.K_RETURN:
                        name = input_text.strip()
                        if name:
                            t = current_time_sec()
                            tags.append(Tag(name=name, t=t))
                            tags.sort(key=lambda x: x.t)
                            selected_idx = max(0, len(tags) - 1)
                        input_mode = False
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        # text input: printable only
                        ch = event.unicode
                        if ch and ch.isprintable():
                            input_text += ch
                    continue  # don't fall through while inputting

                # normal mode
                if event.key in (pygame.K_q,):
                    running = False

                elif event.key == pygame.K_SPACE:
                    playing = not playing

                elif event.key in (pygame.K_r,):
                    show_list = not show_list

                elif event.key in (pygame.K_t,):
                    input_mode = True
                    input_text = ""

                elif event.key in (pygame.K_u,):
                    if tags:
                        tags.pop()
                        selected_idx = min(selected_idx, max(0, len(tags) - 1))

                elif event.key == pygame.K_DELETE:
                    if tags and 0 <= selected_idx < len(tags):
                        tags.pop(selected_idx)
                        selected_idx = min(selected_idx, max(0, len(tags) - 1))

                elif event.key == pygame.K_UP:
                    if tags:
                        selected_idx = max(0, selected_idx - 1)

                elif event.key == pygame.K_DOWN:
                    if tags:
                        selected_idx = min(len(tags) - 1, selected_idx + 1)

                # seek (prefer when paused)
                elif event.key == pygame.K_LEFT and not playing:
                    if mods & pygame.KMOD_SHIFT:
                        jump_seconds(-0.5)
                    else:
                        set_frame(cur_frame - 1)

                elif event.key == pygame.K_RIGHT and not playing:
                    if mods & pygame.KMOD_SHIFT:
                        jump_seconds(0.5)
                    else:
                        set_frame(cur_frame + 1)

        # update frame
        if playing:
            ok, bgr = cap.read()
            if not ok:
                playing = False
            else:
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # if paused, read current frame once per loop (keep in sync)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        ok, bgr = cap.read()
        if not ok:
            # end
            playing = False
            set_frame(max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ok, bgr = cap.read()
            if not ok:
                break

        # convert BGR->RGB, resize to window
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (screen_w, screen_h):
            rgb = cv2.resize(rgb, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
        frame_rgb = rgb

        # draw
        # pygame expects (w,h,3) -> surface via surfarray
        surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # HUD background
        draw_rect(10, 10, screen_w - 20, 120, color=(0, 0, 0), alpha=140)
        t = current_time_sec()
        draw_text(f"{os.path.basename(video_path)}", 20, 18)
        draw_text(f"t={t:8.3f}s ({_sec_to_str(t)})  frame={cur_frame}/{total_frames}  fps={fps:.2f}", 20, 48, small=True)
        draw_text("SPACE play/pause | ←/→ step (paused) | Shift+←/→ 0.5s | T add tag | U undo | Q quit", 20, 72, small=True)
        draw_text("↑/↓ select tag | Delete remove selected | R toggle list", 20, 94, small=True)

        # input mode overlay (no focus change)
        if input_mode:
            draw_rect(10, screen_h - 90, screen_w - 20, 80, color=(0, 0, 0), alpha=170)
            draw_text(input_hint, 20, screen_h - 82, small=True)
            draw_text(f"> {input_text}", 20, screen_h - 54)

        # tag list overlay
        if show_list:
            panel_w = min(520, screen_w - 20)
            panel_h = min(320, screen_h - 150)
            x0 = 10
            y0 = 140
            draw_rect(x0, y0, panel_w, panel_h, color=(0, 0, 0), alpha=140)
            draw_text(f"Tags ({len(tags)})", x0 + 10, y0 + 8)

            # show last N with selection
            # list scroll: keep selected visible
            max_rows = (panel_h - 44) // 22
            max_rows = max(1, max_rows)
            if tags:
                start = max(0, selected_idx - max_rows + 1)
                end = min(len(tags), start + max_rows)
            else:
                start, end = 0, 0

            y = y0 + 34
            for i in range(start, end):
                tg = tags[i]
                line = f"[{i:03d}] {tg.t:8.3f}s  {_sec_to_str(tg.t)}   {tg.name}"
                if i == selected_idx:
                    draw_rect(x0 + 6, y - 2, panel_w - 12, 22, color=(255, 255, 255), alpha=40)
                    draw_text(line, x0 + 10, y, color=(255, 255, 255), small=True)
                else:
                    draw_text(line, x0 + 10, y, color=(220, 220, 220), small=True)
                y += 22

        pygame.display.flip()

        # if paused, don't auto-advance
        if playing:
            # target FPS display (not video FPS; decode speed control)
            clock.tick(60)
        else:
            clock.tick(30)

    cap.release()
    pygame.quit()

    ann = Annotation(video_path=video_path, fps=fps, tags=tags)
    save_annotation(ann, out_json_path)
    print(f"Saved annotation: {out_json_path}")
