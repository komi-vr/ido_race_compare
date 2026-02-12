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
    window_size: Optional[Tuple[int, int]] = (1280, 720),
    font_path: Optional[str] = None,
) -> None:
    """
    pygame UIでタグ付け（画面移動なし）
    JSONは t(秒) と frame(フレーム番号) の両方を保存する。
    旧JSON（tだけ）も読み込み可能で、保存時にframeが補完される。

    キー操作:
      SPACE : 再生/停止
      ←/→  : 1フレーム戻る/進む（停止中）
      Shift + ←/→ : 0.5秒戻る/進む
      T     : タグ入力モード開始（画面内で入力）
      Enter : (通常) 選択中タグの時刻にジャンプ / (入力中) タグ確定
      Esc   : 入力キャンセル（入力モード中）
      Backspace : (入力中) 文字削除 / (通常) 選択タグ削除（macのDelete対策）
      Delete / fn+Delete : 選択タグ削除（環境によって効く）
      X     : 選択タグ削除（保険）
      U     : 直近タグを削除
      ↑/↓  : タグ一覧の選択（停止中推奨）
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
            if prev.video_path and prev.video_path != video_path:
                print(
                    f"[WARN] Existing JSON refers to a different video:\n"
                    f"  json video_path: {prev.video_path}\n"
                    f"  current video:   {video_path}\n"
                    f"  -> tags will be loaded anyway."
                )

            # ★ 旧JSON(tだけ)でも load_annotation が frame を補完してくれる
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

    import pygame
    pygame.init()
    pygame.display.set_caption(window_name)

    if window_size is None:
        screen_w, screen_h = w, h
    else:
        screen_w, screen_h = window_size

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    if font_path:
        font = pygame.font.Font(font_path, 28)
        font_small = pygame.font.Font(font_path, 22)
    else:
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 22)

    playing = False
    show_list = True
    input_mode = False
    input_text = ""
    input_hint = "Type tag name, Enter=OK, Esc=Cancel"
    selected_idx = max(0, len(tags) - 1) if tags else 0

    def set_frame(idx: int) -> None:
        nonlocal cur_frame
        cur_frame = int(_clamp(idx, 0, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    def jump_seconds(dt: float) -> None:
        set_frame(cur_frame + int(dt * fps))

    def jump_to_time_sec(tsec: float) -> None:
        set_frame(int(round(tsec * fps)))

    def current_time_sec() -> float:
        return cur_frame / fps

    def delete_selected_tag() -> None:
        nonlocal selected_idx
        if tags and 0 <= selected_idx < len(tags):
            tags.pop(selected_idx)
            selected_idx = min(selected_idx, max(0, len(tags) - 1))

    def draw_text(s: str, x: int, y: int, color=(255, 255, 255), small=False) -> None:
        f = font_small if small else font
        surf = f.render(s, True, color)
        screen.blit(surf, (x, y))

    def draw_rect(x, y, ww, hh, color=(0, 0, 0), alpha=160) -> None:
        r = pygame.Surface((ww, hh), pygame.SRCALPHA)
        r.fill((color[0], color[1], color[2], alpha))
        screen.blit(r, (x, y))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                # ===== input mode =====
                if input_mode:
                    if event.key == pygame.K_ESCAPE:
                        input_mode = False
                        input_text = ""
                    elif event.key == pygame.K_RETURN:
                        name = input_text.strip()
                        if name:
                            # ★ frame と t を両方保存
                            fr = int(cur_frame)
                            t = fr / fps
                            tags.append(Tag(name=name, t=t, frame=fr))
                            tags.sort(key=lambda x: x.t)
                            selected_idx = max(0, len(tags) - 1)
                        input_mode = False
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        ch = event.unicode
                        if ch and ch.isprintable():
                            input_text += ch
                    continue

                # ===== normal mode =====
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_SPACE:
                    playing = not playing

                elif event.key == pygame.K_r:
                    show_list = not show_list

                elif event.key == pygame.K_t:
                    input_mode = True
                    input_text = ""

                # Enter: jump to selected tag
                elif event.key == pygame.K_RETURN:
                    if tags and 0 <= selected_idx < len(tags):
                        playing = False
                        # frameがあるならそれを優先してジャンプ（精密）
                        tg = tags[selected_idx]
                        if tg.frame is not None:
                            set_frame(tg.frame)
                        else:
                            jump_to_time_sec(tg.t)

                # Undo last tag
                elif event.key == pygame.K_u:
                    if tags:
                        tags.pop()
                        selected_idx = min(selected_idx, max(0, len(tags) - 1))

                # mac delete variants
                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_x):
                    delete_selected_tag()

                elif event.key == pygame.K_UP:
                    if tags:
                        selected_idx = max(0, selected_idx - 1)

                elif event.key == pygame.K_DOWN:
                    if tags:
                        selected_idx = min(len(tags) - 1, selected_idx + 1)

                # seek (paused)
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
            ok, _ = cap.read()
            if not ok:
                playing = False
            else:
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # show current frame (read at cur_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        ok, bgr = cap.read()
        if not ok:
            playing = False
            set_frame(max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ok, bgr = cap.read()
            if not ok:
                break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (screen_w, screen_h):
            rgb = cv2.resize(rgb, (screen_w, screen_h), interpolation=cv2.INTER_AREA)

        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # HUD
        draw_rect(10, 10, screen_w - 20, 130, color=(0, 0, 0), alpha=140)
        t = current_time_sec()
        draw_text(f"{os.path.basename(video_path)}", 20, 18)
        draw_text(f"t={t:8.3f}s ({_sec_to_str(t)})  frame={cur_frame}/{total_frames}  fps={fps:.2f}", 20, 48, small=True)
        draw_text("SPACE play/pause | ←/→ step (paused) | Shift+←/→ 0.5s | T add tag | Enter jump", 20, 72, small=True)
        draw_text("↑/↓ select tag | Backspace/Delete/X remove selected | U undo | R toggle list | Q quit", 20, 94, small=True)

        # input overlay
        if input_mode:
            draw_rect(10, screen_h - 90, screen_w - 20, 80, color=(0, 0, 0), alpha=170)
            draw_text(input_hint, 20, screen_h - 82, small=True)
            draw_text(f"> {input_text}", 20, screen_h - 54)

        # tag list overlay
        if show_list:
            panel_w = min(600, screen_w - 20)
            panel_h = min(340, screen_h - 170)
            x0 = 10
            y0 = 150
            draw_rect(x0, y0, panel_w, panel_h, color=(0, 0, 0), alpha=140)
            draw_text(f"Tags ({len(tags)})", x0 + 10, y0 + 8)

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
                fr = tg.frame if tg.frame is not None else int(round(tg.t * fps))
                line = f"[{i:03d}] fr={fr:7d}  t={tg.t:8.3f}s  {_sec_to_str(tg.t)}   {tg.name}"
                if i == selected_idx:
                    draw_rect(x0 + 6, y - 2, panel_w - 12, 22, color=(255, 255, 255), alpha=40)
                    draw_text(line, x0 + 10, y, color=(255, 255, 255), small=True)
                else:
                    draw_text(line, x0 + 10, y, color=(220, 220, 220), small=True)
                y += 22

        pygame.display.flip()
        clock.tick(60 if playing else 30)

    cap.release()
    pygame.quit()

    # ★ 保存前に、古いtag(tだけ)にも frame を補完して保存される
    # ここでは fps を annotation.fps として保存する
    ann = Annotation(video_path=video_path, fps=fps, tags=tags)
    save_annotation(ann, out_json_path)
    print(f"Saved annotation: {out_json_path}")
