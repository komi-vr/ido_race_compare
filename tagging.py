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
    # 参照（お手本）: JSONを指定すると右側に表示（参照側は保存しない）
    reference_json_path: Optional[str] = None,
    # ★NEW: 参照静止画はタグ時刻の何フレーム前を出すか
    reference_pre_frames: int = 3,
    # ★NEW: タグ追加時に参照タグを自動送りするか
    auto_advance_reference: bool = True,
) -> None:
    """
    pygame UIでタグ付け（画面移動なし）
    - 自分(左)は通常通り再生/停止/シークできる
    - 参照(右)は「静止画」：選択中参照タグの (t - pre_frames) フレームを表示
    - キーで参照タグ切替、タグ追加で自動送り可能

    キー（基本）:
      SPACE : 再生/停止（左のみ）
      ←/→  : 1フレーム戻る/進む（停止中）
      Shift + ←/→ : 0.5秒戻る/進む
      T     : タグ入力モード開始（画面内入力）
      Enter : (通常) 選択中タグの時刻にジャンプ / (入力中) タグ確定
      Esc   : 入力キャンセル（入力モード中）
      Backspace/Delete/X : 選択タグ削除（自分のみ）
      U     : 直近タグを削除（自分のみ）
      ↑/↓  : タグ一覧の選択（自分側）
      R     : タグ一覧表示切替
      Q     : 終了して保存

    参照（お手本）キー:
      [ / ] : 参照タグを前/次へ切替
      J     : 参照タグの時刻へジャンプ（左をその時刻へ）
      A     : 自動送り ON/OFF 切替（auto_advance_reference）
    """
    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame が必要です: uv pip install pygame")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # ===== reference (optional) =====
    ref_enabled = reference_json_path is not None
    ref_cap = None
    ref_fps = None
    ref_total_frames = None
    ref_video_path = None
    ref_tags: List[Tag] = []
    ref_selected_idx = 0
    ref_cached_frame_idx: Optional[int] = None
    ref_cached_rgb = None  # numpy RGB image cache

    if ref_enabled:
        if not os.path.exists(reference_json_path):
            raise RuntimeError(f"reference_json_path not found: {reference_json_path}")

        ref_ann = load_annotation(reference_json_path)
        ref_video_path = ref_ann.video_path
        ref_fps = float(ref_ann.fps if ref_ann.fps and ref_ann.fps > 0 else 30.0)
        ref_tags = ref_ann.tags[:]
        ref_tags.sort(key=lambda x: x.t)
        ref_selected_idx = 0 if ref_tags else 0

        ref_cap = cv2.VideoCapture(ref_video_path)
        if not ref_cap.isOpened():
            raise RuntimeError(f"Failed to open reference video: {ref_video_path}")
        ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # ===== resume: load existing tags =====
    tags: List[Tag] = []
    if resume_if_exists and os.path.exists(out_json_path):
        try:
            prev = load_annotation(out_json_path)
            tags = prev.tags[:]
            tags.sort(key=lambda x: x.t)

            if initial_seek_sec is None and tags:
                initial_seek_sec = tags[-1].t
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON ({out_json_path}): {e}")
            print("[WARN] Start with empty tags.")

    if initial_seek_sec is None:
        initial_seek_sec = 0.0

    cur_frame = int(_clamp(initial_seek_sec * fps, 0, max(0, total_frames - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    import pygame
    pygame.init()
    pygame.display.set_caption(window_name)

    # layout
    if window_size is None:
        screen_w, screen_h = w, h
    else:
        screen_w, screen_h = window_size

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    if font_path:
        font = pygame.font.Font(font_path, 28)
        font_small = pygame.font.Font(font_path, 22)
        font_tiny = pygame.font.Font(font_path, 18)
    else:
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 22)
        font_tiny = pygame.font.SysFont(None, 18)

    playing = False
    show_list = True
    input_mode = False
    input_text = ""

    selected_idx = max(0, len(tags) - 1) if tags else 0
    auto_adv = bool(auto_advance_reference)

    def set_frame(idx: int) -> None:
        nonlocal cur_frame
        cur_frame = int(_clamp(idx, 0, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    def jump_seconds(dt: float) -> None:
        set_frame(cur_frame + int(round(dt * fps)))

    def jump_to_time_sec(tsec: float) -> None:
        set_frame(int(round(tsec * fps)))

    def current_time_sec() -> float:
        return cur_frame / fps

    def delete_selected_tag() -> None:
        nonlocal selected_idx
        if tags and 0 <= selected_idx < len(tags):
            tags.pop(selected_idx)
            selected_idx = min(selected_idx, max(0, len(tags) - 1))

    def draw_text(s: str, x: int, y: int, color=(255, 255, 255), small=False, tiny=False) -> None:
        f = font_tiny if tiny else (font_small if small else font)
        surf = f.render(s, True, color)
        screen.blit(surf, (x, y))

    def draw_rect(x, y, ww, hh, color=(0, 0, 0), alpha=160) -> None:
        r = pygame.Surface((ww, hh), pygame.SRCALPHA)
        r.fill((color[0], color[1], color[2], alpha))
        screen.blit(r, (x, y))

    def read_frame_for(cap_obj, frame_idx: int):
        cap_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap_obj.read()
        return ok, bgr

    def ref_tag_frame_index(idx: int) -> Optional[int]:
        if not ref_enabled or not ref_tags:
            return None
        idx = int(_clamp(idx, 0, len(ref_tags) - 1))
        tg = ref_tags[idx]
        fr = int(round(tg.t * ref_fps)) - int(reference_pre_frames)
        fr = int(_clamp(fr, 0, max(0, ref_total_frames - 1)))
        return fr

    def ref_get_cached_rgb() -> Optional["cv2.Mat"]:
        # cache reference still frame (only when ref_selected changes)
        nonlocal ref_cached_frame_idx, ref_cached_rgb
        if not ref_enabled or ref_cap is None or not ref_tags:
            return None
        fr = ref_tag_frame_index(ref_selected_idx)
        if fr is None:
            return None
        if ref_cached_frame_idx == fr and ref_cached_rgb is not None:
            return ref_cached_rgb
        ok, bgr = read_frame_for(ref_cap, fr)
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ref_cached_frame_idx = fr
        ref_cached_rgb = rgb
        return rgb

    def ref_next(delta: int):
        nonlocal ref_selected_idx, ref_cached_frame_idx, ref_cached_rgb
        if not ref_enabled or not ref_tags:
            return
        ref_selected_idx = int(_clamp(ref_selected_idx + delta, 0, len(ref_tags) - 1))
        ref_cached_frame_idx = None
        ref_cached_rgb = None

    def on_new_self_tag_added(tag_name: str):
        # 左側にタグが追加されたら参照を自動送り
        if not (ref_enabled and auto_adv and ref_tags):
            return
        # 仕様：無条件で次へ（欲しければ「同名を探して次へ」も可能）
        if ref_selected_idx < len(ref_tags) - 1:
            ref_next(+1)

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
                            fr = int(cur_frame)
                            t = fr / fps
                            tags.append(Tag(name=name, t=t, frame=fr))
                            tags.sort(key=lambda x: x.t)
                            selected_idx = max(0, len(tags) - 1)
                            on_new_self_tag_added(name)
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

                elif event.key == pygame.K_a and ref_enabled:
                    auto_adv = not auto_adv

                # ref tag switch
                elif ref_enabled and event.key == pygame.K_LEFTBRACKET:   # [
                    ref_next(-1)
                elif ref_enabled and event.key == pygame.K_RIGHTBRACKET:  # ]
                    ref_next(+1)

                # jump to reference tag time (left moves)
                elif ref_enabled and event.key == pygame.K_j:
                    if ref_tags and 0 <= ref_selected_idx < len(ref_tags):
                        playing = False
                        jump_to_time_sec(ref_tags[ref_selected_idx].t)

                # Enter: jump to selected self tag time
                elif event.key == pygame.K_RETURN:
                    playing = False
                    if tags and 0 <= selected_idx < len(tags):
                        tg = tags[selected_idx]
                        if tg.frame is not None:
                            set_frame(tg.frame)
                        else:
                            jump_to_time_sec(tg.t)

                elif event.key == pygame.K_u:
                    if tags:
                        tags.pop()
                        selected_idx = min(selected_idx, max(0, len(tags) - 1))

                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_x):
                    delete_selected_tag()

                elif event.key == pygame.K_UP:
                    if tags:
                        selected_idx = max(0, selected_idx - 1)

                elif event.key == pygame.K_DOWN:
                    if tags:
                        selected_idx = min(len(tags) - 1, selected_idx + 1)

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

        # update frame (left only)
        if playing:
            ok, _ = cap.read()
            if not ok:
                playing = False
            else:
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # read current left frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        ok, bgr = cap.read()
        if not ok:
            playing = False
            set_frame(max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ok, bgr = cap.read()
            if not ok:
                break

        # ===== draw =====
        screen.fill((0, 0, 0))

        if ref_enabled:
            left_w = screen_w // 2
            right_w = screen_w - left_w

            left_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            left_rgb = cv2.resize(left_rgb, (left_w, screen_h), interpolation=cv2.INTER_AREA)
            surfL = pygame.surfarray.make_surface(left_rgb.swapaxes(0, 1))
            screen.blit(surfL, (0, 0))

            ref_rgb = ref_get_cached_rgb()
            if ref_rgb is not None:
                ref_rgb = cv2.resize(ref_rgb, (right_w, screen_h), interpolation=cv2.INTER_AREA)
                surfR = pygame.surfarray.make_surface(ref_rgb.swapaxes(0, 1))
                screen.blit(surfR, (left_w, 0))
            else:
                draw_rect(left_w, 0, right_w, screen_h, color=(0, 0, 0), alpha=255)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1], rgb.shape[0]) != (screen_w, screen_h):
                rgb = cv2.resize(rgb, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))

        # HUD
        draw_rect(10, 10, screen_w - 20, 160, color=(0, 0, 0), alpha=140)
        tsec = current_time_sec()
        draw_text(f"TARGET: {os.path.basename(video_path)}", 20, 16)
        draw_text(f"t={tsec:8.3f}s ({_sec_to_str(tsec)})  frame={cur_frame}/{total_frames}  fps={fps:.2f}", 20, 44, small=True)

        if ref_enabled:
            ref_info = "REF: (none)"
            if ref_tags:
                rt = ref_tags[ref_selected_idx].t
                rname = ref_tags[ref_selected_idx].name
                rf = ref_tag_frame_index(ref_selected_idx)
                ref_info = f"REF: {os.path.basename(ref_video_path)} | idx {ref_selected_idx+1}/{len(ref_tags)} | tag='{rname}' t={rt:0.3f}s ({_sec_to_str(rt)}) | show frame={rf} (t - {reference_pre_frames}fr)"
            draw_text(ref_info, 20, 70, small=True)
            draw_text(f"[ / ] change ref tag | J jump to ref tag time | A auto-advance={'ON' if auto_adv else 'OFF'} | (ref is STILL image)",
                      20, 96, tiny=True)
        else:
            draw_text("SPACE play/pause | ←/→ step (paused) | Shift+←/→ 0.5s | T add tag | Enter jump",
                      20, 70, small=True)

        draw_text("↑/↓ select self tags | Backspace/Delete/X remove | U undo | R toggle list | Q quit",
                  20, 122, tiny=True)

        # input overlay
        if input_mode:
            draw_rect(10, screen_h - 90, screen_w - 20, 80, color=(0, 0, 0), alpha=170)
            draw_text("Type tag name, Enter=OK, Esc=Cancel", 20, screen_h - 84, small=True)
            draw_text(f"> {input_text}", 20, screen_h - 54)

        # list overlay
        if show_list:
            panel_w = min(760, screen_w - 20)
            panel_h = min(420, screen_h - 190)
            x0 = 10
            y0 = 180
            draw_rect(x0, y0, panel_w, panel_h, color=(0, 0, 0), alpha=140)

            draw_text(f"SELF Tags ({len(tags)})", x0 + 10, y0 + 8, small=True)

            max_rows = (panel_h - 44) // 22
            max_rows = max(1, max_rows)

            y = y0 + 32
            rows = max_rows if not ref_enabled else max_rows // 2
            rows = max(1, rows)

            # self list
            if tags:
                start = max(0, selected_idx - rows + 1)
                end = min(len(tags), start + rows)
            else:
                start = end = 0
            for i in range(start, end):
                tg = tags[i]
                fr = tg.frame if tg.frame is not None else int(round(tg.t * fps))
                line = f"[{i:03d}] fr={fr:7d}  t={tg.t:8.3f}s {_sec_to_str(tg.t)}  {tg.name}"
                if i == selected_idx:
                    draw_rect(x0 + 6, y - 2, panel_w - 12, 22, color=(255, 255, 255), alpha=40)
                draw_text(line, x0 + 10, y, color=(220, 220, 220), tiny=True)
                y += 22

            # ref list (short)
            if ref_enabled:
                y += 8
                draw_rect(x0 + 10, y, panel_w - 20, 1, color=(255, 255, 255), alpha=60)
                y += 10
                draw_text(f"REF Tags ({len(ref_tags)})  [ / ] to change shown still", x0 + 10, y, small=True)
                y += 24

                if ref_tags:
                    start2 = max(0, ref_selected_idx - rows + 1)
                    end2 = min(len(ref_tags), start2 + rows)
                else:
                    start2 = end2 = 0
                for i in range(start2, end2):
                    tg = ref_tags[i]
                    fr = int(round(tg.t * ref_fps)) if ref_fps else 0
                    line = f"[{i:03d}] fr={fr:7d}  t={tg.t:8.3f}s {_sec_to_str(tg.t)}  {tg.name}"
                    if i == ref_selected_idx:
                        draw_rect(x0 + 6, y - 2, panel_w - 12, 22, color=(255, 255, 255), alpha=40)
                    draw_text(line, x0 + 10, y, color=(200, 200, 200), tiny=True)
                    y += 22

        pygame.display.flip()
        clock.tick(60 if playing else 30)

    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    pygame.quit()

    ann = Annotation(video_path=video_path, fps=fps, tags=tags)
    save_annotation(ann, out_json_path)
    print(f"Saved annotation: {out_json_path}")
