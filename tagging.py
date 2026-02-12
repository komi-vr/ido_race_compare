import cv2
import os
from typing import List
from rc_io import Annotation, Tag, load_annotation, save_annotation


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def annotate_video(
    video_path: str,
    out_json_path: str,
    *,
    initial_seek_sec: float | None = None,
    window_name: str = "tagger",
    resume_if_exists: bool = True,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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
                    f"  -> Will still load tags, but be careful."
                )
            tags = prev.tags[:]  # copy
            print(f"[RESUME] Loaded {len(tags)} tags from {out_json_path}")
            # If user didn't give initial_seek_sec, jump to last tag
            if initial_seek_sec is None and len(tags) > 0:
                initial_seek_sec = tags[-1].t
        except Exception as e:
            print(f"[WARN] Failed to load existing JSON ({out_json_path}): {e}")
            print("[WARN] Start with empty tags.")

    if initial_seek_sec is None:
        initial_seek_sec = 0.0

    playing = False
    cur_frame = int(initial_seek_sec * fps)
    cur_frame = int(_clamp(cur_frame, 0, max(0, total_frames - 1)))

    def set_frame(idx: int) -> None:
        nonlocal cur_frame
        cur_frame = int(_clamp(idx, 0, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    set_frame(cur_frame)

    print("Controls:")
    print("  SPACE play/pause | A/D step -/+1 frame (paused)")
    print("  J/L -/+0.5 sec | T add tag | U undo last tag | R list tags | Q quit")
    while True:
        if playing:
            ok, frame = cap.read()
            if not ok:
                break
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            ok, frame = cap.read()
            if not ok:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        t = cur_frame / fps
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"t={t:8.3f}s frame={cur_frame}/{total_frames} fps={fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # show last 5 tags
        y = 80
        for tg in tags[-5:]:
            cv2.putText(
                overlay,
                f"[{tg.t:8.3f}] {tg.name}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(10 if playing else 30) & 0xFF

        if key == ord(" "):  # space
            playing = not playing
            if playing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        elif key in (ord("q"), ord("Q")):
            break

        elif key in (ord("a"), ord("A")) and not playing:
            set_frame(cur_frame - 1)
        elif key in (ord("d"), ord("D")) and not playing:
            set_frame(cur_frame + 1)

        elif key in (ord("j"), ord("J")):
            set_frame(cur_frame - int(0.5 * fps))
        elif key in (ord("l"), ord("L")):
            set_frame(cur_frame + int(0.5 * fps))

        elif key in (ord("t"), ord("T")):
            print(f"\nAdd tag at t={t:.3f}s. Enter tag name (empty cancels): ", end="")
            name = input().strip()
            if name:
                tags.append(Tag(name=name, t=t))
                tags.sort(key=lambda x: x.t)
                print(f"Added: {name} @ {t:.3f}s\n")
            else:
                print("Canceled.\n")

        elif key in (ord("u"), ord("U")):
            if tags:
                removed = tags.pop()
                print(f"\n[UNDO] Removed last tag: {removed.name} @ {removed.t:.3f}s\n")
            else:
                print("\n[UNDO] No tags to remove.\n")

        elif key in (ord("r"), ord("R")):
            print("\n=== TAG LIST ===")
            for tg in tags:
                print(f"{tg.t:8.3f}  {tg.name}")
            print("===============\n")

    cap.release()
    cv2.destroyAllWindows()

    ann = Annotation(video_path=video_path, fps=fps, tags=tags)
    save_annotation(ann, out_json_path)
    print(f"Saved annotation: {out_json_path}")
