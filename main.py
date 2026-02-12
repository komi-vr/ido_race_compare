from typing import List
from rc_io import load_annotation
from tagging import annotate_video
from alignment import build_alignment
from render import render_comparison_video


def main():
    # ========= モード切替 =========
    MODE = "compare"  # "tag" or "compare"

    # ========= tag モード用 =========
    TAG_VIDEO_PATH = "run1.mp4"
    TAG_OUT_JSON = "run1.json"
    TAG_INITIAL_SEEK_SEC = 0.0

    # ========= compare モード用 =========
    JSON_PATHS: List[str] = [
        "run1.json",
        "run2.json",
        # "run3.json",
    ]
    OUT_VIDEO_PATH = "compare_output.mp4"

    # ========= 描画パラメータ =========
    OUT_W = 1280
    BAR_H = 140
    MARGIN = 8
    OUT_FPS = 60

    # 日本語タグ表示したいなら NotoSansCJK 等を指定（無ければ None でOK）
    FONT_PATH = None  # 例: "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

    if MODE == "tag":
        annotate_video(
            TAG_VIDEO_PATH,
            TAG_OUT_JSON,
            initial_seek_sec=TAG_INITIAL_SEEK_SEC,
        )
        return

    if MODE == "compare":
        anns = [load_annotation(p) for p in JSON_PATHS]
        video_paths = [a.video_path for a in anns]

        plan = build_alignment(anns)
        print("Common tags:", plan.common_tags)
        print("Total output duration:", plan.total_out_dur)

        render_comparison_video(
            video_paths=video_paths,
            plan=plan,
            out_path=OUT_VIDEO_PATH,
            out_w=OUT_W,
            bar_h=BAR_H,
            margin=MARGIN,
            font_path=FONT_PATH,
            fps=OUT_FPS,
        )
        return

    raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
