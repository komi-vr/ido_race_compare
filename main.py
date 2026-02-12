from typing import Dict
from rc_io import load_annotation
from tagging import annotate_video
from alignment import build_alignment
from render import render_comparison_video


def main():
    # ========= モード切替 =========
    MODE = "compare"  # "tag" or "compare"

    # ========= tag モード用 =========
    vid_name = "haruna_keitokoyo_249s"
    vid_name = "haruna_dirtyotaku_246s"
    TAG_VIDEO_PATH = f"videos/{vid_name}.mp4"
    TAG_OUT_JSON = f"videos/{vid_name}.json"
    TAG_INITIAL_SEEK_SEC = None  # Noneなら最後のタグから再開（resume実装済み想定）

    # ========= compare モード用 =========
    # ★ ここを dict にする： {"動画名": "annotation.json"}
    JSON_PATHS: Dict[str, str] = {
        "DirtyOtaku": "videos/haruna_dirtyotaku_246s.json",
        "Kei Tokoyo": "videos/haruna_keitokoyo_249s.json",
        # "別視点": "run3.json",
    }
    OUT_VIDEO_PATH = "compare_output.mp4"

    # ========= 描画パラメータ =========
    OUT_W = 1280
    BAR_H = 140
    MARGIN = 8
    OUT_FPS = 60

    # 日本語を確実に出したいなら指定（無ければ None）
    FONT_PATH = "GenShinGothic-Monospace-Medium.ttf"

    if MODE == "tag":
        annotate_video(
            TAG_VIDEO_PATH,
            TAG_OUT_JSON,
            initial_seek_sec=TAG_INITIAL_SEEK_SEC,
            resume_if_exists=True,
            font_path=FONT_PATH,
        )
        return

    if MODE == "compare":
        # JSONをラベル付きで読む（順序は dict の定義順）
        labels = list(JSON_PATHS.keys())
        ann_paths = [JSON_PATHS[k] for k in labels]
        anns = [load_annotation(p) for p in ann_paths]

        # アラインメント計画
        plan = build_alignment(anns)

        # render.py は {"動画名": 動画ファイルパス} を受け取るので組み立てる
        video_paths = {label: ann.video_path for label, ann in zip(labels, anns)}

        print("Common tags:", plan.common_tags)
        print("Total output duration:", plan.total_out_dur)
        print("Video labels:", list(video_paths.keys()))

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
