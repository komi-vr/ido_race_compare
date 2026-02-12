from typing import Dict
from rc_io import load_annotation
from tagging import annotate_video
from alignment import build_alignment
from render import render_comparison_video


def _first_time_by_tag(ann, name: str) -> float:
    # 同名タグが複数ある場合は「最初の1つ」を採用
    tmin = None
    for tg in ann.tags:
        if tg.name == name:
            if tmin is None or tg.t < tmin:
                tmin = tg.t
    if tmin is None:
        raise ValueError(f"Tag '{name}' not found in {ann.video_path}")
    return float(tmin)


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
    JSON_PATHS: Dict[str, str] = {
        "DirtyOtaku": "videos/haruna_dirtyotaku_246s.json",
        "Kei Tokoyo": "videos/haruna_keitokoyo_249s.json",
    }
    OUT_VIDEO_PATH = "compare_output_v2.mp4"

    # ========= タグ定義 =========
    START_TAG = "start"
    END_TAG = "end"

    # ========= 音声モード =========
    # "none"（音声なし） or "seg_fastest"（区間ごと最速の動画の音声）
    AUDIO_MODE = "seg_fastest"

    # ========= 描画パラメータ =========
    OUT_W = 1280
    OUT_H = 720
    BAR_H = 190
    MARGIN = 8
    OUT_FPS = 60

    # 日本語フォント（プロジェクト内パス推奨）
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
        labels = list(JSON_PATHS.keys())
        ann_paths = [JSON_PATHS[k] for k in labels]
        anns = [load_annotation(p) for p in ann_paths]

        # ★ start までの差を計算に入れない（映像には出す）前提のアラインメント
        plan = build_alignment(anns, start_tag=START_TAG)

        video_paths = {label: ann.video_path for label, ann in zip(labels, anns)}

        # ★ 各動画の start/end 時刻（全体タイム表示＆進捗%のため）
        start_times = {label: _first_time_by_tag(ann, START_TAG) for label, ann in zip(labels, anns)}
        end_times = {label: _first_time_by_tag(ann, END_TAG) for label, ann in zip(labels, anns)}

        print("Common tags:", plan.common_tags)
        print("Total output duration:", plan.total_out_dur)
        print("Video labels:", list(video_paths.keys()))
        print("Audio mode:", AUDIO_MODE)

        render_comparison_video(
            video_paths=video_paths,
            plan=plan,
            out_path=OUT_VIDEO_PATH,
            out_w=OUT_W,
            out_h=OUT_H,
            bar_h=BAR_H,
            margin=MARGIN,
            font_path=FONT_PATH,
            fps=OUT_FPS,
            start_times=start_times,  # ★ 追加
            end_times=end_times,      # ★ 追加
            audio_mode=AUDIO_MODE,    # ★ 追加
        )
        return

    raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
