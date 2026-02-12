from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import os
import click
import yaml

from rc_io import load_annotation
from tagging import annotate_video
from alignment import build_alignment
from render import render_comparison_video, preview_comparison_video


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise click.ClickException("config.yaml のトップレベルは dict(map) にしてください。")
    return data


def _get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return default if v is None else v


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d or d[key] is None:
        raise click.ClickException(f"config の '{where}.{key}' が必要です。")
    return d[key]


def _as_tuple2(v, where: str) -> Optional[Tuple[int, int]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise click.ClickException(f"config の '{where}' は [w, h] の形式にしてください。")


def _default_json_path_for_video(video_path: str) -> str:
    # videos/foo.mp4 -> videos/foo.json
    base, _ = os.path.splitext(video_path)
    return base + ".json"


def _first_time_by_tag(ann, name: str) -> float:
    tmin = None
    for tg in ann.tags:
        if tg.name == name:
            if tmin is None or tg.t < tmin:
                tmin = tg.t
    if tmin is None:
        raise click.ClickException(f"タグ '{name}' が見つかりません: {ann.video_path}")
    return float(tmin)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def tag(config_path: str):
    """動画にタグを打ってJSON保存（設定はYAMLから読む）"""
    cfg = _load_yaml(config_path)
    tag_cfg = _get(cfg, "tag", {})

    video_path = _require(tag_cfg, "video_path", "tag")

    # ★ out_json_path は省略可：同じ場所・同じ名前 .json にする
    out_json_path = _get(tag_cfg, "out_json_path", None)
    if out_json_path is None:
        out_json_path = _default_json_path_for_video(video_path)

    initial_seek_sec = _get(tag_cfg, "initial_seek_sec", None)
    resume_if_exists = bool(_get(tag_cfg, "resume_if_exists", True))
    window_size = _as_tuple2(_get(tag_cfg, "window_size", [1280, 720]), "tag.window_size")

    common = _get(cfg, "common", {})
    font_path = _get(common, "font_path", None)

    reference_json_path = tag_cfg.get("reference_json_path", None)

    annotate_video(
        video_path,
        out_json_path,
        initial_seek_sec=initial_seek_sec,
        resume_if_exists=resume_if_exists,
        window_size=window_size,
        font_path=font_path,
        reference_json_path=reference_json_path,  # ★追加
    )



@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def compare(config_path: str):
    """複数動画を比較して動画出力（設定はYAMLから読む）"""
    cfg = _load_yaml(config_path)

    cmp_cfg = _get(cfg, "compare", {})
    common = _get(cfg, "common", {})
    render_cfg = _get(cfg, "render", {})

    json_paths: Dict[str, str] = _require(cmp_cfg, "json_paths", "compare")
    if not isinstance(json_paths, dict) or not json_paths:
        raise click.ClickException("compare.json_paths は {label: path} の dict で指定してください。")

    out_video_path = _get(cmp_cfg, "out_video_path", "compare_output.mp4")
    start_tag = _get(cmp_cfg, "start_tag", "start")
    end_tag = _get(cmp_cfg, "end_tag", "end")
    audio_mode = str(_get(cmp_cfg, "audio_mode", "none")).lower()

    out_w = int(_get(render_cfg, "out_w", 1280))
    out_h = int(_get(render_cfg, "out_h", 720))
    bar_h = int(_get(render_cfg, "bar_h", 210))
    margin = int(_get(render_cfg, "margin", 10))
    out_fps = int(_get(render_cfg, "fps", 60))
    font_path = _get(common, "font_path", None)

    labels = list(json_paths.keys())
    anns = [load_annotation(json_paths[k]) for k in labels]

    plan = build_alignment(anns, start_tag=start_tag)

    video_paths = {label: ann.video_path for label, ann in zip(labels, anns)}
    start_times = {label: _first_time_by_tag(ann, start_tag) for label, ann in zip(labels, anns)}
    end_times = {label: _first_time_by_tag(ann, end_tag) for label, ann in zip(labels, anns)}

    render_comparison_video(
        video_paths=video_paths,
        plan=plan,
        out_path=out_video_path,
        out_w=out_w,
        out_h=out_h,
        bar_h=bar_h,
        margin=margin,
        font_path=font_path,
        fps=out_fps,
        start_times=start_times,
        end_times=end_times,
        audio_mode=audio_mode,
    )


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def preview(config_path: str):
    """出力せずpygameでプレビュー（設定はYAMLから読む）"""
    cfg = _load_yaml(config_path)

    cmp_cfg = _get(cfg, "compare", {})
    common = _get(cfg, "common", {})
    render_cfg = _get(cfg, "render", {})

    json_paths: Dict[str, str] = _require(cmp_cfg, "json_paths", "compare")
    if not isinstance(json_paths, dict) or not json_paths:
        raise click.ClickException("compare.json_paths は {label: path} の dict で指定してください。")

    start_tag = _get(cmp_cfg, "start_tag", "start")
    end_tag = _get(cmp_cfg, "end_tag", "end")

    out_w = int(_get(render_cfg, "out_w", 1280))
    out_h = int(_get(render_cfg, "out_h", 720))
    bar_h = int(_get(render_cfg, "bar_h", 210))
    margin = int(_get(render_cfg, "margin", 10))
    out_fps = int(_get(render_cfg, "fps", 60))
    font_path = _get(common, "font_path", None)

    labels = list(json_paths.keys())
    anns = [load_annotation(json_paths[k]) for k in labels]
    plan = build_alignment(anns, start_tag=start_tag)

    video_paths = {label: ann.video_path for label, ann in zip(labels, anns)}
    start_times = {label: _first_time_by_tag(ann, start_tag) for label, ann in zip(labels, anns)}
    end_times = {label: _first_time_by_tag(ann, end_tag) for label, ann in zip(labels, anns)}

    preview_comparison_video(
        video_paths=video_paths,
        plan=plan,
        out_w=out_w,
        out_h=out_h,
        bar_h=bar_h,
        margin=margin,
        font_path=font_path,
        fps=out_fps,
        start_times=start_times,
        end_times=end_times,
    )


if __name__ == "__main__":
    cli()
