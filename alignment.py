from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from rc_io import Annotation


@dataclass
class Segment:
    name: str
    tag_a: str
    tag_b: str
    out_start: float
    out_dur: float
    per_video_dur: List[float]
    exclude_from_diff: bool = False  # ★ startまでを除外する


@dataclass
class AlignmentPlan:
    segments: List[Segment]
    total_out_dur: float
    common_tags: List[str]

    # seg_idxごと、動画ごとの累積「生時間」（全区間含む）
    cum_raw_at_seg_end: List[List[float]]

    # seg_idxごと、動画ごとの累積「生時間」（exclude_from_diff=False のみ）
    cum_excl_at_seg_end: List[List[float]]


def _sorted_tags(ann: Annotation) -> List[Tuple[str, float]]:
    tags = [(t.name, float(t.t)) for t in ann.tags]
    tags.sort(key=lambda x: x[1])
    return tags


def _first_time_by_tag(ann: Annotation) -> Dict[str, float]:
    # 同名タグが複数あっても「最初の1個」を採用
    d: Dict[str, float] = {}
    for name, t in _sorted_tags(ann):
        if name not in d:
            d[name] = t
    return d


def build_alignment(anns: List[Annotation], start_tag: str = "start") -> AlignmentPlan:
    if len(anns) == 0:
        raise ValueError("anns is empty")

    # 共通タグ（全動画に存在するタグ名）
    per = [_first_time_by_tag(a) for a in anns]
    common = set(per[0].keys())
    for d in per[1:]:
        common &= set(d.keys())

    # 時間順に並べる（基準は先頭動画の時刻）
    common_tags = sorted(list(common), key=lambda k: per[0][k])

    if len(common_tags) < 2:
        raise ValueError("Need at least 2 common tags across all videos")

    segments: List[Segment] = []
    out_cursor = 0.0

    # ★ startタグがあるなら、プレスタート区間(0 -> start)を追加
    if start_tag in common_tags:
        start_times = [per[i][start_tag] for i in range(len(anns))]
        out_dur = max(start_times)
        segments.append(
            Segment(
                name="pre_start",
                tag_a="__BEGIN__",
                tag_b=start_tag,
                out_start=out_cursor,
                out_dur=out_dur,
                per_video_dur=start_times,
                exclude_from_diff=True,  # ★ 差分計算から除外
            )
        )
        out_cursor += out_dur

        # start以降だけを本区間として扱う（start以前はもう追加済）
        start_index = common_tags.index(start_tag)
        common_tags_after = common_tags[start_index:]  # start を含む
    else:
        common_tags_after = common_tags

    # 本区間を構築（tag_i -> tag_{i+1}）
    for i in range(len(common_tags_after) - 1):
        a = common_tags_after[i]
        b = common_tags_after[i + 1]
        durs = []
        for v in range(len(anns)):
            durs.append(per[v][b] - per[v][a])
        out_dur = max(durs)
        segments.append(
            Segment(
                name=f"{a}_to_{b}",
                tag_a=a,
                tag_b=b,
                out_start=out_cursor,
                out_dur=out_dur,
                per_video_dur=durs,
                exclude_from_diff=False,
            )
        )
        out_cursor += out_dur

    # 累積（全区間含む / 除外区間を除いたもの）
    n = len(anns)
    cum_raw_at_seg_end: List[List[float]] = []
    cum_excl_at_seg_end: List[List[float]] = []
    cur_raw = [0.0] * n
    cur_excl = [0.0] * n

    for seg in segments:
        for v in range(n):
            cur_raw[v] += seg.per_video_dur[v]
            if not seg.exclude_from_diff:
                cur_excl[v] += seg.per_video_dur[v]
        cum_raw_at_seg_end.append(cur_raw[:])
        cum_excl_at_seg_end.append(cur_excl[:])

    return AlignmentPlan(
        segments=segments,
        total_out_dur=out_cursor,
        common_tags=common_tags,
        cum_raw_at_seg_end=cum_raw_at_seg_end,
        cum_excl_at_seg_end=cum_excl_at_seg_end,
    )


def find_segment_index(plan: AlignmentPlan, out_t: float) -> int:
    # out_t が属する区間 index を返す
    if out_t <= 0.0:
        return 0
    for i, seg in enumerate(plan.segments):
        if seg.out_start <= out_t < seg.out_start + seg.out_dur:
            return i
    return len(plan.segments) - 1


def map_out_time_to_local(plan: AlignmentPlan, seg_idx: int, video_idx: int, out_t: float) -> float:
    """
    出力時刻 out_t を、各動画のローカル時刻（秒）へ変換。
    区間中は進むが、区間の生時間を超えたらその区間末尾でHOLD。
    """
    seg = plan.segments[seg_idx]
    dt_in_seg = max(0.0, min(seg.out_dur, out_t - seg.out_start))

    # 区間開始までの累積生時間
    prev_raw = plan.cum_raw_at_seg_end[seg_idx - 1][video_idx] if seg_idx > 0 else 0.0
    # 区間中のローカル進行（生時間を超えたら止める）
    local_in_seg = min(dt_in_seg, seg.per_video_dur[video_idx])

    return prev_raw + local_in_seg
