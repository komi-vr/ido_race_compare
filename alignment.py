from dataclasses import dataclass
from typing import List, Dict, Tuple
from rc_io import Annotation


@dataclass
class Segment:
    name: str                 # next tag name (e.g., "S1")
    tag_a: str                # current common tag
    tag_b: str                # next common tag
    out_start: float
    out_dur: float
    per_video_start: List[float]   # local start times per video
    per_video_dur: List[float]     # local durations per video (raw)


@dataclass
class AlignmentPlan:
    common_tags: List[str]
    segments: List[Segment]
    total_out_dur: float
    # cumulative raw time per video at segment boundaries (for deltas)
    cum_raw_at_seg_end: List[List[float]]  # [seg_idx][vid_idx]


def build_alignment(anns: List[Annotation]) -> AlignmentPlan:
    if len(anns) < 2:
        raise ValueError("Need at least 2 annotations for comparison.")

    # tag times per video: name -> time (first occurrence)
    per_vid_map: List[Dict[str, float]] = []
    per_vid_names: List[set] = []
    for ann in anns:
        m: Dict[str, float] = {}
        for tg in ann.tags:
            if tg.name not in m:
                m[tg.name] = tg.t
        per_vid_map.append(m)
        per_vid_names.append(set(m.keys()))

    common = set.intersection(*per_vid_names)
    if len(common) < 2:
        raise ValueError("Need at least 2 common tag names across all videos.")

    # order common tags by the first (reference) video's time
    ref = per_vid_map[0]
    common_list = sorted(list(common), key=lambda name: ref[name])

    # build segments between successive common tags
    segments: List[Segment] = []
    out_t = 0.0

    # cumulative raw time (sum of each segment raw duration) per video
    cum_raw = [0.0 for _ in anns]
    cum_raw_at_seg_end: List[List[float]] = []

    for i in range(len(common_list) - 1):
        a = common_list[i]
        b = common_list[i + 1]

        per_start = []
        per_dur = []
        for vid_idx, m in enumerate(per_vid_map):
            ta = m[a]
            tb = m[b]
            dur = max(0.0, tb - ta)
            per_start.append(ta)
            per_dur.append(dur)

        out_dur = max(per_dur) if per_dur else 0.0
        seg = Segment(
            name=b,
            tag_a=a,
            tag_b=b,
            out_start=out_t,
            out_dur=out_dur,
            per_video_start=per_start,
            per_video_dur=per_dur,
        )
        segments.append(seg)
        out_t += out_dur

        for v in range(len(anns)):
            cum_raw[v] += per_dur[v]
        cum_raw_at_seg_end.append(cum_raw.copy())

    return AlignmentPlan(
        common_tags=common_list,
        segments=segments,
        total_out_dur=out_t,
        cum_raw_at_seg_end=cum_raw_at_seg_end,
    )


def map_out_time_to_local(plan: AlignmentPlan, seg_idx: int, vid_idx: int, out_t: float) -> float:
    seg = plan.segments[seg_idx]
    # time within segment on output timeline
    dt = out_t - seg.out_start
    dt = max(0.0, min(seg.out_dur, dt))
    # local time progresses until raw segment duration, then holds
    raw_dur = seg.per_video_dur[vid_idx]
    local_dt = min(dt, raw_dur)
    return seg.per_video_start[vid_idx] + local_dt


def find_segment_index(plan: AlignmentPlan, out_t: float) -> int:
    # linear scan is OK for small N; can binary-search later
    for i, seg in enumerate(plan.segments):
        if seg.out_start <= out_t < seg.out_start + seg.out_dur:
            return i
    return max(0, len(plan.segments) - 1)
