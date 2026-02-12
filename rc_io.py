import json
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Tag:
    name: str
    t: float  # seconds


@dataclass
class Annotation:
    video_path: str
    fps: float
    tags: List[Tag]


def load_annotation(path: str) -> Annotation:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = [Tag(name=x["name"], t=float(x["t"])) for x in data["tags"]]
    tags.sort(key=lambda x: x.t)
    return Annotation(
        video_path=data["video_path"],
        fps=float(data.get("fps", 0.0)),
        tags=tags,
    )


def save_annotation(ann: Annotation, path: str) -> None:
    data: Dict[str, Any] = {
        "video_path": ann.video_path,
        "fps": ann.fps,
        "tags": [{"name": t.name, "t": t.t} for t in ann.tags],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
