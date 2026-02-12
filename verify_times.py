# verify_times.py
from rc_io import load_annotation

START="start"
END="end"

def first_time(tags, name):
    ts = [t.t for t in tags if t.name == name]
    if not ts:
        raise ValueError(f"tag '{name}' not found")
    return float(min(ts))

def main():
    paths = [
        "videos/haruna_keitokoyo_249s.json",
        "videos/haruna_dirtyotaku_246s.json",
    ]
    for p in paths:
        ann = load_annotation(p)
        st = first_time(ann.tags, START)
        ed = first_time(ann.tags, END)
        print("====", p)
        print("video_path:", ann.video_path)
        print("json_fps:", ann.fps)
        print("start:", st)
        print("end  :", ed)
        print("end-start:", ed - st)

if __name__ == "__main__":
    main()
