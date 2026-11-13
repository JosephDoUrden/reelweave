"""Stage 3 — scenes: detect shot boundaries with PySceneDetect."""

from __future__ import annotations

import json
from pathlib import Path


def detect_file(entry: dict, work_dir: Path, cfg: dict) -> list[dict]:
    fid = entry["id"]
    out = work_dir / "scenes" / f"{fid}.json"
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)

    from scenedetect import AdaptiveDetector, ContentDetector, detect

    fps = entry.get("fps") or 30.0
    min_len_frames = max(1, int(cfg["scenes"].get("min_scene_len_s", 1.0) * fps))
    det = (
        AdaptiveDetector(min_scene_len=min_len_frames)
        if cfg["scenes"].get("detector", "adaptive") == "adaptive"
        else ContentDetector(min_scene_len=min_len_frames)
    )

    print(f"  scenes: {entry['name']} ...", flush=True)
    try:
        scene_list = detect(entry["path"], det)
    except Exception as e:
        print(f"    ! scene detection failed ({e}); treating clip as a single shot.")
        scene_list = []

    shots = []
    if scene_list:
        for start_tc, end_tc in scene_list:
            s, e = start_tc.get_seconds(), end_tc.get_seconds()
            shots.append({"start": round(s, 3), "end": round(e, 3), "dur": round(e - s, 3)})
    else:
        shots.append({"start": 0.0, "end": entry["duration_s"], "dur": entry["duration_s"]})

    out.write_text(json.dumps(shots, ensure_ascii=False, indent=2))
    return shots


def detect_all(manifest: dict, work_dir: Path, cfg: dict) -> dict[str, list[dict]]:
    return {e["id"]: detect_file(e, work_dir, cfg) for e in manifest["files"]}


def load_scenes(fid: str, work_dir: Path) -> list[dict]:
    p = work_dir / "scenes" / f"{fid}.json"
    return json.loads(p.read_text()) if p.exists() else []
