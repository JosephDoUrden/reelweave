"""Stage 4 — score: sample shot frames and rate visual quality via blur/brightness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import util_ffmpeg as ff


def _sample_times(shot: dict, n: int) -> list[float]:
    s, e = shot["start"], shot["end"]
    dur = max(0.0, e - s)
    if dur < 0.4 or n <= 1:
        return [s + dur / 2.0]
    return [s + dur * frac for frac in [(i + 1) / (n + 1) for i in range(n)]]


def _frame_metrics(jpg: Path):
    import cv2
    img = cv2.imread(str(jpg))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > 480:
        scale = 480.0 / w
        img = cv2.resize(img, (480, int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())   # higher = sharper
    brightness = float(gray.mean())                        # 0-255
    return blur, brightness


def score_file(entry: dict, shots: list[dict], work_dir: Path, cfg: dict) -> list[dict]:
    fid = entry["id"]
    out = work_dir / "scores" / f"{fid}.json"
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)

    n = cfg["score"].get("frames_per_shot", 3)
    blur_floor = cfg["score"].get("blur_floor", 80.0)
    scored = []
    print(f"  score: {entry['name']} ({len(shots)} shots) ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, shot in enumerate(shots):
            blurs, brights = [], []
            for j, t in enumerate(_sample_times(shot, n)):
                jpg = tdp / f"{i}_{j}.jpg"
                if ff.extract_frame(entry["path"], t, jpg):
                    m = _frame_metrics(jpg)
                    if m:
                        blurs.append(m[0])
                        brights.append(m[1])
            if blurs:
                avg_blur = sum(blurs) / len(blurs)
                avg_bright = sum(brights) / len(brights)
                sharp = min(1.0, avg_blur / (blur_floor * 3))      # 0-1
                # Good exposure: close to 1 within the ~60-200 band.
                expo = 1.0 - min(1.0, abs(avg_bright - 125) / 125)
                q = round(10.0 * (0.7 * sharp + 0.3 * expo), 2)
            else:
                avg_blur = avg_bright = 0.0
                q = 0.0
            scored.append({**shot,
                           "blur": round(avg_blur, 1),
                           "brightness": round(avg_bright, 1),
                           "sharp_ok": avg_blur >= blur_floor,
                           "quality": q})
    out.write_text(json.dumps(scored, ensure_ascii=False, indent=2))
    return scored


def score_all(manifest, scenes_by_id, work_dir, cfg) -> dict[str, list[dict]]:
    return {e["id"]: score_file(e, scenes_by_id[e["id"]], work_dir, cfg)
            for e in manifest["files"]}


def load_scores(fid: str, work_dir: Path) -> list[dict]:
    p = work_dir / "scores" / f"{fid}.json"
    return json.loads(p.read_text()) if p.exists() else []
