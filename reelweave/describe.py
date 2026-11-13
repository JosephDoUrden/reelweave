"""Stage 5 — describe: deep visual understanding.

Long shots are split into sub-windows; each window is analysed by a vision model
to produce STRUCTURED JSON (place, landmark, activity, objects, mood, OCR, highlight).
The default engine feeds each window as native video to Qwen3-VL via Apple MLX;
an Ollama frame-grid fallback is also supported. This lets the pipeline understand
the CONTENT and MOTION of a shot even when nobody is speaking.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from pathlib import Path

import requests

from . import util_ffmpeg as ff


def _make_grid(src: str, start: float, end: float, k: int, out_jpg: Path, cols: int = 3) -> bool:
    """Sample k frames from the window into a time-labelled mosaic (cv2)."""
    import cv2

    dur = max(0.3, end - start)
    n = max(1, min(k, 16))
    times = [start + dur * (i + 0.5) / n for i in range(n)]
    cells = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(times):
            p = Path(td) / f"{i}.jpg"
            if ff.extract_frame(src, t, p):
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.resize(img, (300, 533))
                    rel = t - start
                    cv2.putText(
                        img,
                        f"{i + 1} ({rel:.0f}s)",
                        (8, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 0),
                        5,
                    )
                    cv2.putText(
                        img,
                        f"{i + 1} ({rel:.0f}s)",
                        (8, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 255),
                        2,
                    )
                    cells.append(img)
        if not cells:
            return False
        cols = cols if len(cells) > 1 else 1
        rows = (len(cells) + cols - 1) // cols
        blank = cells[0] * 0
        grid_rows = []
        for r in range(rows):
            row = cells[r * cols : (r + 1) * cols]
            while len(row) < cols:
                row.append(blank)
            grid_rows.append(cv2.hconcat(row))
        grid = cv2.vconcat(grid_rows) if len(grid_rows) > 1 else grid_rows[0]
        cv2.imwrite(str(out_jpg), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_jpg.exists()


def _vision_call(jpg: Path, prompt: str, cfg: dict) -> dict:
    """Ollama fallback: send a frame-grid image to a vision model → structured JSON."""
    oc = cfg["ollama"]
    vc = cfg["vision"]
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    payload = {
        "model": vc["model"],
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": vc.get("temperature", 0.1), "num_predict": 400},
    }
    r = requests.post(f"{oc['host']}/api/generate", json=payload, timeout=oc.get("timeout_s", 900))
    r.raise_for_status()
    txt = r.json().get("response", "")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


_MLX: dict = {}


def _load_mlx(cfg: dict):
    """Load the MLX vision model once and cache it."""
    if "t" in _MLX:
        return _MLX["t"]
    from mlx_vlm import load

    mp = cfg["vision"]["mlx_model"]
    print(f"  loading MLX vision model: {mp} ...", flush=True)
    model, processor = load(mp)
    try:
        from mlx_vlm.utils import load_config

        config = load_config(mp)
    except Exception:
        config = getattr(model, "config", None)
    _MLX["t"] = (model, processor, config)
    return _MLX["t"]


def _mlx_call(src: str, ws: float, we: float, prompt: str, cfg: dict, tmp: Path) -> dict:
    """Feed the window as native VIDEO to MLX Qwen3-VL → structured JSON."""
    from mlx_vlm import apply_chat_template, generate

    model, processor, config = _load_mlx(cfg)
    win = tmp / "win.mp4"
    ff.run(
        [
            ff.find_ffmpeg(),
            "-y",
            "-ss",
            f"{ws:.2f}",
            "-to",
            f"{we:.2f}",
            "-i",
            src,
            "-vf",
            "scale=360:640",
            "-an",
            str(win),
        ]
    )
    fps = cfg["vision"].get("fps", 2.0)
    pr = apply_chat_template(processor, config, prompt, num_images=0, video=str(win), fps=fps)
    r = generate(
        model,
        processor,
        pr,
        video=str(win),
        fps=fps,
        max_tokens=400,
        temperature=cfg["vision"].get("temperature", 0.1),
        verbose=False,
    )
    txt = getattr(r, "text", str(r))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def _windows(scores: list[dict], entry: dict, cfg: dict) -> list[tuple[float, float]]:
    """Take usable shots and split long ones into ~window_max_s windows."""
    vc = cfg["vision"]
    usable = [s for s in scores if s.get("quality", 0) >= vc.get("min_quality", 2.0)] or list(
        scores
    )
    if not usable:  # no shots → window the whole clip
        usable = [{"start": 0.0, "end": entry["duration_s"]}]
    wmax = vc.get("window_max_s", 13)
    wmin = vc.get("window_min_s", 5)
    wins = []
    for s in sorted(usable, key=lambda x: x["start"]):
        a, b = s["start"], s["end"]
        dur = b - a
        if dur <= wmax:
            wins.append((a, b))
            continue
        nwin = max(1, round(dur / wmax))
        step = dur / nwin
        for i in range(nwin):
            ws = a + i * step
            we = b if i == nwin - 1 else a + (i + 1) * step
            wins.append((round(ws, 2), round(we, 2)))
    # Merge a tiny trailing window into the previous one.
    merged = []
    for w in wins:
        if merged and (w[1] - w[0]) < wmin:
            merged[-1] = (merged[-1][0], w[1])
        else:
            merged.append(list(w))
    merged = [tuple(m) for m in merged]
    cap = vc.get("max_windows_per_file", 40)
    if len(merged) > cap:
        step = len(merged) / cap
        merged = [merged[int(i * step)] for i in range(cap)]
    return merged


def describe_file(
    entry: dict, scores: list[dict], work_dir: Path, cfg: dict, prompt: str
) -> list[dict]:
    fid = entry["id"]
    out = work_dir / "vision" / f"{fid}.json"
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)

    wins = _windows(scores, entry, cfg)
    engine = cfg["vision"].get("engine", "mlx")
    print(f"  vision[{engine}]: {entry['name']} ({len(wins)} windows) ...", flush=True)
    descs = []
    grid_k = cfg["vision"].get("grid", 12)
    grid_cols = cfg["vision"].get("grid_cols", 4)
    with tempfile.TemporaryDirectory() as td:
        for i, (ws, we) in enumerate(wins):
            data = {}
            try:
                if engine == "mlx":
                    data = _mlx_call(entry["path"], ws, we, prompt, cfg, Path(td))
                else:
                    grid = Path(td) / f"g{i}.jpg"
                    if _make_grid(entry["path"], ws, we, grid_k, grid, grid_cols):
                        data = _vision_call(grid, prompt, cfg)
            except Exception as e:
                print(f"    ! vision error (window {i}): {e}")
            descs.append(
                {
                    "start": ws,
                    "end": we,
                    "quality_cv": 0,
                    "place_type": data.get("place_type", ""),
                    "landmark": data.get("landmark"),
                    "landmark_conf": data.get("landmark_confidence", 0),
                    "action": data.get("action", ""),
                    "subjects": data.get("subjects", []),
                    "shot_scale": data.get("shot_scale", ""),
                    "camera_move": data.get("camera_move", ""),
                    "mood": data.get("mood", ""),
                    "ocr": data.get("ocr_text"),
                    "time_of_day": data.get("time_of_day", ""),
                    "activity": data.get("activity", []) or [],
                    "objects": data.get("objects", []) or [],
                    "food_drink": data.get("food_drink", []) or [],
                    "clip_purpose": data.get("clip_purpose", ""),
                    "highlight": float(data.get("highlight_worthiness", 0) or 0),
                    "desc": data.get("caption") or data.get("tr_caption", ""),
                }
            )
    out.write_text(json.dumps(descs, ensure_ascii=False, indent=2))
    return descs


def describe_all(manifest, scores_by_id, work_dir, cfg) -> dict[str, list[dict]]:
    prompt = (Path(cfg["_prompts_dir"]) / "vision_shot.txt").read_text()
    return {
        e["id"]: describe_file(e, scores_by_id.get(e["id"], []), work_dir, cfg, prompt)
        for e in manifest["files"]
    }


def load_vision(fid: str, work_dir: Path) -> list[dict]:
    p = work_dir / "vision" / f"{fid}.json"
    return json.loads(p.read_text()) if p.exists() else []
