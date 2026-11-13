"""Stage 1 — probe: scan source videos and build a rotation-corrected manifest."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from . import util_ffmpeg as ff

# DJI_20260613122106_0004_D.MP4 -> capture timestamp + sequence number
_DJI_RE = re.compile(r"(\d{14})_(\d+)_", re.IGNORECASE)


def _capture_time(name: str, meta: dict) -> str | None:
    """Capture time from the filename timestamp, falling back to ffprobe creation_time."""
    m = _DJI_RE.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            pass
    ct = (meta.get("format", {}).get("tags", {}) or {}).get("creation_time")
    if ct:
        try:
            return datetime.fromisoformat(ct.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return ct
    return None


def build_manifest(
    source_dir: Path, glob: str, work_dir: Path, exclude: list[str] | None = None
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    excl = set(exclude or [])
    files = sorted(source_dir.glob(glob))
    entries = []
    for f in files:
        if f.stem in excl or f.name in excl:
            print(f"  (excluded: {f.name})")
            continue
        try:
            meta = ff.ffprobe_json(f)
        except Exception as e:
            print(f"  ! probe failed: {f.name}: {e}")
            continue
        vs = ff.video_stream(meta)
        if not vs:
            continue
        w, h = ff.display_dimensions(vs)
        dur = float(meta.get("format", {}).get("duration", 0.0))
        entries.append(
            {
                "id": f.stem,
                "path": str(f.resolve()),
                "name": f.name,
                "capture_time": _capture_time(f.name, meta),
                "duration_s": round(dur, 3),
                "disp_w": w,
                "disp_h": h,
                "fps": round(ff.fps_of(vs), 3),
                "is_vertical": h >= w,
            }
        )
    entries.sort(key=lambda e: (e["capture_time"] or "", e["name"]))
    manifest = {
        "source_dir": str(source_dir.resolve()),
        "count": len(entries),
        "total_duration_s": round(sum(e["duration_s"] for e in entries), 1),
        "files": entries,
    }
    out = work_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"  manifest: {len(entries)} files, "
        f"{manifest['total_duration_s'] / 60:.1f} min total -> {out}"
    )
    return manifest


def load_manifest(work_dir: Path) -> dict:
    p = work_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("manifest.json missing — run the 'probe' stage first.")
    return json.loads(p.read_text())
