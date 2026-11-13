"""ffmpeg/ffprobe helpers — auto-detects an ffmpeg build that has libass."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _brew_prefix(formula: str) -> str | None:
    try:
        out = subprocess.run(
            ["brew", "--prefix", formula],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            p = out.stdout.strip()
            return p or None
    except Exception:
        pass
    return None


def _has_ass_filter(ffmpeg: str) -> bool:
    """Does this ffmpeg expose the `ass`/`subtitles` filter? (2nd column is the filter name.)"""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("ass", "subtitles"):
                return True
        return False
    except Exception:
        return False


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """Return an ffmpeg binary that has libass (the ass/subtitles filter)."""
    candidates: list[str] = []
    env = os.environ.get("REELWEAVE_FFMPEG")
    if env:
        candidates.append(env)
    pfx = _brew_prefix("ffmpeg-full")
    if pfx:
        candidates.append(str(Path(pfx) / "bin" / "ffmpeg"))
    # Known keg-only location even when brew is not on PATH.
    candidates.append("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)

    # Prefer a build that has libass.
    for c in candidates:
        if c and Path(c).exists() and _has_ass_filter(c):
            return c
    # Fall back to the first working ffmpeg (caller warns about missing libass).
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "ffmpeg not found. Install one with libass, e.g. `brew install ffmpeg-full`."
    )


@lru_cache(maxsize=1)
def find_ffprobe() -> str:
    ff = find_ffmpeg()
    probe = str(Path(ff).with_name("ffprobe"))
    if Path(probe).exists():
        return probe
    which = shutil.which("ffprobe")
    if which:
        return which
    raise FileNotFoundError("ffprobe not found.")


def ffmpeg_has_libass() -> bool:
    return _has_ass_filter(find_ffmpeg())


def run(cmd: list[str], *, check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command. Captures output when quiet=True."""
    kw = dict(text=True)
    if quiet:
        kw.update(capture_output=True)
    cp = subprocess.run(cmd, **kw)
    if check and cp.returncode != 0:
        err = (cp.stderr or "")[-2000:] if quiet else ""
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(cmd[:3])}...\n{err}")
    return cp


def ffprobe_json(path: str | Path) -> dict:
    cmd = [
        find_ffprobe(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    cp = run(cmd)
    return json.loads(cp.stdout)


def video_stream(meta: dict) -> dict | None:
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return None


def display_dimensions(vstream: dict) -> tuple[int, int]:
    """Apply rotation metadata and return the displayed (width, height)."""
    w = int(vstream.get("width", 0))
    h = int(vstream.get("height", 0))
    rot = 0
    # Rotation from side_data_list (display matrix).
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            with contextlib.suppress(Exception):
                rot = int(float(sd["rotation"]))
    # Legacy tags.rotate.
    tags = vstream.get("tags", {}) or {}
    if "rotate" in tags:
        with contextlib.suppress(Exception):
            rot = int(float(tags["rotate"]))
    if abs(rot) % 180 == 90:
        return h, w
    return w, h


def fps_of(vstream: dict) -> float:
    r = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1"
    try:
        num, den = r.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except Exception:
        return 0.0


def extract_frame(src: str | Path, t: float, out: str | Path) -> bool:
    """Extract a single frame at time t seconds (rotation applied automatically)."""
    cmd = [
        find_ffmpeg(),
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out),
    ]
    cp = run(cmd, check=False)
    return cp.returncode == 0 and Path(out).exists()
