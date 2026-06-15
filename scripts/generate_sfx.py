#!/usr/bin/env python3
"""Synthesise royalty-free transition SFX (whoosh / impact / pop) into sfx/ with ffmpeg.

These are generated locally, so there are no licensing concerns. Drop your own
whoosh.*/impact.*/pop.* into sfx/ to override them.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SFX = Path(__file__).resolve().parent.parent / "sfx"
FF = shutil.which("ffmpeg") or "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"

JOBS = {
    # whoosh: filtered pink-noise sweep
    "whoosh.wav": ('anoisesrc=d=0.45:c=pink:a=0.6:r=48000',
                   'aformat=channel_layouts=stereo,highpass=f=300,lowpass=f=5500,'
                   'flanger=delay=2:depth=4,afade=t=in:d=0.2:curve=ipar,'
                   'afade=t=out:st=0.18:d=0.27,volume=3.5'),
    # impact: low boom with decay
    "impact.wav": ('sine=frequency=56:duration=0.55:r=48000',
                   'aformat=channel_layouts=stereo,afade=t=out:st=0.04:d=0.5:curve=exp,'
                   'bass=g=8:f=80,volume=4'),
    # pop: short tick
    "pop.wav": ('sine=frequency=1000:duration=0.07:r=48000',
                'aformat=channel_layouts=stereo,afade=t=out:d=0.06,volume=2.5'),
}


def main() -> int:
    SFX.mkdir(parents=True, exist_ok=True)
    for name, (src, af) in JOBS.items():
        out = SFX / name
        cmd = [FF, "-y", "-f", "lavfi", "-i", src, "-af", af, str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(("OK  " if r.returncode == 0 else "FAIL") + f" {out}")
        if r.returncode != 0:
            print(r.stderr[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
