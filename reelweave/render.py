"""Stage 7 — render: montage with transitions, cinematic grade, grain, animated captions,
sound design.

Per clip:
  1) Each segment -> upright 1080x1920/30fps intermediate with smooth Ken Burns.
  2) One final filter_complex: xfade(video)+acrossfade(audio) transition chain ->
     colour grade/LUT + film grain + .ass subtitles (hook/location/chapter cards + karaoke)
     -> audio: denoise + side-chain duck (music) + timed SFX (whoosh/impact) + loudnorm.
  3) Hardware encode via VideoToolbox.

PART clips (long-form) use seamless concat (no transitions, accurate card timing);
SUPERCUT clips use transitions.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from . import caption_ass
from . import select as sel_mod
from . import transcribe as tr_mod
from . import util_ffmpeg as ff

PIPE_ROOT = Path(__file__).resolve().parent.parent
_TRANS = ["smoothleft", "smoothright", "slideup", "zoomin", "smoothup", "wiperight", "fade"]
# Subtle teal-orange grade (no external LUT required).
_GRADE = (
    "eq=contrast=1.07:saturation=1.12:gamma=0.98,"
    "curves=r='0/0 0.25/0.21 0.75/0.82 1/1':"
    "g='0/0.02 0.5/0.5 1/0.98':b='0/0.06 0.5/0.47 1/0.94'"
)

# Map a clip "format" to a music mood folder (music/<mood>/).
_FMT2MOOD = {
    "cinematic": "cinematic",
    "history": "cinematic",
    "hidden_gem": "cinematic",
    "pov": "upbeat",
    "itinerary": "upbeat",
    "budget": "upbeat",
    "food": "lofi",
    "coffee": "lofi",
}


def _esc(p: str) -> str:
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


@lru_cache(maxsize=4)
def _find(dir_path: str, *names) -> str | None:
    d = Path(dir_path)
    if not d.is_dir():
        return None
    for stem in names:
        for ext in ("wav", "mp3", "m4a", "aac", "flac"):
            p = d / f"{stem}.{ext}"
            if p.exists():
                return str(p)
    return None


@lru_cache(maxsize=16)
def _music(base: str, mood: str) -> str | None:
    """Pick from music/<mood>/, else anything under music/."""
    bd = Path(base)
    search = [bd / mood, bd] if mood else [bd]
    for d in search:
        if not d.is_dir():
            continue
        for ext in ("*.mp3", "*.m4a", "*.aac", "*.wav", "*.flac"):
            f = sorted(d.rglob(ext))
            if f:
                return str(f[0])
    return None


def _normalize_segment(seg: dict, out: Path, cfg: dict) -> float:
    v = cfg["video"]
    eff = cfg.get("effects", {})
    W, H, fps = v["w"], v["h"], v["fps"]
    dur = seg["end"] - seg["start"]
    if eff.get("ken_burns", True):
        amt = eff.get("kb_zoom", 1.08) - 1.0
        Wb, Hb = round(W * (1 + amt)), round(H * (1 + amt))
        D = max(0.1, dur)
        zf = f"(1+{amt:.4f}*min(t/{D:.3f},1))"
        # Smooth push-in via scale-then-crop (avoids zoompan jitter).
        vf = [
            f"scale={Wb}:{Hb}:force_original_aspect_ratio=increase",
            f"crop={Wb}:{Hb}",
            f"crop=w='{Wb}/{zf}':h='{Hb}/{zf}':x='(in_w-out_w)/2':y='(in_h-out_h)/2'",
            f"scale={W}:{H}",
            "setsar=1",
            f"fps={fps}",
        ]
    else:
        vf = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase",
            f"crop={W}:{H}",
            "setsar=1",
            f"fps={fps}",
        ]
    cmd = [
        ff.find_ffmpeg(),
        "-y",
        "-ss",
        f"{seg['start']:.3f}",
        "-i",
        seg["source_path"],
        "-t",
        f"{dur:.3f}",
        "-vf",
        ",".join(vf),
        "-c:v",
        v["vcodec"],
        "-b:v",
        "12M",
        "-pix_fmt",
        v["pix_fmt"],
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-ar",
        str(cfg["audio"]["rate"]),
        "-ac",
        "2",
        str(out),
    ]
    ff.run(cmd)
    meta = ff.ffprobe_json(out)
    return float(meta.get("format", {}).get("duration", dur))


def _compute_offsets(durs: list[float], td: float, transitions: bool):
    """Segment start offsets + total duration (accounts for transition overlap)."""
    if not transitions or len(durs) == 1:
        offs, run = [], 0.0
        for d in durs:
            offs.append(run)
            run += d
        return offs, run
    offs = [0.0]
    run = durs[0]
    for i in range(1, len(durs)):
        o = run - td
        offs.append(o)
        run = o + durs[i]
    return offs, run


def _video_join(n: int, durs, offs, td: float, transitions: bool) -> tuple[list[str], str]:
    """Normalise segment video inputs and build the xfade chain. (parts, last_label)"""
    parts = [f"[{i}:v]fps=30,setsar=1,format=yuv420p,settb=AVTB[v{i}]" for i in range(n)]
    if n == 1:
        return parts, "[v0]"
    if not transitions:
        ins = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{ins}concat=n={n}:v=1:a=0[vj]")
        return parts, "[vj]"
    prev = "[v0]"
    for i in range(1, n):
        lbl = f"[vx{i}]"
        tr = _TRANS[(i - 1) % len(_TRANS)]
        parts.append(f"{prev}[v{i}]xfade=transition={tr}:duration={td}:offset={offs[i]:.3f}{lbl}")
        prev = lbl
    return parts, prev


def _audio_join(n: int, td: float, transitions: bool) -> tuple[list[str], str]:
    parts = [f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]" for i in range(n)]
    if n == 1:
        return parts, "[a0]"
    if not transitions:
        ins = "".join(f"[a{i}]" for i in range(n))
        parts.append(f"{ins}concat=n={n}:v=0:a=1[aj]")
        return parts, "[aj]"
    prev = "[a0]"
    for i in range(1, n):
        lbl = f"[ax{i}]"
        parts.append(f"{prev}[a{i}]acrossfade=d={td}:c1=tri:c2=tri{lbl}")
        prev = lbl
    return parts, prev


def render_clip(clip: dict, clips_dir: Path, work_dir: Path, cfg: dict) -> dict:
    v = cfg["video"]
    a = cfg["audio"]
    eff = cfg.get("effects", {})
    afx = cfg.get("audio_fx", {})
    out_dir = clips_dir / f"{clip['index']:02d}_{clip['slug']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "clip.mp4"
    segs = clip["segments"]
    n = len(segs)
    # PART (long-form) → seamless concat (accurate card timing); SUPERCUT → transitions.
    transitions = eff.get("transitions", True) and n > 1 and clip.get("type") != "PART"
    td = eff.get("transition_dur_s", 0.28)
    print(
        f"  render [{clip['index']:02d}] {clip['slug']} "
        f"({clip['duration']:.0f}s, {n} seg, {clip['n_sources']} clips"
        f"{', transitions' if transitions else ''}) ...",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as td_dir:
        tdp = Path(td_dir)
        seg_paths, durs = [], []
        for i, seg in enumerate(segs):
            sp = tdp / f"seg{i}.mp4"
            durs.append(_normalize_segment(seg, sp, cfg))
            seg_paths.append(sp)
        offs, total = _compute_offsets(durs, td, transitions)

        # Caption word groups + location cards (placed by timeline offset).
        word_groups, cards = [], []
        for i, seg in enumerate(segs):
            words = []
            if seg.get("has_speech"):
                tr = tr_mod.load_transcript(seg["source_id"], work_dir) or {"segments": []}
                for w in sel_mod.all_words(tr):
                    if w["end"] > seg["start"] and w["start"] < seg["end"]:
                        ws = max(0.0, w["start"] - seg["start"]) + offs[i]
                        we = min(durs[i], w["end"] - seg["start"]) + offs[i]
                        if we > ws:
                            words.append({"w": w["w"], "start": ws, "end": we})
            word_groups.append(words)
            loc = seg.get("location_label", "")
            if loc and eff.get("location_card", True):
                cards.append(
                    {"text": loc, "t_in": offs[i] + 0.2, "t_out": offs[i] + min(durs[i], 2.4)}
                )

        ass_path = work_dir / "ass" / f"{clip['index']:02d}_{clip['slug']}.ass"
        ass = caption_ass.build_ass(
            clip.get("hook_text", "") if eff.get("title_card", True) else "",
            total,
            word_groups,
            cards,
            cfg,
            ass_path,
            chapters=(clip.get("chapters") or []) if eff.get("chapter_card", True) else [],
        )

        # ---- single final filter_complex ----
        cmd = [ff.find_ffmpeg(), "-y", "-filter_complex_threads", "1"]
        for sp in seg_paths:
            cmd += ["-i", str(sp)]
        idx = n  # extra inputs start here

        vparts, vlbl = _video_join(n, durs, offs, td, transitions)
        # grade + grain + subtitles
        lut = eff.get("lut", "")
        grade = (
            f"lut3d=file='{_esc(str(PIPE_ROOT / lut))}':interp=tetrahedral"
            if lut and (PIPE_ROOT / lut).exists()
            else _GRADE
        )
        vchain = [grade] if eff.get("color_grade", True) else []
        if eff.get("film_grain", True):
            vchain.append(f"noise=alls={eff.get('grain_strength', 10)}:allf=t+u")
        if ass:
            vchain.append(f"ass='{_esc(str(ass))}'")
        vfinal = f"{vlbl}{','.join(vchain)}[vout]" if vchain else f"{vlbl}null[vout]"

        aparts, albl = _audio_join(n, td, transitions)
        # denoise
        model = afx.get("denoise_model", "")
        mpath = PIPE_ROOT / model if model else None
        if afx.get("denoise", True) and mpath and mpath.exists():
            aparts.append(f"{albl}highpass=f=85,arnndn=m='{_esc(str(mpath))}':mix=0.85[aclean]")
        elif afx.get("denoise", True):
            aparts.append(f"{albl}highpass=f=85,afftdn=nr={afx.get('denoise_nr', 11)}[aclean]")
        else:
            aparts.append(f"{albl}anull[aclean]")
        prog = "[aclean]"
        labels = []

        # music + side-chain ducking (off by default)
        mood = clip.get("music_mood") or _FMT2MOOD.get(
            str(clip.get("format", "")).lower(), "cinematic"
        )
        mus = (
            _music(str(PIPE_ROOT / cfg["music"]["dir"]), mood)
            if cfg["music"].get("enabled", True)
            else None
        )
        if mus:
            clip["_music_used"] = Path(mus).name
            cmd += ["-stream_loop", "-1", "-i", mus]
            midx = idx
            idx += 1
            mvol = 10 ** (cfg["music"].get("volume_db", -16) / 20.0)
            aparts.append(
                f"[{midx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={mvol:.3f}[mraw]"
            )
            if afx.get("sidechain_duck", True):
                aparts.append(f"{prog}asplit=2[aprog][askey]")
                aparts.append(
                    "[mraw][askey]sidechaincompress=threshold=0.045:ratio=10:attack=20:release=320[mduck]"
                )
                labels += ["[aprog]", "[mduck]"]
            else:
                labels += [prog, "[mraw]"]
        else:
            labels.append(prog)

        # SFX (whoosh on cuts, impact on hook)
        sx = cfg.get("sfx", {})
        if sx.get("enabled", True):
            sdir = str(PIPE_ROOT / sx.get("dir", "sfx"))
            impact = _find(sdir, "impact", "boom")
            whoosh = _find(sdir, "whoosh", "swoosh")
            place = []
            if impact:
                place.append((impact, 0.05, sx.get("impact_db", -5)))
            if whoosh and transitions:
                lead = sx.get("whoosh_lead_s", 0.12)
                for i in range(1, n):
                    place.append((whoosh, max(0.0, offs[i] - lead), sx.get("whoosh_db", -8)))
            for path, t, db in place:
                if t >= total:
                    continue
                cmd += ["-i", path]
                aparts.append(
                    f"[{idx}:a]adelay={int(round(t * 1000))}:all=1,aresample=48000,"
                    f"aformat=channel_layouts=stereo,volume={10 ** (db / 20):.3f}[s{idx}]"
                )
                labels.append(f"[s{idx}]")
                idx += 1

        if len(labels) == 1:
            aparts.append(
                f"{labels[0]}loudnorm=I={a['lufs']}:TP={a['true_peak']}:LRA={a['lra']}[aout]"
            )
        else:
            aparts.append(
                f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[mx];"
                f"[mx]loudnorm=I={a['lufs']}:TP={a['true_peak']}:LRA={a['lra']}[aout]"
            )

        fc = ";".join(vparts + [vfinal] + aparts)
        cmd += [
            "-filter_complex",
            fc,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            v["vcodec"],
            "-b:v",
            v["bitrate"],
            "-pix_fmt",
            v["pix_fmt"],
            "-c:a",
            a["acodec"],
            "-b:a",
            f"{a['kbps']}k",
            "-ar",
            str(a["rate"]),
            "-t",
            f"{total:.3f}",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        ff.run(cmd)

        cover = out_dir / "cover.jpg"
        ff.extract_frame(
            str(seg_paths[0]), min(clip.get("cover_offset_s", 0.8), durs[0] - 0.1), cover
        )

    clip["_out_mp4"] = str(out_mp4)
    clip["_out_dir"] = str(out_dir)
    clip["_cover"] = str(cover) if cover.exists() else None
    return clip


def render_all(selection, clips_dir: Path, work_dir: Path, cfg: dict) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for clip in selection:
        try:
            out.append(render_clip(clip, clips_dir, work_dir, cfg))
        except Exception as e:
            print(f"  ! render failed [{clip.get('index')}]: {e}")
    return out
