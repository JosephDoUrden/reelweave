"""Stage 2 — transcribe: extract audio + mlx-whisper (word-level timestamps)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import util_ffmpeg as ff


def _extract_audio(src: str, out_wav: Path) -> None:
    if out_wav.exists():
        return
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # Handheld vlog: tame wind/ambient noise to improve ASR accuracy.
    cmd = [ff.find_ffmpeg(), "-y", "-i", src,
           "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-25",
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    ff.run(cmd)


# Phrases Whisper commonly hallucinates over silence/ambience (subtitle-data artefacts).
# Turkish + English defaults; extend via config `transcribe.hallucinations`.
_HALLUCINATIONS = (
    "altyazı", "abone ol", "izlediğiniz için teşekkür", "teşekkürler", "amara.org",
    "subtitles by", "subscribe", "thanks for watching", "like and subscribe",
)


def _is_hallucination(s: dict, blocklist: tuple) -> bool:
    """Is this a fabricated segment over silence/ambience? (no_speech / logprob / repetition / blocklist)"""
    txt = (s.get("text") or "").strip()
    low = txt.lower()
    if not txt:
        return True
    if float(s.get("no_speech_prob", 0) or 0) > 0.6 and float(s.get("avg_logprob", 0) or 0) < -0.5:
        return True
    if float(s.get("avg_logprob", 0) or 0) < -1.0:
        return True
    if float(s.get("compression_ratio", 0) or 0) > 2.4:   # runaway repetition
        return True
    if any(h in low for h in blocklist):
        return True
    # Same token repeated 3+ times in a row.
    ws = low.split()
    if len(ws) >= 3 and len(set(ws)) <= max(1, len(ws) // 3):
        return True
    return False


def _fix_names(txt: str, name_fixes: dict) -> str:
    """Deterministically correct frequently-misheard proper nouns (config-driven)."""
    for wrong, right in name_fixes.items():
        txt = re.sub(rf"\b{re.escape(wrong)}\b", right, txt, flags=re.IGNORECASE)
    return txt


def _normalize(result: dict, blocklist: tuple, name_fixes: dict) -> dict:
    segs = []
    for s in result.get("segments", []):
        if _is_hallucination(s, blocklist):
            continue
        words = []
        for w in s.get("words", []) or []:
            txt = (w.get("word") or w.get("text") or "").strip()
            if not txt:
                continue
            words.append({
                "w": _fix_names(txt, name_fixes),
                "start": round(float(w.get("start", s.get("start", 0))), 3),
                "end": round(float(w.get("end", s.get("end", 0))), 3),
            })
        segs.append({
            "start": round(float(s.get("start", 0)), 3),
            "end": round(float(s.get("end", 0)), 3),
            "text": _fix_names((s.get("text") or "").strip(), name_fixes),
            "words": words,
        })
    return {
        "language": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "segments": segs,
    }


def transcribe_file(entry: dict, work_dir: Path, cfg: dict) -> dict:
    fid = entry["id"]
    out_json = work_dir / "transcripts" / f"{fid}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())

    out_json.parent.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "audio" / f"{fid}.wav"
    _extract_audio(entry["path"], wav)

    import mlx_whisper  # heavy import — only when needed

    tcfg = cfg["transcribe"]
    # Optional vocabulary hint (place names, brands) to improve proper-noun accuracy.
    ip_path = Path(cfg.get("_prompts_dir", "")) / "initial_prompt.txt"
    initial = ip_path.read_text().strip() if ip_path.exists() else None
    blocklist = _HALLUCINATIONS + tuple(tcfg.get("hallucinations", []))
    name_fixes = {str(k): str(v) for k, v in (tcfg.get("name_fixes") or {}).items()}

    print(f"  transcribe: {entry['name']} ...", flush=True)
    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=tcfg["model_repo"],
        language=tcfg.get("language") or None,
        word_timestamps=tcfg.get("word_timestamps", True),
        initial_prompt=initial,
        condition_on_previous_text=False,   # break hallucination chains on long takes
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        temperature=(0.0, 0.2, 0.4),
        verbose=False,
    )
    norm = _normalize(result, blocklist, name_fixes)
    out_json.write_text(json.dumps(norm, ensure_ascii=False, indent=2))
    return norm


def transcribe_all(manifest: dict, work_dir: Path, cfg: dict) -> dict[str, dict]:
    return {e["id"]: transcribe_file(e, work_dir, cfg) for e in manifest["files"]}


def load_transcript(fid: str, work_dir: Path) -> dict | None:
    p = work_dir / "transcripts" / f"{fid}.json"
    return json.loads(p.read_text()) if p.exists() else None
