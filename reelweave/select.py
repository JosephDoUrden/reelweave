"""Stage 6 — director: deterministic timetable/parts + LLM naming, narrative and hook verification.

Units = shot windows carrying vision analysis + transcript speech (chronological).
Two output modes (config `output_mode`):
  - "parts": deterministic timetable. Group clips by day -> time-gap/location into chapters,
    pack chapters into ~3-minute parts. The LLM only NAMES parts/chapters and writes narrative.
  - "reels": a two-stage director (storyline -> storyboard) producing short themed clips.
Chronology is enforced in code (never left to the LLM); the hook is the only out-of-order segment.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import requests

from . import describe as desc_mod
from . import transcribe as tr_mod

# --------------------------- helpers ---------------------------


def all_words(transcript: dict) -> list[dict]:
    out = []
    for seg in transcript.get("segments", []):
        out.extend(seg.get("words", []))
    return out


def _words_in(words, s, e):
    return [w for w in words if w["end"] > s and w["start"] < e]


def _speech_text(words, s, e, limit=160):
    txt = " ".join(w["w"] for w in _words_in(words, s, e)).strip()
    return txt[:limit]


def _snap_to_sentences(transcript: dict, start: float, end: float):
    """Extend the range to the full overlapping sentence(s) — avoids mid-sentence cuts."""
    segs = [s for s in transcript.get("segments", []) if s.get("text", "").strip()]
    over = [s for s in segs if s["end"] > start and s["start"] < end]
    if not over:
        return start, end
    return min(s["start"] for s in over), max(s["end"] for s in over)


def _overlap_frac(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    span = min(a1 - a0, b1 - b0)
    return inter / span if span > 0 else 0.0


def _no_rewind(segs: list[dict]) -> list[dict]:
    """Prevent a visible rewind: if consecutive same-source segments overlap, trim the later one."""
    out = []
    for s in segs:
        if out and out[-1]["source_id"] == s["source_id"] and s["start"] < out[-1]["end"]:
            s = {**s, "start": round(out[-1]["end"], 2)}
        if s["end"] - s["start"] >= 1.0:
            out.append(s)
    return out


def _slugify(s: str) -> str:
    s = s.translate(str.maketrans("çğıöşüÇĞİÖŞÜščžćđŠČŽĆĐ", "cgiosuCGIOSUsczcdSCZCD")).lower()
    return (re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "clip")[:40]


def _ollama(prompt: str, cfg: dict, num_predict=4000) -> str:
    oc = cfg["ollama"]
    payload = {
        "model": oc["model"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,  # disable qwen3 thinking (pure JSON)
        "options": {
            "temperature": oc.get("temperature", 0.4),
            "num_ctx": oc.get("num_ctx", 32768),
            "num_predict": num_predict,
        },
    }
    r = requests.post(f"{oc['host']}/api/generate", json=payload, timeout=oc.get("timeout_s", 900))
    r.raise_for_status()
    return r.json().get("response", "")


def _json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# --------------------------- asset graph (units) ---------------------------


def build_units(manifest: dict, work_dir: Path) -> list[dict]:
    """Turn each vision window into a unit and sort chronologically (source_order, start)."""
    units = []
    for e in manifest["files"]:
        fid = e["id"]
        m = re.search(r"_(\d{3,4})_", e["name"])
        order = int(m.group(1)) if m else 0
        ct = e.get("capture_time") or ""
        day = ct[8:10] if len(ct) >= 10 else "?"
        hhmm = ct[11:16] if len(ct) >= 16 else ""
        tr = tr_mod.load_transcript(fid, work_dir) or {"segments": []}
        words = all_words(tr)
        for v in desc_mod.load_vision(fid, work_dir):
            s, en = v["start"], v["end"]
            sp = _speech_text(words, s, en)
            units.append(
                {
                    "source_id": fid,
                    "source_path": e["path"],
                    "source_duration": e["duration_s"],
                    "source_order": order,
                    "day": day,
                    "hhmm": hhmm,
                    "start": s,
                    "end": en,
                    "has_speech": bool(sp),
                    "speech": sp,
                    "place": v.get("place_type", ""),
                    "landmark": v.get("landmark"),
                    "mood": v.get("mood", ""),
                    "action": v.get("action", ""),
                    "activity": v.get("activity", []) or [],
                    "food_drink": v.get("food_drink", []) or [],
                    "clip_purpose": v.get("clip_purpose", ""),
                    "highlight": v.get("highlight", 0),
                    "desc": v.get("desc", ""),
                }
            )
    units.sort(key=lambda u: (u["source_order"], u["start"]))
    for i, u in enumerate(units, 1):
        u["uid"] = f"U{i}"
    return units


def _unit_line(u: dict, full=True) -> str:
    tag = "SPEECH" if u["has_speech"] else "broll"
    lm = u["landmark"] or "-"
    head = (
        f"[{u['uid']}] day{u['day']} {u['hhmm']} #{u['source_order']} "
        f"t={u['start']:.1f}-{u['end']:.1f}({u['end'] - u['start']:.0f}s) {tag} "
        f"place={u['place']} lm={lm} hi={u['highlight']:.1f}"
    )
    if not full:
        return head
    body = f"\n   {u['desc']}"
    if u["speech"]:
        body += f'\n   says: "{u["speech"]}"'
    return head + body


# --------------------------- stage 1: storyline (reels mode) ---------------------------


def _medium_line(u: dict) -> str:
    """Compact single line for storyline (keeps the context small)."""
    tag = "SP" if u["has_speech"] else "br"
    lm = u["landmark"] or "-"
    d = (u["desc"] or "")[:55]
    sp = (u["speech"][:48] + "…") if u["speech"] else ""
    line = (
        f"[{u['uid']}] day{u['day']} {u['hhmm']} #{u['source_order']} "
        f"{u['start']:.0f}-{u['end']:.0f}s {tag} {u['place']}/{lm} hi{u['highlight']:.1f} | {d}"
    )
    return line + (f' | "{sp}"' if sp else "")


def storyline(units: list[dict], cfg: dict, prompts_dir: Path) -> dict:
    tpl = (prompts_dir / "storyline.txt").read_text()
    txt = "\n".join(_medium_line(u) for u in units)
    # Safety: trim well below the context window.
    budget = 52000
    if len(txt) > budget:
        lines = txt.splitlines()
        step = len(lines) / (budget // 130)
        lines = [lines[int(i * step)] for i in range(budget // 130)]
        txt = "\n".join(lines)
    prompt = (
        tpl.replace("{UNITS}", txt)
        .replace("{N}", str(cfg["clip"].get("target_count", 8)))
        .replace("{LANG}", cfg.get("language", "en"))
        .replace("{LOCATION}", cfg.get("location", "the trip"))
    )
    return _json(_ollama(prompt, cfg, num_predict=4000))


def _fallback_plans(units: list[dict], cfg: dict) -> list[dict]:
    """If the LLM storyline is empty, build deterministic theme plans (never zero clips)."""
    loc = cfg.get("location", "Trip")
    plans = []
    top = sorted(units, key=lambda u: u["highlight"], reverse=True)
    hero = [u["uid"] for u in sorted(top[:14], key=lambda u: (u["source_order"], u["start"]))]
    plans.append(
        {
            "type": "HERO",
            "theme": f"{loc} Recap",
            "title_idea": f"{loc} Recap",
            "unit_ids": hero,
            "chrono_rule": "macro_chronological",
        }
    )
    for comp in _compilation_defs(cfg):
        ids = [u["uid"] for u in units if _matches_compilation(u, comp)]
        if len(ids) >= 3:
            plans.append(
                {
                    "type": "SUPERCUT",
                    "theme": comp["title"],
                    "title_idea": comp["title"],
                    "unit_ids": ids[:10],
                    "chrono_rule": "free",
                }
            )
    places = Counter(u["place"] for u in units if u["place"])
    for place, c in places.most_common(3):
        if c >= 4:
            ids = [u["uid"] for u in units if u["place"] == place]
            plans.append(
                {
                    "type": "PLACE",
                    "theme": f"{loc}: {place}",
                    "title_idea": place,
                    "unit_ids": ids[:8],
                    "chrono_rule": "location_internal",
                }
            )
    return plans


# --------------------------- stage 2: storyboard (reels mode) ---------------------------


def _resolve_segments(
    sb: dict, by_uid: dict, work_dir: Path, cfg: dict, used: list | None = None
) -> list[dict]:
    cc = cfg["clip"]
    min_seg = cc.get("min_seg_s", 3.0)
    used = used if used is not None else []
    segs = []
    for s in sb.get("segments") or []:
        ref = str(s.get("ref", "")).upper().strip()
        u = by_uid.get(ref)
        if not u:
            mm = re.match(r"U\d+", ref)
            u = by_uid.get(mm.group(0)) if mm else None
        if not u:
            continue
        try:
            ins = float(s.get("in_s", u["start"]))
            outs = float(s.get("out_s", u["end"]))
        except (TypeError, ValueError):
            ins, outs = u["start"], u["end"]
        ins = max(u["start"], min(ins, u["end"]))
        outs = max(u["start"], min(outs, u["end"]))
        if outs - ins < 1.0:
            ins, outs = u["start"], u["end"]
        if u["has_speech"]:
            # Snap to sentence boundaries so speech is never cut mid-sentence.
            tr = tr_mod.load_transcript(u["source_id"], work_dir) or {"segments": []}
            ins, outs = _snap_to_sentences(tr, ins, outs)
            ins = max(u["start"], ins)
            outs = min(u["end"], outs)
        # Minimum segment duration (let the visual beat complete).
        if outs - ins < min_seg:
            outs = min(u["source_duration"], ins + min_seg)
        # Dedup: skip if it overlaps heavily with an already-used range.
        dup = any(
            sid == u["source_id"] and _overlap_frac(ins, outs, a, b) > 0.5 for sid, a, b in used
        )
        if dup:
            continue
        seg = {
            "source_id": u["source_id"],
            "source_path": u["source_path"],
            "source_order": u["source_order"],
            "start": round(max(0, ins), 2),
            "end": round(min(u["source_duration"], outs), 2),
            "has_speech": u["has_speech"],
            "role": str(s.get("role", "body")),
            "location_label": str(s.get("location_label", "")).strip(),
        }
        if seg["end"] - seg["start"] >= 0.8:
            segs.append(seg)
            used.append((u["source_id"], seg["start"], seg["end"]))
    return segs


def _enforce_chronology(segs: list[dict]) -> list[dict]:
    """Keep the hook first; sort the rest by (source_order, start)."""
    hooks = [s for s in segs if s["role"] == "hook"]
    body = [s for s in segs if s["role"] != "hook"]
    body.sort(key=lambda s: (s["source_order"], s["start"]))
    return (hooks[:1] + body) if hooks else body


def _reel_evidence(segs: list[dict], units: list[dict], work_dir: Path) -> str:
    """The real evidence behind a clip's segments: speech + visual descriptions."""
    lines = []
    for s in segs:
        tr = tr_mod.load_transcript(s["source_id"], work_dir) or {"segments": []}
        sp = _speech_text(all_words(tr), s["start"], s["end"], 200)
        vis = [
            u
            for u in units
            if u["source_id"] == s["source_id"] and u["end"] > s["start"] and u["start"] < s["end"]
        ]
        vd = "; ".join(
            f"{u['desc']}" + (f" [{u['landmark']}]" if u.get("landmark") else "") for u in vis[:3]
        )
        if sp:
            lines.append(f'- says: "{sp}"')
        if vd:
            lines.append(f"- visual: {vd}")
    return "\n".join(lines[:20])


def _verify_hook(hook: str, title: str, evidence: str, cfg: dict) -> dict:
    """Is the hook/title grounded in the footage evidence?

    If fabricated, propose a safe alternative.
    """
    lang = cfg.get("language", "en")
    prompt = (
        "YOU ARE A VERIFIER. Compare a travel clip's HOOK and TITLE against the footage "
        "EVIDENCE (real speech + visuals).\n"
        f'HOOK: "{hook}"\nTITLE: "{title}"\n\nEVIDENCE:\n{evidence or "(no evidence)"}\n\n'
        "Return grounded=false ONLY when: (a) a SPECIFIC number/date/statistic not in the "
        "evidence; (b) a SPECIFIC place/landmark name not seen in the evidence; (c) an "
        "unverifiable absolute claim (first/best/only/never); (d) a clear contradiction. "
        "General true statements (being in the city/market, eating shown food, walking, "
        "drinking coffee) are GROUNDED — do not reject those.\n"
        f"If grounded=false, write safe_hook: a short (<=7 words) {lang} hook that is grounded "
        "in the evidence yet still CURIOSITY-DRIVEN (not a flat description).\n"
        'JSON ONLY: {"grounded":true/false,"safe_hook":"...","safe_title":"..."}'
    )
    try:
        d = _json(_ollama(prompt, cfg, num_predict=300))
    except Exception:
        return {"grounded": True}
    return d if isinstance(d, dict) else {"grounded": True}


_SEG_GUIDE = {
    "HERO": (
        "4-6 segments, ~6-12s each. Narrative flow; do not over-fragment, let lines/moments finish."
    ),
    "SUPERCUT": "5-8 SHORT segments (~2-5s), fast pace; the same theme across different clips.",
    "PLACE": "4-6 segments ~4-9s.",
    "TEMA": "4-6 segments ~4-9s.",
}


def storyboard_reel(
    plan: dict,
    units: list[dict],
    by_uid: dict,
    work_dir: Path,
    cfg: dict,
    prompts_dir: Path,
    used: list,
) -> dict | None:
    cc = cfg["clip"]
    rtype = str(plan.get("type", "PLACE")).upper()
    lens = cc.get("lengths", {})
    tl = lens.get(rtype, lens.get("default", {"min": 20, "max": 45}))
    tmin, tmax = tl["min"], tl["max"]
    tpl = (prompts_dir / "storyboard.txt").read_text()
    reel_uids = [r for r in (plan.get("unit_ids") or []) if r in by_uid]
    reel_units = [by_uid[r] for r in reel_uids] or units[:12]
    reel_txt = "\n".join(_unit_line(u, full=True) for u in reel_units)
    other_txt = "\n".join(_unit_line(u, full=False) for u in units)
    prompt = (
        tpl.replace("{TYPE}", rtype)
        .replace("{THEME}", plan.get("theme", ""))
        .replace("{CHRONO}", plan.get("chrono_rule", "free"))
        .replace("{LEN}", f"{tmin}-{tmax}")
        .replace("{LANG}", cfg.get("language", "en"))
        .replace("{LOCATION}", cfg.get("location", "the trip"))
        .replace("{SEG_GUIDE}", _SEG_GUIDE.get(rtype, _SEG_GUIDE["PLACE"]))
        .replace("{REEL_UNITS}", reel_txt)
        .replace("{OTHER_UNITS}", other_txt)
    )
    sb = _json(_ollama(prompt, cfg, num_predict=3000))
    if not sb:
        return None
    segs = _resolve_segments(sb, by_uid, work_dir, cfg, used)
    if not segs:
        return None
    segs = _enforce_chronology(segs)  # hook first, body chronological

    # Length cap by type (HERO may be long).
    total, kept = 0.0, []
    for s in segs:
        d = s["end"] - s["start"]
        if total + d > tmax:
            d = tmax - total
            if d < cc.get("min_seg_s", 3.0):
                break
            s = {**s, "end": round(s["start"] + d, 2)}
        kept.append(s)
        total += s["end"] - s["start"]
    if total < max(12.0, tmin * 0.5):
        return None
    score = float(sb.get("score", plan.get("score", 7)) or 7)
    if score < cfg["clip"].get("min_score", 5.5):
        score = cfg["clip"]["min_score"]

    # Hook/title verification (no fabrication).
    hook_text = str(sb.get("hook_text", "")).strip()
    title = str(sb.get("title", "")).strip()
    evidence = _reel_evidence(kept, units, work_dir)
    vr = _verify_hook(hook_text, title, evidence, cfg)
    if not vr.get("grounded", True):
        sh = str(vr.get("safe_hook", "")).strip()
        st = str(vr.get("safe_title", "")).strip()
        print(f"    ! hook not grounded, replaced: '{hook_text}' -> '{sh}'")
        if sh:
            hook_text = sh
        if st:
            title = st

    return {
        "type": plan.get("type", ""),
        "theme": str(title or sb.get("title") or plan.get("theme", "")).strip(),
        "format": str(sb.get("format", "cinematic")).strip(),
        "segments": kept,
        "duration": round(total, 2),
        "hook_text": hook_text,
        "title": title,
        "description": str(sb.get("description", "")).strip(),
        "hashtags": [str(h).strip() for h in (sb.get("hashtags") or [])][:6],
        "cover_offset_s": 0.8,
        "music": bool(sb.get("music", False)),
        "chrono_rule": plan.get("chrono_rule", "free"),
        "score": round(score, 2),
        "rationale": str(sb.get("rationale", "")).strip(),
        "n_sources": len({s["source_id"] for s in kept}),
    }


# =========================== PARTS / TIMETABLE MODE ===========================


def _gap_min(a: str, b: str) -> float:
    try:
        h1, m1 = map(int, a.split(":"))
        h2, m2 = map(int, b.split(":"))
        return abs((h2 * 60 + m2) - (h1 * 60 + m1))
    except Exception:
        return 0.0


def build_parts(units: list[dict], cfg: dict) -> list[dict]:
    """Split each day into chapters by time-gap / location change."""
    pc = cfg["parts"]
    by_day: dict = {}
    for u in units:
        by_day.setdefault(u["day"], []).append(u)
    parts = []
    for day in sorted(by_day):
        dus = sorted(by_day[day], key=lambda u: (u["source_order"], u["start"]))
        chapters, cur = [], [dus[0]]
        for prev, u in zip(dus, dus[1:], strict=False):
            if (
                _gap_min(prev["hhmm"], u["hhmm"]) > pc.get("boundary_gap_min", 18)
                or u["place"] != prev["place"]
            ):
                chapters.append(cur)
                cur = []
            cur.append(u)
        if cur:
            chapters.append(cur)
        parts.append({"day": day, "chapters": chapters})
    for i, p in enumerate(parts, 1):
        p["part_no"] = i
    return parts


def _chapter_segments(
    chapter: list[dict], work_dir: Path, cfg: dict, used: list, cap_s: float
) -> list[dict]:
    """Deterministically pick segments from a chapter: merge contiguous windows,
    snap speech to sentences, prefer high-highlight/speech, cap total at cap_s."""
    cc = cfg["clip"]
    min_seg = cc.get("min_seg_s", 3.5)
    # Merge contiguous windows from the same source clip.
    merged = []
    for u in sorted(chapter, key=lambda x: (x["source_order"], x["start"])):
        if (
            merged
            and merged[-1]["source_id"] == u["source_id"]
            and u["start"] - merged[-1]["end"] < 1.0
            and (u["end"] - merged[-1]["start"]) <= 22
        ):
            merged[-1]["end"] = u["end"]
            merged[-1]["has_speech"] = merged[-1]["has_speech"] or u["has_speech"]
            merged[-1]["hl"] = max(merged[-1]["hl"], u["highlight"])
        else:
            merged.append(
                {
                    "source_id": u["source_id"],
                    "source_path": u["source_path"],
                    "source_order": u["source_order"],
                    "source_duration": u["source_duration"],
                    "start": u["start"],
                    "end": u["end"],
                    "has_speech": u["has_speech"],
                    "hl": u["highlight"],
                    "landmark": u.get("landmark"),
                }
            )
    # Rank (speech + high highlight first), keep chronological order in output.
    ranked = sorted(merged, key=lambda m: (m["has_speech"], m["hl"]), reverse=True)
    chosen, total = [], 0.0
    for m in ranked:
        if total >= cap_s:
            break
        s, e = m["start"], m["end"]
        if m["has_speech"]:
            tr = tr_mod.load_transcript(m["source_id"], work_dir) or {"segments": []}
            s, e = _snap_to_sentences(tr, s, e)
            s = max(m["start"], s)
            e = min(m["end"], e)  # stay within the unit (no overlap/rewind)
        if e - s < min_seg:
            e = min(m["source_duration"], s + min_seg)
        if e - s > 20:
            e = s + 20
        if any(sid == m["source_id"] and _overlap_frac(s, e, a, b) > 0.5 for sid, a, b in used):
            continue
        chosen.append(
            {
                "source_id": m["source_id"],
                "source_path": m["source_path"],
                "source_order": m["source_order"],
                "start": round(s, 2),
                "end": round(e, 2),
                "has_speech": m["has_speech"],
                "location_label": (m["landmark"] or "") if m["landmark"] else "",
            }
        )
        used.append((m["source_id"], round(s, 2), round(e, 2)))
        total += e - s
    chosen.sort(key=lambda x: (x["source_order"], x["start"]))
    return chosen


def _chapter_summary(chapter: list[dict]) -> str:
    acts = Counter(a for u in chapter for a in u.get("activity", []))
    place = Counter(u["place"] for u in chapter if u["place"]).most_common(1)
    lm = next((u["landmark"] for u in chapter if u.get("landmark")), None)
    sp = next((u["speech"] for u in chapter if u["speech"]), "")[:80]
    hh = chapter[0]["hhmm"]
    return (
        f"{hh} | place={place[0][0] if place else '?'} lm={lm or '-'} "
        f'activity={",".join(a for a, _ in acts.most_common(2))} | says: "{sp}"'
    )


def _name_parts(parts_summ: list[dict], cfg: dict) -> dict:
    """One small LLM call: name all parts/chapters and write narrative (small context)."""
    lang = cfg.get("language", "en")
    loc = cfg.get("location", "the trip")
    blocks = []
    for p in parts_summ:
        chs = "\n".join(f"  - chapter {j + 1}: {s}" for j, s in enumerate(p["summaries"]))
        blocks.append(f"PART {p['part_no']} (day {p['day']}):\n{chs}")
    prompt = (
        f"You are a travel-documentary editor. A trip to {loc} has been split into PARTS and "
        "chapters (the chronological order is fixed — DO NOT change it). Give every PART and "
        f"chapter a content-driven {lang} title and short narrative. Do not invent anything; rely "
        "on the given place/activity/speech.\n\n"
        + "\n\n".join(blocks)
        + '\n\nJSON ONLY: {"parts":[{"part_no":1,"title":"... — Part 1: ...",'
        '"intro_promise":"what the opening promises (4-6 words)","description":"~150 chars",'
        '"hashtags":["#travel"],'
        '"chapters":[{"title":"chapter title","purpose":"one sentence: what it shows"}]}]}'
    )
    return _json(_ollama(prompt, cfg, num_predict=3000))


def select_parts(manifest: dict, work_dir: Path, cfg: dict) -> list[dict]:
    units = build_units(manifest, work_dir)
    (work_dir / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2))
    print(f"  {len(units)} units built.")
    if not units:
        return []
    pc = cfg["parts"]
    loc = cfg.get("location", "Trip")
    day_parts = build_parts(units, cfg)
    tmax = pc.get("target_max_s", 180)
    tmin = pc.get("target_min_s", 60)

    used: list = []
    parts = []  # day chapters packed into ~tmax parts (Part 1, 2, 3...)

    def _flush(day, bucket):
        if not bucket:
            return
        seg_all, summaries = [], []
        for segs, summ in bucket:
            summaries.append(summ)
            seg_all.extend(segs)
        seg_all = _no_rewind(seg_all)  # prevent same-source rewind
        t = sum(s["end"] - s["start"] for s in seg_all)
        ch_meta, acc = [], 0.0
        for segs, _ in bucket:
            ch_meta.append({"t": round(acc, 2), "n": len(segs)})
            acc += sum(s["end"] - s["start"] for s in segs)
        parts.append(
            {
                "day": day,
                "_segments": seg_all,
                "_ch_meta": ch_meta,
                "summaries": summaries,
                "_dur": t,
            }
        )

    for dp in day_parts:
        bucket, bdur = [], 0.0
        for ch in dp["chapters"]:
            segs = _chapter_segments(ch, work_dir, cfg, used, cap_s=22)
            if not segs:
                continue
            cdur = sum(s["end"] - s["start"] for s in segs)
            if bdur + cdur > tmax and bdur >= tmin:
                _flush(dp["day"], bucket)
                bucket, bdur = [], 0.0
            bucket.append((segs, _chapter_summary(ch)))
            bdur += cdur
        _flush(dp["day"], bucket)
    for i, p in enumerate(parts, 1):
        p["part_no"] = i
    print(f"  {len(day_parts)} day(s) -> {len(parts)} parts (~{tmin}-{tmax}s).")

    try:
        named = _name_parts(
            [
                {"part_no": p["part_no"], "day": p["day"], "summaries": p["summaries"]}
                for p in parts
            ],
            cfg,
        ).get("parts", [])
    except Exception as e:
        print(f"  ! naming error: {e}")
        named = []
    name_by = {n.get("part_no"): n for n in named}

    out = []
    idx = 0
    for p in parts:
        if p["_dur"] < 20 or not p["_segments"]:
            continue
        idx += 1
        nm = name_by.get(p["part_no"], {})
        title = nm.get("title") or f"{loc} — Part {p['part_no']}"
        ch_titles = nm.get("chapters", [])
        chapters = []
        for k, cm in enumerate(p["_ch_meta"]):
            ct = ch_titles[k] if k < len(ch_titles) else {}
            chapters.append({"t": round(cm["t"], 2), "title": ct.get("title", "") or "", "sub": ""})
        out.append(
            {
                "index": idx,
                "type": "PART",
                "format": "trip",
                "theme": title,
                "title": title,
                "slug": _slugify(title),
                "hook_text": nm.get("intro_promise", title),
                "description": nm.get("description", ""),
                "hashtags": [str(h) for h in (nm.get("hashtags") or ["#travel"])][:6],
                "segments": p["_segments"],
                "chapters": chapters,
                "duration": round(p["_dur"], 2),
                "music": False,
                "cover_offset_s": 1.0,
                "score": 8.0,
                "n_sources": len({s["source_id"] for s in p["_segments"]}),
                "rationale": nm.get("intro_promise", ""),
            }
        )

    out += build_compilations(units, work_dir, cfg, used, start_index=idx)
    (work_dir / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  director(parts): {len(out)} parts/compilations produced.")
    return out


def _compilation_defs(cfg: dict) -> list[dict]:
    """Compilation definitions from config, with sensible English defaults."""
    raw = cfg.get("parts", {}).get("compilations", [])
    defs = []
    for c in raw:
        if isinstance(c, dict) and c.get("lexicon"):
            defs.append(
                {
                    "title": c.get("title", "Compilation"),
                    "lexicon": [w.lower() for w in c["lexicon"]],
                    "activity": c.get("activity", ""),
                    "hashtag": c.get("hashtag", "#travel"),
                }
            )
    return defs


def _matches_compilation(u: dict, comp: dict) -> bool:
    hay = (u["desc"] + " " + u["speech"] + " " + " ".join(u.get("food_drink", []))).lower()
    return any(w in hay for w in comp["lexicon"]) or (
        comp["activity"] and comp["activity"] in u.get("activity", [])
    )


def build_compilations(units, work_dir, cfg, used, start_index=0) -> list[dict]:
    out = []
    idx = start_index
    tmax = cfg["parts"].get("target_max_s", 180)
    min_seg = cfg["clip"].get("min_seg_s", 3.5)
    for comp in _compilation_defs(cfg):
        hits = [u for u in units if _matches_compilation(u, comp)]
        if len(hits) < 3:
            continue
        hits.sort(key=lambda u: (u["source_order"], u["start"]))
        segs, t = [], 0.0
        for u in hits:
            if t >= tmax:
                break
            s, e = u["start"], u["end"]
            if u["has_speech"]:
                tr = tr_mod.load_transcript(u["source_id"], work_dir) or {"segments": []}
                s, e = _snap_to_sentences(tr, s, e)
                s = max(u["start"], s)
                e = min(u["end"], e)
            if e - s < min_seg:
                e = min(u["source_duration"], s + min_seg)
            if e - s > 18:
                e = s + 18
            if any(sid == u["source_id"] and _overlap_frac(s, e, a, b) > 0.5 for sid, a, b in used):
                continue
            segs.append(
                {
                    "source_id": u["source_id"],
                    "source_path": u["source_path"],
                    "source_order": u["source_order"],
                    "start": round(s, 2),
                    "end": round(e, 2),
                    "has_speech": u["has_speech"],
                    "location_label": (u.get("landmark") or ""),
                }
            )
            used.append((u["source_id"], round(s, 2), round(e, 2)))
            t += e - s
        segs = _no_rewind(segs)
        t = sum(x["end"] - x["start"] for x in segs)
        if t < 18:
            continue
        idx += 1
        out.append(
            {
                "index": idx,
                "type": "SUPERCUT",
                "format": "food",
                "theme": comp["title"],
                "title": comp["title"],
                "slug": _slugify(comp["title"]),
                "hook_text": comp["title"],
                "description": comp["title"],
                "hashtags": [comp["hashtag"], "#travel"],
                "segments": segs,
                "chapters": [],
                "duration": round(t, 2),
                "music": False,
                "cover_offset_s": 1.0,
                "score": 7.5,
                "n_sources": len({s["source_id"] for s in segs}),
                "rationale": "",
            }
        )
    return out


# --------------------------- orchestration ---------------------------


def select_reels(manifest: dict, work_dir: Path, cfg: dict, prompts_dir: Path) -> list[dict]:
    if cfg.get("output_mode", "parts") == "parts":
        return select_parts(manifest, work_dir, cfg)
    units = build_units(manifest, work_dir)
    (work_dir / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2))
    print(f"  {len(units)} units built.")
    if not units:
        return []
    by_uid = {u["uid"]: u for u in units}

    print(
        f"  storyline: analysing the whole trip -> Ollama ({cfg['ollama']['model']}) ...",
        flush=True,
    )
    sl = storyline(units, cfg, prompts_dir)
    plans = sl.get("reels", []) if isinstance(sl, dict) else []
    if not plans:
        print("  ! storyline returned nothing -> using deterministic fallback plan.")
        plans = _fallback_plans(units, cfg)
        sl = {"reels": plans, "trip_summary": "(fallback)"}
    (work_dir / "storyline.json").write_text(json.dumps(sl, ensure_ascii=False, indent=2))
    print(f"  storyline: {len(plans)} clip plans + summary: {sl.get('trip_summary', '')[:80]}")

    reels = []
    used: list = []  # global used ranges (dedup: do not reuse the same footage)
    # Process SUPERCUTs first so they claim their thematic clips, then place/hero.
    order = sorted(
        enumerate(plans[: cfg["clip"].get("max_count", 12)]),
        key=lambda kv: 0 if str(kv[1].get("type", "")).upper() == "SUPERCUT" else 1,
    )
    for n, (_i, plan) in enumerate(order, 1):
        print(
            f"  storyboard {n}/{len(order)}: {plan.get('theme', '')[:40]} "
            f"({plan.get('type', '')}) ...",
            flush=True,
        )
        try:
            r = storyboard_reel(plan, units, by_uid, work_dir, cfg, prompts_dir, used)
        except Exception as e:
            print(f"    ! storyboard error: {e}")
            r = None
        if r:
            reels.append(r)

    for i, r in enumerate(reels, 1):
        r["index"] = i
        r["slug"] = _slugify(r["theme"] or f"clip{i}")
    out = work_dir / "selection.json"
    out.write_text(json.dumps(reels, ensure_ascii=False, indent=2))
    print(f"  director: {len(reels)} clips produced -> {out}")
    return reels


def load_selection(work_dir: Path) -> list[dict]:
    p = work_dir / "selection.json"
    return json.loads(p.read_text()) if p.exists() else []
