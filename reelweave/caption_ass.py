"""Render helper — caption_ass: animated karaoke captions + optional cards (ASS/libass).

- Bottom-centre karaoke: the spoken word turns yellow and POPS (scale bounce).
- Optional top hook card (scale-in + fade) for the first few seconds.
- Optional bottom-left location pill and centred chapter cards.
seg_word_groups: each segment's words, shifted onto the FINAL timeline.
cards: [{text, t_in, t_out}] location cards (timeline seconds).
chapters: [{t, title, sub}] full-screen chapter cards (timeline seconds).
"""

from __future__ import annotations

import re
from pathlib import Path


def _hex(hexrgb: str, alpha="00") -> str:
    h = hexrgb.lstrip("#")
    return f"&H{alpha}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _ts(t: float) -> str:
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(t: str) -> str:
    return t.replace("\\", "").replace("{", "").replace("}", "").strip()


def _strip_emoji(t: str) -> str:
    # Keep Latin (incl. Turkish/Bosnian diacritics) and basic punctuation; drop emoji.
    return re.sub(r"[^\w\sÀ-ÿĀ-žŠšČčĆćĐđŽž.,!?'\-:]", "", t).strip()


def _chunk(words, n):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= n:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def build_ass(hook_text, duration, seg_word_groups, cards, cfg, out_path: Path, chapters=None):
    c = cfg["captions"]
    v = cfg["video"]
    sz = cfg["safezone_px"]
    eff = cfg.get("effects", {})
    white = _hex(c["primary_color"])
    yellow = _hex(c["highlight_color"])
    outc = _hex("000000")
    font = c["font"]
    size = c["font_size"]
    mv = c.get("margin_v", 430)
    pop = eff.get("caption_word_pop", True)

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {v["w"]}
PlayResY: {v["h"]}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{white},{yellow},{outc},&H64000000,-1,0,0,0,100,100,0,0,1,{c["outline"]},{c["shadow"]},2,{sz["left"]},{sz["right"]},{mv},1
Style: Hook,{font},{int(size * 0.96)},{white},{white},{outc},&HB4000000,-1,0,0,0,100,100,0,0,3,5,2,8,120,120,250,1
Style: Loc,{font},{int(size * 0.6)},{white},{white},{outc},&HB8101010,0,0,0,0,100,100,0,0,4,22,0,1,70,70,640,1
Style: Part,{font},{int(size * 1.4)},{white},{white},{outc},&HC8000000,-1,0,0,0,100,100,0,0,3,6,3,5,120,120,0,1
Style: PartSub,{font},{int(size * 0.7)},{yellow},{yellow},{outc},&H00000000,-1,0,0,0,100,100,0,0,1,4,2,5,120,120,-140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []

    # Hook card (top, large, scale-in + fade)
    hook = _esc(hook_text or "")
    if hook:
        hold = min(duration, max(cfg["hook"].get("min_text_hold_s", 2.0), 3.0))
        ev.append(
            f"Dialogue: 2,{_ts(0)},{_ts(hold)},Hook,,,,,,"
            f"{{\\fad(120,250)\\fscx60\\fscy60\\t(0,220,\\fscx100\\fscy100)}}{hook}"
        )

    # Location cards (bottom-left pill, above the captions — never collides with the hook)
    for cd in cards or []:
        txt = _strip_emoji(str(cd.get("text", "")))
        if not txt:
            continue
        ti, to = float(cd["t_in"]), min(float(cd["t_out"]), float(cd["t_in"]) + 2.4)
        if to <= ti:
            continue
        ev.append(
            f"Dialogue: 1,{_ts(ti)},{_ts(to)},Loc,,,,,,"
            f"{{\\an1\\fad(150,200)\\move(-280,1280,{sz['left'] + 8},1280,0,180)}}"
            f"  {txt.upper()}"
        )

    # Part / chapter cards (full-screen centre, scale-in + fade)
    for ch in chapters or []:
        t = float(ch.get("t", 0))
        title = _esc(str(ch.get("title", "")))
        sub = _esc(str(ch.get("sub", "")))
        if not title:
            continue
        dur_c = 2.2
        ev.append(
            f"Dialogue: 3,{_ts(t)},{_ts(t + dur_c)},Part,,,,,,"
            f"{{\\fad(250,300)\\fscx70\\fscy70\\t(0,250,\\fscx100\\fscy100)}}{title.upper()}"
        )
        if sub:
            ev.append(f"Dialogue: 3,{_ts(t)},{_ts(t + dur_c)},PartSub,,,,,,{{\\fad(250,300)}}{sub}")

    # Bottom karaoke (word pop)
    maxw = c.get("max_words_per_line", 4)
    for words in seg_word_groups:
        allchunks = _chunk(words, maxw)
        for ci, chunk in enumerate(allchunks):
            cend = chunk[-1]["end"]
            if ci + 1 < len(allchunks):
                cend = min(cend + 0.4, allchunks[ci + 1][0]["start"])
            for wi, w in enumerate(chunk):
                es = w["start"]
                ee = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else cend
                if ee <= es:
                    ee = es + 0.12
                parts = []
                for k, ww in enumerate(chunk):
                    tok = _esc(ww["w"])
                    if k == wi and pop:
                        parts.append(
                            f"{{\\c{yellow}\\fscx82\\fscy82"
                            f"\\t(0,90,\\fscx118\\fscy118)\\t(90,180,\\fscx100\\fscy100)}}"
                            f"{tok}{{\\c{white}\\fscx100\\fscy100}}"
                        )
                    elif k == wi:
                        parts.append(f"{{\\c{yellow}}}{tok}{{\\c{white}}}")
                    else:
                        parts.append(tok)
                ev.append(f"Dialogue: 0,{_ts(es)},{_ts(ee)},Cap,,,,,,{' '.join(parts)}")

    if not ev:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")
    return out_path
