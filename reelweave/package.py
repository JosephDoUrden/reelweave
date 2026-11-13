"""Stage 8 — package: write caption.txt, meta.json and the review dashboard (index.html)."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

# Attribution lines for bundled example tracks (only emitted if that track is used).
_MUSIC_CREDIT = {
    "bff_the_great": '🎵 "The Great" — Broke For Free (CC BY 3.0)',
    "bff_night_owl": '🎵 "Night Owl" — Broke For Free (CC BY 3.0)',
    "komiku_battle": '🎵 Komiku — "Battle Theme" (CC0, public domain)',
    "happy_pop": "🎵 Royalty-free music (Pixabay)",
}


def _caption_text(clip: dict) -> str:
    tags = " ".join(clip.get("hashtags", []))
    desc = clip.get("description", "").strip()
    out = f"{desc}\n\n{tags}\n".strip()
    used = clip.get("_music_used", "")
    if used:
        credit = _MUSIC_CREDIT.get(used.rsplit(".", 1)[0])
        if credit:
            out += f"\n\n{credit}"
    return out + "\n"


def package_clip(clip: dict) -> None:
    out_dir = Path(clip["_out_dir"])
    (out_dir / "caption.txt").write_text(_caption_text(clip), encoding="utf-8")
    meta = {k: v for k, v in clip.items() if not k.startswith("_")}
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rel(path: str | None, base: Path) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path or ""


def build_index(clips: list[dict], index_html: Path) -> None:
    base = index_html.parent
    cards = []
    for c in sorted(clips, key=lambda x: x["index"]):
        mp4 = _rel(c.get("_out_mp4"), base)
        cover = _rel(c.get("_cover"), base)
        tags = " ".join(c.get("hashtags", []))
        cap = html.escape(c.get("description", ""))
        music = " · 🎵 music" if c.get("music") else ""
        cards.append(f"""
    <div class="card">
      <video controls preload="none" poster="{html.escape(cover)}" src="{html.escape(mp4)}"></video>
      <div class="meta">
        <div class="row"><span class="badge">{c["index"]:02d}</span>
          <span class="fmt">{html.escape(c.get("format", ""))}</span>
          <span class="score">★ {c.get("score", "")}</span>
          <span class="dur">{c.get("duration", 0):.0f}s</span></div>
        <h3>{html.escape(c.get("theme") or c.get("title", ""))}</h3>
        <p class="hook">🪝 {html.escape(c.get("hook_text", ""))}</p>
        <p class="why">{html.escape(c.get("rationale", ""))}</p>
        <textarea readonly rows="4">{cap}\n\n{html.escape(tags)}</textarea>
        <p class="src">🎬 {len(c.get("segments", []))} cuts · {c.get("n_sources", 1)} source clips{music}</p>
      </div>
    </div>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReelWeave — Review</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d0d10;color:#eee;margin:0;padding:24px}}
  h1{{font-weight:800}}
  .sub{{color:#9aa;margin-bottom:20px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
  .card{{background:#16161c;border-radius:14px;overflow:hidden;border:1px solid #26262e}}
  video{{width:100%;aspect-ratio:9/16;background:#000;display:block}}
  .meta{{padding:12px}}
  .row{{display:flex;gap:8px;align-items:center;font-size:13px;color:#9aa;margin-bottom:6px}}
  .badge{{background:#ffd60a;color:#000;font-weight:800;border-radius:6px;padding:1px 7px}}
  .fmt{{background:#26262e;border-radius:6px;padding:1px 7px}}
  .score{{color:#ffd60a}}
  h3{{margin:4px 0;font-size:16px}}
  .hook{{color:#7dd3fc;font-size:14px;margin:4px 0}}
  .why{{color:#9aa;font-size:13px;margin:4px 0}}
  textarea{{width:100%;box-sizing:border-box;background:#0d0d10;color:#cde;border:1px solid #26262e;border-radius:8px;padding:8px;font-size:12px;resize:vertical}}
  .src{{color:#667;font-size:11px;margin:6px 0 0}}
</style></head><body>
<h1>🎬 ReelWeave</h1>
<div class="sub">{len(clips)} clips · share-ready · subtitles + caption + hashtags + cover included</div>
<div class="grid">{"".join(cards)}</div>
</body></html>"""
    index_html.parent.mkdir(parents=True, exist_ok=True)
    index_html.write_text(doc, encoding="utf-8")
    print(f"  dashboard: {index_html}")


def package_all(clips: list[dict], index_html: Path) -> None:
    for c in clips:
        if c.get("_out_dir"):
            package_clip(c)
    build_index(clips, index_html)
