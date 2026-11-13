"""ReelWeave — local AI video pipeline orchestrator.

Usage:
  reelweave run                         # all stages, all files
  reelweave run --source ./footage      # override the source folder
  reelweave run --limit 2               # quick smoke test (first 2 files)
  reelweave run --stage transcribe      # a single stage
  reelweave run --from select           # from this stage onwards
  reelweave setup                       # pull/prepare the local models

Stages: probe -> transcribe -> scenes -> score -> describe -> select -> render -> package
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import yaml

from . import describe as desc_mod
from . import package as pkg_mod
from . import probe as probe_mod
from . import render as render_mod
from . import scenes as scenes_mod
from . import score as score_mod
from . import select as sel_mod
from . import transcribe as tr_mod
from . import util_ffmpeg as ff

HERE = Path(__file__).resolve().parent
STAGES = ["probe", "transcribe", "scenes", "score", "describe", "select", "render", "package"]


def _default_config() -> Path:
    """Prefer ./config.yaml in the working directory, else the repo's config.yaml."""
    cwd = Path.cwd() / "config.yaml"
    if cwd.exists():
        return cwd
    return HERE.parent / "config.yaml"


def load_cfg(config_path: Path) -> tuple[dict, dict[str, Path]]:
    cfg = yaml.safe_load(config_path.read_text())
    cdir = config_path.parent
    p = cfg["paths"]
    paths = {
        "source_dir": (cdir / p["source_dir"]).resolve(),
        "work_dir": (cdir / p["work_dir"]).resolve(),
        "clips_dir": (cdir / p["clips_dir"]).resolve(),
        "index_html": (cdir / p["index_html"]).resolve(),
        "prompts_dir": (HERE.parent / "prompts").resolve(),
    }
    return cfg, paths


def _collect_rendered(selection: list[dict], clips_dir: Path) -> list[dict]:
    out = []
    for c in selection:
        d = clips_dir / f"{c['index']:02d}_{c['slug']}"
        mp4 = d / "clip.mp4"
        if mp4.exists():
            c = dict(c)
            c["_out_dir"] = str(d)
            c["_out_mp4"] = str(mp4)
            cov = d / "cover.jpg"
            c["_cover"] = str(cov) if cov.exists() else None
            out.append(c)
    return out


def cmd_run(args) -> int:
    cfg, paths = load_cfg(Path(args.config).resolve())
    cfg["_prompts_dir"] = str(paths["prompts_dir"])
    if args.source:
        paths["source_dir"] = Path(args.source).resolve()

    print(f"ffmpeg: {ff.find_ffmpeg()}  (libass: {'yes' if ff.ffmpeg_has_libass() else 'NO ⚠️'})")
    if not ff.ffmpeg_has_libass():
        print("  ⚠️ libass missing — subtitles cannot be burned in. Install ffmpeg with libass.")

    if args.stage:
        run_stages = [args.stage]
    elif args.from_stage:
        run_stages = STAGES[STAGES.index(args.from_stage) :]
    else:
        run_stages = STAGES

    work = paths["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if "probe" in run_stages or not (work / "manifest.json").exists():
        print("\n[1/8] probe")
        manifest = probe_mod.build_manifest(
            paths["source_dir"],
            cfg["paths"]["source_glob"],
            work,
            exclude=cfg["paths"].get("exclude"),
        )
    else:
        manifest = probe_mod.load_manifest(work)

    if args.limit > 0:
        manifest["files"] = manifest["files"][: args.limit]
        print(f"  (--limit {args.limit}: {len(manifest['files'])} files)")

    if "transcribe" in run_stages:
        print("\n[2/8] transcribe (mlx-whisper)")
        tr_mod.transcribe_all(manifest, work, cfg)

    scenes_by_id = {}
    if "scenes" in run_stages:
        print("\n[3/8] scenes (PySceneDetect)")
        scenes_by_id = scenes_mod.detect_all(manifest, work, cfg)

    scores_by_id = {}
    if "score" in run_stages:
        print("\n[4/8] score (visual quality)")
        if not scenes_by_id:
            scenes_by_id = {
                e["id"]: scenes_mod.load_scenes(e["id"], work) for e in manifest["files"]
            }
        scores_by_id = score_mod.score_all(manifest, scenes_by_id, work, cfg)

    if "describe" in run_stages:
        print("\n[5/8] describe (vision understanding)")
        if not scores_by_id:
            scores_by_id = {
                e["id"]: score_mod.load_scores(e["id"], work) for e in manifest["files"]
            }
        desc_mod.describe_all(manifest, scores_by_id, work, cfg)

    selection = []
    if "select" in run_stages:
        print("\n[6/8] select (director: timetable / storyboard)")
        selection = sel_mod.select_reels(manifest, work, cfg, paths["prompts_dir"])
    else:
        selection = sel_mod.load_selection(work)

    if "render" in run_stages:
        print("\n[7/8] render (montage + captions + grade + encode)")
        if not selection:
            print("  ! no selection — run the 'select' stage first.")
        else:
            selection = render_mod.render_all(selection, paths["clips_dir"], work, cfg)

    if "package" in run_stages:
        print("\n[8/8] package (caption + meta + dashboard)")
        done = (
            selection
            if (selection and selection[0].get("_out_dir"))
            else _collect_rendered(selection, paths["clips_dir"])
        )
        pkg_mod.package_all(done, paths["index_html"])

    print(f"\n✅ Done ({time.time() - t0:.0f}s). Output: {paths['clips_dir']}")
    if paths["index_html"].exists():
        print(f"   Review dashboard: open '{paths['index_html']}'")
    return 0


def cmd_setup(args) -> int:
    """Pull/prepare the local models. MLX models auto-download from Hugging Face on first use."""
    cfg, _ = load_cfg(Path(args.config).resolve())
    models = [cfg["ollama"]["model"]]
    if cfg["vision"].get("engine") == "ollama":
        models.append(cfg["vision"]["model"])
    print("Pulling Ollama models:", ", ".join(models))
    for m in models:
        try:
            subprocess.run(["ollama", "pull", m], check=True)
        except Exception as e:
            print(f"  ! could not pull {m}: {e} (is Ollama installed and running?)")
    print(f"\nMLX vision model '{cfg['vision'].get('mlx_model', '')}' will download on first run.")
    print("ffmpeg:", ff.find_ffmpeg(), "(libass:", "yes)" if ff.ffmpeg_has_libass() else "NO ⚠️)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="reelweave", description="Local AI travel-vlog editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--config", default=str(_default_config()))
    r.add_argument("--source", help="source footage folder (overrides config)")
    r.add_argument("--limit", type=int, default=0, help="first N source files (smoke test)")
    r.add_argument("--stage", choices=STAGES, help="run only this stage")
    r.add_argument("--from", dest="from_stage", choices=STAGES, help="run from this stage onwards")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("setup", help="pull/prepare local models")
    s.add_argument("--config", default=str(_default_config()))
    s.set_defaults(func=cmd_setup)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
