#!/usr/bin/env python3
"""Pull/prepare the local models ReelWeave needs.

- Ollama director LLM (and the optional Ollama vision fallback) via `ollama pull`.
- MLX models (Whisper, Qwen3-VL) download automatically from Hugging Face on first run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "examples" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    models = [cfg["ollama"]["model"]]
    if cfg["vision"].get("engine") == "ollama":
        models.append(cfg["vision"]["model"])

    for m in models:
        print(f"==> ollama pull {m}")
        try:
            subprocess.run(["ollama", "pull", m], check=True)
        except Exception as e:
            print(f"  ! failed: {e} — is Ollama installed and running? (https://ollama.com)")

    print("\nMLX models download on first use:")
    print(f"  - ASR:    {cfg['transcribe']['model_repo']}")
    print(f"  - vision: {cfg['vision'].get('mlx_model', '')}")
    print("\nTip: run `python scripts/generate_sfx.py` to synthesise transition SFX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
