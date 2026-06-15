# Contributing to ReelWeave

Thanks for your interest! ReelWeave targets **Apple Silicon + local models**, so most work happens
on a Mac with `ffmpeg` (libass) and `ollama` installed.

## Setup

```bash
uv sync                       # create the environment
uv run reelweave setup        # pull local models
uv run ruff check . && uv run ruff format --check .
```

## Guidelines

- Keep the **understand → direct → render** separation: the renderer stays deterministic; only the
  director talks to the LLM.
- Run `ruff check`/`ruff format` before opening a PR. Code and comments are in English.
- Don't commit footage, renders, models or large binaries (see `.gitignore`).
- Small, focused PRs with a clear description are easiest to review.

## Reporting issues

Use the issue templates. For bugs, include your macOS/chip, `ffmpeg -version`, the failing command
and the relevant log output.
