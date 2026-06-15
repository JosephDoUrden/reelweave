# Examples

- **`config.yaml`** — a worked localisation (Turkish narration, Sarajevo). Start from the
  repository-root `config.yaml` and override the keys shown here for your own trip:
  set `language`, `location`, `paths.source_glob`, the compilation lexicons, and (optionally)
  `transcribe.name_fixes` plus a `prompts/initial_prompt.txt` vocabulary hint.

To localise transcription accuracy for proper nouns, create `prompts/initial_prompt.txt` with a
short line of place names/brands you expect, e.g.:

```
A trip to Sarajevo. Baščaršija, Sebilj, Miljacka, Latin Bridge, Vijećnica. ćevapi, burek.
```

> Sample output clips are **not** committed (private footage). Drop your own clips into
> `footage/` and run `reelweave run`.
