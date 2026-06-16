# Project Instructions for Pi Agents

This repository is being prepared as a **pixel-art dataset auto-captioning** project. The long-term pipeline will support image ingestion, caption generation, model comparison, batch processing, and dataset export. For now, keep changes small and scaffold-first.

## Current phase

- Setup, documentation, and architecture only.
- Do **not** add new caption generation logic unless the user explicitly asks for an implementation step.
- Treat `main.py` as the current prototype/legacy script. Do not rewrite it during scaffolding tasks unless asked.

## Agent role split

Use two lightweight roles inside Pi:

1. **Planning agent** (`deepseek-v4-pro` style)
   - Use for architecture, task breakdowns, risk review, and acceptance criteria.
   - Prefer read-only inspection and written plans.
   - Output small implementation tickets.

2. **Implementation agent** (`deepseek-v4-flash` style)
   - Use for executing an approved, narrow ticket.
   - Make minimal diffs.
   - Run targeted checks where practical.
   - Report exactly what changed.

Pi does not include built-in sub-agents by default, so this project uses prompt templates, skills, and workflow docs to make the split explicit.

## Development guardrails

- Keep the project incremental and easy to revert.
- Prefer documentation and interfaces before model-specific implementation.
- Avoid hard-coding local machine paths except in examples.
- Keep large/generated assets out of git (`Models/`, `input/`, `output/`, dataset contents).
- Do not commit API keys, model weights, generated captions, or private datasets.
- Add tests before or alongside real implementation work.

## Expected commands

Use the Windows virtual environment when running Python from this WSL-mounted workspace:

```bash
./.env/Scripts/python.exe --version
./.env/Scripts/python.exe -m pip list
```

No project test command is established yet. Add one when the first testable implementation lands.

## Folder conventions

- `src/pixel_art_auto_captioner/` - future importable package skeleton.
- `docs/` - architecture, workflow, project state, and dataset layout notes.
- `.pi/` - Pi project settings, skills, prompt templates, and workflow docs.
- `configs/` - future checked-in example config files only.
- `data/` - local dataset workspace; contents ignored except `.gitkeep`.
- `Models/`, `input/`, `output/` - existing local prototype assets; ignored by git.
