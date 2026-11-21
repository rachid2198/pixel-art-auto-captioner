# CLAUDE.md

@AGENTS.md

## Where to start

- `SPEC.md` is the source of truth. Read §0.4 (decisions R1–R13) first, then only the sections the current step names.
- `TODO.md` is the ledger. §2.1 lists the v1.2 steps; work only on the one marked ▶.

## Claude Code on this PC

Shell is PowerShell; the repo is `F:\Research\Code\pixel-art-auto-captioner`. Always use the project venv, never a global Python.

```powershell
# CPU test suite (must stay green)
.\.env\Scripts\python.exe -m pytest tests -q

# GPU-gated tests (RTX 3060, JoyCaption weights in .\Models)
$env:JOYCAPTION_MODEL_PATH=".\Models"; .\.env\Scripts\python.exe -m pytest tests/test_model_gpu.py -q -m gpu

# Captioning CLI (dry run prints the plan as JSON, no model load)
.\.env\Scripts\python.exe scripts\run_caption.py --config configs\local_hf_config.json --dry_run
```

Permissions are in `.claude/settings.json` (shared) and `.claude/settings.local.json` (this machine only, git-ignored).
