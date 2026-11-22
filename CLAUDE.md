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

# GPU-gated tests (RTX 3060, bf16 JoyCaption weights on D:, quantized to NF4 at load)
$env:JOYCAPTION_MODEL_PATH="D:\Models\llama-joycaption-beta-one-bf16"; .\.env\Scripts\python.exe -m pytest tests/test_model_gpu.py -q -m gpu

# Captioning CLI (dry run prints the plan as JSON, no model load)
.\.env\Scripts\python.exe scripts\run_caption.py --config configs\local_hf_config.json --model_path D:/Models/llama-joycaption-beta-one-bf16 --dry_run
```

Weights on this PC: `D:\Models\llama-joycaption-beta-one-bf16` (bf16, revision `ebf414e…`). `Models\llama-joycaption-beta-one-hf-llava` in the repo is a pre-quantized NF4 copy that does not work with the HF backend (TODO C9 notes, W5); leave it alone. The shipped config stays portable, so pass `--model_path` here.

Permissions are in `.claude/settings.json` (shared) and `.claude/settings.local.json` (this machine only, git-ignored).
