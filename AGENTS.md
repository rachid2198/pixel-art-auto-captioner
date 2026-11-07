# AGENTS.md — Working Rules for Implementation Agents

**Project:** Pixel Art Auto-Captioner
**Specification:** `SPEC.md` v1.1 (2026-11-06)
**Progress ledger:** `TODO.md`
**Applies to:** any coding agent working on this repo (currently Claude; previously Pi/DeepSeek/Qwen).

---

## 0. Authority & change control

1. `SPEC.md` is the source of truth for architecture, interfaces, config, output formats and error semantics.
2. **The user approves every change to the spec.** An agent may:
   - **Implement** what the spec says.
   - **Choose** internal details the spec leaves open (private helpers, variable names, log wording), as long as they stay inside the spec.
   - **Propose** anything else (new features, different models, new dependencies, interface changes): write it to `TODO.md → Proposals` with motivation, cost and alternatives, then ask the user, **one proposal at a time**. Do not implement a proposal until the user accepts it. When accepted, update `SPEC.md` first, then the code.
3. If the spec is ambiguous or wrong, stop and ask. Record the question in `TODO.md → Open questions`.
4. Small deviations forced by reality (e.g. a library API differs from the spec) are allowed if they keep the spec's intent. Record each one in the step's TODO entry under **Deviations** and tell the user.

---

## 1. Working on the user's computer

The repo and dataset live on the user's Windows machine (`F:\Research\Code\pixel-art-auto-captioner`, `F:\Work\datasets\pixel-art`). When an agent operates there:

- **Narrate every step** as it happens (what you are about to read, run or write, and why).
- **Change only what the current step needs.** Never modify, move or delete anything else: no dataset files, `main.py`, `.env/`, `Models/`, `input/` or `output/`.
- **The dataset is read-only.** No pipeline step writes inside `input_dir`.
- **Ask before any new access** (new folders, installing packages, network, credentials).
- **Git:** do not commit, branch, push or rewrite history unless the user asks. When asked, use the branch `v1.1` and one commit per TODO step.
- Large scans of the dataset (≥ 280k images + ~1M msgpack files) are slow. Use bounded commands (`timeout`, `head`, per-folder sampling) and say so.

---

## 2. Per-step protocol

1. **Read `TODO.md`.** Work only on the step marked ▶ *next*, and only after all prerequisite steps are ✅. If prior results are missing, ask.
2. **Re-read the relevant `SPEC.md` sections** before writing code.
3. **Implement the module and its tests together.** Tests must cover the success paths, the error paths and every item listed for that module in SPEC §13.
4. **Run the CPU suite.** It must be green with no `torch`, `transformers` or `vllm` installed (use the Linux sandbox or the local `.env`).
5. **Update `TODO.md`:** status, test counts (`N passed, M skipped`), files touched, any **Deviations**, and the next step.
6. **Report to the user in 3–6 lines** (what changed, test result, anything needing a decision) and wait for the go-ahead before the next step, unless the user has approved a batch of steps.

**Transition rule (v1.0 → v1.1 rework, steps 1–10).** Reworking a shared module (e.g. `types.py` in step 1) breaks v1.0 modules and tests that a *later* step will rewrite. Do not patch those modules ahead of their step. Instead:

- Skip the broken v1.0 tests, naming the step that rewrites them: `reason="v1.0 module; reworked in v1.1 step N"`.
  - If most of a test module breaks, use a module-level `pytestmark = pytest.mark.skip(...)`.
  - Otherwise mark the individual broken tests (or test classes) with `@pytest.mark.skip(...)`.
- **Never delete** v1.0 tests before their step; the step that rewrites the module replaces them.
- **Only skip what actually breaks.** Tests that still pass stay enabled.
- Shared fixtures in `tests/conftest.py` are updated to the new types in the step that changes the types.
- List every skipped test module in the step's TODO entry.

The CPU suite counts as green when the new tests pass and every skip is one of these documented transition skips.

There is no "friction note" requirement. If something cost real time and could recur, add a short entry to §6 *Known pitfalls* below.

---

## 3. Coding standards

- Python ≥ 3.10. Absolute imports (`from pixel_art_auto_captioner.common.types import ImageRecord`).
- Full type hints on public functions; `pathlib.Path` for paths in memory, `as_posix()` strings on disk.
- Docstrings on every public class and function (params, returns, raises).
- `logging` only, never `print()` (the CLI's `--dry_run` JSON plan goes to stdout via `sys.stdout.write`). `logger = logging.getLogger(__name__)`.
- **Heavy imports are lazy:** `torch`/`transformers` only in `joycaption_hf.py`, `vllm` only inside `JoyCaptionVLLMModel.load()`, never at package import time.
- No hard-coded local paths. Everything comes from config/CLI.
- No new dependencies without an accepted proposal.

---

## 4. Package boundaries

```text
common  ◄──  ingestion  ◄──  batch
common  ◄──  captioning ◄──  batch
```

- `common` imports nothing from the package and never imports torch/vllm.
- `ingestion` and `captioning` never import each other.
- `batch` is the only orchestrator. Scripts in `scripts/` only wire modules together.

---

## 5. Testing rules

- From step 13 on: install once with `pip install -e ".[dev]"` (pyproject, SPEC §15); tests import the installed package and never edit `sys.path`. Until step 13, `tests/conftest.py` keeps adding `src/` to `sys.path`.
- Run: `python -m pytest tests -q` (local Windows: `./.env/Scripts/python.exe -m pytest tests -q`; Linux: `python3 -m pytest tests -q`).
- Fixtures in `tests/conftest.py`; synthetic images via `PIL.Image.new()`; temp dirs via `tmp_path`.
- No network, no real model, no real dataset in the CPU suite.
- Backends are tested with **fakes** (fake processor/model, fake vLLM `LLM`), injected via constructor parameters or monkeypatch, never by importing the real library.
- GPU tests: `@pytest.mark.gpu` + **dual guard** (§6.3).

---

## 6. Known pitfalls

### 6.1 PIL `verify()` closes the handle

`Image.verify()` leaves the image unusable for pixel access. Keep `validate_image()` and `load_image()` as separate code paths with separate `Image.open()` calls. In `load_image`, call `.convert("RGB")` *inside* the `with Image.open(...)` block so the pixels are decoded before the file closes.

### 6.2 Boundary tests need sibling temp dirs

When testing "path outside root" errors, create `tmp_path/"root"` and `tmp_path/"outside"` as **siblings**. A file nested under the root never triggers the error.

### 6.3 Dual guard for GPU tests

```python
_MODEL_PATH = Path(os.environ.get("JOYCAPTION_MODEL_PATH", "./Models"))
_MODEL_AVAILABLE = (_MODEL_PATH / "config.json").exists()
requires_cuda  = pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
requires_model = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="JoyCaption weights not found")
```

`_cuda_available()` must import torch lazily and return `False` if torch is missing.

### 6.4 Relative path is the only unique key

Stems like `000001` repeat across ~110 source folders. Never key, name or deduplicate by stem alone. Use `rel_path` (with extension).

### 6.5 Linux is case-sensitive

`Path.glob("*.jpg")` misses `.JPG` on Linux (the cloud) but not on Windows (local), so tests pass locally and files vanish in the cloud. Always compare `suffix.lower()`.

### 6.6 HF LLaVA chat handling is fragile

Use exactly `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` and then `processor(text=[convo], images=[image], return_tensors="pt")`, as `main.py` does. Do not tokenize inside the template call, and do not batch several conversations in one HF call.

### 6.7 Reproducibility is best-effort

Same seed + same backend + same versions + same GPU type ≈ same text. vLLM output can still differ when batch composition changes. Never write tests that assert bit-identical generated text from a real model.

### 6.8 Resume reads JSONL, not `.txt`

`.txt` files are optional exports. Never use their existence to decide what is done.

---

## 7. Environments

| Where | What | How |
|---|---|---|
| Local Windows (RTX 3060) | HF backend, NF4 | `.env` venv: torch 2.12.0+cu130, transformers 5.12, bitsandbytes 0.49.2, accelerate 1.13; `./.env/Scripts/python.exe` |
| Agent sandbox (Linux, CPU) | CPU test suite | `python3 -m pytest tests -q`; torch/vllm absent by design |
| Cloud GPU | vLLM backend, bf16 | Docker image from `docker/Dockerfile` (SPEC §11.2) |

---

## 8. Git hygiene

- Never commit `Models/`, `input/`, `output/`, `data/`, `.env/`, `.pi/sessions/`, caption outputs, datasets, model weights or tokens.
- Keep `.gitignore` at least as strict as it is now.
- `main.py` is never modified.

---

## 9. Reviews

External adversarial review (`REVIEW.md`, previously done with GPT) is **optional** and happens on the user's request. When a review is added, the agent:

1. evaluates each critique against `SPEC.md`,
2. marks it ✅ fixed (with test name), ❌ rejected (with reason) or ⏸ deferred, and
3. updates the review's status line.

---

## 10. Final guardrail

> If the spec doesn't cover it, or an action would touch something outside the current step, stop and ask the user.
