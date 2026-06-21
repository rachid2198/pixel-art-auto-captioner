# AGENTS.md — Operational Guardrails for Pi

**Project:** Pixel Art Auto-Captioner  
**Specification:** `SPEC.md` (version 1.0, finalized 2026-06-16)  
**Role:** Implementation Agent operating under strict constraints

---

## 0. Authority

You are an **implementation agent** executing the architecture defined in `SPEC.md`. You do not change the architecture. You do not invent new modules, data types, interfaces, or output formats. If you find a gap, ambiguity, or conflict in the spec, **stop and flag it** — do not resolve it unilaterally.

`SPEC.md` is the single source of truth for:
- Module boundaries and responsibilities
- Data types and their fields
- Function signatures and return types
- Configuration schema
- Output formats
- Error handling semantics
- Implementation sequence

---

## 1. What You Are Allowed To Do

| Action | Rule |
|--------|------|
| Create files under `src/pixel_art_auto_captioner/` | Only the modules listed in SPEC `§13` |
| Create files under `tests/` | One test file per module, per SPEC `§10.2` |
| Create `scripts/run_caption.py` | Thin CLI entry point per SPEC `§9.1` |
| Create `configs/example_config.json` | Example config per SPEC `§8.2` |
| Modify `requirements.txt` | Add `pytest`, remove `torchaudio` per SPEC `§14` |
| Run Python from the project virtual environment | Use `./.env/Scripts/python.exe` |
| Create `.pi/skills/pixel-art-captioning-project/SKILL.md` | Project skill prompt |
| Modify `TODO.md` | State ledger — mark steps complete, record test results |

---

## 2. What You Are NOT Allowed To Do

| Prohibition | Rationale |
|-------------|-----------|
| **Modify `main.py`** | Reference prototype, preserved verbatim (SPEC `§1.2`) |
| **Add new modules outside `src/pixel_art_auto_captioner/`** | No code outside the package (SPEC `§13`) |
| **Add new public data types** | `ImageRecord` and `CaptionRecord` are the only data types (SPEC `§2.3`) |
| **Change function signatures from the spec** | Spec defines the exact signatures (SPEC `§3-6`) |
| **Change the default prompt text** | Must match `main.py` exactly (SPEC `§8.3`) |
| **Add multi-GPU support** | Out of scope for v1.0 (SPEC `§1.3`) |
| **Add support for VLMs other than JoyCaption** | Out of scope (SPEC `§1.3`) |
| **Use `print()` instead of `logging`** | SPEC `§12.2` |
| **Hard-code local paths** | Use config/CLI, not hard-coded paths (SPEC `§11.4`) |
| **Add new dependencies without justification** | SPEC `§14` is definitive |
| **Delete or move existing `main.py` or `.pi/` files** | Preserve existing project state |
| **Skip tests** | Every module must have tests (SPEC `§10`) |
| **Merge multiple implementation steps into one commit** | Execute steps sequentially per SPEC `§15` |
| **Write code for step N without `TODO.md` confirmation** | State Ledger Rule — see Section 4 |

---

## 3. Implementation Sequence (Mandatory Order)

Follow SPEC `§15` exactly. One step at a time. Do not proceed to step N+1 until step N is complete and tested.

| Step | File to write | Test file | GPU Required? | Target Engine |
|------|---------------|-----------|:---:|:---:|
| 1 | `common/types.py` | `test_types.py` | No | `qwen/qwen3-coder:free` |
| 2 | `common/image_utils.py` | `test_image_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` |
| 3 | `ingestion/dataloader.py` | `test_dataloader.py` | No | `openrouter/deepseek/deepseek-v4-flash` |
| 4 | `common/export_utils.py` | `test_export_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` |
| 5 | `captioning/base.py` | (via step 6) | No | `qwen/qwen3-coder:free` |
| 6 | `captioning/joycaption.py` | `test_model.py` | Yes | `openrouter/deepseek/deepseek-v4-pro` |
| 7 | `batch/runner.py` | `test_runner.py` | Yes | `openrouter/deepseek/deepseek-v4-flash` |
| 8 | `scripts/run_caption.py` | Manual | Yes | `openrouter/deepseek/deepseek-v4-flash` |
| 9 | `configs/example_config.json` | — | No | `qwen/qwen3-coder:free` |

- **GPU-free steps (1-4, 5, 9):** Fully implement and test now.
- **GPU-requiring steps (6-8):** Implement code, verify syntax and imports. Mark tests with `@pytest.mark.gpu` and design them to `pytest.skip` when CUDA is unavailable.

---

## 4. Per-Step Execution Protocol

For each implementation step:

1. **Read** the relevant section of `SPEC.md` again before writing any code.
2. **Write the module file** with the exact signatures and behaviour specified.
3. **Write the test file** with the tests listed in SPEC `§10.3`.
4. **Run the tests** (GPU-free steps only; for GPU steps, verify syntax and imports).
5. **Report** exactly what files were created/modified and test results.
6. **Update `TODO.md`** — mark the step complete, record test output, declare the next step.

Do not combine multiple steps. Each step is a discrete task.

### 4.1 State Ledger Rule (MANDATORY)

**You are strictly forbidden from writing code for step N unless `TODO.md` explicitly shows all preceding steps as `[x]` completed.** This is non-negotiable. The ledger exists to maintain state continuity across rolled-over context windows where you have no memory of prior work.

On every single turn, before writing any implementation code:

1. **Read `TODO.md`** to establish which step is currently targeted.
2. **Verify** all previous steps are marked `[x]` with test results documented.
3. **If a step is marked complete but lacks test output**, stop and request the user provide the missing results before proceeding.
4. **Immediately upon finishing a step and running its tests**, modify `TODO.md` to:
   - Check off the item (`[ ]` → `[x]`)
   - Document exact test results (pass/fail counts, any terminal output)
   - Declare the next targeted file
5. **Never skip or combine steps.** If `TODO.md` shows step 3 is next, you write step 3 — not step 4.

### 4.2 Iterative Workflow Improvement Clause

Immediately upon completing step N and running its unit tests, **before** marking it as complete in `TODO.md`, you must output a concise 2-sentence **Friction &amp; Streamlining Note** inside the `TODO.md` completion log. The note must explicitly cover:

1. **Friction point:** What was the single biggest mechanical or structural friction point during this coding step (e.g., unexpected library behaviour, type mismatch, platform-specific path parsing)?
2. **Streamlining guardrail:** What specific guardrail could be added to `AGENTS.md` or `SPEC.md` to ensure an agent executing this code in the future avoids this block?

This clause creates a self-improving feedback loop: every step that costs time leaves a trail that prevents the same cost from recurring.

---

## 5. Coding Standards

### 5.1 Imports
- Use absolute imports from the package root: `from pixel_art_auto_captioner.common.types import ImageRecord`
- In `__init__.py` files, re-export public classes for convenient imports.

### 5.2 Type hints
- All public functions must have complete type annotations (SPEC defines them).
- Use `Path` from `pathlib`, not `str` for file paths.

### 5.3 Docstrings
- Every public class and function must have a docstring.
- Include parameter descriptions and return type explanations.

### 5.4 Logging
- Use `import logging; logger = logging.getLogger(__name__)`.
- Format: `"%(asctime)s  %(levelname)-8s  %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`.
- Log levels: INFO for progress, WARNING for skips, ERROR for failures.

### 5.5 Error handling
- **Fail fast:** Config errors and model load errors exit immediately.
- **Resilient:** Per-image errors are logged and the batch continues.
- **Cleanup:** Use `try` / `finally` to ensure `model.unload()` is always called.

---

## 6. GPU Awareness

- This project targets GPUs with >= 12 GB VRAM (SPEC `§11.3`).
- The reference `main.py` runs on an RTX 3060 with NF4 quantization.
- When writing GPU code, **always** guard device moves and CUDA calls.
- Tests that require GPU must be decorated with `@pytest.mark.gpu` and skip gracefully.

---

## 7. Testing Requirements

### 7.1 Framework
- `pytest` (add to `requirements.txt` or `requirements-dev.txt`).
- Run with: `./.env/Scripts/python.exe -m pytest tests/ -v`

### 7.2 Test fixture location
- All shared fixtures go in `tests/conftest.py` (SPEC `§10.4`).
- Use `tmp_path` from pytest for temp directories and files.

### 7.3 Test isolation
- Tests must not depend on real images, real models, or network access (except GPU tests which need the model).
- GPU-free tests use synthetic PIL images created via `Image.new()`.

### 7.4 Coverage requirement
- Every public function in every module must have at least one test.
- Tests must cover both success and error paths where feasible.

---

## 8. Package Boundaries (Do Not Cross)

The package enforces a strict unidirectional dependency graph:

```
common  <--  ingestion  <--  batch
common  <--  captioning <--  batch
```

- `common` has **no internal dependencies** — it is the leaf module.
- `ingestion` depends **only** on `common`.
- `captioning` depends **only** on `common` (via `PIL.Image.Image` and shared types).
- `batch` depends on `common`, `ingestion`, and `captioning` — it is the orchestrator.

**No circular imports. `ingestion` must not import from `captioning`. `common` must not import from any other module.**

```
src/pixel_art_auto_captioner/
├── __init__.py              # version, package docstring
├── common/                  # shared types + utilities (leaf module)
│   ├── __init__.py
│   ├── types.py             # ImageRecord, CaptionRecord
│   ├── image_utils.py       # load_image, validate_image
│   └── export_utils.py      # save_txt_sidecar, save_jsonl_entry, build_record
├── ingestion/               # image discovery + loading
│   ├── __init__.py
│   └── dataloader.py        # ImageDataLoader
├── captioning/              # model loading + inference
│   ├── __init__.py
│   ├── base.py              # CaptionModel ABC
│   └── joycaption.py        # JoyCaptionModel
└── batch/                   # orchestration runner
    ├── __init__.py
    └── runner.py            # CaptionRunner
```

---

## 9. Virtual Environment

All Python commands execute inside the project virtual environment. The source of truth for dependencies is `requirements.txt`.

**Primary entry point:** `./.env/Scripts/python.exe`

```bash
# Verify Python version
./.env/Scripts/python.exe --version

# Install runtime dependencies
./.env/Scripts/python.exe -m pip install -r requirements.txt

# Install test dependencies
./.env/Scripts/python.exe -m pip install pytest

# Run all tests
./.env/Scripts/python.exe -m pytest tests/ -v

# Run a single test file
./.env/Scripts/python.exe -m pytest tests/test_types.py -v
```

---

## 10. Git Hygiene

- Do not commit `Models/`, `input/`, `output/`, `data/`, `.env/`, `.pi/sessions/`.
- Do not commit model weights, generated captions, or private datasets.
- Do not commit API keys.
- The `.gitignore` already covers these — do not weaken it.

---

## 11. Final Guardrail

> **If you are unsure whether an action is allowed, consult `SPEC.md`. If `SPEC.md` does not cover it, stop and ask. Never assume.**
