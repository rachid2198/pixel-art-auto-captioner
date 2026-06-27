# TODO.md — Implementation State Ledger

**Project:** Pixel Art Auto-Captioner  
**Governed by:** `SPEC.md` `§15` | `AGENTS.md` `§4`  
**Rule:** No code for step N unless all preceding steps are `[x]` completed.

---

| Step | Status | File | Test File | GPU | Target Engine | Test Results |
|------|--------|------|-----------|:---:|:---:|--------------|
| 1 | `[x]` Complete | `common/types.py` | `test_types.py` | No | `qwen/qwen3-coder:free` | 10 passed, 0 failed |
| 2 | `[x]` Complete | `common/image_utils.py` | `test_image_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | 17 passed, 0 failed |
| 3 | `[ ]` Pending | `ingestion/dataloader.py` | `test_dataloader.py` | No | `openrouter/deepseek/deepseek-v4-flash` | — |
| 4 | `[ ]` Pending | `common/export_utils.py` | `test_export_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | — |
| 5 | `[ ]` Pending | `captioning/base.py` | (via step 6) | No | `qwen/qwen3-coder:free` | — |
| 6 | `[ ]` Pending | `captioning/joycaption.py` | `test_model.py` | Yes | `openrouter/deepseek/deepseek-v4-pro` | — |
| 7 | `[ ]` Pending | `batch/runner.py` | `test_runner.py` | Yes | `openrouter/deepseek/deepseek-v4-pro` | — |
| 8 | `[ ]` Pending | `scripts/run_caption.py` | Manual | Yes | `openrouter/deepseek/deepseek-v4-flash` | — |
| 9 | `[ ]` Pending | `configs/example_config.json` | — | No | `qwen/qwen3-coder:free` | — |

---

## Completion Log

- **2026-06-16 — Step 1 complete:** `common/types.py` (ImageRecord + CaptionRecord dataclasses). Tests: 10 passed, 0 failed. Cross-platform path fixes applied (WindowsPath handling in asdict). Next targeted file: `common/image_utils.py`.

- **2026-06-21 — Step 2 complete:** `common/image_utils.py` (`load_image`, `validate_image`). Tests: 17 passed, 0 failed. Files created: `src/pixel_art_auto_captioner/common/image_utils.py`, `tests/test_image_utils.py`, `tests/__init__.py`. Updated `common/__init__.py` to export the new functions.

  **Friction & Streamlining Note:**
  - **Friction:** The biggest friction point was the PIL `img.verify()` behaviour — `verify()` closes the file after reading, so you cannot subsequently use the same image handle. The `validate_image` function correctly opens the image in a `with` block and calls `verify()`, but if you accidentally tried to reuse the same PIL handle after `verify()`, it would fail. This is a well-known PIL pitfall, but worth noting.
  - **Streamlining guardrail:** Add a note to `SPEC.md` *§6.1* (or as a comment in `image_utils.py`) that `validate_image` uses `Image.open` + `verify()` which closes the file handle, and that `load_image` uses a separate `Image.open` call to get a usable handle. This prevents future refactors from accidentally merging the two patterns.

  Next targeted file: `ingestion/dataloader.py`.

- **2026-06-24 — Fix: Duplicate filename collision.**
  `load_image()` now accepts an `input_root: Path` parameter and computes
  ``stem`` as ``"_".join(path.relative_to(input_root).with_suffix("").parts)``.
  This gives unique identifiers like ``"folder1_sprite"`` instead of the
  ambiguous bare stem ``"sprite"``, preventing output-file collisions when
  duplicate filenames exist in different subdirectories.

  **Changes:**
  - `types.py`: Updated docstrings for `ImageRecord.stem` and
    `CaptionRecord.image_stem`.
  - `image_utils.py`: Added `input_root` parameter to `load_image()`;
    stem computed via ``path.relative_to(input_root)`` with separator
    replacement. Raises `ValueError` if path not under input_root.
  - `SPEC.md` §2.3 and §6.1: Updated signatures and docstrings.
  - `test_types.py`: Stem values changed to relative-path identifiers.
  - `test_image_utils.py`: All `load_image` tests pass `input_root`;
    4 new tests added (subdir stem, deeply nested, not-under-root,
    duplicate-filename uniqueness).

  Tests: 31 passed, 0 failed (10 types + 21 image_utils).
