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

- **2026-06-27 — Fix: Output path strategy changed to directory-preserving.**
  Reverted the flattened-stem approach. ``ImageRecord.stem`` is now the
  bare filename (e.g. ``"sprite"``), and a new ``rel_path`` field captures
  the full relative path from ``input_root`` (e.g. ``Path("folder1/sprite.png")``).
  ``CaptionRecord`` gains ``image_rel_path``. Export utilities will use
  ``image_rel_path`` to reconstruct output paths that mirror the input
  directory tree (``output/folder1/sprite.txt`` instead of
  ``output/folder1_sprite.txt``).

  **Changes:**
  - `types.py`: Added ``rel_path: Path`` to ``ImageRecord``, ``image_rel_path: Path``
    to ``CaptionRecord``. Reverted ``stem``/``image_stem`` docstrings to bare
    filename semantics.
  - `image_utils.py`: ``load_image()`` now returns ``stem=path.stem`` and
    ``rel_path=relative_to(input_root)``. Removed the underscore-joining logic.
  - `test_types.py`: All 10 tests updated with ``rel_path``/``image_rel_path`` fields.
  - `test_image_utils.py``: Stem tests now assert bare filenames.  Added
    ``test_load_image_stem_is_bare_filename``, ``test_load_image_rel_path_in_subdir``,
    ``test_load_image_rel_path_deeply_nested``,
    ``test_load_image_rel_path_distinguishes_duplicate_names``. Removed obsolete
    underscore-collision tests.  Total: 21 tests.
  - `SPEC.md`: Updated §2.3 (data types), §6.1 (load_image), §6.2 (export utils),
    §7.1 (output paths).

  Tests: 32 passed, 0 failed.
