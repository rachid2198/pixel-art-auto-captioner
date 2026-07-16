# TODO.md — Implementation State Ledger

**Project:** Pixel Art Auto-Captioner  
**Governed by:** `SPEC.md` `§15` | `AGENTS.md` `§4`  
**Rule:** No code for step N unless all preceding steps are `[x]` completed.

---

| Step | Status | File | Test File | GPU | Target Engine | Test Results |
|------|--------|------|-----------|:---:|:---:|--------------|
| 1 | `[x]` Complete | `common/types.py` | `test_types.py` | No | `qwen/qwen3-coder:free` | 10 passed, 0 failed |
| 2 | `[x]` Complete | `common/image_utils.py` | `test_image_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | 17 passed, 0 failed |
| 3 | `[x]` Complete | `ingestion/dataloader.py` | `test_dataloader.py` | No | `openrouter/deepseek/deepseek-v4-flash` | 22 passed, 0 failed |
| 4 | `[x]` Complete | `common/export_utils.py` | `test_export_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | 19 passed, 0 failed |
| 5 | `[x]` Complete | `captioning/base.py` | `test_base.py` | No | `qwen/qwen3-coder:free` | 8 passed, 0 failed (suite: 92/92) |
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

- **2026-06-27 — Step 3 complete:** `ingestion/dataloader.py` (`ImageDataLoader`). Tests: 22 passed, 0 failed. Files created/modified: `src/pixel_art_auto_captioner/ingestion/dataloader.py`, `tests/test_dataloader.py`, `src/pixel_art_auto_captioner/ingestion/__init__.py` (added export). All 8 SPEC §10.3 required tests implemented plus 14 additional edge-case tests covering config validation, deduplication, sorting, empty/nonexistent directories, max_images=None, resize=None, skip_existing disabled, and len-after-filtering.

  **Friction & Streamlining Note:**
  - **Friction:** The `test_raises_on_path_outside_source_dirs` test initially created the "orphan" image at `tmp_path/other/orphan.png` while using `tmp_path` itself as the sole source_dir. Since `other/` is a child of `tmp_path`, the path resolved *under* the source directory — the `load()` call succeeded instead of raising `ValueError`. The root cause was a single `tmp_path` serving double duty as both the source root and the test-scratch root, making the intended boundary invisible.
  - **Streamlining guardrail:** When writing tests that validate source-dir boundary enforcement, always create **explicit sibling directories** — one for the source and a separate one for the out-of-bounds file — rather than nesting the orphan inside what happens to be the source parent. A comment in `test_dataloader.py` or a short note in SPEC §10.3 calling out this pattern would prevent the same mental slip for future agents.

  Next targeted file: `common/export_utils.py`.

- **2026-06-27 — Harness update: Friction & Streamlining guardrails applied.**
  Both guardrails from Steps 2 and 3 have been baked into the project harness:
  - **SPEC.md §6.1** — ``validate_image`` row now warns that ``PIL.Image.verify()`` closes the file handle and that the two code paths must remain separate.
  - **SPEC.md §10.3** — ``test_dataloader.py`` section now includes a blockquote with the sibling-directory pattern for source-dir boundary tests, with a concrete ``source_dir`` / ``outside_dir`` code example.
  - **AGENTS.md §7.5** — New "Known Pitfalls" subsection added under Testing Requirements, documenting both the PIL verify() handle lifetime issue and the tmp_path sibling-directory pattern with correct/incorrect examples.

  Next targeted file: `captioning/base.py` (Step 5).

- **2026-06-28 — Step 4 complete:** `common/export_utils.py` (`save_txt_sidecar`, `save_jsonl_entry`, `build_record`). Tests: **19 passed, 0 failed** (full suite: 80 passed, 0 failed). Files created/modified: `src/pixel_art_auto_captioner/common/export_utils.py`, `tests/test_export_utils.py`, `src/pixel_art_auto_captioner/common/__init__.py` (added exports), `tests/conftest.py` (added `sample_image_record` and `sample_caption_record` fixtures). All 3 SPEC §10.3 required tests implemented plus 16 additional edge-case tests covering: directory-structure preservation, parent-directory creation, overwrite idempotency, multi-entry JSONL append, schema key completeness, POSIX string serialisation, empty captions, nonempty `extra` dict, and generation-params round-trip fidelity.

  **Friction & Streamlining Note:**
  - **Friction:** SPEC §6.2 lists only three functions for `export_utils.py` (`save_txt_sidecar`, `save_jsonl_entry`, `build_record`), but §7.3 assigns a fourth function (`generate_visual_deck`) to the same module with no function signature, no test specification, and no mention in the §15 implementation sequence. This creates a decision point: implement a stub now or defer. The choice was to implement the three functions per §6.2/§15 and flag the gap — `generate_visual_deck` has no typed interface, no output spec beyond "a lightweight, standalone index.html," and no tests, making it impossible to implement with the same confidence.
  - **Streamlining guardrail:** Every function in the §15 step description should have a corresponding row in its module's § function table **and** at least one test in §10.3. If `generate_visual_deck` is in-scope for v1.0, SPEC.md should add a `generate_visual_deck(something) -> Path` signature to §6.2, concrete test cases to §10.3, and explicit input/output specs to §7.3. Until then, agents should implement only what the §15 step table and § function table agree on.

  Next targeted file: `captioning/base.py` (Step 5).

- **2026-06-28 — Step 5 complete:** `captioning/base.py` (`CaptionModel` ABC). Tests: **8 passed, 0 failed** (full suite: 92 passed, 0 failed). Files created/modified: `src/pixel_art_auto_captioner/captioning/base.py`, `tests/test_base.py`, `src/pixel_art_auto_captioner/captioning/__init__.py` (added `CaptionModel` export). All 8 tests cover: ABC cannot be instantiated directly, concrete subclass is instantiable, missing abstract method raises `TypeError`, `model_name` attribute is accessible and persists across `load()`/`unload()`, `caption()` accepts and forwards `**gen_kwargs`, return type is `tuple[str, dict]`, and `load()`/`unload()` are callable without error.

  **Friction & Streamlining Note:**
  - **Friction:** SPEC §15 says step 5 has "(no separate tests — tested via concrete impl)" and the TODO.md ledger column said "(via step 6)", but AGENTS.md §7.4 mandates "every public function in every module must have at least one test." The ABC has no concrete logic, but it does have testable behaviour: abstract enforcement (direct instantiation is forbidden, missing methods raise `TypeError`) and interface contracts (`caption()` return shape, `model_name` persistence). The ambiguity between the two documents forced a deliberation pause — write `test_base.py` or trust the spec's deferral to step 6?
  - **Streamlining guardrail:** SPEC §15 and AGENTS.md §7.4 must agree on the testing policy for ABC-only modules. Either (a) SPEC §15 should list a test file for every step including ABCs (even if minimal, e.g. `test_base.py` with 3–4 contract tests), or (b) AGENTS.md should explicitly exempt ABC-only steps, stating "modules containing only abstract classes with no concrete logic are tested via their concrete implementations in the next GPU step." The latter is cleaner but either choice eliminates the ambiguity.

  Next targeted file: `captioning/joycaption.py` (Step 6).
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
