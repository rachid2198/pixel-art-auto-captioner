# ADVERSARY REVIEW LEDGER

<!-- The external adversary will append new reviews below this line -->

### Review Date: 2026-06-28
**[STATUS: CLOSED - IMPLEMENTED]**
*   **Task Audited:** Step 3 complete — `ingestion/dataloader.py` (`ImageDataLoader`)
*   **Target Files:** `SPEC.md`, `TODO.md`, `src/pixel_art_auto_captioner/ingestion/dataloader.py`, `src/pixel_art_auto_captioner/ingestion/__init__.py`, `tests/test_dataloader.py`
*   **Critique 1 — ✅ FIXED:** `_get_filtered_paths()` now calls `validate_image()` and excludes corrupt files before caching. `__len__()` and `__iter__()` agree. Test: `test_corrupt_image_excluded_from_len_and_iter`.
*   **Critique 2 — ✅ ADDRESSED:** The structural accounting gap is eliminated by Critique 1's pre-validation. The claim about runner's `failed` count tracking image-load errors is a misreading of SPEC §5.1 (`failed` = "errors during captioning"). No dataloader change beyond Critique 1 required.
*   **Critique 3 — ✅ FIXED:** `__init__()` now validates `source_dirs` is a `list` (not a string) and that each element is a `str`. Also validates `extensions` is a `list`. Tests: `test_source_dirs_string_raises`, `test_source_dirs_elements_must_be_strings`, `test_extensions_must_be_list`.
*   **Critique 4 — ✅ FIXED:** Dedup test rewritten to use `[tmp_path, tmp_path]` — two source dirs resolving the same physical directory, exercising actual `set[Path]` deduplication. Old distinct-files test preserved as `test_distinct_files_with_same_name_are_not_deduped`.
*   **Critique 5 — ✅ FIXED:** Added `test_skip_existing_preserves_directory_structure` (nested sidecar at correct path triggers skip) and `test_skip_existing_flat_sidecar_wrong_location` (flat sidecar does NOT match nested image — regression guardrail).
*   **Result:** 61 tests passed, 0 failed (29 dataloader + 21 image_utils + 10 types + 1 conftest).

### Review Date: 2026-07-16
**[STATUS: CLOSED - IMPLEMENTED]**
*   **Task Audited:** Step 4 — `common/export_utils.py` (`save_txt_sidecar`, `save_jsonl_entry`, `build_record`)
*   **Target Files:** `TODO.md`, `SPEC.md`, `src/pixel_art_auto_captioner/common/export_utils.py`, `src/pixel_art_auto_captioner/common/__init__.py`, `src/pixel_art_auto_captioner/common/types.py`, `tests/test_export_utils.py`, `tests/conftest.py`
*   **Critique 1 — ✅ FIXED:** `save_txt_sidecar()` now resolves the output path and validates containment via `relative_to(output_root)` before calling `mkdir`/`write_text`. Raises `ValueError` on absolute `image_rel_path` or `..` escape. Tests: `test_save_txt_sidecar_raises_on_absolute_image_rel_path`, `test_save_txt_sidecar_raises_on_parent_traversal`.
*   **Critique 2 — ❌ REJECTED:** The adversary cites SPEC §6.2's JSONL example as exhaustive, but SPEC §2.3 (the authoritative `CaptionRecord` dataclass definition) explicitly includes `image_rel_path: Path` and `extra: dict`. The §6.2 example is illustrative, not restrictive. The implementation faithfully serialises every §2.3 field. Adding fields is backwards-compatible; no JSON consumer breaks on extra keys. No code change needed.
*   **Critique 3 — ✅ FIXED:** Error-path tests added per AGENTS.md §7.4. Tests: `test_save_jsonl_entry_raises_typeerror_on_unserializable` (bytes in `generation_params` → `TypeError` from `json.dumps`), `test_save_txt_sidecar_raises_on_unwritable_output` (file-as-directory blocker → `OSError`/`FileExistsError` from `mkdir`).
*   **Result:** 84 tests passed, 0 failed (29 dataloader + 23 export_utils + 21 image_utils + 10 types + 1 conftest).

### Review Date: 2026-07-16
**[STATUS: OPEN]**
*   **Task Audited:** Step 7 complete (Review) — `batch/runner.py` (`CaptionRunner`)
*   **Target Files:** `TODO.md`, `SPEC.md`, `src/pixel_art_auto_captioner/batch/runner.py`, `src/pixel_art_auto_captioner/batch/__init__.py`, `tests/test_runner.py`, `src/pixel_art_auto_captioner/ingestion/dataloader.py`, `src/pixel_art_auto_captioner/common/export_utils.py`
*   **Critique 1:** `CaptionRunner.run()` swallows `model.load()` failures and returns a zeroed summary. SPEC §12.1/§12.2 require model-load errors to fail fast with traceback/code 1 semantics, and AGENTS §5.5 says config/model load errors exit immediately. Returning a normal summary lets a CLI mistakenly report success after OOM or missing weights.
*   **Critique 2:** `model.unload()` is not guaranteed if `model.load()` partially allocates resources and then raises. The `finally` block begins only after a successful load, which violates the cleanup principle in SPEC §12.2/AGENTS §5.5 for partial-load failures.
*   **Critique 3:** Runner-level `resume` is effectively dead configuration. `self.resume` is stored but never applied to the dataloader, so a caller can pass `resume=True` to `CaptionRunner` and still reprocess existing files if the dataloader was built with `skip_existing=False`; the test only passes because it manually toggles the dataloader config instead of proving the runner config works.
*   **Critique 4:** `output_formats` is not validated. Any invalid value or an empty list is accepted; the run can mark images as `succeeded` while writing no outputs at all, violating SPEC §5.1/§7's supported output formats (`txt`, `jsonl`, or both).
*   **Critique 5:** The summary invariant can be broken by load-time iterator drops. `total`/`skipped` are computed before iteration, but `ImageDataLoader.__iter__()` catches `load()` exceptions and yields nothing for that path; the runner never increments `failed`, so `total == succeeded + failed + skipped` can become false despite SPEC §5.1 declaring it invariant.
*   **Critique 6:** Required runner logging is incomplete. SPEC §5.2 calls for model load start/success/failure, image discovery count, per-image progress, skipping, and final summary from the runner; this implementation logs load start/failure and final summary but no model-load success and delegates most progress/skip logging implicitly to the dataloader.
*   **Critique 7:** Tests miss the fragile paths above: no assertion that `resume=True` controls skipping through the runner, no invalid/empty `output_formats` test, no partial `load()` failure cleanup test, no invariant test for iterator load failures, and the existing model-load failure test bakes in the spec-violating swallow-and-return behavior.
