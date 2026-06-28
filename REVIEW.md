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
