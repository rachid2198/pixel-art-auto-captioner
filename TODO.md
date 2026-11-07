# TODO.md — Implementation Ledger

**Project:** Pixel Art Auto-Captioner
**Governed by:** `SPEC.md` v1.1 · `AGENTS.md` §2 (per-step protocol)
**Last updated:** 2026-11-07

Legend: ✅ done · ▶ next · ⬜ pending · 🔁 done in v1.0, needs rework for v1.1 · 👤 needs the user

---

## 1. Status at a glance

| Phase | State |
|---|---|
| v1.0 implementation (steps 1–7 of the old plan) | Done, 111 CPU tests passing (verified 2026-11-06 in a Linux sandbox with torch stubbed). **Not runnable end-to-end** (bug B1). |
| v1.1 review ("grill") with the user | ✅ Done 2026-11-06. 17 decisions, see §3. |
| SPEC / AGENTS / TODO rewrite | ✅ Done 2026-11-06. SPEC v1.1 approved by the user 2026-11-07 (incl. D1–D11). |
| v1.1 implementation (steps 1–13) | ▶ In progress. Step 1 ✅ 2026-11-07 (94 passed, 45 transition-skipped). Step 2 next. |
| GPU validation (steps 14–16) | ⬜ |

### Overnight log

Batch run of steps 2–13 (approved 2026-11-07). One commit per step on `origin/v1.1`. Times are local (UTC+1).

- 22:35 step 1 ✅ f5df4b7 94 passed, 45 skipped
- 22:37 step 2 ✅ 5a8b228 265 passed, 45 skipped
- 22:45 step 3 ✅ 8b97952 420 passed, 45 skipped
- 22:51 step 4 ✅ c643983 470 passed, 36 skipped
- 23:02 step 5 ✅ bb3931c 588 passed, 42 skipped

---

## 2. v1.1 implementation plan

Each step: code + tests → CPU suite green → update this table → report to the user. See AGENTS.md §2.

| Step | Status | Work | Builds on (v1.0) | Tests | Result |
|---|---|---|---|---|---|
| 1 | ✅ | `common/types.py`: new `ImageRecord` fields (`source`, original size), `LoadFailure`, `CaptionRequest`, `CaptionResult`, new `CaptionRecord` (SPEC §3) | 🔁 `types.py` | `test_types.py` | 94 passed, 45 skipped (excl. `test_model.py`, see notes) |
| 2 | ✅ | `common/config.py`: `DEFAULT_CONFIG`, `load_config`, `validate_config` (SPEC §9) | new | `test_config.py` | 265 passed, 45 skipped (excl. `test_model.py`, see step 1 notes) |
| 3 | ✅ | `common/prompts.py`, `common/seeding.py`, `common/quality.py` (SPEC §5) | new | `test_prompts.py`, `test_seeding.py`, `test_quality.py` | 420 passed, 45 skipped (excl. `test_model.py`, see step 1 notes) |
| 4 | ✅ | `common/image_utils.py` fixes: close handle, original size, broad except in `validate_image` (SPEC §4.5) | 🔁 `image_utils.py` | `test_image_utils.py` | 470 passed, 36 skipped (excl. `test_model.py`, see step 1 notes) |
| 5 | ✅ | `ingestion/dataloader.py` rework: single `input_dir`, one cached walk, case-insensitive ext, include/exclude, hash sharding, `plan()`, threaded `iter_images()`, `LoadFailure` (SPEC §4) | 🔁 `dataloader.py` | `test_dataloader.py` | 588 passed, 42 skipped (excl. `test_model.py`, see step 1 notes) |
| 6 | ✅ | `common/export_utils.py` rework: `ShardWriter`, `ResumeIndex`, txt naming + collisions, `generate_qa_page` (SPEC §8, §10.3) | 🔁 `export_utils.py` | `test_export_utils.py` | 780 passed, 19 skipped (excl. `test_model.py`, see step 1 notes) |
| 7 | ▶ | `captioning/base.py`: `caption_batch`, `caption`, `describe`, `get_model()` with lazy imports; `captioning/__init__.py` stops importing torch (SPEC §6.1) | 🔁 `base.py` | `test_base.py` | — |
| 8 | ⬜ | `captioning/joycaption_hf.py` (replaces `joycaption.py`): skip-modules quantization, no `out_proj` hack, per-request seeds, finish_reason (SPEC §6.2) | 🔁 `joycaption.py` | `test_joycaption_hf.py` (fakes) + `test_model_gpu.py` (gated) | — |
| 9 | ⬜ | `captioning/joycaption_vllm.py`: pin + verify the vLLM version/API (SPEC §6.3) | new | `test_joycaption_vllm.py` (fake LLM) | — |
| 10 | ⬜ | `batch/runner.py` rework: model_config only to `load()`, plan → batches → retry → writer, SIGTERM, summary invariants, cost logging (SPEC §7, §11.4) | 🔁 `runner.py` | `test_runner.py` | — |
| 11 | ⬜ | `scripts/run_caption.py` CLI (+ `--dry_run`, exit codes); `configs/example_config.json`, `configs/local_hf_config.json`; draft `configs/sources.json` 👤 **user approves mapping** | new | `test_cli.py` | — |
| 12 | ⬜ | `scripts/export_txt.py`, `scripts/make_qa_page.py`, `scripts/pack_dataset.py`, `scripts/unpack_dataset.sh` (SPEC §10) | new | `test_pack_dataset.py` (+ export/QA tests in `test_export_utils.py`) | — |
| 13 | ⬜ | `pyproject.toml`; split requirements; `docker/Dockerfile`; README rewrite (local setup, RunPod volume + upload + launch) (SPEC §11, §15) | new | build check only | — |
| 14 | ⬜ 👤 | Local smoke test on RTX 3060, HF backend NF4, ~20 images; user runs the command I provide | — | manual | — |
| 15 | ⬜ 👤 | RunPod pilot: pack + upload, ~500 images across categories, RTX 4090 vs L40S; report img/h and $/1k images; QA page review; tune prompts and `sources.json` | — | manual | — |
| 16 | ⬜ 👤 | Full run with the chosen caption types and GPU, within the $50–$200 budget | — | manual | — |

Dependencies: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8/9 → 10 → 11 → 12 → 13 → 14 → 15 → 16. Steps 8 and 9 are independent of each other.

### Step 1 notes (2026-11-07)

- **Files touched:** `src/pixel_art_auto_captioner/common/types.py` (rewritten per SPEC §3: `ImageRecord`, `LoadFailure`, `CaptionRequest`, `CaptionResult`, `CaptionRecord` + constants `CAPTION_STATUSES`, `CAPTION_FLAGS`, `FINISH_REASONS`); `common/__init__.py` (new exports); `tests/test_types.py` (rewritten, 38 tests); `tests/conftest.py` (`sample_image_record` / `sample_caption_record` rebuilt for v1.1). Transition skips only, no logic changes, in `test_image_utils.py`, `test_dataloader.py`, `test_export_utils.py`, `test_runner.py`.
- **Transition skips (45 tests):**
  - `test_image_utils.py`: 9 `load_image` tests → step 4. `validate_image` and the `load_image` error-path tests still run.
  - `test_dataloader.py`: 9 tests (`TestLoad` ×3, `TestIteration` ×6) → step 5. Discovery, config-validation and the other tests still run.
  - `test_export_utils.py`: module-level `pytestmark` (16 of 23 broke) → step 6. The 7 txt-sidecar tests that still passed are skipped with the module, per the "most of a module" rule.
  - `test_runner.py`: 4 `TestCaptionRunnerIntegration` tests (`test_runner_integration`, `test_runner_resume`, `test_runner_respects_output_formats`, `test_runner_iterator_load_drop_tracks_failed`) → step 10.
- **Test environment:** Linux sandbox, Python 3.13, pytest 9.1, Pillow 12.3, torch stubbed outside the repo. The no-torch subprocess test runs without the stub.
- **`test_model.py` (not touched):** 7 CPU tests pass, 6 GPU/model skips. The 4 `TestJoyCaptionLoadConfigValidation` tests fail in the sandbox because `transformers` is not installed. This is pre-existing on v1.0 (the 111 baseline also excluded `test_model.py`) and unrelated to step 1, so they are not skipped. They should pass in the local `.env`; `test_model.py` is replaced in step 8.
- **Deviations from SPEC §3:** none.

### Step 2 notes (2026-11-07)

- **Files touched:** new `src/pixel_art_auto_captioner/common/config.py` (`DEFAULT_CONFIG`, `load_config`, `validate_config` + private helpers); new `tests/test_config.py` (171 tests, mostly parametrized: defaults vs SPEC §9.2, merge precedence, sources resolution, file errors, every §9.3 rule failing and passing, all-problems-together, no heavy imports); `src/pixel_art_auto_captioner/common/__init__.py` (exports the three names).
- **Transition skips:** none. **Obsolete files:** none. `test_model.py` unchanged (same 4 sandbox-only failures as step 1).
- **Test environment:** as step 1 (Linux sandbox, torch stubbed outside the repo). 94 existing + 171 new = 265 passed, 45 skipped (all step 1 transition skips).
- **Choices inside the spec:**
  - Deep merge: nested objects merge key by key; lists, scalars and `null` replace. `overrides` uses the nested config shape (e.g. `{"runtime": {"shard": "1/4"}}`); the CLI (step 11) must omit flags the user did not give, because `None` is applied as an explicit `null`.
  - `load_config` resolves the effective `prompts.sources`: inline non-null `prompts.sources` wins; otherwise the JSON object in `prompts.sources_file` (relative paths resolved against the current directory); `sources_file: null` gives `{}`. Missing config file → `FileNotFoundError`; invalid JSON or a non-object top level (config or sources file) → `ValueError`.
  - `max_new_tokens`, `top_p`, `temperature` are checked for every defined caption type, not only the enabled ones. Booleans are rejected where ints/numbers are required. Unknown keys pass through without warning.
  - Shard parsing is a private `_parse_shard` (steps 5/10/11 will need the `(i, N)` tuple; exposing it publicly would be an interface addition).
- **Deviations:**
  1. A missing `prompts.sources_file` (when `prompts.sources` is not given inline) is reported by `validate_config` as a config error. §9.3 does not list it explicitly, but the sources mapping is needed to check "every category referenced by `sources`", and silently falling back to `default_category` for every source would waste a paid run. Consequence: until step 11 creates `configs/sources.json`, a run must set `sources_file` to `null` (or give sources inline), and the default relative path only works when started from the repo root.
  2. `top_p` and `temperature` may be `null` (= backend default, SPEC §3 `CaptionRequest.gen_params`); the §9.3 ranges apply when a value is set.
  3. A `sources` entry without a `category` key is accepted (how it resolves is `prompts.py`'s job, step 3); an entry that is not an object is an error.
- **Note for the user:** `DEFAULT_CONFIG` mirrors §9.2 literally, so `input.input_dir` defaults to `/data/pixel-art` and `output.output_dir` to `/data/captions` (the spec's example paths), although §4.1 calls `input_dir` required and §11.4 uses `/workspace/...`. Validation still fails if `input_dir` does not exist. Worth deciding at step 11 whether these two defaults should be `null` (spec change, needs approval).

### Step 3 notes (2026-11-07)

- **Files touched:** new `src/pixel_art_auto_captioner/common/prompts.py` (`render_prompt`, `resolve_source`, `prompt_config_hash`); new `common/seeding.py` (`derive_seed`); new `common/quality.py` (`check_caption`); `common/__init__.py` (exports the five names); new `tests/test_prompts.py` (86 tests: source resolution + default names, every caption type × category with the draft defaults, template mechanics, error paths, hash golden values / key order / sensitivity / cross-process stability), `tests/test_seeding.py` (17: SPEC formula, golden values, cross-process with different `PYTHONHASHSEED`, 31-bit range, Windows vs POSIX paths, sensitivity to each input), `tests/test_quality.py` (52: each flag, combinations and order, 4-gram and compression thresholds incl. boundaries, config defaults and bad values, unknown `finish_reason`). Each test file also checks that its module imports no torch/vllm.
- **Transition skips:** none. **Obsolete files:** none. `test_model.py` unchanged (same 4 sandbox-only failures).
- **Test environment:** as step 1 (Linux sandbox, Python 3.13, torch stubbed outside the repo). 265 existing + 155 new = 420 passed, 45 skipped (all step 1 transition skips). The three modules and test files also compile on Python 3.10 (Cowork VM), where the seed and hash golden values were reproduced; pytest is not installed there, so the suite itself was not run on 3.10.
- **Choices inside the spec:**
  - `resolve_source`: a `sources` entry with a missing or `null` `category` uses `default_category`; a missing or `null` `name` uses the default name; an explicit `""` name is kept. A missing/`null` `sources` mapping counts as empty. It does not check that the category exists (`validate_config` does; `render_prompt` raises). A non-object entry, a non-object `sources` or a missing `default_category` → `ValueError`.
  - Default source name: dashes → spaces, then each word `str.capitalize()`-d (`stardew-valley` → `Stardew Valley`). Not `str.title()`, which also capitalizes after digits and apostrophes (`rpg2k` → `Rpg2K`). Runs of dashes collapse.
  - `render_prompt`: only the category context is formatted (with `source_name`); instruction, constraints and source name are inserted verbatim, so braces in them stay literal. Missing/`null` `constraints` = `""`. Unknown caption type, missing instruction/category/template, or an unknown placeholder in a context or the template → `ValueError`.
  - `prompt_config_hash`: canonical JSON = `sort_keys=True`, separators `(",", ":")`, `ensure_ascii=False`, UTF-8. Non-JSON values (e.g. a `Path`) → `TypeError`. Golden values are pinned in the tests.
  - `check_caption(result, cfg)`: `cfg` is the `quality` section; missing keys use the §5.3 defaults, invalid values (`repeat_ngram_threshold` not an int ≥ 1, `min_compression_ratio` not a finite number ≥ 0) → `ValueError`. Every rule is evaluated independently (an errored empty result gives `["error", "empty"]`); flags come in the fixed order `error, empty, truncated, repetitive`. Words = whitespace tokens of the lowercased stripped text, punctuation attached; 4-grams over overlapping windows. Compression ratio = `len(zlib.compress(utf8)) / len(utf8)` at zlib's default level, strict `<`. The repetition rules are skipped for empty text. An unknown `finish_reason` sets no flag but logs a WARNING (backend mapping bug).
  - No status helper: §5.3 defines the status rule (`error`/`empty` → `failed`, other flags → `flagged`, none → `ok`) and the retry rule (`attempt <= max_retries`) but no function, so both stay in the runner (step 10) rather than adding a public name.
- **Deviations:**
  1. **`prompt_config` shape (interpretation, please confirm).** §5.1 builds the prompt from `caption_types`, `categories` and `sources` and indexes `prompt_config["template"]`, but in §9.2 `caption_types` lives under `generation`, not `prompts`. All three functions therefore take the `prompts` section plus one extra top-level key `caption_types` (= `generation.caption_types`). The runner (step 10) builds it as `{**cfg["prompts"], "caption_types": cfg["generation"]["caption_types"]}`. Consequence: `prompt_config_hash` also changes when a caption type's generation params (e.g. `temperature`), `system` or `sources_file` change, not only the prompt wording.

### Step 4 notes (2026-11-07)

- **Files touched:** `src/pixel_art_auto_captioner/common/image_utils.py` (rewritten per SPEC §4.5; line endings CRLF → LF, like the other rewritten modules); `tests/test_image_utils.py` (rewritten, 63 tests: record fields incl. `source` / `rel_path` / `stem`, RGB conversion of 6 modes, formats and upper-case extensions, original size with and without `target_size` (B5), handle closed for single- and multi-frame images and on decode errors via an `Image.open` spy on the underlying file object + a `ResourceWarning` check + a file-replace check (B10), every error path, `validate_image` returning `False` for any `Exception` incl. a CRC-corrupt PNG whose `verify()` raises `SyntaxError`, `KeyboardInterrupt` not swallowed, no torch/vllm import). The new handle and original-size tests were checked to fail against the v1.0 module.
- **Transition skips:** removed the 9 step-4 skips in `test_image_utils.py` (file replaced). Added: none. The step-5 skips in `test_dataloader.py` stay as they are (some would pass again now that `load_image` sets `source`, but `dataloader.py` and its tests are replaced in step 5). **Obsolete files:** none. `test_model.py` unchanged (same 4 sandbox-only failures).
- **Test environment:** as step 1 (Linux sandbox, Python 3.13, Pillow 12.3, torch stubbed outside the repo). 420 − 13 previously enabled v1.0 image_utils tests + 63 new = 470 passed, 36 skipped (all step 1 transition skips, for steps 5, 6 and 10). The module was also smoke-run on Python 3.10 + Pillow 12.3 (Cowork VM; pytest is not installed there).
- **Choices inside the spec:**
  - `load_image` keeps the §4.5 signature. Order: resolve `path` and `input_root`, check the boundary (`ValueError`, before the file is opened, so a missing file outside the root also gives `ValueError`), then inside `with Image.open(...)` read the original size and call `.convert("RGB")`; resize (Lanczos) after the block. `ImageRecord.path` is the resolved absolute path; `stem` comes from it.
  - `source` = first component of `rel_path` when it has ≥ 2 parts, else `""` (SPEC §3).
  - Dropped v1.0's `path.exists()` pre-check and its ERROR log: `Image.open` raises `FileNotFoundError` itself (message is now Python's, not "Image not found"), and decode failures are logged once by the loader/runner as `LoadFailure`s (steps 5/10).
  - Errors propagate unchanged: `ValueError` (outside root), `FileNotFoundError`, `PIL.UnidentifiedImageError` (not an image), `OSError` (truncated / undecodable, directory). `target_size` is not validated (Pillow raises).
  - `validate_image` catches `Exception` (not `BaseException`), logs at DEBUG and returns `False`.
- **Deviations:** none.
- **Note for step 5:** v1.0 `ImageDataLoader.load()` still passes `image_size` as `target_size`. §4.1 has no resize key and §3 says the image is not resized, so step 5 should call `load_image(path, input_dir)` without `target_size`.

### Step 5 notes (2026-11-07)

- **Files touched:** `src/pixel_art_auto_captioner/ingestion/dataloader.py` (rewritten per SPEC §4: `ImageDataLoader`, `discover()`, `plan()`, `iter_images()`, `Plan`; LF); `tests/test_dataloader.py` (v1.0 tests and their 9 step-5 skips replaced by 153 v1.1 tests: config defaults, every invalid value, all problems reported together; discovery: case-insensitive extensions, `.msgpack`/other files ignored and never opened (spy on `open` / `Image.open`), `as_posix()` sort, include/exclude incl. pruned folders never listed (`os.scandir` spy), **single walk** proved by counting `os.walk` / `os.scandir` calls (one walk, one listing per directory, no new calls on later `discover()`/`plan()`), no symlink following, progress and final-count logs, stem-collision WARNING; sharding: golden values, SPEC formula, Windows path = POSIX path, partition disjoint + complete for N ∈ {1, 2, 3, 7}, membership stable when files are added/removed; plan: pending types per image, already-complete, `max_images` after resume and repeated limited runs advance (B4), summary invariant per shard, `captions_by_source()`; iter_images: plan order with 0/1/4 workers, not resized + RGB, `load_image` called without `target_size`, `LoadFailure` for text/empty/truncated files and any `Exception` (B8), `KeyboardInterrupt` propagates, deterministic order with uneven decode times, exact bounded look-ahead (submit count), early `close()`/`break` joins the threads, empty plan starts no pool, `num_workers=0` decodes inline and lazily; package boundary in a subprocess). Key tests were mutation-checked: case-sensitive suffix, `max_images` before resume, no discovery cache, out-of-order yield and `prefetch + 1` each make tests fail. `tests/test_runner.py`: module-level transition skip only. `ingestion/__init__.py` unchanged (exports `ImageDataLoader`; `Plan` is imported from `ingestion.dataloader`).
- **Transition skips:** removed the 9 step-5 skips in `test_dataloader.py` (file replaced). Added: module-level `pytestmark` in `test_runner.py` → step 10 (all 15 still-enabled tests broke because the shared fixture builds the v1.0 loader with `source_dirs`; the 4 existing per-test step-10 skips stay). Skipped now: `test_export_utils.py` 23 → step 6, `test_runner.py` 19 → step 10. **Obsolete files:** none. `test_model.py` unchanged (same 4 sandbox-only failures).
- **Test environment:** as step 1 (Linux sandbox, Python 3.13, Pillow 12.3, torch stubbed outside the repo). 470 − 20 enabled v1.0 dataloader tests − 15 runner tests + 153 new = 588 passed, 42 skipped. Also smoke-run on Python 3.10 + Pillow (Cowork VM; pytest not installed there).
- **Public API:** `ImageDataLoader(config: dict)`; `discover() -> list[Path]`; `plan(is_complete: Callable[[Path, str], bool], caption_types: list[str]) -> Plan`; `iter_images(plan: Plan) -> Iterator[ImageRecord | LoadFailure]`; constant `DEFAULT_EXTENSIONS`. Attributes: `input_dir` (resolved), `extensions` (lowercased frozenset), `recursive`, `include_sources`, `exclude_sources`, `shard` (tuple), `max_images`, `num_workers`, `prefetch`.
- **`Plan` (§4.3 leaves fields open; minimal set for the runner and `--dry_run`):** `shard: tuple[int, int]`, `caption_types: list[str]`, `images_discovered`, `images_in_shard`, `images_already_complete`, `images_deferred` (ints, = the §7.3 summary fields), `items: list[tuple[Path, list[str]]]` (sorted `rel_path`, pending types in `caption_types` order); properties `images_planned`, `captions_planned`; method `captions_by_source() -> {source: {caption_type: n}}` for D7.
- **Choices inside the spec:**
  - Config is the flat §4.1 dict. Unknown keys are ignored (the CLI can pass `{**cfg["input"], **cfg["runtime"]}`); `shard` accepts `(i, N)`, `[i, N]` or the config-file string `"i/N"` (reuses `common.config._parse_shard`); `exclude_sources: null` = `[]`; extensions are lowercased and get a leading dot if missing; `prefetch` must be ≥ 1. All problems → one `ValueError`.
  - Discovery: filters match exact top-level folder names, only at the top level; exclude wins over include; with `include_sources` set, files directly in `input_dir` (source `""`) are excluded; a WARNING names filter entries that are not top-level folders. `recursive=False` → only files directly in `input_dir`. Symlinked folders are not entered; a symlinked file that resolves to another path becomes a `LoadFailure` (otherwise `load_image` would return it under the target's key and resume would never mark the planned key done).
  - Stem collisions (§8.2): detected in `discover()` per folder, stems compared case-insensitively (Windows file systems are case-insensitive), one WARNING listing up to 20 paths. The set is kept privately (`_collisions`), see the note for steps 6/10.
  - Shard hash input is `rel_path.as_posix()` encoded as UTF-8.
  - `plan()`: `caption_types` must be non-empty unique strings (`ValueError`); exceptions from `is_complete` propagate; one INFO line; not cached (discovery is).
  - `iter_images()`: exactly one item per plan item, in plan order; at most `prefetch` items submitted and not yet yielded; a decode `Exception` → `LoadFailure(error="<Type>: <message>")` + one WARNING log; `BaseException` propagates; closing the generator cancels queued decodes and joins the threads (name prefix `pac-decode`). Images are not resized (`load_image(path, input_dir)`, per the step 4 note).
- **Deviations:** none.
- **Notes for later steps:**
  - Steps 6/10: `save_txt(..., collision_set)` needs the collision set. §4 defines no public accessor, so the loader keeps it private; recompute it from `discover()` with the same rule (same folder, case-insensitive stem), or propose a public accessor.
  - Step 10: zip `iter_images(plan)` with `plan.items` (one-to-one, same order) to get each image's pending types; wrap the generator in `contextlib.closing` so an interrupt joins the decode threads at once; do not log `LoadFailure`s again (the loader already does).
  - Step 11: `--dry_run` can print the `Plan` counts and `captions_by_source()`.

### Step 6 notes (2026-11-07)

- **Files touched:** `src/pixel_art_auto_captioner/common/export_utils.py` (rewritten per SPEC §8 / §10.3; LF); `common/__init__.py` (removed the exports `save_jsonl_entry`, `save_txt_sidecar`; added `ShardWriter`, `ResumeIndex`, `save_txt`, `read_records`, `generate_qa_page`; `build_record` kept with its new signature); `tests/test_export_utils.py` (v1.0 tests and the module-level step-6 skip replaced by 192 v1.1 tests: JSONL key order = dataclass order, POSIX paths (also nested), `ensure_ascii=False`, one physical line per record incl. `\n`/U+2028/U+0085 captions, nothing written on a serialization error; writer: shard file names, invalid shard / `fsync_every`, flush per record, append, other shards untouched, JSONL written + flushed **before** the `.txt` (spy), `.txt` only for ok/flagged and only with `write_txt`, collision set passed through; **fsync cadence** via an `os.fsync` spy (every N records, on close, data flushed first, default 100); **malformed final line** skipped with WARNING (partial JSON, non-object, cut UTF-8, followed by blank lines, per file), malformed middle line → `ValueError` with `file:line`; tail repair before appending (partial line, terminated bad line, valid unterminated line, blank file, line longer than the 64 KiB read block); `ResumeIndex`: status rule, key = full `rel_path` + caption type (not stem), `Path`/str/`PureWindowsPath` lookups, **all `captions/*.jsonl`** incl. other shard counts, merged and upper-case names (D3), `failures/`, sub-folders and non-JSONL ignored, **never opens `.txt`** (spy on `open`), **latest wins** in one file and across files, ties → last read, times compared as instants (offsets), invalid records → error, prompt-change WARNING; `build_record`: fields, `Z` timestamp, shard formatting, `LoadFailure`, error into `extra`, copies, validation; `save_txt`: naming, no trailing newline / no newline translation, **collision rule (B2)** with `Path` and string sets, unsafe paths and caption types; QA page: **self-contained** (no script/link/iframe, every `src` a `data:image/jpeg;base64` thumbnail ≤ 256 px, no `url(`/`http`), escaping of captions and prompts, all caption types side by side, status/flag highlighting and hover details, `n` limit, `n=0` disables, stratified by source, deterministic per seed, status and source filters, latest record across shards, missing/broken/outside images → placeholder, invalid args, input dir untouched; package boundary in a subprocess; exports). Key tests were mutation-checked: 22 mutations (txt before JSONL, no flush, fsync off by one, no fsync on close, first-wins, string timestamp compare, error on final line, lenient middle line, own shard only, no collision naming, `ensure_ascii=True`, sorted keys, no tail repair, `+00:00` timestamps, failed counted complete, unescaped caption, external CSS, no stratification, txt for failed, trailing newline in txt, no prompt warning, prompt check keyed by category only) each make a test fail.
- **Transition skips:** removed the module-level step-6 `pytestmark` in `test_export_utils.py` (its 23 v1.0 tests are replaced). Added: none. Remaining: `test_runner.py` 19 → step 10. The v1.0 `batch/runner.py` now fails at import (it imports the removed `save_jsonl_entry` / `save_txt_sidecar`), so `import pixel_art_auto_captioner.batch` raises `ImportError` until step 10. Not patched: `test_runner.py` imports the runner only inside its (skipped) tests, so the suite still collects (816 tests incl. `test_model.py`). **Obsolete files:** none. `test_model.py` unchanged (same 4 sandbox-only failures).
- **Test environment:** as step 1 (Linux sandbox, Python 3.13, Pillow 12.3, torch stubbed outside the repo). 588 + 192 new = 780 passed, 42 − 23 = 19 skipped. Also smoke-run on Python 3.10 + Pillow 12.3 (Cowork VM; pytest not installed there): writer + collisions, crash tail, resume, QA page.
- **Public API:**
  - `ShardWriter(output_dir: Path, shard: str | tuple[int, int], write_txt: bool, *, fsync_every: int = 100, collision_set: Collection[Path | str] | None = None)`; `write(record: CaptionRecord) -> None`; `close() -> None` (idempotent); context manager; attributes `output_dir`, `shard` (tuple), `write_txt`, `fsync_every`, `path`, `records_written`, property `closed`.
  - `ResumeIndex()` (empty, for `resume=False`); `ResumeIndex.from_output_dir(output_dir: Path) -> ResumeIndex`; `is_complete(rel_path: Path | str, caption_type: str) -> bool`; `num_complete`, `len()`, `files_read`, `records_read`.
  - `build_record(image: ImageRecord | LoadFailure, result: CaptionResult, *, caption_type, category, prompt, generation_params, seed, attempt, status, flags, model_name, model_path, model_revision, backend, run_id, shard, extra=None) -> CaptionRecord`.
  - `save_txt(record: CaptionRecord, output_dir: Path, collision_set: Collection[Path | str] | None) -> Path`.
  - `read_records(paths: Iterable[Path]) -> Iterator[dict]`.
  - `generate_qa_page(output_dir: Path, input_dir: Path, *, n: int = 200, seed: int = 0, statuses: Collection[str] = ("ok", "flagged"), sources: Collection[str] | None = None, name: str | None = None) -> Path | None` (writes `qa/<name>.html`; default name `qa-<YYYYMMDDTHHMMSSZ>`; `n=0` → `None`).
- **Choices inside the spec:**
  - JSONL: `json.dumps(..., ensure_ascii=False)` with default separators; any `PurePath` (also inside `extra` / `generation_params`) → `as_posix()`; other non-JSON values → `TypeError` before anything is written. The file is opened in binary append mode (no `\r\n` on Windows); readers split on `\n` only, so U+2028 / U+0085 inside captions stay on one line.
  - Reading: blank lines are ignored; "final line" = the last non-blank line of each file; malformed = invalid UTF-8 or JSON, or not a JSON object. Resume also requires non-empty string `image_rel_path`, `caption_type`, `status` and a parseable `timestamp_utc`, else `ValueError` (also on a final line: a crash cannot produce valid JSON with missing keys). An unknown status counts as not complete (no error).
  - Latest wins by the parsed instant (`Z`, offsets and v1.0 `+00:00` accepted, naive = UTC); ties → the record read last (files sorted by name, then line order). Files = regular files directly in `captions/` with a case-insensitive `.jsonl` suffix; `failures/` is never read. Per key only (time, position, status, small group id) is kept, so ~1.1M records need a few hundred MB.
  - `build_record`: validates `status` and `flags` against `CAPTION_STATUSES` / `CAPTION_FLAGS` (not `finish_reason`, so a backend mapping bug stays visible in the data); `LoadFailure` → width/height `0`; `result.error` → `extra["error"]` unless `extra` already has it; lists/dicts copied; shard stored as `"i/N"`.
  - `save_txt` refuses `failed` (and unknown) status with `ValueError`; the collision name is `<file name>.<caption_type>.txt`, keeping the extension's case; `collision_set` holds `Path`s or POSIX strings; absolute / `..` / drive paths and caption types containing `/ \ :` → `ValueError`. If the `.txt` write fails inside `ShardWriter.write`, the JSONL line is already written and the error propagates.
  - QA page: an image is eligible if any of its latest records has a status in `statuses` (and passes the source filter); its row shows all its latest caption types (columns sorted by name). Stratification: seeded round-robin over sources (each source's images seeded-shuffled), rows sorted by (source, rel_path). Thumbnails: longest side ≤ 256 px (NEAREST), JPEG q85, displayed at 256 px with `image-rendering: pixelated`. Missing, undecodable or outside-`input_dir` images → placeholder + WARNING. `input_dir` not a directory → `FileNotFoundError`. Two passes over the JSONL (light index, then only the sampled records) to keep memory low.
- **Deviations:**
  1. `ShardWriter` takes two keyword-only parameters not in the §8.4 signature: `fsync_every` (§8.4 requires fsync every `output.fsync_every` records) and `collision_set` (§8.2 naming needs it for the optional `.txt`). The 3-argument call from the spec still works (defaults: 100, no collisions).
  2. **Tail repair.** On open, `ShardWriter` removes a malformed final line of its shard file (WARNING) and terminates a valid unterminated one. Without this, the next append would turn the skipped partial line into a malformed line in the *middle* of the file, which §8.4 makes a hard error on the following resume. Only bytes that resume would skip anyway are dropped.
  3. **Prompt-change WARNING (interpretation, please confirm; see Open questions).** `from_output_dir(output_dir)` receives no prompt config, so it cannot compare with the *current* prompts. Implemented: WARNING when the complete (latest ok/flagged) records of one (caption type, category, **source**) were produced with ≥ 2 different prompts. Keyed by source as well because `game_crop` prompts legitimately differ by `{source_name}`; keying by category only (literal §8.4) would warn on every run. Consequence: a prompt change is detected from the second run after the change, not at the first resume.
- **Notes for later steps:**
  - Step 10: `ShardWriter(output_dir, plan.shard, cfg["output"]["write_txt"], fsync_every=cfg["output"]["fsync_every"], collision_set=...)`; `ResumeIndex.from_output_dir(output_dir)` if `resume` else `ResumeIndex()`; build a `CaptionResult(text="", finish_reason="error", num_output_tokens=0, error=failure.error)` to pass a `LoadFailure` to `build_record`; `generate_qa_page(output_dir, input_dir, n=cfg["output"]["qa_sample"], name=run_id)` (its `seed` is the sampling seed). `failures/` writing is not in `export_utils` (§8.4 does not list it); reuse the same file name (private `_shard_filename`), and note §7.2 ("one failed record per pending caption type") vs §8.1 ("LoadFailure entries + systemic errors") for the record shape.
  - Steps 10/12: the collision set is still computed nowhere public. `ingestion.dataloader._find_stem_collisions` (same folder, case-insensitive stem) is private and `common` cannot import `ingestion`; the runner can recompute it from `discover()`, `export_txt.py` from the JSONL `rel_path`s with the same rule, or propose a public helper.
  - Step 12: `ResumeIndex` keeps no caption text; `export_txt.py` needs its own latest-wins pass over `read_records` (same rule: parsed timestamp, ties → last read).

---

## 3. Decisions log (2026-11-06 review)

| # | Topic | Decision |
|---|---|---|
| Q1 | Purpose | Research dataset; training architecture undecided; `.msgpack` files are leftovers and ignored |
| Q2 | Engine | vLLM on the cloud, HF transformers locally |
| Q3 | Precision | bf16 cloud; NF4 local with vision tower + projector unquantized; drop the `out_proj` hack |
| Q4 | Prompts | Per-source category prompts + game name from the folder |
| Q5 | Caption types | Configurable list, chosen per run |
| Q6 | Output | JSONL primary; `.txt` optional / exported |
| Q7 | Cloud runs | Hash sharding + crash-safe resume |
| Q8 | Environment | Docker image on the official vLLM base |
| Q9 | Reproducibility | Per-image deterministic seeds |
| Q10 | Quality | Flag truncated/repetitive/empty, retry once, QA HTML page; raw text untouched |
| Q11 | Bugs | Fix all of B1–B10 |
| Q12 | Process | Rewrite SPEC/AGENTS/TODO; improvements only after user approval |
| Q13 | Budget | $50–$200 for a full pass |
| Q14 | Provider | RunPod Community Cloud; pilot RTX 4090 vs L40S |
| Q15 | Data transfer | Pack JPGs only into ~1–2 GB tars; upload to the network volume |
| Q16 | Model comparison (proposal) | **Rejected**: JoyCaption only |
| Q17 | `pyproject.toml` (proposal) | **Accepted** |

---

## 4. v1.0 bugs found in review (all scheduled for fix, Q11)

| ID | Bug | Fixed in step |
|---|---|---|
| B1 | Runner passes its own config to `model.load()` (missing `model_path` → crash); spec had CLI and runner both loading the model | 10, 11 |
| B2 | `sprite.png` + `sprite.jpg` overwrite the same `sprite.txt` | 6 |
| B3 | Resume only checks `.txt`; JSONL-only runs duplicate; crash between txt and JSONL writes leaves gaps | 6, 10 |
| B4 | `max_images` applied before resume → repeated dry run processes 0 | 5 |
| B5 | `width`/`height` recorded after resize | 4 |
| B6 | Case-sensitive extension matching on Linux | 5 |
| B7 | Discovery twice × once per extension + upfront validation of every image | 5 |
| B8 | Corrupt images counted as skipped, not failed | 5, 10 |
| B9 | `requirements.txt` replaces PyPI with the cu130 index; version floors too low | 13 |
| B10 | Unclosed file handles; `+00:00` timestamps; truncation not flagged; no CPU tests for model code | 4, 6, 3, 8 |

Also found: importing `captioning.base` pulls in torch via `captioning/__init__.py` (fixed in step 7). The `out_proj` hack silently skips at DEBUG level if its attribute path doesn't match (removed in step 8).

---

## 5. Proposals

| ID | Proposal | Status |
|---|---|---|
| P1 | Compare JoyCaption with Qwen3-VL-8B in the pilot | ❌ Rejected 2026-11-06 (Q16) |
| P2 | `pyproject.toml` packaging | ✅ Accepted 2026-11-06 (Q17), in SPEC v1.1 |
| P3 | JPG-only tar packing for upload | ✅ Accepted 2026-11-06 (Q15), in SPEC v1.1 |
| P4 | Always record the **resolved** Hugging Face commit hash of the downloaded model in the run manifest, even when `model.revision` is null | ⏳ Suggested 2026-11-07, awaiting the user's answer. Tiny change to `describe()` (step 8/9) + manifest (step 10). |

New proposals go here first (motivation, cost, alternatives), one at a time, and need the user's approval (AGENTS.md §0).

---

## 6. Open questions / items for the user

| Item | When |
|---|---|
| Decide proposal P4 (record resolved model commit hash) | Before step 8 |
| Approve `configs/sources.json` (folder → category/name). Note: some sources are not pixel art (`hades`, `prey`, `lies-of-p`, `fallout`, …); keeping or removing them from the dataset is your call (out of pipeline scope) | Step 11 |
| Final prompt wording; whether game-name injection helps or causes hallucinated details | Step 15 |
| Pilot GPU choice (4090 vs L40S) by measured cost | Step 15 |
| Caption types for the full run | Step 16 |
| Resume prompt-change WARNING (SPEC §8.4): confirm the step 6 interpretation (compare complete records per caption type + category + source among themselves), or approve an optional argument so the runner can compare against the current prompts (step 6 notes, deviation 3) | Before step 10 |

---

## 7. History (v1.0, 2026-06-16 → 2026-07-16)

Built with Pi + DeepSeek/Qwen under the v1.0 spec, reviewed externally by GPT 5.5 (`REVIEW.md`). Full step notes are in git history (`TODO.md` before 2026-11-06).

| v1.0 step | File | Result at the time | v1.1 status |
|---|---|---|---|
| 1 | `common/types.py` | 10 tests | 🔁 step 1 |
| 2 | `common/image_utils.py` | 21 tests | 🔁 step 4 |
| 3 | `ingestion/dataloader.py` | 29 tests (after review 2026-06-28) | 🔁 step 5 |
| 4 | `common/export_utils.py` | 23 tests (after review 2026-07-16) | 🔁 step 6 |
| 5 | `captioning/base.py` | 8 tests | 🔁 step 7 |
| 6 | `captioning/joycaption.py` | 11 passed, 6 GPU-skipped | 🔁 step 8 |
| 7 | `batch/runner.py` | 13 tests; runner fixes for review critiques 1–7 in commit `beb626f` | 🔁 step 10 |
| 8 | `scripts/run_caption.py` | not started | → step 11 |
| 9 | `configs/example_config.json` | not started | → step 11 |

Lessons carried into AGENTS.md §6: PIL `verify()` handle lifetime; sibling temp dirs for boundary tests; dual guard for GPU tests.

**REVIEW.md:** the 2026-07-16 runner review is still marked OPEN. Code inspection on 2026-11-06 shows critiques 1, 2, 4, 5 and 6 addressed in `beb626f` (load errors re-raised, unload in `finally`, `output_formats` validated, dropped-image accounting, load-success log); critique 3 was only downgraded to a warning. The whole runner is redesigned in step 10, which supersedes that review. It will be marked closed when step 10 lands.
