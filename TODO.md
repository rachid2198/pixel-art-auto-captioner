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

---

## 2. v1.1 implementation plan

Each step: code + tests → CPU suite green → update this table → report to the user. See AGENTS.md §2.

| Step | Status | Work | Builds on (v1.0) | Tests | Result |
|---|---|---|---|---|---|
| 1 | ✅ | `common/types.py`: new `ImageRecord` fields (`source`, original size), `LoadFailure`, `CaptionRequest`, `CaptionResult`, new `CaptionRecord` (SPEC §3) | 🔁 `types.py` | `test_types.py` | 94 passed, 45 skipped (excl. `test_model.py`, see notes) |
| 2 | ✅ | `common/config.py`: `DEFAULT_CONFIG`, `load_config`, `validate_config` (SPEC §9) | new | `test_config.py` | 265 passed, 45 skipped (excl. `test_model.py`, see step 1 notes) |
| 3 | ▶ | `common/prompts.py`, `common/seeding.py`, `common/quality.py` (SPEC §5) | new | `test_prompts.py`, `test_seeding.py`, `test_quality.py` | — |
| 4 | ⬜ | `common/image_utils.py` fixes: close handle, original size, broad except in `validate_image` (SPEC §4.5) | 🔁 `image_utils.py` | `test_image_utils.py` | — |
| 5 | ⬜ | `ingestion/dataloader.py` rework: single `input_dir`, one cached walk, case-insensitive ext, include/exclude, hash sharding, `plan()`, threaded `iter_images()`, `LoadFailure` (SPEC §4) | 🔁 `dataloader.py` | `test_dataloader.py` | — |
| 6 | ⬜ | `common/export_utils.py` rework: `ShardWriter`, `ResumeIndex`, txt naming + collisions, `generate_qa_page` (SPEC §8, §10.3) | 🔁 `export_utils.py` | `test_export_utils.py` | — |
| 7 | ⬜ | `captioning/base.py`: `caption_batch`, `caption`, `describe`, `get_model()` with lazy imports; `captioning/__init__.py` stops importing torch (SPEC §6.1) | 🔁 `base.py` | `test_base.py` | — |
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
