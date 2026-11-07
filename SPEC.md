# SPECIFICATION — Pixel Art Auto-Captioner

**Version:** 1.1
**Date:** 2026-11-06
**Status:** Approved by the user on 2026-11-07 (decisions Q1–Q17 and derived choices D1–D11). Ready for implementation.
**Supersedes:** v1.0 (2026-06-16)

---

## Table of Contents

0. [Changes from v1.0](#0-changes-from-v10)
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Types (`common/types.py`)](#3-data-types-commontypespy)
4. [Ingestion (`ingestion`)](#4-ingestion-ingestion)
5. [Prompts, Seeds & Quality (`common`)](#5-prompts-seeds--quality-common)
6. [Captioning Backends (`captioning`)](#6-captioning-backends-captioning)
7. [Runner (`batch`)](#7-runner-batch)
8. [Output Formats & Resume](#8-output-formats--resume)
9. [Configuration](#9-configuration)
10. [Scripts & CLI](#10-scripts--cli)
11. [Cloud Execution](#11-cloud-execution)
12. [Error Handling](#12-error-handling)
13. [Testing](#13-testing)
14. [Package Layout](#14-package-layout)
15. [Dependencies](#15-dependencies)
16. [Implementation Sequence](#16-implementation-sequence)
17. [Open Items](#17-open-items)

---

## 0. Changes from v1.0

### 0.1 Decisions made by the user (2026-11-06 review)

| # | Decision | Effect on spec |
|---|----------|----------------|
| Q1 | Captions are for a **research** dataset; training architecture is undecided. The `*_clip_h.msgpack` / `*_vae_latent_*.msgpack` files in the dataset are leftovers and are ignored. | Caption style and length are configurable, and full metadata is kept so captions can be regenerated or re-filtered later. |
| Q2 | **vLLM backend for the cloud, HF transformers backend for local** (RTX 3060). | Batched `caption_batch()` interface (§6.1); new `joycaption_vllm.py` (§6.3). |
| Q3 | **bf16 on the cloud; NF4 locally with the vision tower left unquantized.** The `out_proj` hack is removed. | §6.2 / §6.3. |
| Q4 | **Per-source category prompts plus the game name** injected from the folder name. | §5.1, `sources` config (§9.2). |
| Q5 | **Caption types are configurable** and chosen per run (e.g. long, short, tags, SD prompt). | `caption_types` + `enabled_caption_types` (§5.1, §9.2). One JSONL record per (image, caption type). |
| Q6 | **JSONL is the primary output; `.txt` files are optional** (runner flag, or exported later). | §8. |
| Q7 | **Sharding (`--shard i/N`) and crash-safe resume.** | §4.3, §8.4, §11. |
| Q8 | **Docker image** (built on the official vLLM image) for the cloud. | §11.2, §15. |
| Q9 | **Per-image deterministic seed.** | §5.2. |
| Q10 | **Flag + retry + QA page.** Detect truncated, repetitive and empty captions; retry once with a new seed; flag if still bad; generate an HTML spot-check page. Raw text is never modified. | §5.3, §10.3. |
| Q11 | **Fix all confirmed v1.0 bugs** (B1–B10 of the review). | Folded into the redesigned modules; listed in §0.3. |
| Q12 | Spec, AGENTS and TODO are rewritten for the new workflow. **Further improvements must be proposed and approved by the user before being added.** | This document; `AGENTS.md`; `TODO.md`. |
| Q13 | GPU budget for a full pass: **$50–$200**. | Pilot (step 15) must report cost per 1 000 images and a projected full-run cost before the full run starts. |
| Q14 | Provider: **RunPod** (Community Cloud). Pilot on **RTX 4090**, compared with **L40S**. | §11: RunPod network volume layout and launch notes. The Docker image itself stays provider-neutral. |
| Q15 | Dataset transfer: **pack only the JPGs into ~1–2 GB tar files**, upload once to the network volume, unpack there. | New `scripts/pack_dataset.py` (§10.4). |
| Q16 | Proposal to compare with Qwen3-VL-8B: **rejected. JoyCaption only.** | No change; backends stay JoyCaption-only. |
| Q17 | Proposal to add `pyproject.toml`: **accepted.** | Package installable with `pip install -e .`; the `sys.path` hack in `conftest.py` and `PYTHONPATH` in Docker are removed (§14, §15). |

### 0.2 Implementation choices derived from those decisions (approved 2026-11-07)

These are not new features. They are the concrete mechanics needed to implement the decisions above. The user approved all of them on 2026-11-07.

| ID | Choice | Why |
|----|--------|-----|
| D1 | A single dataset root `input_dir` replaces the list `source_dirs`, with optional `include_sources` / `exclude_sources` filters (top-level folder names). | Resume and sharding are keyed by the path relative to one root. Several roots can produce colliding relative paths. |
| D2 | Shard membership is decided by a hash of the relative path, not by position in a sorted list. | Each image stays in the same shard even when files are added or removed. |
| D3 | Resume reads **all** `captions/*.jsonl` files in the output directory, not just the current shard's file. | Allows changing the shard count between runs, and merging outputs from several machines by copying files into one folder. |
| D4 | `max_images` limits images **still pending after resume**, not images discovered. | Fixes bug B4: a repeated dry run now advances instead of processing zero images. |
| D5 | Images are no longer validated in a separate pass before captioning. A decode failure becomes a `LoadFailure` (counted as *failed*, logged to `failures/`). | Fixes B7 (slow, silent startup at 280k+ images) and B8 (failed vs skipped). |
| D6 | Model backends are imported lazily through a factory (`get_model(backend)`). | CPU tests and the runner must not require `torch` or `vllm` to be installed. Today, importing `captioning.base` pulls in torch. |
| D7 | `--dry_run` CLI flag: discover, shard, resume-filter and print a plan (counts per source and caption type) without loading the model. | Needed to size cloud runs and check sharding before paying for a GPU. |
| D8 | Each run writes a manifest `runs/<run_id>.json` (config snapshot, package versions, git commit, summary). | Research reproducibility (Q1, Q9). |
| D9 | The HF backend processes requests one at a time inside `caption_batch()`. | HF LLaVA batched chat-templating is fragile, and per-request seeds need sequential generation. The HF backend is only for local testing; speed comes from vLLM. |
| D10 | Background thread pool decodes images ahead of the GPU (`runtime.num_workers`, default 4). | Keeps the GPU busy; JPEG decoding of 512×512 images is CPU-bound. |
| D11 | Optional `model.revision` (Hugging Face commit hash) is recorded and used when set. | Reproducibility (Q9). |

### 0.3 v1.0 bugs fixed by this spec (Q11)

| Bug | Fixed by |
|-----|----------|
| B1 Runner passes its own config to `model.load()`; CLI and runner would both load the model | §7.1: the runner owns the model lifecycle and receives an explicit `model_config`. The CLI never calls `load()`. |
| B2 `sprite.png` and `sprite.jpg` overwrite the same `sprite.txt` | §8.2 collision rule; JSONL is keyed by full `image_rel_path` (with extension). |
| B3 Resume ignores JSONL; JSONL-only runs duplicate records | §8.4: resume is driven by JSONL. |
| B4 `max_images` applied before resume | D4. |
| B5 Width/height recorded after resize | §3: `width`/`height` are always the original size. |
| B6 Extension matching case-sensitive on Linux | §4.2: case-insensitive. |
| B7 Discovery runs twice, walks the tree once per extension, and validates every image upfront | §4.2 single cached walk; D5. |
| B8 Corrupt images counted as skipped | D5: counted as failed. |
| B9 `requirements.txt` replaces PyPI with the CUDA 13 index; version floors too low | §15. |
| B10 Unclosed file handles; `+00:00` timestamps; truncation not flagged; no CPU tests for model code | §4.4, §8.1, §5.3, §13. |

---

## 1. Project Overview

### 1.1 Purpose

Caption a large (≥ 280k images, 110+ source folders) dataset of mostly pixel-art video-game imagery for research use. The captions will feed a future, not-yet-chosen generative-model training setup. Captions are produced with **JoyCaption Beta One** (`fancyfeast/llama-joycaption-beta-one-hf-llava`, LLaVA architecture: Llama 3.1 8B + SigLIP2 so400m @ 384px).

### 1.2 Dataset facts (observed 2026-11-06)

- Layout: `<dataset_root>/<source>/<NNNN>/<NNNNNN>.jpg`, where `<source>` is a game name (e.g. `celeste`, `stardew-valley`) or a collection (`pinterest-pixel-art-2023-07-01-v3`, `midjourney-dataset`, `pixel-art-dataset`, …).
- Images: RGB JPEG, 512×512 (crops of screenshots, sprite sheets, AI-generated pixel art).
- Each image sits next to legacy `*_clip_h.msgpack` / `*_vae_latent_{64,256}.msgpack` files (about 3 per image). These **are not inputs** and must be ignored cheaply by discovery.
- File stems repeat across folders (`000001.jpg` exists in many sources). The **relative path is the only unique key**.
- Not all sources are pixel art (e.g. `hades`, `prey`, `lies-of-p`, `fallout`). Category prompts (§5.1) describe each source neutrally. Deciding which sources to keep in the dataset is out of scope for this pipeline.

### 1.3 Guiding principles

- **Separation of concerns.** Ingestion, captioning backends, and orchestration are independent and replaceable.
- **Narrow typed interfaces.** All module boundaries use the dataclasses in §3.
- **Throughput first on the cloud, correctness first everywhere.** Batching via vLLM. Every caption is traceable to its image, prompt, parameters, seed and model revision.
- **Crash-safe and restartable.** A run can be killed at any moment (spot instances) and resumed with no duplicates and no lost work beyond the in-flight batch.
- **Raw output is never modified.** Quality flags are metadata. Post-processing happens downstream.
- **Reference implementation preserved.** `main.py` stays untouched as the original prototype.

### 1.4 Scope (v1.1)

| In scope | Out of scope |
|---|---|
| JoyCaption via vLLM (cloud) and HF transformers (local) | Other VLMs (only via an approved proposal) |
| Configurable caption types, category/source-aware prompts | Dataset curation, deduplication, filtering |
| Single GPU per process; several GPUs via several processes and shards | Tensor/pipeline parallelism across GPUs |
| JSONL output; optional / exported `.txt` | Training manifests for a specific trainer |
| Resume, sharding, per-image seeds, quality flags, retries | Caption rewriting or cleanup |
| QA HTML spot-check page | Interactive GUI / web app |
| Docker image for cloud; RunPod launch notes | Provider API automation (creating/stopping pods by script) |
| JPG-only tar packing for upload | Dataset hosting or versioning |

---

## 2. Architecture

### 2.1 Data flow

```text
 dataset_root ──► ImageDataLoader ──► CaptionRunner ──► CaptionModel.caption_batch()
   (disk)        discover/shard/       builds requests      (vLLM or HF backend)
                 resume/decode          (prompts, seeds)            │
                       ▲                     │  ◄── CaptionResult ──┘
                       │                     ▼
              ResumeIndex ◄──────── quality check → retry? → CaptionRecord
              (reads JSONL)                                        │
                                                    ShardWriter (JSONL, optional txt)
                                                                   │
                                                          output_dir (§8.1)
```

### 2.2 Dependency graph (strict, no cycles)

```text
common  ◄──  ingestion  ◄──  batch
common  ◄──  captioning ◄──  batch
```

- `common` imports nothing from the package and **never** imports `torch` or `vllm`.
- `captioning/__init__.py` exports `CaptionModel` and `get_model()` only. Backend modules are imported lazily inside `get_model()`.
- `vllm` is imported only inside `JoyCaptionVLLMModel.load()`. `torch`/`transformers` only inside `joycaption_hf.py`.

---

## 3. Data Types (`common/types.py`)

All types are `@dataclass`. Paths are `pathlib.Path` in memory and POSIX strings (`as_posix()`) on disk.

```python
@dataclass
class ImageRecord:
    """An image decoded from disk, ready for captioning."""
    path: Path              # absolute path
    rel_path: Path          # relative to input_dir, e.g. Path("celeste/0001/000001.jpg"); the unique key
    source: str             # first component of rel_path ("celeste"); "" for files directly in input_dir
    stem: str               # "000001"
    image: PIL.Image.Image  # RGB, NOT resized (the backend's processor resizes)
    width: int              # original width
    height: int             # original height

@dataclass
class LoadFailure:
    """An image that was discovered but could not be decoded."""
    path: Path
    rel_path: Path
    source: str
    error: str              # "<ExceptionType>: <message>"

@dataclass
class CaptionRequest:
    """One (image, caption type) generation job sent to a backend."""
    image_rel_path: Path    # routing key
    caption_type: str       # e.g. "long"
    image: PIL.Image.Image
    prompt: str             # fully rendered user prompt (§5.1)
    gen_params: dict        # max_new_tokens, temperature, top_p, top_k (None = backend default)
    seed: int               # §5.2
    attempt: int = 1        # 1 = first try, 2 = retry, ...

@dataclass
class CaptionResult:
    """Backend output for one CaptionRequest (same order as requests)."""
    text: str               # raw decoded text, .strip()-ed only
    finish_reason: str      # "stop" | "length" | "error"
    num_output_tokens: int
    error: str | None = None

@dataclass
class CaptionRecord:
    """One persisted caption = one JSONL line (§8.3)."""
    image_rel_path: Path
    image_path: Path
    source: str
    category: str
    caption_type: str
    caption_text: str
    status: str             # "ok" | "flagged" | "failed"
    flags: list[str]        # subset of {"truncated", "repetitive", "empty", "error"}
    model_name: str         # "joycaption-beta-one"
    model_path: str         # repo id or local path
    model_revision: str | None
    backend: str            # "vllm" | "hf"
    prompt: str
    generation_params: dict
    seed: int
    attempt: int
    finish_reason: str
    num_output_tokens: int
    image_width: int
    image_height: int
    run_id: str
    shard: str              # "i/N"
    timestamp_utc: str      # ISO 8601, second precision, "Z" suffix
    extra: dict             # free-form; e.g. {"source_name": "Celeste", "latency_s": 1.9}
```

`ImageRecord.stem` and `CaptionRecord.image_stem` from v1.0 are kept on `ImageRecord` only. `image_stem` is dropped from `CaptionRecord` because it is derivable from `image_rel_path`.

---

## 4. Ingestion (`ingestion/dataloader.py`)

### 4.1 `ImageDataLoader.__init__(config: dict)`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_dir` | `str` | required | Dataset root. Must exist (config error otherwise). |
| `extensions` | `list[str]` | `[".png", ".jpg", ".jpeg", ".webp"]` | Matched **case-insensitively**. |
| `recursive` | `bool` | `True` | Walk subdirectories. |
| `include_sources` | `list[str] \| None` | `None` | If set, only these top-level folders. |
| `exclude_sources` | `list[str]` | `[]` | Top-level folders to skip. |
| `shard` | `tuple[int,int]` | `(0, 1)` | `(index, count)`, `0 <= index < count`. |
| `max_images` | `int \| None` | `None` | Cap on **pending** images after resume (D4). |
| `num_workers` | `int` | `4` | Decode threads (D10). `0` = decode inline. |
| `prefetch` | `int` | `64` | Max decoded images held ahead of the consumer. |

The loader does **not** know about output files. The runner supplies a resume predicate (§4.3).

### 4.2 `discover() -> list[Path]`

- One `os.walk` over `input_dir` (no symlink following). Each file name is matched once against the lowercased extension set. Non-image files (e.g. `.msgpack`) are rejected by a suffix check only, never opened.
- Applies `include_sources` / `exclude_sources` while walking (pruned directories are not entered).
- Returns relative paths sorted by `as_posix()`.
- **Cached** on the instance; repeated calls do not re-walk.
- Logs an INFO progress line every 50 000 files examined, and a final count.

### 4.3 Planning: `plan(is_complete: Callable[[Path, str], bool], caption_types: list[str]) -> Plan`

1. Takes `discover()` results and keeps only paths in this shard: `int(sha1(rel_path.as_posix()).hexdigest()[:8], 16) % count == index` (D2).
2. For each image, computes the **pending caption types**: the requested types for which `is_complete(rel_path, type)` is `False`.
3. Images with no pending types are *already complete*.
4. Applies `max_images` to the pending images in sorted order.
5. Returns a `Plan`, an internal (non-public) structure with counts and the ordered list of `(rel_path, pending_types)`.

`Plan` is internal to `ingestion` and `batch`. It is a plain dataclass in `ingestion/dataloader.py`, not part of `common/types.py`.

### 4.4 `iter_images(plan) -> Iterator[ImageRecord | LoadFailure]`

- Yields in plan order. Decodes in a thread pool with bounded look-ahead (`prefetch`).
- Decoding uses `common.image_utils.load_image()`, which opens the file in a `with` block, calls `.convert("RGB")` (which forces a full decode inside the block), and records the original size.
- Any exception during decoding yields a `LoadFailure`. Nothing is silently dropped.

### 4.5 `common/image_utils.py`

| Function | Signature | Notes |
|---|---|---|
| `load_image` | `(path: Path, input_root: Path, target_size: tuple[int,int] \| None = None) -> ImageRecord` | Closes the file handle. `width`/`height` are the **original** size, even if `target_size` is given. Raises `ValueError` if `path` is not under `input_root`. |
| `validate_image` | `(path: Path) -> bool` | Kept as a utility (unused by the pipeline). Catches all `Exception` subclasses. The `verify()` handle-lifetime warning from v1.0 still applies. |

---

## 5. Prompts, Seeds & Quality (`common`)

### 5.1 Prompts (`common/prompts.py`)

The prompt for a request is built from three config sections (§9.2):

- `caption_types`: name → `{instruction, max_new_tokens, temperature, top_p, top_k}`
- `categories`: name → `context` template (may reference `{source_name}`)
- `sources`: top-level folder → `{category, name}`; unknown folders use `default_category`, and `name` defaults to the folder name title-cased with dashes replaced by spaces

```python
def render_prompt(caption_type: str, source: str, prompt_config: dict) -> str
```

Rendering: `prompt_config["template"].format(instruction=..., context=..., constraints=...)`, with whitespace collapsed and stripped. `constraints` is a global string (e.g. "Do not mention the image resolution.").

```python
def resolve_source(source: str, prompt_config: dict) -> tuple[str, str]   # (category, source_name)
def prompt_config_hash(prompt_config: dict) -> str                        # sha256 of canonical JSON, first 12 hex chars
```

**Draft defaults.** Wording will be tuned during the pilot (step 15):

| Caption type | Instruction | max_new_tokens |
|---|---|---|
| `long` | "Write a long detailed description for this image." | 512 |
| `short` | "Write a short description for this image in one or two sentences." | 96 |
| `tags` | "Write a list of Booru-like tags for this image." | 160 |
| `sd_prompt` | "Write a stable diffusion prompt for this image." | 160 |

| Category | Context (draft) |
|---|---|
| `game_crop` | "This image is a crop from a screenshot of the video game {source_name}." |
| `sprite_sheet` | "This image is a sprite sheet or a collection of video game assets." |
| `ai_pixel_art` | "This image is AI-generated pixel art." |
| `mixed_collection` | "" (no context) |

The draft `configs/sources.json` mapping all observed folders to a category is produced in step 11 and **must be approved by the user** before the pilot.

### 5.2 Seeds (`common/seeding.py`)

```python
def derive_seed(global_seed: int, rel_path: Path, caption_type: str, attempt: int) -> int
```

`int.from_bytes(sha256(f"{global_seed}|{rel_path.as_posix()}|{caption_type}|{attempt}".encode()).digest()[:4], "big") & 0x7FFFFFFF`

Reproducibility is **best-effort**. Identical output requires the same backend, model revision, dtype, library versions and GPU type. vLLM may also differ slightly when the batch composition changes. The seed, backend, revision and versions are recorded so differences can be explained.

### 5.3 Quality checks (`common/quality.py`)

```python
def check_caption(result: CaptionResult, cfg: dict) -> list[str]   # returns flags
```

| Flag | Rule (defaults in `quality` config) |
|---|---|
| `error` | `result.finish_reason == "error"` |
| `empty` | stripped text is empty |
| `truncated` | `finish_reason == "length"` |
| `repetitive` | any word 4-gram occurs ≥ `repeat_ngram_threshold` (default 4) times **or** zlib compression ratio < `min_compression_ratio` (default 0.25) |

Retry policy (applied by the runner): if any flag is set and `attempt <= max_retries` (default 1), re-request with `attempt+1` (new seed). After the last attempt:

- `error` or `empty` → status `failed` (caption text stored as-is, may be empty)
- other flags → status `flagged`
- no flags → status `ok`

---

## 6. Captioning Backends (`captioning`)

### 6.1 Interface (`captioning/base.py`)

```python
class CaptionModel(ABC):
    model_name: str            # "joycaption-beta-one"
    backend: str               # "vllm" | "hf"

    @abstractmethod
    def load(self, config: dict) -> None: ...
    @abstractmethod
    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]: ...
    @abstractmethod
    def unload(self) -> None: ...

    def caption(self, request: CaptionRequest) -> CaptionResult:
        """Convenience wrapper: caption_batch([request])[0]."""

    def describe(self) -> dict:
        """Versions & settings for the run manifest (model_path, revision, dtype, library versions)."""
```

Contract:

- `caption_batch` returns exactly `len(requests)` results in the same order.
- Per-request problems are returned as `CaptionResult(finish_reason="error", error=...)`, not raised.
- Raising is reserved for systemic failures (e.g. CUDA OOM for the whole batch). The runner handles that (§7.2).
- `unload()` must be safe to call at any time, including after a failed or partial `load()` and more than once.

```python
def get_model(backend: str, model_name: str = "joycaption-beta-one") -> CaptionModel   # lazy imports
```

### 6.2 HF backend (`captioning/joycaption_hf.py`, class `JoyCaptionHFModel`)

Local/dev backend. It replaces `joycaption.py`; `JoyCaptionModel` stays as an alias for one release.

`load()` keys (from `model` + `model.hf` config):

| Key | Default | Notes |
|---|---|---|
| `model_path` | `fancyfeast/llama-joycaption-beta-one-hf-llava` | Repo id or local dir |
| `revision` | `None` | Passed to `from_pretrained` |
| `dtype` | `bfloat16` | `bfloat16` \| `float16` \| `float32` |
| `quantization` | `nf4` | `nf4` \| `int8` \| `None` |
| `device_map` | `0` | |

Behaviour:

1. Quantization configs (NF4/int8) set `llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]`, so the vision encoder and projector stay in `dtype`.
2. **No `out_proj` replacement hack.**
3. The processor is used with its default 384×384 resize (no override).
4. For each request, in order: `torch.manual_seed(seed)` (and CUDA), build `[system, user]` messages, `apply_chat_template(tokenize=False, add_generation_prompt=True)`, `processor(text=[...], images=[...])`, move tensors to the device, cast `pixel_values` to the model dtype, and `generate(**inputs, do_sample=True, temperature, top_p, top_k, max_new_tokens, use_cache=True, suppress_tokens=None)` under `torch.inference_mode()`.
5. Trim prompt tokens and decode with `skip_special_tokens=True, clean_up_tokenization_spaces=False`, then `.strip()`. `finish_reason = "length"` if `num_output_tokens >= max_new_tokens` and the last token is not EOS, else `"stop"`.
6. Exceptions for one request become an error result. `torch.cuda.OutOfMemoryError` re-raises (systemic).

System prompt: `"You are a helpful image captioner."` (unchanged from `main.py`).

### 6.3 vLLM backend (`captioning/joycaption_vllm.py`, class `JoyCaptionVLLMModel`)

Cloud backend. `load()` keys (from `model` + `model.vllm`):

| Key | Default | Notes |
|---|---|---|
| `model_path`, `revision`, `dtype` | as §6.2 (`dtype` default `bfloat16`) | |
| `max_model_len` | `4096` | |
| `gpu_memory_utilization` | `0.90` | |
| `max_num_seqs` | `128` | Concurrent sequences |
| `enable_prefix_caching` | `True` | The system prompt and instructions are shared |
| `enforce_eager` | `False` | Set `True` if CUDA-graph capture fails |

Behaviour:

1. `vllm.LLM(model=..., revision=..., dtype=..., max_model_len=..., limit_mm_per_prompt={"image": 1}, ...)`.
2. `caption_batch` builds one chat conversation per request (same system and user structure as HF, image passed as a PIL image) and one `SamplingParams(temperature, top_p, top_k or -1, max_tokens, seed=request.seed)` per request. It calls `llm.chat(conversations, sampling_params=[...], use_tqdm=False)`.
3. Maps outputs to `CaptionResult` (`text.strip()`, `finish_reason`, `len(token_ids)`).
4. `unload()` deletes the engine, runs `gc.collect()` and `torch.cuda.empty_cache()`. It must not crash if vLLM was never loaded.

The exact vLLM version and any API differences (e.g. how `llm.chat` accepts PIL images) are pinned and verified in step 9. Deviations are recorded in `TODO.md`.

---

## 7. Runner (`batch/runner.py`)

### 7.1 `CaptionRunner.__init__(dataloader, model, config)`

`config` is the **full merged config** (§9). The runner reads `model`, `generation`, `prompts`, `quality`, `output`, `runtime`. It passes **only** the `model` section to `model.load()` (fixes B1). The CLI never calls `load()`.

Config validation happens in `__init__` (§9.3). Invalid config raises `ValueError`.

### 7.2 `run() -> dict`

1. `plan = dataloader.plan(resume_index.is_complete, enabled_caption_types)`, where `resume_index` is built from existing output (§8.4); with `resume=False`, nothing counts as complete. Log the plan.
2. If the plan is empty, return the summary without loading the model.
3. `try: model.load(model_config)` … `finally: model.unload()`. A load error is logged and **re-raised** (fail fast).
4. Consume `dataloader.iter_images(plan)` in groups of `runtime.batch_size` images:
   - A `LoadFailure` writes one `failed` record per pending caption type to `failures/` (§8.1) and counts as an image load failure.
   - Each `ImageRecord` expands into one `CaptionRequest` per pending caption type (prompt from §5.1, seed from §5.2, `attempt=1`).
   - Call `model.caption_batch(requests)`. If it **raises**, retry those requests one at a time. A request that still raises gets an error result. If the exception is a CUDA OOM on a single request, log CRITICAL and abort the run (re-raise after flushing).
   - Quality-check each result (§5.3). Collect retryable requests and re-submit them as a follow-up batch (attempt + 1) until resolved.
   - Build a `CaptionRecord` for each final result and hand it to the `ShardWriter`.
5. Progress log every `runtime.log_every_s` seconds (default 30): images done / planned, captions per second, flagged %, failed %, ETA.
6. Close the writer (flush + fsync), write the run manifest (§8.1), return the summary.

`KeyboardInterrupt`/`SIGTERM`: stop taking new batches, flush the writer, unload, write the manifest with `"interrupted": true`, then re-raise. Spot instances usually send SIGTERM before stopping.

### 7.3 Summary

```python
{
  "run_id": str, "shard": "i/N",
  "images_discovered": int,        # whole dataset after include/exclude
  "images_in_shard": int,
  "images_already_complete": int,
  "images_deferred": int,          # pending but cut by max_images
  "images_planned": int,
  "images_load_failed": int,
  "captions_planned": int,
  "captions_ok": int, "captions_flagged": int, "captions_failed": int,
  "retries": int,
  "interrupted": bool,
  "output_dir": str,
}
```

Invariants (asserted by tests):

- `images_in_shard == images_already_complete + images_deferred + images_planned`
- If not interrupted: `captions_planned == captions_ok + captions_flagged + captions_failed`

---

## 8. Output Formats & Resume

### 8.1 Output directory layout

```text
<output_dir>/
├── captions/shard-<iii>-of-<NNN>.jsonl     # one CaptionRecord per line (status ok|flagged|failed)
├── failures/shard-<iii>-of-<NNN>.jsonl     # LoadFailure entries + systemic errors
├── txt/<rel_dir>/<stem>.<caption_type>.txt # only if output.write_txt (or exported, §10.2)
├── runs/<run_id>.json                      # manifest: config, model.describe(), versions, git sha, summary
└── qa/<run_id>.html                        # QA page (§10.3)
```

`run_id` = `YYYYMMDDTHHMMSSZ-<shard i>-<4 hex random>`.

### 8.2 `.txt` files

- Content: `caption_text` only, UTF-8, no trailing newline.
- Written for status `ok` and `flagged`; never for `failed`.
- **Collision rule (B2):** if two images in the same directory share a stem (e.g. `sprite.png`, `sprite.jpg`), both use `<stem>.<ext>.<caption_type>.txt`. Discovery detects collisions once and logs a WARNING listing them.

### 8.3 JSONL record

One line per `CaptionRecord` (§3), keys in dataclass order, `ensure_ascii=False`. Paths are POSIX strings. Timestamps look like `"2026-11-06T19:14:05Z"`.

### 8.4 Writing & resume (`common/export_utils.py`)

`ShardWriter(output_dir, shard, write_txt)`:

- Opens its own shard file in append mode. Only one process writes a given shard file.
- `write(record)`: JSON line + `flush()`, then the `.txt` file if enabled. The JSONL is written **first** because it is the source of truth for resume.
- `fsync` every `output.fsync_every` records (default 100) and on `close()`.

`ResumeIndex.from_output_dir(output_dir) -> ResumeIndex`:

- Reads **every** `captions/*.jsonl` file (D3).
- A malformed final line (crash mid-write) is skipped with a WARNING. A malformed line elsewhere is an error.
- `is_complete(rel_path, caption_type)` is `True` if a record exists with status `ok` or `flagged`. `failed` records are retried on the next run.
- With duplicate records for the same key, the latest `timestamp_utc` wins.
- Logs a WARNING if existing records were produced with a different `prompt` for the same caption type and category (the prompt config changed between runs). They still count as complete. Use `--no-resume` or a new output dir to regenerate.

`build_record(...)`, `save_txt(record, output_dir, collision_set) -> Path` and `read_records(paths) -> Iterator[dict]` are public helpers.

---

## 9. Configuration

### 9.1 Precedence

CLI flags > `--config` JSON file > built-in defaults (`common/config.py: DEFAULT_CONFIG`). Loading, merging and validation live in `common/config.py`:

```python
def load_config(path: Path | None, overrides: dict) -> dict      # deep-merge
def validate_config(cfg: dict) -> None                            # raises ValueError with all problems listed
```

### 9.2 Config file (`configs/example_config.json`)

```json
{
  "input":   { "input_dir": "/data/pixel-art", "extensions": [".png", ".jpg", ".jpeg", ".webp"],
               "recursive": true, "include_sources": null, "exclude_sources": [] },
  "model":   { "backend": "vllm",
               "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
               "revision": null, "dtype": "bfloat16",
               "hf":   { "quantization": "nf4", "device_map": 0 },
               "vllm": { "max_model_len": 4096, "gpu_memory_utilization": 0.9, "max_num_seqs": 128,
                         "enable_prefix_caching": true, "enforce_eager": false } },
  "generation": {
    "enabled_caption_types": ["long", "short"],
    "caption_types": {
      "long":      { "instruction": "Write a long detailed description for this image.",
                     "max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": null },
      "short":     { "instruction": "Write a short description for this image in one or two sentences.",
                     "max_new_tokens": 96,  "temperature": 0.6, "top_p": 0.9, "top_k": null },
      "tags":      { "instruction": "Write a list of Booru-like tags for this image.",
                     "max_new_tokens": 160, "temperature": 0.6, "top_p": 0.9, "top_k": null },
      "sd_prompt": { "instruction": "Write a stable diffusion prompt for this image.",
                     "max_new_tokens": 160, "temperature": 0.6, "top_p": 0.9, "top_k": null }
    }
  },
  "prompts": {
    "system": "You are a helpful image captioner.",
    "template": "{instruction} {context} {constraints}",
    "constraints": "Do not mention the image resolution.",
    "default_category": "mixed_collection",
    "categories": {
      "game_crop":        "This image is a crop from a screenshot of the video game {source_name}.",
      "sprite_sheet":     "This image is a sprite sheet or a collection of video game assets.",
      "ai_pixel_art":     "This image is AI-generated pixel art.",
      "mixed_collection": ""
    },
    "sources_file": "configs/sources.json"
  },
  "quality": { "max_retries": 1, "repeat_ngram_threshold": 4, "min_compression_ratio": 0.25 },
  "output":  { "output_dir": "/data/captions", "write_txt": false, "resume": true, "fsync_every": 100,
               "qa_sample": 100 },
  "runtime": { "shard": "0/1", "batch_size": 64, "num_workers": 4, "prefetch": 64,
               "max_images": null, "seed": 0, "log_every_s": 30, "log_level": "INFO",
               "log_file": null, "gpu_hourly_cost": null }
}
```

`configs/sources.json`: `{"celeste": {"category": "game_crop", "name": "Celeste"}, ...}`. Inline `prompts.sources` is also accepted and takes precedence over `sources_file`.

### 9.3 Validation rules (all reported together)

- `input_dir` exists and is a directory.
- `backend ∈ {vllm, hf}`; `dtype ∈ {bfloat16, float16, float32}`; `hf.quantization ∈ {nf4, int8, null}`.
- `enabled_caption_types` is non-empty and each entry exists in `caption_types`.
- Every category referenced by `sources` exists in `categories`. `default_category` exists.
- `shard` parses as `i/N` with `0 <= i < N`.
- Positive ints: `batch_size`, `max_new_tokens`, `fsync_every`, `log_every_s`; `num_workers >= 0`; `max_images` is null or > 0.
- `0 < top_p <= 1`, `temperature >= 0`.

---

## 10. Scripts & CLI

### 10.1 `scripts/run_caption.py`

```text
usage: run_caption.py [--config CONFIG] [--input_dir DIR] [--output_dir DIR]
                      [--backend {vllm,hf}] [--model_path PATH] [--revision REV]
                      [--caption_types TYPE [TYPE ...]] [--shard I/N]
                      [--max_images N] [--batch_size N] [--seed N]
                      [--write_txt | --no-write_txt] [--no-resume]
                      [--gpu_hourly_cost USD] [--dry_run]
                      [--log_level {DEBUG,INFO,WARNING,ERROR}]
```

Thin wiring only: parse → `load_config` → `validate_config` → configure logging → build `ImageDataLoader` and `get_model(backend)` → `CaptionRunner(...).run()` (or `--dry_run`: print the plan as JSON, exit 0) → log the summary.

Exit codes: `0` completed; `1` config or model-load error; `2` interrupted; `3` completed but zero captions succeeded while > 0 were planned (systemic problem).

### 10.2 `scripts/export_txt.py`

`export_txt.py --output_dir DIR [--caption_types ...] [--include_flagged]`: rebuilds `txt/` from the JSONL files (latest record per key), using the §8.2 naming and collision rule.

### 10.3 `scripts/make_qa_page.py`

`make_qa_page.py --output_dir DIR --input_dir DIR [--n 200] [--seed 0] [--status ok,flagged] [--source ...]`

Writes a self-contained `qa/<name>.html`:

- Random sample of N images (stratified by source when possible).
- Image embedded as a base64 JPEG thumbnail, with all caption types side by side.
- Status and flags highlighted. Prompt, seed and token count shown on hover.
- No external CSS/JS. The runner calls the same function (`common/export_utils.generate_qa_page`) at the end of a run with `n = output.qa_sample` (default 100; 0 disables).

### 10.4 `scripts/pack_dataset.py`

`pack_dataset.py --input_dir DIR --out_dir DIR [--target_size_gb 1.5] [--extensions .jpg ...] [--include_sources ...] [--exclude_sources ...]`

- **Read-only on `input_dir`.** Writes only into `out_dir`, which must not be inside `input_dir`.
- Uses the same discovery as the dataloader (§4.2), so only image files are packed and `.msgpack` files are skipped.
- Writes `pixel-art-<NNN>.tar` files (uncompressed; JPEG doesn't compress further) of about `target_size_gb` each. Member names are the POSIX relative paths, so unpacking recreates the folder tree.
- Writes `manifest.json`: per tar, file count, bytes and sha256; total counts per source.
- Restartable: complete tars listed in the manifest are not rewritten.
- Companion: `scripts/unpack_dataset.sh` (cloud side) verifies sha256 and extracts all tars into a target dir.

---

## 11. Cloud Execution

### 11.1 Model

- One process per GPU. Several GPUs on one machine = several processes, each with `CUDA_VISIBLE_DEVICES=k --shard k/N`.
- Several machines = disjoint shard indices. All write to their own shard files and can be merged by copying `captions/*.jsonl` into one folder (D3).
- Spot/preemptible instances are supported via SIGTERM handling (§7.2) and resume (§8.4).

### 11.2 Docker (`docker/Dockerfile`)

- `FROM vllm/vllm-openai:<pinned tag>` (version chosen and verified in step 13).
- Copies the repo and runs `pip install --no-deps .` (pyproject, §15). Sets `ENTRYPOINT ["python", "/app/scripts/run_caption.py"]` (overrides the image's server entrypoint).
- `HF_HOME` points to a mounted volume (`/workspace/hf-cache`) so the ~16 GB model is downloaded once per volume, not baked into the image.
- The dataset and output directories are mounted volumes. No credentials are baked in. `HF_TOKEN` is an environment variable if ever needed.

### 11.3 Hardware guidance

- vLLM bf16 needs about 17 GB for weights plus KV cache: use ≥ 24 GB GPUs (L4 24GB is marginal; A10/L40S/A100/H100/4090/5090 are fine).
- HF NF4 (local) runs on 12 GB (RTX 3060).
- On a 24 GB card the KV cache is small, which limits concurrency; 48 GB cards (L40S/A6000) allow larger batches.
- Throughput is measured in the pilot (step 15). No numbers are assumed in this spec.

### 11.4 RunPod setup (Q14, Q15)

One **network volume** (same region as the pods) mounted at `/workspace`:

```text
/workspace/
├── hf-cache/           # HF_HOME — model downloaded once
├── dataset-tars/       # uploaded output of pack_dataset.py
├── pixel-art/          # unpacked dataset (input_dir)
└── captions/           # output_dir — survives pod termination
```

Flow:

1. Locally: `pack_dataset.py` → tars.
2. Upload the tars to the volume (`runpodctl send`/`receive`, or the volume's S3-compatible API; the exact method is verified and documented in README at step 13).
3. On a cheap CPU pod or the first GPU pod: `unpack_dataset.sh`.
4. GPU pod from the Docker image with `--shard i/N`.
5. On interruption, start a new pod with the same command; resume continues.
6. Download `captions/` (small) when done.

Budget guard (Q13): the runner logs images/hour, and if `runtime.gpu_hourly_cost` (optional, CLI `--gpu_hourly_cost`) is set, also $/1 000 images and the projected cost of the remaining work. The pilot must report these before the full run.

---

## 12. Error Handling

| Category | Example | Behaviour |
|---|---|---|
| Config error | missing `input_dir`, bad shard | Exit 1 before any GPU work; list all problems |
| Model load error | OOM, missing weights | Log traceback, `unload()`, exit 1 |
| Image decode error | truncated JPEG | `LoadFailure` → `failures/` + `failed` caption records; continue |
| Per-request generation error | bad input | Error result → retry policy → `failed` record; continue |
| Batch-level exception | transient CUDA error | Retry requests singly; continue |
| CUDA OOM on a single request | | Flush, unload, manifest, exit 1 |
| Write error | disk full | Log CRITICAL, flush what is possible, unload, exit 1 (no silent loss) |
| SIGTERM / Ctrl-C | spot reclaim | Graceful stop, exit 2 |

Principles: fail fast on config/model/disk errors; be resilient per image; always unload; use `logging` only (format `"%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"`, stderr, optional `runtime.log_file`).

---

## 13. Testing

- `pytest`. **The CPU suite must pass without `torch`, `transformers` or `vllm` installed.**
- Backends are tested on CPU with fakes: a fake processor/model for HF (checks message construction, dtype cast, prompt trimming, finish_reason, per-request seeding, error mapping) and a fake `LLM` object injected for vLLM (checks SamplingParams construction, ordering, finish_reason mapping).
- GPU tests: `@pytest.mark.gpu` plus the dual guard (CUDA available **and** model weights present), as in AGENTS.md.
- Required coverage per module:
  - `types` (construction)
  - `config` (merge precedence, every validation rule)
  - `prompts` (rendering, source resolution, defaults, hash stability)
  - `seeding` (determinism, sensitivity to each input)
  - `quality` (each flag, thresholds)
  - `image_utils` (original size, handle closed, errors)
  - `dataloader` (case-insensitive ext, msgpack ignored, include/exclude, shard partition is disjoint + complete + stable, max_images after resume, LoadFailure yielded, single walk)
  - `export_utils` (writer order, fsync cadence, malformed last line, latest-wins, collision naming, QA page is self-contained)
  - `runner` (summary invariants, B1 regression: `load()` receives only the model section, retry flow, batch-exception fallback, empty plan → no load, load error re-raised + unload called, interrupt flush)
  - CLI (`--dry_run`, exit codes)
  - `pack_dataset` (only images packed, tree preserved, size splitting, manifest checksums, restart skips complete tars, refuses `out_dir` inside `input_dir`, never writes to `input_dir`)
- Real-model checks are manual steps: local HF smoke test (step 14) and cloud pilot (step 15).

---

## 14. Package Layout

```text
pixel-art-auto-captioner/
├── main.py                         # v0 prototype, preserved, never modified
├── SPEC.md  AGENTS.md  TODO.md  REVIEW.md  README.md
├── pyproject.toml                  # package metadata; `pip install -e .`
├── requirements.txt                # shared runtime (pillow)
├── requirements-local.txt          # HF backend (Windows/Linux local)
├── requirements-dev.txt            # pytest
├── configs/
│   ├── example_config.json
│   ├── local_hf_config.json        # 3060 / NF4 settings
│   └── sources.json                # folder → category/name (user-approved)
├── docker/
│   └── Dockerfile
├── scripts/
│   ├── run_caption.py
│   ├── export_txt.py
│   ├── make_qa_page.py
│   ├── pack_dataset.py
│   └── unpack_dataset.sh
├── src/pixel_art_auto_captioner/
│   ├── __init__.py
│   ├── common/
│   │   ├── types.py  config.py  prompts.py  seeding.py  quality.py
│   │   ├── image_utils.py  export_utils.py
│   ├── ingestion/
│   │   └── dataloader.py
│   ├── captioning/
│   │   ├── base.py  joycaption_hf.py  joycaption_vllm.py
│   └── batch/
│       └── runner.py
└── tests/
    ├── conftest.py
    ├── test_types.py  test_config.py  test_prompts.py  test_seeding.py  test_quality.py
    ├── test_image_utils.py  test_dataloader.py  test_export_utils.py
    ├── test_base.py  test_joycaption_hf.py  test_joycaption_vllm.py
    ├── test_runner.py  test_cli.py  test_pack_dataset.py  test_model_gpu.py
```

`captioning/joycaption.py` is replaced by `joycaption_hf.py`, and `tests/test_model.py` by `test_joycaption_hf.py` + `test_model_gpu.py`.

---

## 15. Dependencies

| File | Contents |
|---|---|
| `pyproject.toml` | Package `pixel-art-auto-captioner`, `src/` layout, `requires-python >=3.10`, dependency `pillow>=10.0`; optional extras `local` (= requirements-local) and `dev` (= pytest). Tests import the installed package (no `sys.path` edits). |
| `requirements.txt` | `pillow>=10.0` (shared; no index-url lines) |
| `requirements-local.txt` | `transformers==5.12.*`, `accelerate==1.13.*`, `bitsandbytes==0.49.*`. **torch is installed separately** per the CUDA instructions in README (currently `torch 2.12.0+cu130` locally). |
| `requirements-dev.txt` | `pytest>=8` |
| Docker image | vLLM's pinned torch/transformers from the base image; the package adds only pillow. |

`torchaudio` and `torchvision` are removed (unused).

---

## 16. Implementation Sequence

Each step: implement → tests green → TODO.md updated → user notified. The step list and status live in `TODO.md`, which is authoritative for progress. Summary:

| Step | Work | GPU |
|---|---|---|
| 1 | `types.py` rework | No |
| 2 | `config.py` (+ defaults, validation) | No |
| 3 | `prompts.py`, `seeding.py`, `quality.py` | No |
| 4 | `image_utils.py` fixes | No |
| 5 | `dataloader.py` rework (walk, filters, shard, plan, prefetch) | No |
| 6 | `export_utils.py` rework (ShardWriter, ResumeIndex, txt, QA page) | No |
| 7 | `base.py` (`caption_batch`, `get_model`, lazy imports) | No |
| 8 | `joycaption_hf.py` + fake-based tests | No (GPU test gated) |
| 9 | `joycaption_vllm.py` + fake-based tests | No (GPU test gated) |
| 10 | `runner.py` rework | No |
| 11 | `run_caption.py` CLI, configs, draft `sources.json` (**user approval**) | No |
| 12 | `export_txt.py`, `make_qa_page.py`, `pack_dataset.py`, `unpack_dataset.sh` | No |
| 13 | `pyproject.toml`, requirements split, Dockerfile, README (local + RunPod instructions) | No |
| 14 | Local smoke test on RTX 3060, HF backend, ~20 images (**user runs**) | Yes |
| 15 | Cloud pilot: ~500 images across categories; measure throughput; review QA page; tune prompts (**user approval**) | Yes |
| 16 | Full dataset run | Yes |

---

## 17. Open Items

| Item | Owner | When |
|---|---|---|
| Approve `configs/sources.json` (folder → category/name) | User | Step 11 |
| Final prompt wording per caption type and category; decide whether game-name injection helps or causes hallucination | User + pilot | Step 15 |
| Pilot GPU decision: RTX 4090 vs L40S, by measured $/1 000 images | User + pilot | Step 15 |
| Which caption types to run in the full pass | User | Step 16 |
| Improvement proposals awaiting approval (see `TODO.md` → Proposals) | User | Any time |
