# SPECIFICATION — Pixel Art Auto-Captioner

**Version:** 1.0  
**Date:** 2026-06-16  
**Status:** Draft for implementation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Module 1 — Model (`captioning`)](#3-module-1--model-captioning)
4. [Module 2 — DataLoader (`ingestion`)](#4-module-2--dataloader-ingestion)
5. [Module 3 — Runner (`batch`)](#5-module-3--runner-batch)
6. [Utilities (`common`)](#6-utilities-common)
7. [Output Formats](#7-output-formats)
8. [Configuration](#8-configuration)
9. [Scripts](#9-scripts)
10. [Testing](#10-testing)
11. [GPU & Cloud Considerations](#11-gpu--cloud-considerations)
12. [Error Handling](#12-error-handling)
13. [Package Layout](#13-package-layout)
14. [Dependencies](#14-dependencies)
15. [Implementation Sequence](#15-implementation-sequence)

---

## 1. Project Overview

### 1.1 Purpose

A modular image-captioning pipeline that captions a dataset of pixel-art images using **Joy Caption Beta One** (a LLaVA-based vision-language model), with a clean architecture extensible to other VLMs.

### 1.2 Guiding principles

- **Separation of concerns.** The pipeline is split into three independent modules: model, dataloader, and runner. Each module can be developed, tested, and replaced independently.
- **Standardized interfaces.** Every module exposes a narrow, typed interface. This makes it easy to swap backends (e.g., replace JoyCaption with a different VLM) without touching the dataloader or orchestration code.
- **Reference implementation.** The existing `main.py` is a working monolithic prototype. It is preserved as a reference. The new pipeline lives under `src/pixel_art_auto_captioner/` and is invoked from a thin script inside `scripts/`.
- **GPU-ready.** Designed to run on cloud-provided GPUs (single-GPU for now, multi-GPU considered in architecture but not implemented yet).
- **Standardized output.** Captions are saved in formats that link each caption back to its source image and include metadata (model name, prompt, generation parameters, timestamp).

### 1.3 Scope (v1.0)

| In scope | Out of scope |
|---|---:|
| Modular JoyCaption pipeline | Captioning with other VLMs (architecturally allowed, not implemented) |
| Single-GPU execution | Multi-GPU data-parallel or sharding |
| Local model loading | Remote/API-based model inference |
| JSONL + sidecar `.txt` output | Dataset versioning or training manifests |
| Basic tests for each module | Evaluation/metric comparison between models |
| CLI entry point in `scripts/` | GUI or web interface |

---

## 2. Architecture

### 2.1 Data flow

```text
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────────┐
│  images  │ ──► │  Dataloader  │ ──► │  Model   │ ──► │   Exporter   │
│ (disk)   │     │  (ingestion) │     │(caption) │     │  (common)    │
└──────────┘     └──────────────┘     └──────────┘     └──────────────┘
                                                        │
                                                  ┌─────▼──────┐
                                                  │  captions   │
                                                  │  (disk)     │
                                                  └────────────┘
```

1. **Dataloader** discovers images from one or more directories/globs, validates them, and yields structured image records.
2. **Model** loads the VLM once, accepts an image + prompt, returns a caption string + metadata.
3. **Runner** wires them together: iterates over the dataloader, feeds each image to the model, and saves results through the exporter.
4. **Exporter** (in `common`) writes captions to disk in standardized formats.

### 2.2 Module ownership

```text
src/pixel_art_auto_captioner/
├── __init__.py
├── captioning/          # Module 1: model loading & inference
│   ├── __init__.py
│   ├── base.py          # Abstract CaptionModel interface
│   └── joycaption.py    # JoyCaption concrete implementation
├── ingestion/           # Module 2: image dataloader
│   ├── __init__.py
│   └── dataloader.py    # Image discovery, validation, batching
├── batch/               # Module 3: orchestration runner
│   ├── __init__.py
│   └── runner.py        # Wiring: dataloader → model → exporter
└── common/              # Shared utilities
    ├── __init__.py
    ├── types.py         # Dataclasses: ImageRecord, CaptionRecord, BatchConfig
    ├── image_utils.py   # Image loading, validation, resizing helpers
    └── export_utils.py  # Caption serialization (JSONL, txt sidecar)
```

### 2.3 Data types

```python
# common/types.py

@dataclass
class ImageRecord:
    """A single image ready for captioning."""
    path: Path              # absolute path to image file
    stem: str               # filename without extension (e.g. "sprite")
    rel_path: Path          # path relative to input_root
                            # (e.g. Path("folder1/sprite.png"))
    image: Image.Image      # loaded PIL image (RGB)
    width: int              # original pixel width
    height: int             # original pixel height

@dataclass
class CaptionRecord:
    """A generated caption linked to its source image."""
    image_path: Path        # source image path
    image_stem: str         # filename without extension (matches ImageRecord.stem)
    image_rel_path: Path    # relative path from input_root
                            # (matches ImageRecord.rel_path)
    caption_text: str       # generated caption string
    model_name: str         # e.g. "joycaption-beta-one"
    prompt_template: str    # the user prompt used
    generation_params: dict # temperature, max_tokens, top_p, etc.
    timestamp_utc: str      # ISO 8601 UTC timestamp
    image_width: int
    image_height: int
    extra: dict             # optional additional metadata
```

---

## 3. Module 1 — Model (`captioning`)

### 3.1 Abstract interface

```python
# captioning/base.py

class CaptionModel(ABC):
    """Abstract interface for any captioning VLM."""

    model_name: str

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load model weights, move to device, set eval mode."""
        ...

    @abstractmethod
    def caption(self, image: Image.Image, prompt: str, **gen_kwargs) -> tuple[str, dict]:
        """Generate a caption for a single image.

        Returns:
            (caption_text, generation_metadata_dict)
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory."""
        ...
```

### 3.2 JoyCaption implementation (`joycaption.py`)

**Class:** `JoyCaptionModel(CaptionModel)`

**`__init__(self, model_name="joycaption-beta-one")`**

Sets the model name.

**`load(self, config: dict)`**

Accepts a config dict with keys:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_path` | `str` | required | Local path or HuggingFace repo ID (e.g. `"fancyfeast/llama-joycaption-beta-one-hf-llava"`) |
| `torch_dtype` | `str` | `"bfloat16"` | `"bfloat16"`, `"float16"`, or `"float32"` |
| `device_map` | `str \| int` | `0` | `"auto"`, `"cuda:0"`, or integer GPU index |
| `quantization` | `str \| None` | `"nf4"` | `"nf4"`, `"int8"`, or `None` (full precision) |
| `image_size` | `tuple[int,int]` | `(384, 384)` | Image processor resize target |

Behaviour:
1. If `quantization == "nf4"`, create `BitsAndBytesConfig` with NF4 settings (matching the reference `main.py`).
2. If `quantization == "int8"`, create 8-bit config.
3. If `quantization is None`, load in the requested `torch_dtype` with no quantization.
4. Load `AutoProcessor` and configure `image_processor.size` and `do_resize`.
5. Load `LlavaForConditionalGeneration` with the chosen dtype/quantization and `device_map`.
6. Set `model.eval()`.
7. Apply the vision-tower head fix (the `out_proj` `Linear` replacement from `main.py`), guarded by a try/except since future model versions may not need it.
8. Log success.

**`caption(self, image: Image.Image, prompt: str, **gen_kwargs) -> tuple[str, dict]`**

Accepts:
- `image`: a PIL Image (already validated, assumed RGB).
- `prompt`: the user-prompt text (system prompt is added internally by the model).
- `**gen_kwargs`: optional overrides for generation parameters.

Default generation parameters (from reference `main.py`):

| Parameter | Default |
|-----------|---------|
| `max_new_tokens` | `512` |
| `do_sample` | `True` |
| `temperature` | `0.6` |
| `top_p` | `0.9` |
| `top_k` | `None` |
| `use_cache` | `True` |

Behaviour:
1. Build the message list: `[{"role": "system", "content": "You are a helpful image captioner."}, {"role": "user", "content": prompt}]`.
2. Apply `processor.apply_chat_template()`.
3. Tokenize text + process image with `processor(text=..., images=..., return_tensors="pt")`.
4. Move tensors to model device, cast `pixel_values` to the model's dtype.
5. Run `model.generate(**inputs, **merged_gen_kwargs)` under `torch.inference_mode()`.
6. Trim prompt tokens from output.
7. Decode → strip → return `(caption_text, {"gen_params": {...}})`.

**`unload(self)`**

Calls `del self.model`, `del self.processor`, and `torch.cuda.empty_cache()`.

---

## 4. Module 2 — DataLoader (`ingestion`)

### 4.1 `ImageDataLoader`

**`__init__(self, config: dict)`**

Accepts a config dict:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `source_dirs` | `list[str]` | required | One or more directory paths to scan |
| `extensions` | `list[str]` | `[".png", ".jpg", ".jpeg", ".webp"]` | Allowed file extensions |
| `recursive` | `bool` | `True` | Recurse into subdirectories |
| `max_images` | `int \| None` | `None` | Limit total images (for dry runs) |
| `skip_existing` | `bool` | `True` | Skip images that already have output files |
| `output_dir` | `str \| None` | `None` | Output directory for skip_existing check |
| `image_size` | `tuple[int,int] \| None` | `None` | If set, resize images during loading |

**`discover(self) -> list[Path]`**

Discovers image paths from `source_dirs`, deduplicates, sorts, and optionally limits by `max_images`. Returns a sorted list of unique `Path` objects.

**`load(self, path: Path) -> ImageRecord`**

Loads a single image from disk:
1. Open with PIL.
2. Convert to RGB.
3. Optionally resize.
4. Return an `ImageRecord`.

**`__iter__(self) -> Iterator[ImageRecord]`**

Discovers images, then yields `ImageRecord` objects one at a time. Skips images whose output file already exists when `skip_existing` is enabled.

**`__len__(self) -> int`**

Returns the number of images that will be processed (after filtering).

---

## 5. Module 3 — Runner (`batch`)

### 5.1 `CaptionRunner`

**`__init__(self, dataloader: ImageDataLoader, model: CaptionModel, config: dict)`**

Accepts a config dict:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | `str` | required | Directory for caption output files |
| `output_formats` | `list[str]` | `["txt", "jsonl"]` | Output format(s): `"txt"`, `"jsonl"`, or both |
| `prompt_template` | `str` | required | Caption prompt text |
| `generation_params` | `dict` | `{}` | Overrides for model generation parameters |
| `resume` | `bool` | `True` | Skip images that already have output |

**`run(self) -> dict`**

1. Loads the model via `model.load(config)`.
2. Iterates over `dataloader`.
3. For each `ImageRecord`, calls `model.caption(...)`.
4. On success, creates a `CaptionRecord` and passes it to the exporter.
5. On failure, logs the error and continues.
6. After all images, returns a summary dict:

```python
{
    "total": int,        # images discovered
    "succeeded": int,    # successfully captioned
    "failed": int,       # errors during captioning
    "skipped": int,      # already existed
    "output_dir": str,   # where captions were saved
}
```

7. Calls `model.unload()` on completion (or in a `finally` block).

### 5.2 Logging

The runner uses Python's `logging` module with this format:

```text
YYYY-MM-DD HH:MM:SS  LEVEL     message
```

Logged events:
- Model load start / success / failure.
- Image discovery count.
- Per-image progress: `"[N/total] Processing: <filename>"`, `"Saved: <filename>"`, `"Skipping (exists): <filename>"`.
- Per-image errors.
- Final summary.

---

## 6. Utilities (`common`)

### 6.1 `image_utils.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_image` | `(path: Path, input_root: Path, target_size: tuple[int,int] \| None = None) -> ImageRecord` | Opens an image, converts to RGB, optionally resizes. ``stem`` is the bare filename without extension; ``rel_path`` captures the path relative to *input_root* for directory-preserving output. Raises ``ValueError`` if *path* is not under *input_root*. |
| `validate_image` | `(path: Path) -> bool` | Returns `True` if the file can be opened as a valid image. **⚠️ Uses ``PIL.Image.verify()`` which closes the underlying file handle** — the verified handle is not reusable for pixel access. ``load_image()`` uses a separate ``Image.open()`` call; keep these two code paths separate. |

### 6.2 `export_utils.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `save_txt_sidecar` | `(record: CaptionRecord, output_dir: Path) -> Path` | Writes ``{image_rel_path.with_suffix('.txt')}`` under *output_dir*, preserving the input directory structure (e.g. ``output/folder1/sprite.txt``) |
| `save_jsonl_entry` | `(record: CaptionRecord, output_dir: Path) -> Path` | Appends one JSON line to `captions.jsonl` in output_dir |
| `build_record` | `(image: ImageRecord, caption: str, model_name: str, prompt: str, gen_params: dict) -> CaptionRecord` | Constructs a `CaptionRecord` with UTC timestamp |

**JSONL record schema:**

```json
{
  "image_path": "/absolute/path/to/image.png",
  "image_stem": "image",
  "caption_text": "A pixel art scene showing...",
  "model_name": "joycaption-beta-one",
  "prompt_template": "Write a highly detailed...",
  "generation_params": {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 512},
  "timestamp_utc": "2026-06-16T12:00:00Z",
  "image_width": 64,
  "image_height": 64
}
```

### 6.3 `types.py`

Contains `ImageRecord` and `CaptionRecord` dataclasses as specified in section 2.3.

---

## 7. Output Formats

Two formats are supported, both produced from the same `CaptionRecord`:

### 7.1 Sidecar text (`.txt`)

- **Path:** `{output_dir}/{image_rel_path.with_suffix('.txt')}`, preserving the input directory structure (e.g. ``output/folder1/sprite.txt``).
- **Content:** caption text only, UTF-8 encoded.
- **Purpose:** Human-readable, matches the reference `main.py` output convention.

### 7.2 JSONL manifest (`captions.jsonl`)

- **Path:** `{output_dir}/captions.jsonl`
- **Content:** One JSON object per line, per the schema in §6.2.
- **Purpose:** Machine-readable, full metadata, filterable by model/prompt/params.

<!-- ### 7.3 Visual Evaluation Deck (`index.html`)

To eliminate human evaluation friction, the runner will automatically compile a lightweight, standalone `index.html` file in the output directory upon execution completion.

- **Layout:** A clean browser grid displaying all processed images.
- **Content:** Hovering or clicking on a pixel-art image dynamically displays its generated caption, token counts, processing time, and confidence parameters side-by-side.
- **Validation:** The HTML file includes a client-side JavaScript validator that checks whether the companion `.txt` and `.jsonl` entries are structurally present and aligned with the images on disk. If a sidecar file is missing or malformed, the corresponding image card highlights in red to flag a pipeline failure visually.

Implementation responsibility: `export_utils.py` provides a `generate_visual_deck()` function; the runner calls it after all images are processed. The `index.html` is self-contained (no external CSS/JS dependencies) and viewable directly in any modern browser. -->

---

## 8. Configuration

The runner accepts configuration in two ways, with this precedence:

1. **CLI arguments** (highest priority, override config file values).
2. **JSON config file** (optional, provides defaults).

### 8.1 CLI interface (in `scripts/run_caption.py`)

```text
usage: run_caption.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR
                      [--model_path MODEL_PATH] [--prompt PROMPT]
                      [--config CONFIG] [--extensions EXTENSIONS [EXTENSIONS ...]]
                      [--no-recursive] [--max_images MAX_IMAGES]
                      [--output_formats {txt,jsonl} [{txt,jsonl} ...]]
                      [--temperature TEMPERATURE] [--top_p TOP_P]
                      [--max_new_tokens MAX_NEW_TOKENS]
                      [--device DEVICE] [--quantization {nf4,int8,none}]
                      [--no-resume] [--log_level {DEBUG,INFO,WARNING,ERROR}]
```

Key arguments:
- `--input_dir` (required): directory containing images.
- `--output_dir` (required): directory for caption output.
- `--model_path`: local path or HuggingFace repo ID (default: `"fancyfeast/llama-joycaption-beta-one-hf-llava"`).
- `--prompt`: caption prompt text (default: pixel-art captioning prompt from `main.py`).
- `--config`: path to JSON config file for additional/default settings.
- `--device`: GPU index or `"auto"` (default: `0`).
- `--quantization`: `nf4`, `int8`, or `none` (default: `nf4`).
- `--output_formats`: `txt`, `jsonl`, or both (default: both).

### 8.2 JSON config file format

```json
{
  "input": {
    "source_dirs": ["./input"],
    "extensions": [".png", ".jpg", ".jpeg", ".webp"],
    "recursive": true,
    "max_images": null
  },
  "model": {
    "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "torch_dtype": "bfloat16",
    "device_map": 0,
    "quantization": "nf4",
    "image_size": [384, 384]
  },
  "generation": {
    "prompt_template": "Write a highly detailed, descriptive caption for this pixel art video game screenshot image.",
    "max_new_tokens": 512,
    "do_sample": true,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": null,
    "use_cache": true
  },
  "output": {
    "output_dir": "./output",
    "output_formats": ["txt", "jsonl"],
    "resume": true
  }
}
```

### 8.3 Default prompt

The default captioning prompt (from `main.py`):

> Write a highly detailed, descriptive caption for this pixel art video game screenshot image.

This is configurable via `--prompt` or the config file, but the default should remain the same as the prototype.

---

## 9. Scripts

### 9.1 `scripts/run_caption.py` — Entry point

A thin script that:

1. Parses CLI arguments (using `argparse`).
2. If `--config` is provided, merges with CLI overrides.
3. Constructs an `ImageDataLoader`.
4. Constructs a `JoyCaptionModel` and calls `load()`.
5. Constructs a `CaptionRunner` and calls `run()`.
6. Prints the summary dict and exits with code 0 on success or 1 on failure.

**No model logic lives in this script.** It only wires the three modules together.

### 9.2 `scripts/` directory

Future scripts (e.g., `compare_models.py`, `validate_dataset.py`) will also live here. Only `run_caption.py` is in scope for v1.0.

---

## 10. Testing

### 10.1 Framework

Use `pytest`. Add it to `requirements.txt` or a `requirements-dev.txt`.

### 10.2 Test layout

```text
tests/
├── __init__.py
├── conftest.py              # shared fixtures: sample images, dummy configs
├── test_dataloader.py       # ImageDataLoader unit tests
├── test_model.py            # JoyCaptionModel unit tests (may need GPU)
├── test_runner.py           # CaptionRunner integration tests
├── test_export_utils.py     # export_utils unit tests
├── test_image_utils.py      # image_utils unit tests
└── test_types.py            # dataclass construction/serialization tests
```

### 10.3 Required tests

**`test_dataloader.py`:**
- `test_discover_finds_images` — discovers PNGs in a temp directory.
- `test_discover_filters_by_extension` — ignores non-image files.
- `test_discover_recursive` — finds images in subdirectories.
- `test_discover_max_images` — respects the limit.
- `test_load_returns_image_record` — returns correct shape/type.
- `test_load_resize` — resizes when target_size is set.
- `test_skip_existing` — skips images with existing output files.
- `test_len` — returns correct count.

> **⚠️ Source-dir boundary testing pattern:** When testing that a path is rejected because it falls outside configured source directories, do **not** nest the out-of-bounds file inside a source directory tree. Create explicit sibling directories under ``tmp_path`` instead:
>
> ```python
> source_dir = tmp_path / "source"
> outside_dir = tmp_path / "outside"   # sibling, not child of source_dir
> ```
>
> Nesting the orphan inside the source directory (e.g. ``tmp_path / "source" / "orphan.png"`` while also using ``tmp_path / "source"`` as a source dir) produces a false negative — the path unintentionally resolves under a source dir and the expected ``ValueError`` never fires.

**`test_model.py`:**
- `test_model_load_nf4` — loads without error (requires GPU).
- `test_model_load_no_quant` — loads in bfloat16 without quantization.
- `test_model_caption_returns_tuple` — returns (str, dict).
- `test_model_caption_non_empty` — produces non-empty caption for a simple image.
- `test_model_unload_frees_memory` — unload clears CUDA memory.
- These tests can be marked with `@pytest.mark.gpu` and skipped on CPU.

**`test_runner.py`:**
- `test_runner_integration` — end-to-end on a small temp image set.
- `test_runner_resume` — second run skips already-captioned images.
- `test_runner_summary` — returns correct counts.

**`test_export_utils.py`:**
- `test_save_txt_sidecar_creates_file` — file exists with correct content.
- `test_save_jsonl_appends_line` — JSONL file contains valid JSON per line.
- `test_build_record_includes_timestamp` — timestamp is ISO 8601 UTC.

### 10.4 Test fixtures (`conftest.py`)

- `sample_image_rgb` — creates a small synthetic RGB image (e.g., 64×64 pixel art) via PIL.
- `sample_image_dir` — temp directory with 3 synthetic images.
- `sample_config_dict` — a valid config dict for the runner.
- `sample_image_record` — a pre-built `ImageRecord` for export tests.
- `sample_caption_record` — a pre-built `CaptionRecord` for export tests.

---

## 11. GPU & Cloud Considerations

### 11.1 Single-GPU (v1.0)

- `device_map=0` by default.
- All tensors move to `cuda:0`.
- Model loads once, stays resident during the entire batch run.
- Images are processed one at a time (no batched inference).

### 11.2 Multi-GPU (future)

The architecture allows future multi-GPU support without changing the interface:
- `device_map="auto"` can be passed to `JoyCaptionModel.load()` for `accelerate`-based sharding across multiple GPUs.
- A future `ParallelRunner` subclass could shard the image list across multiple model replicas.

### 11.3 Cloud GPU requirements

- VRAM: ≥ 12 GB for 4-bit NF4 quantization (matching the RTX 3060 baseline from `main.py`).
- For fp16/bf16 without quantization: ≥ 24 GB VRAM.
- CUDA toolkit compatible with the installed PyTorch version.
- `bitsandbytes` requires a CUDA-compatible GPU; it does not work on CPU or macOS.

### 11.4 Environment

- Python 3.10+ (tested on 3.10).
- The virtual environment and dependency management should work with `pip` and a `requirements.txt`.
- The model itself is a ~16 GB download (two 8 GB safetensors shards). On cloud instances, cache it at a known path and pass `--model_path`.

---

## 12. Error Handling

### 12.1 Error categories

| Category | Example | Behaviour |
|----------|---------|-----------|
| **Config error** | Missing `--input_dir` | Exit immediately with message and code 1 |
| **Image load error** | Corrupt file | Log warning, increment `failed`, continue |
| **Model load error** | OOM, missing files | Exit immediately with traceback and code 1 |
| **Caption generation error** | CUDA error mid-batch | Log error, increment `failed`, continue |
| **Export error** | Disk full, permission | Log error, increment `failed`, continue |

### 12.2 Principles

- **Fail fast on config/model errors.** If the model can't load or the input directory doesn't exist, exit immediately — don't waste time.
- **Resilient on per-image errors.** A single bad image should not crash the entire batch. Log the error and move on.
- **Always unload the model.** Use a `try/finally` block in the runner to ensure `model.unload()` is called even on failure.
- **Use Python's `logging` module.** Do not use `print()`. Log to stderr by default; allow redirect to a file via a future config option.

---

## 13. Package Layout

Final file tree (files to create in implementation):

```text
pixel-art-auto-captioner/
├── README.md
├── SPEC.md                          # This file
├── requirements.txt
├── .gitignore
│
├── main.py                          # Reference prototype (preserved, not modified)
│
├── src/
│   └── pixel_art_auto_captioner/
│       ├── __init__.py              # Version, package docstring
│       ├── captioning/
│       │   ├── __init__.py
│       │   ├── base.py              # CaptionModel ABC
│       │   └── joycaption.py        # JoyCaptionModel
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── dataloader.py        # ImageDataLoader
│       ├── batch/
│       │   ├── __init__.py
│       │   └── runner.py            # CaptionRunner
│       └── common/
│           ├── __init__.py
│           ├── types.py             # ImageRecord, CaptionRecord
│           ├── image_utils.py       # load_image, validate_image
│           └── export_utils.py      # save_txt_sidecar, save_jsonl_entry
│
├── scripts/
│   └── run_caption.py               # Thin CLI entry point
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_dataloader.py
│   ├── test_model.py
│   ├── test_runner.py
│   ├── test_export_utils.py
│   ├── test_image_utils.py
│   └── test_types.py
│
├── configs/
│   └── example_config.json          # Example config file
│
└── .pi/                             # Pi agent configuration
    ├── settings.json
    └── skills/
        └── pixel-art-captioning-project/
            └── SKILL.md
```

---

## 14. Dependencies

### 14.1 Runtime

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.38.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
pillow>=10.0.0
```

### 14.2 Development / testing

```
pytest>=7.0.0
```

No other heavy dependencies. The reference prototype uses `torchaudio` but the new pipeline does not need it (it was probably pulled in by the PyTorch CUDA index but is unused). Remove `torchaudio` from `requirements.txt` unless actually needed.

---

## 15. Implementation Sequence

Recommended order, smallest useful increment first:

| Step | What | Tests? |
|------|------|--------|
| 1 | `common/types.py` — `ImageRecord` and `CaptionRecord` dataclasses | `test_types.py` |
| 2 | `common/image_utils.py` — `load_image`, `validate_image` | `test_image_utils.py` |
| 3 | `ingestion/dataloader.py` — `ImageDataLoader` | `test_dataloader.py` |
| 4 | `common/export_utils.py` — `save_txt_sidecar`, `save_jsonl_entry`, `build_record` | `test_export_utils.py` |
| 5 | `captioning/base.py` — `CaptionModel` ABC | (no separate tests — tested via concrete impl) |
| 6 | `captioning/joycaption.py` — `JoyCaptionModel` | `test_model.py` |
| 7 | `batch/runner.py` — `CaptionRunner` | `test_runner.py` |
| 8 | `scripts/run_caption.py` — CLI entry point | Manual integration test |
| 9 | `configs/example_config.json` | — |

Steps 1-4 can be implemented and tested without a GPU. Steps 5-7 require a GPU for meaningful testing. Steps 1-4 produce useful, testable code immediately and establish the data types the rest of the pipeline depends on.