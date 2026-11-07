"""Configuration defaults, loading and validation (SPEC v1.1 §9).

Precedence (SPEC §9.1): CLI flags > ``--config`` JSON file > built-in defaults
(:data:`DEFAULT_CONFIG`).

- :func:`load_config` deep-merges the three layers and resolves the effective
  ``prompts.sources`` mapping (inline ``prompts.sources`` takes precedence over
  ``prompts.sources_file``, SPEC §9.2).
- :func:`validate_config` checks the merged config against SPEC §9.3 and raises
  one ``ValueError`` that lists **every** problem found.

This module uses the standard library only. It imports nothing from the
package and never imports ``torch`` or ``vllm`` (SPEC §2.2).
"""

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Built-in defaults. Mirrors SPEC §9.2 (``configs/example_config.json``) key for
#: key and value for value. Never mutate it; :func:`load_config` returns a deep
#: copy. ``prompts.sources`` is intentionally absent: it is filled in by
#: :func:`load_config` from ``prompts.sources_file`` unless given inline.
DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "input_dir": "/data/pixel-art",
        "extensions": [".png", ".jpg", ".jpeg", ".webp"],
        "recursive": True,
        "include_sources": None,
        "exclude_sources": [],
    },
    "model": {
        "backend": "vllm",
        "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "revision": None,
        "dtype": "bfloat16",
        "hf": {"quantization": "nf4", "device_map": 0},
        "vllm": {
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 128,
            "enable_prefix_caching": True,
            "enforce_eager": False,
        },
    },
    "generation": {
        "enabled_caption_types": ["long", "short"],
        "caption_types": {
            "long": {
                "instruction": "Write a long detailed description for this image.",
                "max_new_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": None,
            },
            "short": {
                "instruction": "Write a short description for this image in one or two sentences.",
                "max_new_tokens": 96,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": None,
            },
            "tags": {
                "instruction": "Write a list of Booru-like tags for this image.",
                "max_new_tokens": 160,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": None,
            },
            "sd_prompt": {
                "instruction": "Write a stable diffusion prompt for this image.",
                "max_new_tokens": 160,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": None,
            },
        },
    },
    "prompts": {
        "system": "You are a helpful image captioner.",
        "template": "{instruction} {context} {constraints}",
        "constraints": "Do not mention the image resolution.",
        "default_category": "mixed_collection",
        "categories": {
            "game_crop": "This image is a crop from a screenshot of the video game {source_name}.",
            "sprite_sheet": "This image is a sprite sheet or a collection of video game assets.",
            "ai_pixel_art": "This image is AI-generated pixel art.",
            "mixed_collection": "",
        },
        "sources_file": "configs/sources.json",
    },
    "quality": {"max_retries": 1, "repeat_ngram_threshold": 4, "min_compression_ratio": 0.25},
    "output": {
        "output_dir": "/data/captions",
        "write_txt": False,
        "resume": True,
        "fsync_every": 100,
        "qa_sample": 100,
    },
    "runtime": {
        "shard": "0/1",
        "batch_size": 64,
        "num_workers": 4,
        "prefetch": 64,
        "max_images": None,
        "seed": 0,
        "log_every_s": 30,
        "log_level": "INFO",
        "log_file": None,
        "gpu_hourly_cost": None,
    },
}

_BACKENDS = ("vllm", "hf")
_DTYPES = ("bfloat16", "float16", "float32")
_QUANTIZATIONS = ("nf4", "int8", None)


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` deep-merged with ``override``.

    Nested dicts merge key by key. Any other value in ``override`` (including
    lists and ``None``) replaces the value in ``base``. Inputs are not mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_json_object(path: Path, what: str) -> dict[str, Any]:
    """Read a JSON file whose top level must be an object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not valid JSON or its top level is not an object.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{what} {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{what} {path} must contain a JSON object, got {type(data).__name__}")
    return data


def _resolve_sources(cfg: dict[str, Any]) -> None:
    """Fill ``cfg["prompts"]["sources"]`` in place (SPEC §9.2).

    Inline ``prompts.sources`` (any non-null value) wins. Otherwise, if
    ``prompts.sources_file`` is set and the file exists, its JSON object is
    loaded; a relative path is resolved against the current working directory.
    If ``sources_file`` is null, ``sources`` becomes ``{}``. If the file is
    missing, ``sources`` is left unset so :func:`validate_config` reports it
    together with the other problems.
    """
    prompts = cfg.get("prompts")
    if not isinstance(prompts, dict) or prompts.get("sources") is not None:
        return
    sources_file = prompts.get("sources_file")
    if sources_file is None:
        prompts["sources"] = {}
        return
    if not isinstance(sources_file, str):
        return  # reported by validate_config
    path = Path(sources_file)
    if not path.is_file():
        logger.debug("Sources file %s not found; left for validate_config", path)
        return
    prompts["sources"] = _read_json_object(path, "Sources file")
    logger.info("Loaded %d source mappings from %s", len(prompts["sources"]), path.as_posix())


def load_config(path: Path | None, overrides: dict) -> dict:
    """Build the effective config: defaults < config file < CLI overrides.

    The three layers are deep-merged (SPEC §9.1): nested objects merge key by
    key; any other value (lists, scalars, ``null``) replaces the lower layer's
    value. ``overrides`` uses the same nested shape as the config file, e.g.
    ``{"runtime": {"shard": "1/4"}}``; callers must omit flags the user did not
    give, because a ``None`` value is applied as an explicit ``null``.

    After merging, the effective ``prompts.sources`` mapping is resolved:
    inline ``prompts.sources`` wins over ``prompts.sources_file`` (a relative
    path is resolved against the current working directory). A missing sources
    file is not raised here; :func:`validate_config` reports it.

    The result is not validated; call :func:`validate_config` on it.
    :data:`DEFAULT_CONFIG` and ``overrides`` are never mutated.

    Args:
        path: JSON config file, or ``None`` to use only defaults and overrides.
        overrides: Nested dict of values that take precedence over the file.

    Returns:
        A new, fully merged config dict.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If the config file or the sources file is not valid JSON or
            does not contain a JSON object, or if ``overrides`` is not a dict.
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"overrides must be a dict, got {type(overrides).__name__}")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        path = Path(path)
        file_cfg = _read_json_object(path, "Config file")
        cfg = _deep_merge(cfg, file_cfg)
        logger.info("Loaded config file %s", path.as_posix())
    cfg = _deep_merge(cfg, overrides)
    _resolve_sources(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _section(cfg: dict[str, Any], name: str, errors: list[str]) -> dict[str, Any] | None:
    """Return ``cfg[name]`` if it is a dict; otherwise record an error and return ``None``."""
    value = cfg.get(name)
    if not isinstance(value, dict):
        errors.append(f"{name}: must be an object, got {value!r}")
        return None
    return value


def _is_int(value: Any) -> bool:
    """True for real ints (``bool`` excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """True for finite ints/floats (``bool`` excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_positive_int(value: Any, key: str, errors: list[str]) -> None:
    """Record an error unless ``value`` is an int > 0."""
    if not (_is_int(value) and value > 0):
        errors.append(f"{key}: must be a positive integer, got {value!r}")


def _parse_shard(value: Any) -> tuple[int, int] | None:
    """Parse ``"i/N"`` into ``(i, N)``; ``None`` if malformed or out of range."""
    if not isinstance(value, str):
        return None
    parts = value.strip().split("/")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        return None
    index, count = int(parts[0]), int(parts[1])
    if not 0 <= index < count:
        return None
    return index, count


def _validate_input(cfg: dict[str, Any], errors: list[str]) -> None:
    """``input_dir`` exists and is a directory."""
    section = _section(cfg, "input", errors)
    if section is None:
        return
    input_dir = section.get("input_dir")
    if not isinstance(input_dir, str) or not input_dir:
        errors.append(f"input.input_dir: is required (a directory path), got {input_dir!r}")
    elif not Path(input_dir).exists():
        errors.append(f"input.input_dir: does not exist: {input_dir}")
    elif not Path(input_dir).is_dir():
        errors.append(f"input.input_dir: is not a directory: {input_dir}")


def _validate_model(cfg: dict[str, Any], errors: list[str]) -> None:
    """``backend``, ``dtype`` and ``hf.quantization`` are allowed values."""
    section = _section(cfg, "model", errors)
    if section is None:
        return
    backend = section.get("backend")
    if backend not in _BACKENDS:
        errors.append(f"model.backend: must be one of {list(_BACKENDS)}, got {backend!r}")
    dtype = section.get("dtype")
    if dtype not in _DTYPES:
        errors.append(f"model.dtype: must be one of {list(_DTYPES)}, got {dtype!r}")
    hf = section.get("hf")
    if not isinstance(hf, dict):
        errors.append(f"model.hf: must be an object, got {hf!r}")
    else:
        quantization = hf.get("quantization")
        if quantization not in _QUANTIZATIONS:
            errors.append(
                f"model.hf.quantization: must be one of ['nf4', 'int8', null], got {quantization!r}"
            )


def _validate_generation(cfg: dict[str, Any], errors: list[str]) -> None:
    """Enabled caption types exist; per-type ``max_new_tokens``, ``top_p``, ``temperature``."""
    section = _section(cfg, "generation", errors)
    if section is None:
        return
    caption_types = section.get("caption_types")
    if not isinstance(caption_types, dict) or not caption_types:
        errors.append(f"generation.caption_types: must be a non-empty object, got {caption_types!r}")
        caption_types = {}

    enabled = section.get("enabled_caption_types")
    if not isinstance(enabled, list) or not enabled:
        errors.append(
            f"generation.enabled_caption_types: must be a non-empty list, got {enabled!r}"
        )
    else:
        for name in enabled:
            if not isinstance(name, str) or name not in caption_types:
                errors.append(
                    f"generation.enabled_caption_types: {name!r} is not defined in "
                    f"generation.caption_types (defined: {sorted(caption_types)})"
                )

    for name, spec in caption_types.items():
        key = f"generation.caption_types.{name}"
        if not isinstance(spec, dict):
            errors.append(f"{key}: must be an object, got {spec!r}")
            continue
        _check_positive_int(spec.get("max_new_tokens"), f"{key}.max_new_tokens", errors)
        top_p = spec.get("top_p")
        if top_p is not None and not (_is_number(top_p) and 0 < top_p <= 1):
            errors.append(f"{key}.top_p: must satisfy 0 < top_p <= 1 (or null), got {top_p!r}")
        temperature = spec.get("temperature")
        if temperature is not None and not (_is_number(temperature) and temperature >= 0):
            errors.append(
                f"{key}.temperature: must be >= 0 (or null), got {temperature!r}"
            )


def _validate_prompts(cfg: dict[str, Any], errors: list[str]) -> None:
    """Sources resolvable; every referenced category and ``default_category`` exist."""
    section = _section(cfg, "prompts", errors)
    if section is None:
        return
    categories = section.get("categories")
    if not isinstance(categories, dict):
        errors.append(f"prompts.categories: must be an object, got {categories!r}")
        categories = {}

    default_category = section.get("default_category")
    if default_category not in categories:
        errors.append(
            f"prompts.default_category: {default_category!r} is not defined in "
            f"prompts.categories (defined: {sorted(categories)})"
        )

    sources = section.get("sources")
    if sources is None:
        sources_file = section.get("sources_file")
        if sources_file is not None:
            errors.append(
                f"prompts.sources_file: file not found: {sources_file!r} (relative paths "
                f"are resolved against the current directory; set it to null or give "
                f"prompts.sources inline to run without a source mapping)"
            )
        return
    if not isinstance(sources, dict):
        errors.append(f"prompts.sources: must be an object, got {sources!r}")
        return
    for folder, entry in sources.items():
        if not isinstance(entry, dict):
            errors.append(f"prompts.sources.{folder}: must be an object, got {entry!r}")
            continue
        if "category" in entry and entry["category"] not in categories:
            errors.append(
                f"prompts.sources.{folder}.category: {entry['category']!r} is not defined "
                f"in prompts.categories (defined: {sorted(categories)})"
            )


def _validate_output(cfg: dict[str, Any], errors: list[str]) -> None:
    """``fsync_every`` is a positive int."""
    section = _section(cfg, "output", errors)
    if section is None:
        return
    _check_positive_int(section.get("fsync_every"), "output.fsync_every", errors)


def _validate_runtime(cfg: dict[str, Any], errors: list[str]) -> None:
    """``shard``, ``batch_size``, ``log_every_s``, ``num_workers``, ``max_images``."""
    section = _section(cfg, "runtime", errors)
    if section is None:
        return
    shard = section.get("shard")
    if _parse_shard(shard) is None:
        errors.append(f"runtime.shard: must be 'i/N' with 0 <= i < N, got {shard!r}")
    _check_positive_int(section.get("batch_size"), "runtime.batch_size", errors)
    _check_positive_int(section.get("log_every_s"), "runtime.log_every_s", errors)
    num_workers = section.get("num_workers")
    if not (_is_int(num_workers) and num_workers >= 0):
        errors.append(f"runtime.num_workers: must be an integer >= 0, got {num_workers!r}")
    max_images = section.get("max_images")
    if max_images is not None and not (_is_int(max_images) and max_images > 0):
        errors.append(f"runtime.max_images: must be null or a positive integer, got {max_images!r}")


def validate_config(cfg: dict) -> None:
    """Validate a merged config against SPEC §9.3, reporting all problems at once.

    Rules:

    - ``input.input_dir`` exists and is a directory.
    - ``model.backend`` in {vllm, hf}; ``model.dtype`` in {bfloat16, float16,
      float32}; ``model.hf.quantization`` in {nf4, int8, null}.
    - ``generation.enabled_caption_types`` is non-empty and each entry exists in
      ``generation.caption_types``.
    - Every category referenced by ``prompts.sources`` exists in
      ``prompts.categories``; ``prompts.default_category`` exists. If the
      sources mapping could not be resolved because ``prompts.sources_file``
      does not exist, that is reported too.
    - ``runtime.shard`` parses as ``i/N`` with ``0 <= i < N``.
    - Positive ints: ``runtime.batch_size``, every caption type's
      ``max_new_tokens``, ``output.fsync_every``, ``runtime.log_every_s``;
      ``runtime.num_workers >= 0``; ``runtime.max_images`` is null or > 0.
    - Every caption type: ``0 < top_p <= 1`` and ``temperature >= 0`` (``null``
      is allowed for both and means "backend default", SPEC §3).

    A section that is missing or not an object is reported as one problem and
    its rules are skipped. ``cfg`` is not modified.

    Args:
        cfg: Merged config, normally from :func:`load_config`.

    Raises:
        ValueError: If any rule fails. The message lists every problem, one per
            line.
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid configuration: must be an object, got {type(cfg).__name__}")
    _validate_input(cfg, errors)
    _validate_model(cfg, errors)
    _validate_generation(cfg, errors)
    _validate_prompts(cfg, errors)
    _validate_output(cfg, errors)
    _validate_runtime(cfg, errors)
    if errors:
        lines = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Invalid configuration ({len(errors)} problem(s)):\n{lines}")
