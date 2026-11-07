"""Tests for ``common/config.py`` (SPEC v1.1 §9).

Covers ``DEFAULT_CONFIG`` (mirrors SPEC §9.2), ``load_config`` merge precedence
(CLI overrides > config file > defaults, deep merge), sources resolution
(inline ``prompts.sources`` > ``prompts.sources_file``), file errors, every
validation rule of SPEC §9.3 (each failing and passing), the "all problems
reported together" contract and the no-heavy-imports rule (SPEC §2.2).
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pixel_art_auto_captioner.common.config import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
)

# Verbatim copy of the SPEC §9.2 example config. DEFAULT_CONFIG must equal it.
SPEC_9_2_JSON = """
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
"""

SOURCES = {
    "celeste": {"category": "game_crop", "name": "Celeste"},
    "midjourney-dataset": {"category": "ai_pixel_art", "name": "Midjourney"},
}


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def valid_cfg(tmp_path: Path) -> dict:
    """A merged config that passes validation (real input_dir, inline sources)."""
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    return load_config(
        None,
        {
            "input": {"input_dir": str(input_dir)},
            "output": {"output_dir": str(tmp_path / "out")},
            "prompts": {"sources": copy.deepcopy(SOURCES)},
        },
    )


def _errors(cfg: dict) -> str:
    """Run validate_config and return the error message ('' if valid)."""
    try:
        validate_config(cfg)
    except ValueError as exc:
        return str(exc)
    return ""


def _problem_count(message: str) -> int:
    return sum(1 for line in message.splitlines() if line.startswith("  - "))


# ---------------------------------------------------------------------------
# 1. DEFAULT_CONFIG mirrors SPEC §9.2
# ---------------------------------------------------------------------------


def test_default_config_equals_spec_9_2():
    assert DEFAULT_CONFIG == json.loads(SPEC_9_2_JSON)


def test_default_config_section_and_key_order_matches_spec():
    spec = json.loads(SPEC_9_2_JSON)
    assert list(DEFAULT_CONFIG) == list(spec)
    for section in spec:
        assert list(DEFAULT_CONFIG[section]) == list(spec[section]), section


def test_default_config_has_no_inline_sources():
    assert "sources" not in DEFAULT_CONFIG["prompts"]


def test_default_config_is_json_serializable():
    assert json.loads(json.dumps(DEFAULT_CONFIG)) == DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# 2. load_config: merge precedence (CLI > file > defaults)
# ---------------------------------------------------------------------------


def test_load_config_defaults_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no configs/sources.json here
    cfg = load_config(None, {})
    assert cfg == DEFAULT_CONFIG
    assert cfg is not DEFAULT_CONFIG


def test_load_config_returns_deep_copy_of_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = copy.deepcopy(DEFAULT_CONFIG)
    cfg = load_config(None, {})
    cfg["runtime"]["batch_size"] = 1
    cfg["generation"]["enabled_caption_types"].append("tags")
    cfg["generation"]["caption_types"]["long"]["max_new_tokens"] = 1
    assert DEFAULT_CONFIG == before


def test_file_overrides_defaults_and_keeps_siblings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "c.json", {"runtime": {"batch_size": 8}, "model": {"hf": {"quantization": "int8"}}})
    cfg = load_config(path, {})
    assert cfg["runtime"]["batch_size"] == 8
    assert cfg["runtime"]["num_workers"] == 4  # sibling from defaults
    assert cfg["model"]["hf"] == {"quantization": "int8", "device_map": 0}
    assert cfg["model"]["vllm"] == DEFAULT_CONFIG["model"]["vllm"]


def test_overrides_beat_file_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "c.json", {"runtime": {"batch_size": 8, "seed": 7}})
    cfg = load_config(path, {"runtime": {"batch_size": 2}, "model": {"backend": "hf"}})
    assert cfg["runtime"]["batch_size"] == 2  # CLI > file
    assert cfg["runtime"]["seed"] == 7  # file > default
    assert cfg["model"]["backend"] == "hf"  # CLI > default
    assert cfg["runtime"]["shard"] == "0/1"  # default kept


def test_lists_are_replaced_not_concatenated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "c.json", {"generation": {"enabled_caption_types": ["tags"]}})
    cfg = load_config(path, {})
    assert cfg["generation"]["enabled_caption_types"] == ["tags"]
    cfg = load_config(path, {"generation": {"enabled_caption_types": ["short", "long"]}})
    assert cfg["generation"]["enabled_caption_types"] == ["short", "long"]


def test_explicit_null_replaces_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "c.json", {"model": {"hf": {"quantization": None}}})
    cfg = load_config(path, {"runtime": {"max_images": None}})
    assert cfg["model"]["hf"]["quantization"] is None
    assert cfg["runtime"]["max_images"] is None


def test_caption_types_deep_merge_adds_and_patches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(
        tmp_path / "c.json",
        {"generation": {"caption_types": {
            "long": {"max_new_tokens": 300},
            "alt": {"instruction": "Alt text.", "max_new_tokens": 40, "temperature": 0.5, "top_p": 1.0, "top_k": None},
        }}},
    )
    cfg = load_config(path, {})
    types = cfg["generation"]["caption_types"]
    assert types["long"]["max_new_tokens"] == 300
    assert types["long"]["instruction"] == DEFAULT_CONFIG["generation"]["caption_types"]["long"]["instruction"]
    assert set(types) == {"long", "short", "tags", "sd_prompt", "alt"}


def test_unknown_keys_pass_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(None, {"runtime": {"note": "x"}, "extra_section": {"a": 1}})
    assert cfg["runtime"]["note"] == "x"
    assert cfg["extra_section"] == {"a": 1}


def test_load_config_does_not_mutate_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    overrides = {"runtime": {"batch_size": 2}, "generation": {"enabled_caption_types": ["long"]}}
    snapshot = copy.deepcopy(overrides)
    cfg = load_config(None, overrides)
    cfg["generation"]["enabled_caption_types"].append("short")
    assert overrides == snapshot


def test_load_config_accepts_str_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "c.json", {"runtime": {"seed": 3}})
    assert load_config(str(path), {})["runtime"]["seed"] == 3


# ---------------------------------------------------------------------------
# 3. load_config: file errors
# ---------------------------------------------------------------------------


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json", {})


def test_invalid_json_config_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(path, {})


def test_non_object_config_raises_value_error(tmp_path):
    path = _write_json(tmp_path / "list.json", [1, 2])
    with pytest.raises(ValueError, match="JSON object"):
        load_config(path, {})


def test_non_dict_overrides_raise_value_error():
    with pytest.raises(ValueError, match="overrides"):
        load_config(None, [("runtime", {})])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. load_config: sources resolution (inline > sources_file)
# ---------------------------------------------------------------------------


def test_sources_file_loaded_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_json(tmp_path / "configs" / "sources.json", SOURCES)
    cfg = load_config(None, {})
    assert cfg["prompts"]["sources"] == SOURCES


def test_sources_file_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_json(tmp_path / "elsewhere" / "s.json", SOURCES)
    cfg = load_config(None, {"prompts": {"sources_file": str(path)}})
    assert cfg["prompts"]["sources"] == SOURCES


def test_inline_sources_take_precedence_over_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_json(tmp_path / "configs" / "sources.json", SOURCES)
    inline = {"celeste": {"category": "sprite_sheet", "name": "Celeste!"}}
    path = _write_json(tmp_path / "c.json", {"prompts": {"sources": inline}})
    cfg = load_config(path, {})
    assert cfg["prompts"]["sources"] == inline


def test_null_sources_file_gives_empty_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(None, {"prompts": {"sources_file": None}})
    assert cfg["prompts"]["sources"] == {}


def test_missing_sources_file_left_for_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(None, {"input": {"input_dir": str(tmp_path)}})
    assert "sources" not in cfg["prompts"]
    message = _errors(cfg)
    assert "prompts.sources_file" in message and "not found" in message
    assert _problem_count(message) == 1


def test_invalid_sources_file_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "sources.json").write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="Sources file"):
        load_config(None, {})


def test_non_object_sources_file_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_json(tmp_path / "configs" / "sources.json", ["celeste"])
    with pytest.raises(ValueError, match="JSON object"):
        load_config(None, {})


# ---------------------------------------------------------------------------
# 5. validate_config: the valid baseline
# ---------------------------------------------------------------------------


def test_valid_config_passes(valid_cfg):
    validate_config(valid_cfg)


def test_validate_does_not_mutate(valid_cfg):
    snapshot = copy.deepcopy(valid_cfg)
    validate_config(valid_cfg)
    assert valid_cfg == snapshot


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_both_backends_valid(valid_cfg, backend):
    valid_cfg["model"]["backend"] = backend
    validate_config(valid_cfg)


# ---------------------------------------------------------------------------
# 6. Rule: input_dir exists and is a directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", 5])
def test_input_dir_required(valid_cfg, value):
    valid_cfg["input"]["input_dir"] = value
    assert "input.input_dir: is required" in _errors(valid_cfg)


def test_input_dir_missing_key(valid_cfg):
    del valid_cfg["input"]["input_dir"]
    assert "input.input_dir: is required" in _errors(valid_cfg)


def test_input_dir_does_not_exist(valid_cfg, tmp_path):
    valid_cfg["input"]["input_dir"] = str(tmp_path / "missing")
    assert "input.input_dir: does not exist" in _errors(valid_cfg)


def test_input_dir_is_a_file(valid_cfg, tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    valid_cfg["input"]["input_dir"] = str(f)
    assert "input.input_dir: is not a directory" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 7. Rule: backend / dtype / hf.quantization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["tgi", "VLLM", None, ""])
def test_bad_backend(valid_cfg, value):
    valid_cfg["model"]["backend"] = value
    assert "model.backend" in _errors(valid_cfg)


@pytest.mark.parametrize("value", ["bfloat16", "float16", "float32"])
def test_good_dtype(valid_cfg, value):
    valid_cfg["model"]["dtype"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", ["bf16", "fp8", "int8", None])
def test_bad_dtype(valid_cfg, value):
    valid_cfg["model"]["dtype"] = value
    assert "model.dtype" in _errors(valid_cfg)


@pytest.mark.parametrize("value", ["nf4", "int8", None])
def test_good_quantization(valid_cfg, value):
    valid_cfg["model"]["hf"]["quantization"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", ["fp4", "awq", "NF4", "none"])
def test_bad_quantization(valid_cfg, value):
    valid_cfg["model"]["hf"]["quantization"] = value
    assert "model.hf.quantization" in _errors(valid_cfg)


def test_hf_section_not_object(valid_cfg):
    valid_cfg["model"]["hf"] = "nf4"
    assert "model.hf: must be an object" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 8. Rule: enabled_caption_types non-empty and defined
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [[], None, "long"])
def test_enabled_caption_types_must_be_non_empty_list(valid_cfg, value):
    valid_cfg["generation"]["enabled_caption_types"] = value
    assert "generation.enabled_caption_types: must be a non-empty list" in _errors(valid_cfg)


def test_enabled_caption_type_not_defined(valid_cfg):
    valid_cfg["generation"]["enabled_caption_types"] = ["long", "poem"]
    message = _errors(valid_cfg)
    assert "'poem' is not defined in generation.caption_types" in message
    assert _problem_count(message) == 1


def test_all_default_caption_types_can_be_enabled(valid_cfg):
    valid_cfg["generation"]["enabled_caption_types"] = ["long", "short", "tags", "sd_prompt"]
    validate_config(valid_cfg)


def test_caption_type_entry_not_object(valid_cfg):
    valid_cfg["generation"]["caption_types"]["tags"] = "Write tags."
    assert "generation.caption_types.tags: must be an object" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 9. Rule: sources categories exist; default_category exists
# ---------------------------------------------------------------------------


def test_source_with_unknown_category(valid_cfg):
    valid_cfg["prompts"]["sources"]["celeste"]["category"] = "screenshot"
    message = _errors(valid_cfg)
    assert "prompts.sources.celeste.category: 'screenshot' is not defined" in message
    assert _problem_count(message) == 1


def test_source_entry_not_object(valid_cfg):
    valid_cfg["prompts"]["sources"]["celeste"] = "game_crop"
    assert "prompts.sources.celeste: must be an object" in _errors(valid_cfg)


def test_sources_not_object(valid_cfg):
    valid_cfg["prompts"]["sources"] = ["celeste"]
    assert "prompts.sources: must be an object" in _errors(valid_cfg)


def test_source_from_file_with_unknown_category(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_json(tmp_path / "configs" / "sources.json", {"hades": {"category": "3d_game", "name": "Hades"}})
    cfg = load_config(None, {"input": {"input_dir": str(tmp_path)}})
    assert "prompts.sources.hades.category: '3d_game'" in _errors(cfg)


def test_source_category_can_be_user_defined(valid_cfg):
    valid_cfg["prompts"]["categories"]["3d_game"] = "This is a 3D game {source_name}."
    valid_cfg["prompts"]["sources"]["hades"] = {"category": "3d_game", "name": "Hades"}
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", ["unknown", None, ""])
def test_default_category_must_exist(valid_cfg, value):
    valid_cfg["prompts"]["default_category"] = value
    assert "prompts.default_category" in _errors(valid_cfg)


def test_categories_not_object(valid_cfg):
    valid_cfg["prompts"]["categories"] = None
    message = _errors(valid_cfg)
    assert "prompts.categories: must be an object" in message
    assert "prompts.default_category" in message


# ---------------------------------------------------------------------------
# 10. Rule: shard parses as i/N with 0 <= i < N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0/1", "3/4", "0/16", "15/16"])
def test_good_shard(valid_cfg, value):
    valid_cfg["runtime"]["shard"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize(
    "value", ["1/1", "4/4", "5/4", "-1/2", "0/0", "a/b", "1", "1/2/3", "", "/2", None, 0, [0, 1]]
)
def test_bad_shard(valid_cfg, value):
    valid_cfg["runtime"]["shard"] = value
    assert "runtime.shard" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 11. Rule: positive ints; num_workers >= 0; max_images null or > 0
# ---------------------------------------------------------------------------

_POSITIVE_INT_KEYS = [
    (("runtime", "batch_size"), "runtime.batch_size"),
    (("output", "fsync_every"), "output.fsync_every"),
    (("runtime", "log_every_s"), "runtime.log_every_s"),
    (("generation", "caption_types", "long", "max_new_tokens"), "generation.caption_types.long.max_new_tokens"),
    (("generation", "caption_types", "tags", "max_new_tokens"), "generation.caption_types.tags.max_new_tokens"),
]


def _set(cfg: dict, keys: tuple, value) -> None:
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


@pytest.mark.parametrize("keys,label", _POSITIVE_INT_KEYS, ids=[k[1] for k in _POSITIVE_INT_KEYS])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "8", None])
def test_positive_int_rejected(valid_cfg, keys, label, value):
    _set(valid_cfg, keys, value)
    message = _errors(valid_cfg)
    assert f"{label}: must be a positive integer" in message
    assert _problem_count(message) == 1


@pytest.mark.parametrize("keys,label", _POSITIVE_INT_KEYS, ids=[k[1] for k in _POSITIVE_INT_KEYS])
def test_positive_int_accepts_one(valid_cfg, keys, label):
    _set(valid_cfg, keys, 1)
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", [0, 1, 16])
def test_good_num_workers(valid_cfg, value):
    valid_cfg["runtime"]["num_workers"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", [-1, 2.0, None, False, "4"])
def test_bad_num_workers(valid_cfg, value):
    valid_cfg["runtime"]["num_workers"] = value
    assert "runtime.num_workers" in _errors(valid_cfg)


@pytest.mark.parametrize("value", [None, 1, 500])
def test_good_max_images(valid_cfg, value):
    valid_cfg["runtime"]["max_images"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", [0, -5, 2.5, True, "10"])
def test_bad_max_images(valid_cfg, value):
    valid_cfg["runtime"]["max_images"] = value
    assert "runtime.max_images" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 12. Rule: 0 < top_p <= 1, temperature >= 0 (per caption type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 1.0, 0.01, 0.9, None])
def test_good_top_p(valid_cfg, value):
    valid_cfg["generation"]["caption_types"]["short"]["top_p"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", [0, 0.0, -0.1, 1.01, 2, float("nan"), True, "0.9"])
def test_bad_top_p(valid_cfg, value):
    valid_cfg["generation"]["caption_types"]["short"]["top_p"] = value
    assert "generation.caption_types.short.top_p" in _errors(valid_cfg)


@pytest.mark.parametrize("value", [0, 0.0, 0.6, 1.5, None])
def test_good_temperature(valid_cfg, value):
    valid_cfg["generation"]["caption_types"]["sd_prompt"]["temperature"] = value
    validate_config(valid_cfg)


@pytest.mark.parametrize("value", [-0.1, -1, float("inf"), False, "0.6"])
def test_bad_temperature(valid_cfg, value):
    valid_cfg["generation"]["caption_types"]["sd_prompt"]["temperature"] = value
    assert "generation.caption_types.sd_prompt.temperature" in _errors(valid_cfg)


# ---------------------------------------------------------------------------
# 13. All problems reported together; malformed sections
# ---------------------------------------------------------------------------


def test_all_problems_reported_together(valid_cfg, tmp_path):
    cfg = valid_cfg
    cfg["input"]["input_dir"] = str(tmp_path / "missing")
    cfg["model"]["backend"] = "tgi"
    cfg["model"]["dtype"] = "bf16"
    cfg["model"]["hf"]["quantization"] = "fp4"
    cfg["generation"]["enabled_caption_types"] = ["poem"]
    cfg["generation"]["caption_types"]["long"]["max_new_tokens"] = 0
    cfg["generation"]["caption_types"]["long"]["top_p"] = 0
    cfg["generation"]["caption_types"]["long"]["temperature"] = -1
    cfg["prompts"]["sources"]["celeste"]["category"] = "nope"
    cfg["prompts"]["default_category"] = "nope"
    cfg["runtime"]["shard"] = "2/2"
    cfg["runtime"]["batch_size"] = 0
    cfg["output"]["fsync_every"] = 0
    cfg["runtime"]["log_every_s"] = 0
    cfg["runtime"]["num_workers"] = -1
    cfg["runtime"]["max_images"] = 0

    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    message = str(excinfo.value)
    expected = [
        "input.input_dir", "model.backend", "model.dtype", "model.hf.quantization",
        "generation.enabled_caption_types", "caption_types.long.max_new_tokens",
        "caption_types.long.top_p", "caption_types.long.temperature",
        "prompts.sources.celeste.category", "prompts.default_category",
        "runtime.shard", "runtime.batch_size", "output.fsync_every",
        "runtime.log_every_s", "runtime.num_workers", "runtime.max_images",
    ]
    for key in expected:
        assert key in message, key
    assert _problem_count(message) == len(expected)
    assert message.startswith(f"Invalid configuration ({len(expected)} problem(s)):")


@pytest.mark.parametrize("section", ["input", "model", "generation", "prompts", "output", "runtime"])
def test_section_not_object_reported(valid_cfg, section):
    valid_cfg[section] = None
    message = _errors(valid_cfg)
    assert f"{section}: must be an object" in message
    assert _problem_count(message) == 1


def test_empty_section_still_checked(valid_cfg):
    valid_cfg["runtime"] = {}
    message = _errors(valid_cfg)
    for key in ["runtime.shard", "runtime.batch_size", "runtime.log_every_s", "runtime.num_workers"]:
        assert key in message, key
    assert "runtime.max_images" not in message  # absent == null, which is allowed


def test_non_dict_config_raises():
    with pytest.raises(ValueError, match="must be an object"):
        validate_config(["not", "a", "dict"])  # type: ignore[arg-type]


def test_end_to_end_file_and_overrides_validate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    _write_json(tmp_path / "configs" / "sources.json", SOURCES)
    path = _write_json(
        tmp_path / "configs" / "local.json",
        {"input": {"input_dir": "data"}, "model": {"backend": "hf"}, "runtime": {"batch_size": 1}},
    )
    cfg = load_config(path, {"runtime": {"shard": "1/2"}, "generation": {"enabled_caption_types": ["tags"]}})
    validate_config(cfg)
    assert cfg["model"]["backend"] == "hf"
    assert cfg["runtime"]["shard"] == "1/2"
    assert cfg["prompts"]["sources"] == SOURCES


# ---------------------------------------------------------------------------
# 14. No heavy imports (SPEC §2.2)
# ---------------------------------------------------------------------------


def test_config_import_pulls_in_no_torch_or_vllm():
    """Import ``config`` in a clean interpreter: no torch, no vllm.

    ``PYTHONPATH`` is replaced (not extended) with the repo's ``src`` so that
    any test-only torch stub on the parent's path is not visible.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(src)
    code = (
        "import sys, pixel_art_auto_captioner.common.config; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
