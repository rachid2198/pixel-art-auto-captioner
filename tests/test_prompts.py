"""Tests for ``common/prompts.py`` (SPEC v1.1 §5.1, §13).

Covers rendering, source resolution, the draft defaults (via ``DEFAULT_CONFIG``)
and the stability of ``prompt_config_hash``.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pixel_art_auto_captioner.common.config import DEFAULT_CONFIG
from pixel_art_auto_captioner.common.prompts import (
    prompt_config_hash,
    render_prompt,
    resolve_source,
)

SRC = Path(__file__).resolve().parent.parent / "src"


def _prompt_config(sources: dict | None = None) -> dict:
    """``prompts`` section of DEFAULT_CONFIG + ``caption_types``, as the runner builds it."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    pc = {**cfg["prompts"], "caption_types": cfg["generation"]["caption_types"]}
    pc["sources"] = (
        sources
        if sources is not None
        else {
            "celeste": {"category": "game_crop", "name": "Celeste"},
            "sprites-collection": {"category": "sprite_sheet", "name": "Sprite Collection"},
            "midjourney-dataset": {"category": "ai_pixel_art", "name": "Midjourney"},
            "pinterest-pixel-art": {"category": "mixed_collection", "name": "Pinterest"},
        }
    )
    return pc


@pytest.fixture
def pc() -> dict:
    return _prompt_config()


# ---------------------------------------------------------------------------
# 1. resolve_source
# ---------------------------------------------------------------------------


def test_resolve_known_source(pc):
    assert resolve_source("celeste", pc) == ("game_crop", "Celeste")


def test_resolve_unknown_source_uses_default_category_and_name(pc):
    assert resolve_source("stardew-valley", pc) == ("mixed_collection", "Stardew Valley")


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("celeste", "Celeste"),
        ("stardew-valley", "Stardew Valley"),
        ("lies-of-p", "Lies Of P"),
        ("pinterest-pixel-art-2023-07-01-v3", "Pinterest Pixel Art 2023 07 01 V3"),
        ("rpg2k", "Rpg2k"),  # not str.title()'s "Rpg2K"
        ("hollow--knight", "Hollow Knight"),  # dash runs collapse
        ("SNES-Games", "Snes Games"),
        ("", ""),
    ],
)
def test_default_source_name(folder, expected):
    pc = _prompt_config(sources={})
    assert resolve_source(folder, pc) == ("mixed_collection", expected)


def test_entry_without_category_uses_default_category(pc):
    pc["sources"]["hades"] = {"name": "Hades II"}
    assert resolve_source("hades", pc) == ("mixed_collection", "Hades II")


def test_entry_without_name_uses_default_name(pc):
    pc["sources"]["stardew-valley"] = {"category": "game_crop"}
    assert resolve_source("stardew-valley", pc) == ("game_crop", "Stardew Valley")


def test_entry_with_null_fields_uses_defaults(pc):
    pc["sources"]["stardew-valley"] = {"category": None, "name": None}
    assert resolve_source("stardew-valley", pc) == ("mixed_collection", "Stardew Valley")


def test_explicit_empty_name_is_kept(pc):
    pc["sources"]["celeste"] = {"category": "game_crop", "name": ""}
    assert resolve_source("celeste", pc) == ("game_crop", "")


def test_source_lookup_is_case_sensitive(pc):
    # Folder names are keys; a differently-cased folder is a different source.
    assert resolve_source("Celeste", pc) == ("mixed_collection", "Celeste")


@pytest.mark.parametrize("sources", ["missing", None, {}])
def test_missing_null_or_empty_sources_mapping(sources):
    pc = _prompt_config(sources={})
    if sources == "missing":
        pc.pop("sources")
    else:
        pc["sources"] = sources
    assert resolve_source("celeste", pc) == ("mixed_collection", "Celeste")


def test_resolve_does_not_check_category_existence(pc):
    pc["sources"]["weird"] = {"category": "nope", "name": "Weird"}
    assert resolve_source("weird", pc) == ("nope", "Weird")


def test_resolve_non_object_entry_raises(pc):
    pc["sources"]["celeste"] = "game_crop"
    with pytest.raises(ValueError, match="celeste"):
        resolve_source("celeste", pc)


def test_resolve_non_object_sources_raises(pc):
    pc["sources"] = ["celeste"]
    with pytest.raises(ValueError, match="sources"):
        resolve_source("celeste", pc)


@pytest.mark.parametrize("value", ["missing", None, 3])
def test_resolve_bad_default_category_raises(pc, value):
    if value == "missing":
        pc.pop("default_category")
    else:
        pc["default_category"] = value
    with pytest.raises(ValueError, match="default_category"):
        resolve_source("unknown", pc)


def test_resolve_does_not_mutate(pc):
    before = copy.deepcopy(pc)
    resolve_source("celeste", pc)
    resolve_source("unknown-folder", pc)
    assert pc == before


# ---------------------------------------------------------------------------
# 2. render_prompt with the draft defaults (SPEC §5.1 tables, DEFAULT_CONFIG)
# ---------------------------------------------------------------------------

CONSTRAINT = "Do not mention the image resolution."
INSTRUCTIONS = {
    "long": "Write a long detailed description for this image.",
    "short": "Write a short description for this image in one or two sentences.",
    "tags": "Write a list of Booru-like tags for this image.",
    "sd_prompt": "Write a stable diffusion prompt for this image.",
}
CONTEXTS = {
    "celeste": "This image is a crop from a screenshot of the video game Celeste.",
    "sprites-collection": "This image is a sprite sheet or a collection of video game assets.",
    "midjourney-dataset": "This image is AI-generated pixel art.",
    "pinterest-pixel-art": "",
    "unknown-folder": "",
}


@pytest.mark.parametrize("caption_type", sorted(INSTRUCTIONS))
@pytest.mark.parametrize("source", sorted(CONTEXTS))
def test_render_defaults_every_type_and_category(pc, caption_type, source):
    parts = [INSTRUCTIONS[caption_type], CONTEXTS[source], CONSTRAINT]
    expected = " ".join(p for p in parts if p)
    assert render_prompt(caption_type, source, pc) == expected


def test_render_long_game_crop_exact(pc):
    assert render_prompt("long", "celeste", pc) == (
        "Write a long detailed description for this image. "
        "This image is a crop from a screenshot of the video game Celeste. "
        "Do not mention the image resolution."
    )


def test_empty_context_leaves_no_double_space(pc):
    out = render_prompt("short", "pinterest-pixel-art", pc)
    assert "  " not in out
    assert out == (
        "Write a short description for this image in one or two sentences. "
        "Do not mention the image resolution."
    )


def test_game_crop_with_default_name(pc):
    pc["sources"]["stardew-valley"] = {"category": "game_crop"}
    assert "video game Stardew Valley." in render_prompt("long", "stardew-valley", pc)


def test_default_config_constants_match_spec_tables():
    caption_types = DEFAULT_CONFIG["generation"]["caption_types"]
    assert {k: v["instruction"] for k, v in caption_types.items()} == INSTRUCTIONS
    assert {k: v["max_new_tokens"] for k, v in caption_types.items()} == {
        "long": 512, "short": 96, "tags": 160, "sd_prompt": 160,
    }
    assert DEFAULT_CONFIG["prompts"]["default_category"] == "mixed_collection"
    assert DEFAULT_CONFIG["prompts"]["categories"]["mixed_collection"] == ""


# ---------------------------------------------------------------------------
# 3. render_prompt mechanics
# ---------------------------------------------------------------------------


def test_custom_template_order(pc):
    pc["template"] = "{context}\n\n{constraints}\t{instruction}"
    assert render_prompt("long", "midjourney-dataset", pc) == (
        "This image is AI-generated pixel art. "
        "Do not mention the image resolution. "
        "Write a long detailed description for this image."
    )


def test_whitespace_collapsed_and_stripped(pc):
    pc["template"] = "   {instruction}\n\n  {context}   {constraints}   "
    pc["caption_types"]["long"]["instruction"] = "Describe\tthis\n image."
    out = render_prompt("long", "celeste", pc)
    assert out == (
        "Describe this image. This image is a crop from a screenshot of the video game "
        "Celeste. Do not mention the image resolution."
    )


@pytest.mark.parametrize("constraints", [None, "", "missing"])
def test_missing_or_empty_constraints(pc, constraints):
    if constraints == "missing":
        pc.pop("constraints")
    else:
        pc["constraints"] = constraints
    assert render_prompt("tags", "midjourney-dataset", pc) == (
        "Write a list of Booru-like tags for this image. This image is AI-generated pixel art."
    )


def test_source_name_used_more_than_once(pc):
    pc["categories"]["game_crop"] = "From {source_name}. Game: {source_name}."
    assert render_prompt("long", "celeste", pc).count("Celeste") == 2


def test_braces_in_instruction_constraints_and_name_are_literal(pc):
    pc["caption_types"]["long"]["instruction"] = "Answer in {json}."
    pc["constraints"] = "Never write {curly}."
    pc["sources"]["celeste"]["name"] = "Cel{este}"
    assert render_prompt("long", "celeste", pc) == (
        "Answer in {json}. This image is a crop from a screenshot of the video game "
        "Cel{este}. Never write {curly}."
    )


def test_non_ascii_text_kept(pc):
    pc["sources"]["pokemon"] = {"category": "game_crop", "name": "Pokémon"}
    assert "video game Pokémon." in render_prompt("short", "pokemon", pc)


def test_render_is_deterministic_and_does_not_mutate(pc):
    before = copy.deepcopy(pc)
    first = render_prompt("long", "celeste", pc)
    assert render_prompt("long", "celeste", pc) == first
    assert pc == before


# ---------------------------------------------------------------------------
# 4. render_prompt errors
# ---------------------------------------------------------------------------


def test_unknown_caption_type_raises(pc):
    with pytest.raises(ValueError, match="poem"):
        render_prompt("poem", "celeste", pc)


def test_missing_caption_types_section_raises(pc):
    pc.pop("caption_types")
    with pytest.raises(ValueError, match="caption_types"):
        render_prompt("long", "celeste", pc)


@pytest.mark.parametrize("instruction", ["missing", None, 5])
def test_bad_instruction_raises(pc, instruction):
    if instruction == "missing":
        pc["caption_types"]["long"].pop("instruction")
    else:
        pc["caption_types"]["long"]["instruction"] = instruction
    with pytest.raises(ValueError, match="instruction"):
        render_prompt("long", "celeste", pc)


def test_undefined_category_raises(pc):
    pc["sources"]["celeste"]["category"] = "nope"
    with pytest.raises(ValueError, match="nope"):
        render_prompt("long", "celeste", pc)


def test_missing_categories_raises(pc):
    pc.pop("categories")
    with pytest.raises(ValueError, match="categories"):
        render_prompt("long", "celeste", pc)


@pytest.mark.parametrize("context", ["About {game}.", "Bad {", "Index {0}."])
def test_bad_context_template_raises(pc, context):
    pc["categories"]["game_crop"] = context
    with pytest.raises(ValueError, match="game_crop"):
        render_prompt("long", "celeste", pc)


@pytest.mark.parametrize("template", ["{instruction} {source_name}", "{instruction", None])
def test_bad_template_raises(pc, template):
    pc["template"] = template
    with pytest.raises(ValueError, match="template"):
        render_prompt("long", "celeste", pc)


# ---------------------------------------------------------------------------
# 5. prompt_config_hash
# ---------------------------------------------------------------------------


def _reference_hash(obj: dict) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def test_hash_format(pc):
    h = prompt_config_hash(pc)
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_matches_canonical_json_definition(pc):
    assert prompt_config_hash(pc) == _reference_hash(pc)


def test_hash_golden_values():
    # Pinned values (sha256 of the literal canonical bytes '{}' and
    # '{"a":[1,"é",null],"b":1}'). A change here means every recorded hash changes.
    assert prompt_config_hash({}) == "44136fa355b3"
    assert prompt_config_hash({"b": 1, "a": [1, "é", None]}) == "bc1cf18b35fa"


def test_hash_ignores_key_order_at_every_level():
    a = {"template": "{instruction}", "categories": {"x": "1", "y": "2"}, "k": {"p": 1, "q": 2}}
    b = {"k": {"q": 2, "p": 1}, "categories": {"y": "2", "x": "1"}, "template": "{instruction}"}
    assert prompt_config_hash(a) == prompt_config_hash(b)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pc: pc.__setitem__("template", "{context} {instruction} {constraints}"),
        lambda pc: pc.__setitem__("constraints", "Be brief."),
        lambda pc: pc["categories"].__setitem__("ai_pixel_art", "AI art."),
        lambda pc: pc["sources"]["celeste"].__setitem__("name", "Celeste Classic"),
        lambda pc: pc["caption_types"]["long"].__setitem__("instruction", "Describe."),
        lambda pc: pc.__setitem__("default_category", "game_crop"),
        lambda pc: pc.__setitem__("system", "You are a captioner."),
    ],
)
def test_hash_changes_with_content(pc, mutate):
    before = prompt_config_hash(pc)
    mutate(pc)
    assert prompt_config_hash(pc) != before


def test_hash_does_not_mutate(pc):
    before = copy.deepcopy(pc)
    prompt_config_hash(pc)
    assert pc == before


def test_hash_unserializable_raises_type_error(pc):
    pc["sources_file"] = Path("configs/sources.json")
    with pytest.raises(TypeError):
        prompt_config_hash(pc)


def test_hash_stable_across_processes(pc):
    """Same hash in fresh interpreters with different PYTHONHASHSEED values."""
    payload = json.dumps(pc)
    code = (
        "import json, sys; "
        "from pixel_art_auto_captioner.common.prompts import prompt_config_hash; "
        "sys.stdout.write(prompt_config_hash(json.loads(sys.stdin.read())))"
    )
    results = set()
    for seed in ("0", "1", "12345"):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(SRC)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", code], input=payload, env=env,
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        results.add(proc.stdout)
    assert results == {prompt_config_hash(pc)}


# ---------------------------------------------------------------------------
# 6. No heavy imports (SPEC §2.2)
# ---------------------------------------------------------------------------


def test_prompts_import_pulls_in_no_torch_or_vllm():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.common.prompts; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
