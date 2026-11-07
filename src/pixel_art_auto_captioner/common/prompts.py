"""Prompt rendering and source resolution (SPEC v1.1 §5.1).

A request's user prompt is built from three config parts:

- ``caption_types``: name -> ``{instruction, max_new_tokens, temperature,
  top_p, top_k}`` (``generation.caption_types`` in the config file, §9.2);
- ``categories``: name -> ``context`` template, which may reference
  ``{source_name}``;
- ``sources``: top-level folder -> ``{category, name}``. Unknown folders use
  ``default_category``; ``name`` defaults to the folder name with dashes
  replaced by spaces, title-cased.

Shape of ``prompt_config``
--------------------------
All three public functions take the same ``prompt_config`` dict: the
``prompts`` config section (``template``, ``constraints``, ``default_category``,
``categories``, ``sources``, plus any other keys such as ``system``) with one
extra top-level key, ``caption_types``, holding ``generation.caption_types``.
The runner builds it once per run::

    prompt_config = {**cfg["prompts"], "caption_types": cfg["generation"]["caption_types"]}

Rendering (SPEC §5.1)::

    prompt_config["template"].format(instruction=..., context=..., constraints=...)

with all whitespace runs collapsed to one space and the result stripped.

This module uses the standard library only. It imports nothing from the
package and never imports ``torch`` or ``vllm`` (SPEC §2.2).
"""

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Number of hex characters kept by prompt_config_hash (SPEC §5.1).
_HASH_HEX_CHARS = 12


def _default_source_name(source: str) -> str:
    """Folder name -> display name: dashes become spaces, each word capitalized.

    ``"stardew-valley"`` -> ``"Stardew Valley"``. Words are capitalized with
    :meth:`str.capitalize` rather than :meth:`str.title`, which would also
    capitalize after digits and apostrophes (``"rpg2k"`` -> ``"Rpg2K"``).
    Runs of dashes/whitespace collapse to one space.
    """
    return " ".join(word.capitalize() for word in source.replace("-", " ").split())


def _collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends."""
    return " ".join(text.split())


def resolve_source(source: str, prompt_config: dict) -> tuple[str, str]:
    """Return the ``(category, source_name)`` of a top-level source folder.

    Lookup in ``prompt_config["sources"]`` (a missing or ``null`` mapping is
    treated as empty):

    - Known folder: its ``category`` and ``name``. A missing or ``null``
      ``category`` falls back to ``default_category``; a missing or ``null``
      ``name`` falls back to the default display name.
    - Unknown folder (including ``""``, files directly in ``input_dir``):
      ``(default_category, default display name)``.

    The default display name is the folder name with dashes replaced by spaces
    and each word capitalized (``"stardew-valley"`` -> ``"Stardew Valley"``).

    The category is returned as configured; whether it exists in
    ``categories`` is checked by ``validate_config`` and by
    :func:`render_prompt`.

    Args:
        source: Top-level folder name (``ImageRecord.source``).
        prompt_config: Prompt config (see the module docstring).

    Returns:
        ``(category, source_name)``.

    Raises:
        ValueError: If ``prompt_config`` has no ``default_category``, if
            ``sources`` is not an object, or if the entry for ``source`` is not
            an object.
    """
    default_category = prompt_config.get("default_category")
    if not isinstance(default_category, str):
        raise ValueError(
            f"prompt_config['default_category'] must be a string, got {default_category!r}"
        )
    sources = prompt_config.get("sources") or {}
    if not isinstance(sources, dict):
        raise ValueError(f"prompt_config['sources'] must be an object, got {sources!r}")

    entry = sources.get(source)
    if entry is None:
        return default_category, _default_source_name(source)
    if not isinstance(entry, dict):
        raise ValueError(f"prompt_config['sources'][{source!r}] must be an object, got {entry!r}")

    category = entry.get("category")
    if category is None:
        category = default_category
    name = entry.get("name")
    if name is None:
        name = _default_source_name(source)
    return str(category), str(name)


def render_prompt(caption_type: str, source: str, prompt_config: dict) -> str:
    """Render the full user prompt for one (caption type, source) pair.

    Steps:

    1. ``instruction`` = ``prompt_config["caption_types"][caption_type]["instruction"]``.
    2. ``(category, source_name)`` = :func:`resolve_source`.
    3. ``context`` = ``prompt_config["categories"][category]`` formatted with
       ``source_name=source_name``.
    4. ``constraints`` = ``prompt_config["constraints"]`` (missing or ``null``
       means ``""``).
    5. ``prompt_config["template"].format(instruction=..., context=...,
       constraints=...)``, whitespace collapsed and stripped.

    The instruction and constraints are inserted verbatim (not formatted), so
    braces in them are kept as-is.

    Args:
        caption_type: Caption type name, e.g. ``"long"``.
        source: Top-level folder name (``ImageRecord.source``).
        prompt_config: Prompt config (see the module docstring).

    Returns:
        The rendered prompt, e.g. ``"Write a long detailed description for
        this image. This image is a crop from a screenshot of the video game
        Celeste. Do not mention the image resolution."``

    Raises:
        ValueError: If the caption type, its instruction, the category or the
            template is missing, or if the context or template references a
            placeholder other than the documented ones (or is malformed).
    """
    caption_types = prompt_config.get("caption_types")
    if not isinstance(caption_types, dict):
        raise ValueError(
            "prompt_config['caption_types'] must be an object (generation.caption_types), "
            f"got {caption_types!r}"
        )
    spec = caption_types.get(caption_type)
    if not isinstance(spec, dict):
        raise ValueError(
            f"Unknown caption type {caption_type!r} (defined: {sorted(caption_types)})"
        )
    instruction = spec.get("instruction")
    if not isinstance(instruction, str):
        raise ValueError(
            f"caption_types[{caption_type!r}]['instruction'] must be a string, got {instruction!r}"
        )

    category, source_name = resolve_source(source, prompt_config)
    categories = prompt_config.get("categories")
    if not isinstance(categories, dict):
        raise ValueError(f"prompt_config['categories'] must be an object, got {categories!r}")
    context_template = categories.get(category)
    if not isinstance(context_template, str):
        raise ValueError(
            f"Category {category!r} (source {source!r}) is not defined in "
            f"prompt_config['categories'] (defined: {sorted(categories)})"
        )
    try:
        context = context_template.format(source_name=source_name)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Context of category {category!r} is not a valid template "
            f"(only {{source_name}} is allowed): {type(exc).__name__}: {exc}"
        ) from exc

    constraints = prompt_config.get("constraints")
    if constraints is None:
        constraints = ""
    template = prompt_config.get("template")
    if not isinstance(template, str):
        raise ValueError(f"prompt_config['template'] must be a string, got {template!r}")
    try:
        rendered = template.format(
            instruction=instruction, context=context, constraints=str(constraints)
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            "prompt_config['template'] is not a valid template (allowed placeholders: "
            f"{{instruction}}, {{context}}, {{constraints}}): {type(exc).__name__}: {exc}"
        ) from exc
    return _collapse_whitespace(rendered)


def prompt_config_hash(prompt_config: dict) -> str:
    """Short, stable fingerprint of a prompt config (SPEC §5.1).

    SHA-256 of the canonical JSON encoding, first 12 hex characters. Canonical
    JSON: keys sorted at every level, separators ``(",", ":")``, non-ASCII
    characters kept as-is (``ensure_ascii=False``) and encoded as UTF-8. The
    result does not depend on dict insertion order, the process, the platform
    or ``PYTHONHASHSEED``.

    Args:
        prompt_config: Prompt config (see the module docstring). Must be
            JSON-serializable.

    Returns:
        12 lowercase hex characters.

    Raises:
        TypeError: If ``prompt_config`` contains a value that JSON cannot
            encode (e.g. a ``Path``) or a non-string key that cannot be sorted
            with the others.
    """
    canonical = json.dumps(
        prompt_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_HASH_HEX_CHARS]
