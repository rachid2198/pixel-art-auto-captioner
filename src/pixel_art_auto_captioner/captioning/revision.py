"""Resolve the model revision actually loaded (SPEC §6.2, R4).

Both backends report ``resolved_revision`` in ``describe()``: the commit hash
of the Hugging Face snapshot that was loaded, or ``None`` for a plain local
folder without commit information. Resolution never fails ``load()``: any
error is logged at WARNING and gives ``None``.

Sources, in order:

1. a commit hash the library already knows (transformers sets
   ``config._commit_hash`` for files from the Hub cache);
2. a local folder that is a Hub cache snapshot (``.../snapshots/<hash>``) or a
   ``huggingface_hub`` ``--local-dir`` download (its
   ``.cache/huggingface/download/config.json.metadata`` starts with the hash);
3. for a repo id, the Hub cache (``huggingface_hub.try_to_load_from_cache``,
   imported lazily, offline).

Depends on: nothing from the package; ``huggingface_hub`` only inside
:func:`_hub_cache_lookup`.
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEX40 = re.compile(r"[0-9a-f]{40}")

#: Relative path of the commit metadata ``huggingface_hub`` writes for ``config.json``
#: in a ``--local-dir`` download.
_LOCAL_DIR_METADATA = Path(".cache") / "huggingface" / "download" / "config.json.metadata"


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and _HEX40.fullmatch(value) is not None


def resolve_revision(model_path: str, revision: str | None, commit_hash: Any = None) -> str | None:
    """Return the commit hash of the model that was loaded, if it can be known.

    Args:
        model_path: ``model.model_path`` (a Hub repo id or a local folder).
        revision: ``model.revision`` as configured (``None`` = the repo's default).
        commit_hash: A hash reported by the loading library, if any (e.g. the
            loaded config's ``_commit_hash``); used when it is a 40-hex string.

    Returns:
        A 40-character commit hash, or ``None`` for a plain local folder, a
        repo id not found in the cache, or any resolution error (logged at
        WARNING). Never raises.
    """
    try:
        if _is_hash(commit_hash):
            return commit_hash
        path = Path(model_path)
        if path.is_dir():
            return _from_local_dir(path)
        found = _hub_cache_lookup(model_path, revision)
        if found is None:
            logger.warning("resolved_revision: %s (revision=%s) not found in the Hugging Face cache",
                           model_path, revision)
        return found
    except Exception as exc:  # never fail load() because of this (R4)
        logger.warning("resolved_revision: could not resolve %s: %s: %s", model_path, type(exc).__name__, exc)
        return None


def _from_local_dir(path: Path) -> str | None:
    """Hash of a cache snapshot or ``--local-dir`` download; ``None`` for a plain folder."""
    resolved = path.resolve()
    if resolved.parent.name == "snapshots" and _is_hash(resolved.name):
        return resolved.name
    metadata = resolved / _LOCAL_DIR_METADATA
    if metadata.is_file():
        first = metadata.read_text(encoding="utf-8").splitlines()[:1]
        if first and _is_hash(first[0].strip()):
            return first[0].strip()
    return None


def _hub_cache_lookup(repo_id: str, revision: str | None) -> str | None:
    """Look ``repo_id`` up in the local Hub cache (offline) and return the snapshot hash."""
    hub = importlib.import_module("huggingface_hub")
    found = hub.try_to_load_from_cache(repo_id, "config.json", revision=revision)
    if not isinstance(found, (str, Path)):
        return None  # not cached (None) or cached as missing (_CACHED_NO_EXIST)
    snapshot = Path(found).parent
    if snapshot.parent.name == "snapshots" and _is_hash(snapshot.name):
        return snapshot.name
    return None
