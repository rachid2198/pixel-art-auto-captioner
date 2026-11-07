"""Per-request deterministic seeds (SPEC v1.1 §5.2).

Every (image, caption type, attempt) gets its own seed, derived from the run's
global seed with SHA-256, so it is identical across processes, machines,
platforms and ``PYTHONHASHSEED`` values (Python's built-in ``hash()`` is never
used). A retry (``attempt + 1``) gets a new seed.

Reproducibility of the generated *text* is best-effort: it also needs the same
backend, model revision, dtype, library versions and GPU type, and vLLM may
differ slightly when batch composition changes (SPEC §5.2, AGENTS.md §6.7).

This module uses the standard library only. It imports nothing from the
package and never imports ``torch`` or ``vllm`` (SPEC §2.2).
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Seeds are masked to 31 bits so every backend accepts them as a signed int32.
_SEED_MASK = 0x7FFFFFFF


def derive_seed(global_seed: int, rel_path: Path, caption_type: str, attempt: int) -> int:
    """Derive the seed of one generation request (SPEC §5.2).

    ``int.from_bytes(sha256(f"{global_seed}|{rel_path.as_posix()}|{caption_type}|{attempt}"
    .encode()).digest()[:4], "big") & 0x7FFFFFFF``

    ``rel_path`` is converted with ``as_posix()``, so a Windows path and the
    equivalent POSIX path give the same seed.

    Args:
        global_seed: Run-level seed (``runtime.seed``).
        rel_path: Image path relative to ``input_dir`` (the unique key, with
            extension).
        caption_type: Caption type name, e.g. ``"long"``.
        attempt: Attempt number (1 = first try, 2 = first retry, ...).

    Returns:
        A seed in ``[0, 2**31 - 1]``.
    """
    key = f"{global_seed}|{rel_path.as_posix()}|{caption_type}|{attempt}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & _SEED_MASK
