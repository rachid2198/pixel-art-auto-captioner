"""Caption quality checks (SPEC v1.1 §5.3).

:func:`check_caption` inspects one backend result and returns quality flags.
It never modifies the text: flags are metadata (SPEC §1.3).

| Flag         | Rule                                                              |
|--------------|-------------------------------------------------------------------|
| ``error``      | ``finish_reason == "error"``                                    |
| ``empty``      | stripped text is empty                                          |
| ``truncated``  | ``finish_reason == "length"``                                   |
| ``repetitive`` | a word 4-gram occurs >= ``repeat_ngram_threshold`` times, or    |
|                | zlib compression ratio < ``min_compression_ratio``              |

The retry policy and the final ``status`` (``failed`` for ``error``/``empty``,
``flagged`` for other flags, ``ok`` otherwise) are applied by the runner
(SPEC §5.3, §7.2).

This module uses the standard library only. It imports nothing from the
package except ``common`` modules and never imports ``torch`` or ``vllm``
(SPEC §2.2).
"""

import logging
import math
import zlib
from collections import Counter
from typing import Any

from pixel_art_auto_captioner.common.types import FINISH_REASONS, CaptionResult

logger = logging.getLogger(__name__)

#: Defaults for the ``quality`` config section (SPEC §5.3, §9.2), used for
#: keys missing from ``cfg``.
_DEFAULT_REPEAT_NGRAM_THRESHOLD = 4
_DEFAULT_MIN_COMPRESSION_RATIO = 0.25
_NGRAM_SIZE = 4


def _max_ngram_count(words: list[str], n: int = _NGRAM_SIZE) -> int:
    """Highest number of occurrences of any word ``n``-gram (overlapping windows).

    Returns 0 when there are fewer than ``n`` words.
    """
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def _compression_ratio(text: str) -> float:
    """``len(zlib.compress(utf8)) / len(utf8)`` for a non-empty string.

    Uses zlib's default compression level. Short texts have a ratio above 1
    (header overhead); highly repetitive long texts approach 0.
    """
    data = text.encode("utf-8")
    return len(zlib.compress(data)) / len(data)


def _read_thresholds(cfg: dict) -> tuple[int, float]:
    """Read and check ``repeat_ngram_threshold`` and ``min_compression_ratio``.

    Raises:
        ValueError: If a value is present but not a positive int / a finite
            number >= 0.
    """
    threshold: Any = cfg.get("repeat_ngram_threshold", _DEFAULT_REPEAT_NGRAM_THRESHOLD)
    if not (isinstance(threshold, int) and not isinstance(threshold, bool) and threshold >= 1):
        raise ValueError(
            f"quality.repeat_ngram_threshold must be a positive integer, got {threshold!r}"
        )
    ratio: Any = cfg.get("min_compression_ratio", _DEFAULT_MIN_COMPRESSION_RATIO)
    if not (
        isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
        and ratio >= 0
    ):
        raise ValueError(
            f"quality.min_compression_ratio must be a finite number >= 0, got {ratio!r}"
        )
    return threshold, float(ratio)


def check_caption(result: CaptionResult, cfg: dict) -> list[str]:
    """Return the quality flags of one backend result (SPEC §5.3).

    Each rule is evaluated independently, so several flags can be set (an
    errored request with no text gives ``["error", "empty"]``).

    - ``error``: ``result.finish_reason == "error"``.
    - ``empty``: ``result.text.strip() == ""``.
    - ``truncated``: ``result.finish_reason == "length"``.
    - ``repetitive`` (non-empty text only): some word 4-gram occurs at least
      ``repeat_ngram_threshold`` times, **or** the zlib compression ratio of
      the UTF-8 text is below ``min_compression_ratio``. Words are the
      whitespace-separated tokens of the lowercased text (punctuation stays
      attached); 4-grams are counted over overlapping windows.

    An unknown ``finish_reason`` (not in ``FINISH_REASONS``) sets no flag but
    is logged as a WARNING, because it indicates a backend mapping bug.

    Args:
        result: Backend output for one request.
        cfg: The ``quality`` config section (SPEC §9.2). Missing keys use the
            defaults ``repeat_ngram_threshold=4`` and
            ``min_compression_ratio=0.25``; ``max_retries`` is not used here.

    Returns:
        Flags in the fixed order ``error``, ``empty``, ``truncated``,
        ``repetitive`` (only those that apply); ``[]`` for a clean caption.

    Raises:
        ValueError: If ``repeat_ngram_threshold`` is not a positive integer or
            ``min_compression_ratio`` is not a finite number >= 0.
    """
    threshold, min_ratio = _read_thresholds(cfg)
    flags: list[str] = []

    if result.finish_reason not in FINISH_REASONS:
        logger.warning(
            "Unknown finish_reason %r (expected one of %s); no flag set for it",
            result.finish_reason,
            sorted(FINISH_REASONS),
        )

    text = result.text.strip()
    if result.finish_reason == "error":
        flags.append("error")
    if not text:
        flags.append("empty")
    if result.finish_reason == "length":
        flags.append("truncated")
    if text:
        max_count = _max_ngram_count(text.lower().split())
        if max_count >= threshold or _compression_ratio(text) < min_ratio:
            flags.append("repetitive")

    if flags:
        logger.debug("Caption flagged %s (finish_reason=%r)", flags, result.finish_reason)
    return flags
