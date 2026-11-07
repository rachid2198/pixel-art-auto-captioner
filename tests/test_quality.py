"""Tests for ``common/quality.py`` (SPEC v1.1 §5.3, §13).

Covers each flag, the thresholds and their boundaries, flag combinations and
order, config defaults and config errors.
"""

import copy
import logging
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from pixel_art_auto_captioner.common.config import DEFAULT_CONFIG
from pixel_art_auto_captioner.common.quality import check_caption
from pixel_art_auto_captioner.common.types import CAPTION_FLAGS, CaptionResult

SRC = Path(__file__).resolve().parent.parent / "src"

GOOD = (
    "This pixel art image shows a small knight in silver armor standing on a grassy "
    "ledge beside a wooden signpost. The background is a dusk sky with purple and "
    "orange gradients, distant mountains, and a few twinkling stars. A red banner "
    "flutters on the right, and a treasure chest sits half buried in the soil near "
    "the bottom left corner. The palette is limited and the outlines are crisp, "
    "typical of 16-bit era platformers."
)
LOOP = "the cat sat on the mat. " * 20  # 4-gram count 20, compression ratio ~0.08

# Isolate one repetition rule by disabling the other.
NGRAM_ONLY = {"repeat_ngram_threshold": 4, "min_compression_ratio": 0.0}
RATIO_ONLY = {"repeat_ngram_threshold": 10**9, "min_compression_ratio": 0.25}


def _r(text: str = GOOD, finish_reason: str = "stop", error: str | None = None) -> CaptionResult:
    return CaptionResult(text=text, finish_reason=finish_reason, num_output_tokens=10, error=error)


@pytest.fixture
def qcfg() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG["quality"])


def _ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(zlib.compress(data)) / len(data)


# ---------------------------------------------------------------------------
# 1. Clean caption
# ---------------------------------------------------------------------------


def test_clean_caption_has_no_flags(qcfg):
    assert check_caption(_r(), qcfg) == []


def test_short_clean_caption_has_no_flags(qcfg):
    # Short texts compress to a ratio > 1; they must not be flagged.
    assert check_caption(_r("A knight."), qcfg) == []


def test_tags_caption_not_repetitive(qcfg):
    tags = "pixel art, knight, armor, sword, castle, night sky, stars, 16-bit, side view"
    assert check_caption(_r(tags), qcfg) == []


# ---------------------------------------------------------------------------
# 2. Each flag
# ---------------------------------------------------------------------------


def test_error_flag(qcfg):
    assert check_caption(_r("partial text", "error", "RuntimeError: boom"), qcfg) == ["error"]


def test_error_with_empty_text_is_error_and_empty(qcfg):
    assert check_caption(_r("", "error", "RuntimeError: boom"), qcfg) == ["error", "empty"]


@pytest.mark.parametrize("text", ["", " ", "\n\t  \n"])
def test_empty_flag(qcfg, text):
    assert check_caption(_r(text), qcfg) == ["empty"]


def test_truncated_flag(qcfg):
    assert check_caption(_r(GOOD, "length"), qcfg) == ["truncated"]


def test_empty_and_truncated(qcfg):
    assert check_caption(_r("", "length"), qcfg) == ["empty", "truncated"]


def test_repetitive_flag_default_cfg(qcfg):
    assert check_caption(_r(LOOP), qcfg) == ["repetitive"]


def test_truncated_and_repetitive_typical_loop(qcfg):
    # A generation stuck in a loop usually also hits max_new_tokens.
    assert check_caption(_r(LOOP, "length"), qcfg) == ["truncated", "repetitive"]


def test_error_and_repetitive_order(qcfg):
    assert check_caption(_r(LOOP, "error"), qcfg) == ["error", "repetitive"]


def test_flags_are_known_values(qcfg):
    for result in (_r(), _r("", "error"), _r(LOOP, "length"), _r("", "length")):
        assert set(check_caption(result, qcfg)) <= CAPTION_FLAGS


# ---------------------------------------------------------------------------
# 3. Repetition: word 4-grams
# ---------------------------------------------------------------------------


def _ngram_text(repeats: int) -> str:
    """Text where the 4-gram 'red brick wall here' occurs exactly ``repeats`` times."""
    fillers = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    parts = []
    for i in range(repeats):
        parts.append("red brick wall here")
        parts.append(f"{fillers[i]} {fillers[i]}x {fillers[i]}y")
    return " ".join(parts)


@pytest.mark.parametrize("repeats, flagged", [(1, False), (3, False), (4, True), (5, True)])
def test_ngram_threshold_boundary(repeats, flagged):
    flags = check_caption(_r(_ngram_text(repeats)), NGRAM_ONLY)
    assert (flags == ["repetitive"]) is flagged


@pytest.mark.parametrize("threshold, flagged", [(2, True), (3, True), (4, False)])
def test_ngram_threshold_is_configurable(threshold, flagged):
    cfg = {"repeat_ngram_threshold": threshold, "min_compression_ratio": 0.0}
    assert (check_caption(_r(_ngram_text(3)), cfg) == ["repetitive"]) is flagged


def test_ngrams_count_overlapping_windows():
    # 7 identical words -> the 4-gram (a, a, a, a) occurs 4 times (positions 0..3).
    assert check_caption(_r("go go go go go go go"), NGRAM_ONLY) == ["repetitive"]
    assert check_caption(_r("go go go go go go"), NGRAM_ONLY) == []


def test_ngrams_are_case_insensitive():
    text = "Red brick wall here x RED BRICK WALL HERE y red Brick wall Here z red brick WALL here"
    assert check_caption(_r(text), NGRAM_ONLY) == ["repetitive"]


def test_ngrams_split_on_any_whitespace():
    text = (
        "red brick\twall here a red\nbrick wall  here b "
        "red brick wall here c red brick wall\rhere"
    )
    assert check_caption(_r(text), NGRAM_ONLY) == ["repetitive"]


def test_ngrams_keep_punctuation_attached():
    # "here." and "here," are different tokens, so no 4-gram repeats 4 times.
    text = "red brick wall here. red brick wall here, red brick wall here! red brick wall here?"
    assert check_caption(_r(text), NGRAM_ONLY) == []


def test_fewer_than_four_words_never_ngram_repetitive():
    cfg = {"repeat_ngram_threshold": 1, "min_compression_ratio": 0}
    assert check_caption(_r("go go go"), cfg) == []


def test_threshold_one_flags_any_four_words():
    cfg = {"repeat_ngram_threshold": 1, "min_compression_ratio": 0}
    assert check_caption(_r("a b c d"), cfg) == ["repetitive"]


# ---------------------------------------------------------------------------
# 4. Repetition: compression ratio
# ---------------------------------------------------------------------------


def test_low_compression_ratio_flags_without_ngrams():
    text = "abcdefgh" * 100  # a single "word": no 4-grams at all
    assert _ratio(text) < 0.25
    assert check_caption(_r(text), RATIO_ONLY) == ["repetitive"]


def test_normal_text_ratio_not_flagged():
    assert _ratio(GOOD) > 0.4
    assert check_caption(_r(GOOD), RATIO_ONLY) == []


def test_ratio_threshold_boundary_is_strict():
    text = "abcdefgh" * 100
    ratio = _ratio(text)
    exact = {"repeat_ngram_threshold": 10**9, "min_compression_ratio": ratio}
    assert check_caption(_r(text), exact) == []
    bumped = {"repeat_ngram_threshold": 10**9, "min_compression_ratio": ratio + 1e-9}
    assert check_caption(_r(text), bumped) == ["repetitive"]


def test_ratio_threshold_is_configurable():
    # GOOD has a ratio around 0.6: a strict threshold flags it, zero never does.
    strict = {"repeat_ngram_threshold": 10**9, "min_compression_ratio": 0.9}
    off = {"repeat_ngram_threshold": 10**9, "min_compression_ratio": 0}
    assert check_caption(_r(GOOD), strict) == ["repetitive"]
    assert check_caption(_r(LOOP), off) == []


def test_ratio_uses_stripped_text():
    # Surrounding whitespace must not change the outcome.
    padded = "   " + GOOD + "\n\n" + " " * 2000
    assert check_caption(_r(padded), RATIO_ONLY) == []


def test_non_ascii_text_ok(qcfg):
    text = "Une image en pixel art d'un chevalier près d'un château, ciel étoilé, palette limitée."
    assert check_caption(_r(text), qcfg) == []


# ---------------------------------------------------------------------------
# 5. Config handling
# ---------------------------------------------------------------------------


def test_defaults_when_keys_missing():
    assert check_caption(_r(GOOD), {}) == []
    assert check_caption(_r(LOOP), {}) == ["repetitive"]
    assert check_caption(_r(_ngram_text(4)), {}) == ["repetitive"]
    assert check_caption(_r(_ngram_text(3)), {}) == []


def test_default_config_quality_section_values():
    assert DEFAULT_CONFIG["quality"] == {
        "max_retries": 1, "repeat_ngram_threshold": 4, "min_compression_ratio": 0.25,
    }


@pytest.mark.parametrize("value", [0, -1, 2.5, "4", None, True])
def test_bad_ngram_threshold_raises(value):
    with pytest.raises(ValueError, match="repeat_ngram_threshold"):
        check_caption(_r(), {"repeat_ngram_threshold": value})


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), "0.25", None, False])
def test_bad_compression_ratio_raises(value):
    with pytest.raises(ValueError, match="min_compression_ratio"):
        check_caption(_r(), {"min_compression_ratio": value})


def test_int_compression_ratio_accepted():
    assert check_caption(_r(GOOD), {"min_compression_ratio": 0}) == []


def test_does_not_mutate_inputs(qcfg):
    result = _r("  " + LOOP + "  ", "length")
    before_cfg = copy.deepcopy(qcfg)
    before_text = result.text
    check_caption(result, qcfg)
    assert qcfg == before_cfg
    assert result.text == before_text


# ---------------------------------------------------------------------------
# 6. Unknown finish_reason
# ---------------------------------------------------------------------------


def test_unknown_finish_reason_warns_and_sets_no_flag(qcfg, caplog):
    with caplog.at_level(logging.WARNING, logger="pixel_art_auto_captioner.common.quality"):
        assert check_caption(_r(GOOD, "abort"), qcfg) == []
    assert "abort" in caplog.text


def test_known_finish_reasons_do_not_warn(qcfg, caplog):
    with caplog.at_level(logging.WARNING, logger="pixel_art_auto_captioner.common.quality"):
        for reason in ("stop", "length", "error"):
            check_caption(_r(GOOD, reason), qcfg)
    assert caplog.records == []


# ---------------------------------------------------------------------------
# 7. No heavy imports (SPEC §2.2)
# ---------------------------------------------------------------------------


def test_quality_import_pulls_in_no_torch_or_vllm():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.common.quality; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
