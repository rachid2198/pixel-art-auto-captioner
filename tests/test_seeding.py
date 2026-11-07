"""Tests for ``common/seeding.py`` (SPEC v1.1 §5.2, §13).

Covers the exact SPEC formula, determinism (in-process and across processes
with different ``PYTHONHASHSEED``), the output range, path normalization, and
sensitivity to each of the four inputs.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from pixel_art_auto_captioner.common.seeding import derive_seed

SRC = Path(__file__).resolve().parent.parent / "src"
REL = Path("celeste/0001/000001.jpg")


def _spec_formula(global_seed: int, rel_posix: str, caption_type: str, attempt: int) -> int:
    """SPEC §5.2, written out independently of the implementation."""
    key = f"{global_seed}|{rel_posix}|{caption_type}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# 1. Formula and golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (0, "celeste/0001/000001.jpg", "long", 1),
        (42, "stardew-valley/0003/000123.jpg", "short", 2),
        (-7, "a.png", "tags", 3),
        (2**40, "pokémon/0001/000001.jpg", "sd_prompt", 1),
    ],
)
def test_matches_spec_formula(args):
    global_seed, rel, caption_type, attempt = args
    assert derive_seed(global_seed, Path(rel), caption_type, attempt) == _spec_formula(*args)


def test_golden_values():
    # Pinned: if these change, every seed recorded in existing JSONL changes.
    assert derive_seed(0, Path("celeste/0001/000001.jpg"), "long", 1) == 1838564669
    assert derive_seed(42, Path("stardew-valley/0003/000123.jpg"), "short", 2) == 82409684


# ---------------------------------------------------------------------------
# 2. Determinism and range
# ---------------------------------------------------------------------------


def test_deterministic_in_process():
    assert derive_seed(3, REL, "long", 1) == derive_seed(3, Path(REL.as_posix()), "long", 1)


def test_stable_across_processes():
    """Fresh interpreters with different PYTHONHASHSEED give the same seed."""
    code = (
        "import sys; from pathlib import Path; "
        "from pixel_art_auto_captioner.common.seeding import derive_seed; "
        "sys.stdout.write(str(derive_seed(7, Path('celeste/0001/000001.jpg'), 'long', 2)))"
    )
    outputs = set()
    for hash_seed in ("0", "1", "98765"):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(SRC)
        env["PYTHONHASHSEED"] = hash_seed
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.add(proc.stdout)
    assert outputs == {str(derive_seed(7, REL, "long", 2))}


def test_range_is_31_bits():
    seeds = [
        derive_seed(s, Path(f"src{i}/0001/{i:06d}.jpg"), t, a)
        for s in (0, 1, 2**31, -5)
        for i in range(50)
        for t in ("long", "short")
        for a in (1, 2)
    ]
    assert all(isinstance(x, int) and 0 <= x <= 0x7FFFFFFF for x in seeds)
    # Well spread: no collisions in this small sample, and the top bit range is used.
    assert len(set(seeds)) == len(seeds)
    assert max(seeds) > 2**30


def test_windows_and_posix_paths_give_same_seed():
    win = PureWindowsPath("celeste\\0001\\000001.jpg")
    posix = PurePosixPath("celeste/0001/000001.jpg")
    assert derive_seed(0, win, "long", 1) == derive_seed(0, posix, "long", 1)
    assert derive_seed(0, posix, "long", 1) == derive_seed(0, REL, "long", 1)


# ---------------------------------------------------------------------------
# 3. Sensitivity to each input
# ---------------------------------------------------------------------------


def test_sensitive_to_global_seed():
    assert derive_seed(0, REL, "long", 1) != derive_seed(1, REL, "long", 1)


def test_sensitive_to_rel_path():
    other = Path("celeste/0001/000002.jpg")
    assert derive_seed(0, REL, "long", 1) != derive_seed(0, other, "long", 1)


def test_same_stem_in_different_sources_differs():
    # Stems repeat across sources (AGENTS.md §6.4); the seed is keyed by rel_path.
    other = Path("stardew-valley/0001/000001.jpg")
    assert derive_seed(0, REL, "long", 1) != derive_seed(0, other, "long", 1)


def test_extension_matters():
    png = Path("celeste/0001/000001.png")
    assert derive_seed(0, REL, "long", 1) != derive_seed(0, png, "long", 1)


def test_sensitive_to_caption_type():
    assert derive_seed(0, REL, "long", 1) != derive_seed(0, REL, "short", 1)


def test_retry_gets_new_seed():
    seeds = {derive_seed(0, REL, "long", a) for a in (1, 2, 3)}
    assert len(seeds) == 3


def test_all_single_input_changes_are_distinct():
    base = derive_seed(0, REL, "long", 1)
    variants = [
        derive_seed(1, REL, "long", 1),
        derive_seed(0, Path("celeste/0001/000002.jpg"), "long", 1),
        derive_seed(0, REL, "tags", 1),
        derive_seed(0, REL, "long", 2),
    ]
    assert base not in variants
    assert len(set(variants)) == len(variants)


# ---------------------------------------------------------------------------
# 4. No heavy imports (SPEC §2.2)
# ---------------------------------------------------------------------------


def test_seeding_import_pulls_in_no_torch_or_vllm():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.common.seeding; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
