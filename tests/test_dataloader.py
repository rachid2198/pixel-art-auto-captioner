"""Tests for ``pixel_art_auto_captioner.ingestion.dataloader`` (SPEC v1.1 §4, §13).

Covers config validation, the single cached walk, case-insensitive extensions,
``.msgpack`` files ignored without being opened, include/exclude sources, hash
sharding (disjoint, complete, stable), planning with ``max_images`` applied
after resume, threaded decoding in plan order with bounded look-ahead,
``LoadFailure`` for undecodable files, and the package boundary.
"""

from __future__ import annotations

import builtins
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

import pytest
from PIL import Image

from pixel_art_auto_captioner.common.types import ImageRecord, LoadFailure
from pixel_art_auto_captioner.ingestion import dataloader as dl_module
from pixel_art_auto_captioner.ingestion.dataloader import (
    DEFAULT_EXTENSIONS,
    ImageDataLoader,
    Plan,
    _shard_index,
)

SRC = Path(__file__).resolve().parent.parent / "src"
LOGGER = "pixel_art_auto_captioner.ingestion.dataloader"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _img(path: Path, size: tuple[int, int] = (8, 8), mode: str = "RGB", fmt: str | None = None) -> Path:
    """Write a small synthetic image at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    color = 1 if mode in ("L", "P", "1") else tuple(range(10, 10 + len(mode)))
    Image.new(mode, size, color=color).save(path, format=fmt)
    return path


def _file(path: Path, data: bytes = b"not an image") -> Path:
    """Write raw bytes at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _loader(root: Path, **overrides) -> ImageDataLoader:
    """An ``ImageDataLoader`` over ``root``; inline decoding unless overridden."""
    cfg: dict = {"input_dir": str(root), "num_workers": 0}
    cfg.update(overrides)
    return ImageDataLoader(cfg)


def _never(rel_path: Path, caption_type: str) -> bool:
    """Resume predicate: nothing is complete."""
    return False


def _posix(paths) -> list[str]:
    return [p.as_posix() for p in paths]


def _dataset(root: Path) -> list[str]:
    """A small dataset in the SPEC §1.2 layout, with msgpack leftovers.

    Returns the expected image rel paths, sorted.
    """
    images = [
        "celeste/0001/000001.jpg",
        "celeste/0001/000002.jpg",
        "celeste/0002/000001.jpg",
        "hades/0001/000001.jpg",
        "stardew-valley/0001/000001.jpg",
        "stardew-valley/0001/000002.png",
    ]
    for rel in images:
        _img(root / rel, fmt="PNG" if rel.endswith(".png") else "JPEG")
        stem = (root / rel).with_suffix("")
        _file(Path(f"{stem}_clip_h.msgpack"), b"\x00\x01")
        _file(Path(f"{stem}_vae_latent_64.msgpack"), b"\x00\x01")
        _file(Path(f"{stem}_vae_latent_256.msgpack"), b"\x00\x01")
    return sorted(images)


def _many(root: Path, n: int, start: int = 0, sources: tuple[str, ...] = ("a", "b", "c")) -> list[str]:
    """Create ``n`` tiny empty image-named files (discovery and planning never open them)."""
    rels = []
    for i in range(start, start + n):
        rel = f"{sources[i % len(sources)]}/{i // 50:04d}/{i:06d}.jpg"
        _file(root / rel, b"")
        rels.append(rel)
    return rels


def _decoder_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(dl_module._THREAD_NAME_PREFIX)]


# ---------------------------------------------------------------------------
# __init__ / config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self, tmp_path: Path):
        loader = ImageDataLoader({"input_dir": str(tmp_path)})
        assert loader.input_dir == tmp_path.resolve()
        assert loader.extensions == frozenset(DEFAULT_EXTENSIONS)
        assert DEFAULT_EXTENSIONS == (".png", ".jpg", ".jpeg", ".webp")
        assert loader.recursive is True
        assert loader.include_sources is None
        assert loader.exclude_sources == []
        assert loader.shard == (0, 1)
        assert loader.max_images is None
        assert loader.num_workers == 4
        assert loader.prefetch == 64

    def test_accepts_path_object(self, tmp_path: Path):
        assert ImageDataLoader({"input_dir": tmp_path}).input_dir == tmp_path.resolve()

    def test_input_dir_is_resolved(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        loader = ImageDataLoader({"input_dir": str(tmp_path / "sub" / ".." / "sub")})
        assert loader.input_dir == (tmp_path / "sub").resolve()
        assert loader.input_dir.is_absolute()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [((0, 1), (0, 1)), ((2, 4), (2, 4)), ([3, 4], (3, 4)), ("1/4", (1, 4)), (" 0/2 ", (0, 2))],
    )
    def test_shard_forms(self, tmp_path: Path, value, expected):
        assert _loader(tmp_path, shard=value).shard == expected

    def test_extensions_are_lowercased_and_dotted(self, tmp_path: Path):
        loader = _loader(tmp_path, extensions=[".PNG", "JPG", ".Webp"])
        assert loader.extensions == frozenset({".png", ".jpg", ".webp"})

    def test_exclude_sources_none_means_empty(self, tmp_path: Path):
        assert _loader(tmp_path, exclude_sources=None).exclude_sources == []

    def test_explicit_values_are_kept(self, tmp_path: Path):
        loader = _loader(
            tmp_path,
            recursive=False,
            include_sources=["celeste"],
            exclude_sources=["hades"],
            max_images=5,
            num_workers=2,
            prefetch=3,
        )
        assert (loader.recursive, loader.include_sources, loader.exclude_sources) == (
            False,
            ["celeste"],
            ["hades"],
        )
        assert (loader.max_images, loader.num_workers, loader.prefetch) == (5, 2, 3)

    def test_unknown_keys_are_ignored(self, tmp_path: Path):
        # e.g. the merged ``input`` + ``runtime`` sections (batch_size, seed, ...)
        loader = _loader(tmp_path, batch_size=64, seed=0, log_every_s=30, shard="0/1")
        assert loader.shard == (0, 1)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("extensions", ".jpg"),
            ("extensions", []),
            ("extensions", [1]),
            ("extensions", ["."]),
            ("extensions", [""]),
            ("recursive", "yes"),
            ("recursive", 1),
            ("include_sources", "celeste"),
            ("include_sources", [1]),
            ("exclude_sources", "hades"),
            ("exclude_sources", [None]),
            ("shard", (1, 1)),
            ("shard", (-1, 2)),
            ("shard", (0, 0)),
            ("shard", (0,)),
            ("shard", (0, 1, 2)),
            ("shard", (True, 2)),
            ("shard", (0.0, 1)),
            ("shard", "2/2"),
            ("shard", "a/b"),
            ("shard", "1"),
            ("shard", None),
            ("max_images", 0),
            ("max_images", -1),
            ("max_images", 1.5),
            ("max_images", True),
            ("max_images", "10"),
            ("num_workers", -1),
            ("num_workers", 1.5),
            ("num_workers", True),
            ("num_workers", None),
            ("prefetch", 0),
            ("prefetch", None),
            ("prefetch", 2.0),
        ],
    )
    def test_invalid_value_raises(self, tmp_path: Path, key, value):
        with pytest.raises(ValueError, match=key):
            _loader(tmp_path, **{key: value})

    @pytest.mark.parametrize("value", [None, "", 5])
    def test_input_dir_required(self, value):
        with pytest.raises(ValueError, match="input_dir"):
            ImageDataLoader({"input_dir": value})

    def test_input_dir_missing_key(self):
        with pytest.raises(ValueError, match="input_dir: is required"):
            ImageDataLoader({})

    def test_input_dir_must_exist(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            ImageDataLoader({"input_dir": str(tmp_path / "missing")})

    def test_input_dir_must_be_directory(self, tmp_path: Path):
        f = _file(tmp_path / "file.jpg")
        with pytest.raises(ValueError, match="not a directory"):
            ImageDataLoader({"input_dir": str(f)})

    def test_v1_source_dirs_is_not_accepted(self, tmp_path: Path):
        with pytest.raises(ValueError, match="input_dir"):
            ImageDataLoader({"source_dirs": [str(tmp_path)]})

    def test_all_problems_reported_together(self, tmp_path: Path):
        with pytest.raises(ValueError) as excinfo:
            ImageDataLoader(
                {"input_dir": str(tmp_path / "missing"), "shard": "3/2", "max_images": 0, "prefetch": 0}
            )
        message = str(excinfo.value)
        for key in ("input_dir", "shard", "max_images", "prefetch"):
            assert key in message

    def test_config_must_be_dict(self):
        with pytest.raises(ValueError, match="dict"):
            ImageDataLoader(["/data"])  # type: ignore[arg-type]

    def test_init_does_not_walk(self, tmp_path: Path, monkeypatch):
        _dataset(tmp_path)
        calls = []
        monkeypatch.setattr(os, "walk", lambda *a, **k: calls.append(a) or iter(()))
        _loader(tmp_path)
        assert calls == []


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_finds_images_relative_and_sorted(self, tmp_path: Path):
        expected = _dataset(tmp_path)
        found = _loader(tmp_path).discover()
        assert _posix(found) == expected
        assert all(not p.is_absolute() for p in found)

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path):
        # B6 / AGENTS.md §6.5: upper-case suffixes vanish on Linux with glob("*.jpg").
        names = ["a.JPG", "b.Png", "c.jpeg", "d.JPEG", "e.WEBP", "f.jpg", "g.PnG"]
        for n in names:
            _file(tmp_path / "src" / n, b"")
        assert _posix(_loader(tmp_path).discover()) == [f"src/{n}" for n in names]

    def test_configured_extensions_case_insensitive(self, tmp_path: Path):
        for n in ["a.PNG", "b.png", "c.jpg", "d.JPG"]:
            _file(tmp_path / "s" / n, b"")
        found = _loader(tmp_path, extensions=[".PNG"]).discover()
        assert _posix(found) == ["s/a.PNG", "s/b.png"]

    def test_non_images_are_ignored(self, tmp_path: Path):
        for n in [
            "000001_clip_h.msgpack",
            "000001_vae_latent_64.msgpack",
            "000001.jpg.msgpack",
            "notes.txt",
            "noext",
            "jpg",
            ".jpg.bak",
            "archive.tar",
        ]:
            _file(tmp_path / "src" / "0001" / n, b"")
        _file(tmp_path / "src" / "0001" / "000001.jpg", b"")
        assert _posix(_loader(tmp_path).discover()) == ["src/0001/000001.jpg"]

    def test_files_are_never_opened(self, tmp_path: Path, monkeypatch):
        # msgpack files (and images) are rejected / accepted by suffix only.
        _dataset(tmp_path)
        _file(tmp_path / "celeste" / "0001" / "broken.jpg")  # not decodable, still discovered
        opened: list = []
        real_open = builtins.open
        monkeypatch.setattr(builtins, "open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
        monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(a[0]) or pytest.fail("opened"))
        loader = _loader(tmp_path)
        found = loader.discover()
        loader.plan(_never, ["long"])
        assert opened == []
        assert "celeste/0001/broken.jpg" in _posix(found)

    def test_sorted_by_posix_string_not_path_parts(self, tmp_path: Path):
        # as_posix order: "a-b/..." < "a.jpg" < "a/..." ('-' < '.' < '/');
        # Path ordering by parts would put "a/b.jpg" first.
        for rel in ["a/b.jpg", "a-b/c.jpg", "a.jpg", "B/x.jpg"]:
            _file(tmp_path / rel, b"")
        assert _posix(_loader(tmp_path).discover()) == ["B/x.jpg", "a-b/c.jpg", "a.jpg", "a/b.jpg"]

    def test_same_stem_in_many_sources_all_kept(self, tmp_path: Path):
        # AGENTS.md §6.4: stems repeat across sources; rel_path is the key.
        for s in ["a", "b", "c"]:
            _file(tmp_path / s / "0001" / "000001.jpg", b"")
        assert len(_loader(tmp_path).discover()) == 3

    def test_recursive_disabled_only_root_files(self, tmp_path: Path):
        _file(tmp_path / "top.jpg", b"")
        _file(tmp_path / "src" / "deep.jpg", b"")
        assert _posix(_loader(tmp_path, recursive=False).discover()) == ["top.jpg"]

    def test_recursive_enabled_includes_root_files(self, tmp_path: Path):
        _file(tmp_path / "top.jpg", b"")
        _file(tmp_path / "src" / "0001" / "deep.jpg", b"")
        assert _posix(_loader(tmp_path).discover()) == ["src/0001/deep.jpg", "top.jpg"]

    def test_empty_directory(self, tmp_path: Path):
        assert _loader(tmp_path).discover() == []

    def test_include_sources(self, tmp_path: Path):
        _dataset(tmp_path)
        _file(tmp_path / "root.jpg", b"")
        found = _loader(tmp_path, include_sources=["celeste", "hades"]).discover()
        assert {p.parts[0] for p in found} == {"celeste", "hades"}
        assert "root.jpg" not in _posix(found)  # source "" is not an included source
        assert len(found) == 4

    def test_include_sources_empty_list_means_nothing(self, tmp_path: Path):
        _dataset(tmp_path)
        assert _loader(tmp_path, include_sources=[]).discover() == []

    def test_exclude_sources(self, tmp_path: Path):
        _dataset(tmp_path)
        _file(tmp_path / "root.jpg", b"")
        found = _loader(tmp_path, exclude_sources=["celeste"]).discover()
        assert {p.parts[0] for p in found} == {"hades", "stardew-valley", "root.jpg"}

    def test_exclude_wins_over_include(self, tmp_path: Path):
        _dataset(tmp_path)
        found = _loader(tmp_path, include_sources=["celeste", "hades"], exclude_sources=["hades"]).discover()
        assert {p.parts[0] for p in found} == {"celeste"}

    def test_filters_apply_to_top_level_only(self, tmp_path: Path):
        _file(tmp_path / "other" / "celeste" / "1.jpg", b"")
        _file(tmp_path / "celeste" / "1.jpg", b"")
        found = _loader(tmp_path, exclude_sources=["celeste"]).discover()
        assert _posix(found) == ["other/celeste/1.jpg"]

    def test_pruned_sources_are_not_entered(self, tmp_path: Path, monkeypatch):
        _dataset(tmp_path)
        scanned: list[str] = []
        real_scandir = os.scandir

        def spy(path=".", *args):
            scanned.append(Path(os.fsdecode(path)).resolve().as_posix())
            return real_scandir(path, *args)

        monkeypatch.setattr(os, "scandir", spy)
        _loader(tmp_path, include_sources=["celeste", "stardew-valley"], exclude_sources=["stardew-valley"]).discover()
        root = tmp_path.resolve().as_posix()
        assert root in scanned
        assert not [s for s in scanned if s.startswith(root + "/hades") or s.startswith(root + "/stardew-valley")]
        assert [s for s in scanned if s.startswith(root + "/celeste")]

    def test_unknown_source_names_warn(self, tmp_path: Path, caplog):
        _dataset(tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _loader(tmp_path, include_sources=["celeste", "celest"], exclude_sources=["haddes"]).discover()
        assert "include_sources" in caplog.text and "celest'" in caplog.text
        assert "exclude_sources" in caplog.text and "haddes" in caplog.text

    def test_single_walk_and_cached(self, tmp_path: Path, monkeypatch):
        # B7: one walk over the tree, reused by later discover()/plan() calls.
        expected = _dataset(tmp_path)
        n_dirs = sum(1 for _ in os.walk(tmp_path))
        walk_calls: list = []
        scandir_calls: list = []
        real_walk, real_scandir = os.walk, os.scandir
        monkeypatch.setattr(os, "walk", lambda *a, **k: walk_calls.append(a) or real_walk(*a, **k))
        monkeypatch.setattr(os, "scandir", lambda *a: scandir_calls.append(a) or real_scandir(*a))

        loader = _loader(tmp_path)
        first = loader.discover()
        assert len(walk_calls) == 1
        assert len(scandir_calls) == n_dirs  # each directory listed exactly once
        second = loader.discover()
        loader.plan(_never, ["long", "short"])
        loader.plan(_never, ["tags"])
        assert len(walk_calls) == 1
        assert len(scandir_calls) == n_dirs
        assert _posix(first) == _posix(second) == expected

    def test_walk_does_not_follow_symlinks(self, tmp_path: Path, monkeypatch):
        calls: list = []
        real_walk = os.walk
        monkeypatch.setattr(os, "walk", lambda *a, **k: calls.append(k) or real_walk(*a, **k))
        _loader(tmp_path).discover()
        assert calls == [{"followlinks": False}]

    def test_symlinked_directory_not_entered(self, tmp_path: Path):
        root, outside = tmp_path / "root", tmp_path / "outside"
        _file(outside / "x.jpg", b"")
        _file(root / "real" / "a.jpg", b"")
        try:
            os.symlink(outside, root / "real" / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert _posix(_loader(root).discover()) == ["real/a.jpg"]

    def test_cached_result_is_a_copy(self, tmp_path: Path):
        _dataset(tmp_path)
        loader = _loader(tmp_path)
        loader.discover().clear()
        assert len(loader.discover()) == 6

    def test_cache_ignores_later_changes(self, tmp_path: Path):
        _dataset(tmp_path)
        loader = _loader(tmp_path)
        loader.discover()
        _file(tmp_path / "new" / "1.jpg", b"")
        assert len(loader.discover()) == 6
        assert len(_loader(tmp_path).discover()) == 7

    def test_progress_and_final_count_logged(self, tmp_path: Path, monkeypatch, caplog):
        _dataset(tmp_path)  # 6 images + 18 msgpack files
        monkeypatch.setattr(dl_module, "_PROGRESS_EVERY", 5)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _loader(tmp_path).discover()
        progress = [r for r in caplog.records if "files examined," in r.getMessage()]
        assert len(progress) == 24 // 5
        assert progress[0].levelno == logging.INFO
        assert "Discovery done: 6 images found" in caplog.text
        assert "(24 files examined)" in caplog.text

    def test_default_progress_interval(self):
        assert dl_module._PROGRESS_EVERY == 50_000

    def test_stem_collisions_warned(self, tmp_path: Path, caplog):
        # SPEC §8.2 / B2: same stem, same folder -> .txt names would collide.
        _file(tmp_path / "s" / "sprite.png", b"")
        _file(tmp_path / "s" / "sprite.jpg", b"")
        _file(tmp_path / "s" / "Hero.PNG", b"")
        _file(tmp_path / "s" / "hero.webp", b"")
        _file(tmp_path / "s" / "single.jpg", b"")
        _file(tmp_path / "t" / "sprite.jpg", b"")  # other folder: no collision
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loader = _loader(tmp_path)
            loader.discover()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "4 images share a stem" in warnings[0]
        for rel in ["s/sprite.png", "s/sprite.jpg", "s/Hero.PNG", "s/hero.webp"]:
            assert rel in warnings[0]
        assert "single" not in warnings[0] and "t/sprite.jpg" not in warnings[0]
        assert _posix(sorted(loader._collisions, key=Path.as_posix)) == [
            "s/Hero.PNG", "s/hero.webp", "s/sprite.jpg", "s/sprite.png"
        ]

    def test_no_collision_warning_without_collisions(self, tmp_path: Path, caplog):
        _dataset(tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _loader(tmp_path).discover()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_collision_warning_is_truncated(self, tmp_path: Path, caplog):
        for i in range(15):
            _file(tmp_path / "s" / f"{i:02d}.png", b"")
            _file(tmp_path / "s" / f"{i:02d}.jpg", b"")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _loader(tmp_path).discover()
        assert "30 images share a stem" in caplog.text
        assert "(+10 more)" in caplog.text


# ---------------------------------------------------------------------------
# Sharding (D2)
# ---------------------------------------------------------------------------


def _spec_shard(rel: str, count: int) -> int:
    """SPEC §4.3 formula, written out independently of the module."""
    return int(hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8], 16) % count


class TestSharding:
    def test_golden_values(self):
        assert _shard_index(Path("celeste/0001/000001.jpg"), 4) == 3
        assert _shard_index(Path("celeste/0001/000001.jpg"), 7) == 2
        assert _shard_index(Path("stardew-valley/0003/000123.png"), 4) == 1
        assert int(hashlib.sha1(b"celeste/0001/000001.jpg").hexdigest()[:8], 16) == 3295457895

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 16])
    def test_matches_spec_formula(self, count):
        for i in range(100):
            rel = f"src{i % 5}/{i // 10:04d}/{i:06d}.jpg"
            assert _shard_index(Path(rel), count) == _spec_shard(rel, count)

    def test_windows_and_posix_paths_agree(self):
        win = PureWindowsPath("celeste\\0001\\000001.jpg")
        assert _shard_index(win, 4) == _shard_index(Path("celeste/0001/000001.jpg"), 4) == 3  # type: ignore[arg-type]

    def test_extension_is_part_of_the_key(self):
        # sprite.png and sprite.jpg are different images (B2, §6.4).
        values = {_shard_index(Path(f"s/sprite{ext}"), 1_000_003) for ext in (".png", ".jpg", ".jpeg", ".webp")}
        assert len(values) == 4

    @pytest.mark.parametrize("count", [1, 2, 3, 7])
    def test_partition_is_disjoint_and_complete(self, tmp_path: Path, count):
        _many(tmp_path, 90)
        everything = set(_posix(_loader(tmp_path).discover()))
        seen: list[str] = []
        total_in_shard = 0
        for index in range(count):
            plan = _loader(tmp_path, shard=(index, count)).plan(_never, ["long"])
            shard_paths = _posix(p for p, _ in plan.items)
            assert all(_spec_shard(p, count) == index for p in shard_paths)
            assert plan.images_discovered == 90
            total_in_shard += plan.images_in_shard
            seen.extend(shard_paths)
        assert len(seen) == len(set(seen))  # disjoint
        assert set(seen) == everything  # complete
        assert total_in_shard == 90

    def test_every_shard_gets_work(self, tmp_path: Path):
        _many(tmp_path, 200)
        sizes = [_loader(tmp_path, shard=(i, 4)).plan(_never, ["long"]).images_in_shard for i in range(4)]
        assert sum(sizes) == 200
        assert min(sizes) > 20

    def test_membership_stable_when_files_added_or_removed(self, tmp_path: Path):
        count = 4
        root, removed = tmp_path / "root", tmp_path / "removed"
        removed.mkdir()
        rels = _many(root, 60)

        def membership() -> dict[str, int]:
            out = {}
            for index in range(count):
                plan = _loader(root, shard=(index, count)).plan(_never, ["long"])
                out.update({p.as_posix(): index for p, _ in plan.items})
            return out

        before = membership()
        for rel in rels[::3]:  # remove a third (moved to a sibling dir outside the root)
            (root / rel).replace(removed / Path(rel).name)
        _many(root, 25, start=1000)  # add new files, some in new folders
        after = membership()
        common = set(before) & set(after)
        assert len(common) == 40
        assert all(before[p] == after[p] for p in common)
        assert len(after) == 65

    def test_string_shard_equals_tuple_shard(self, tmp_path: Path):
        _many(tmp_path, 30)
        a = _loader(tmp_path, shard="2/3").plan(_never, ["long"])
        b = _loader(tmp_path, shard=(2, 3)).plan(_never, ["long"])
        assert a == b


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


class TestPlan:
    def test_everything_pending(self, tmp_path: Path):
        expected = _dataset(tmp_path)
        plan = _loader(tmp_path).plan(_never, ["long", "short"])
        assert isinstance(plan, Plan)
        assert _posix(p for p, _ in plan.items) == expected
        assert all(types == ["long", "short"] for _, types in plan.items)
        assert plan.shard == (0, 1)
        assert plan.caption_types == ["long", "short"]
        assert (plan.images_discovered, plan.images_in_shard) == (6, 6)
        assert (plan.images_already_complete, plan.images_deferred) == (0, 0)
        assert (plan.images_planned, plan.captions_planned) == (6, 12)

    def test_predicate_receives_rel_path_and_type(self, tmp_path: Path):
        _file(tmp_path / "celeste" / "0001" / "000001.jpg", b"")
        calls: list = []
        _loader(tmp_path).plan(lambda p, t: calls.append((p, t)) or False, ["long", "tags"])
        assert calls == [
            (Path("celeste/0001/000001.jpg"), "long"),
            (Path("celeste/0001/000001.jpg"), "tags"),
        ]
        assert all(isinstance(p, Path) and not p.is_absolute() for p, _ in calls)

    def test_partial_completion_keeps_pending_types_in_order(self, tmp_path: Path):
        _dataset(tmp_path)
        done = {
            ("celeste/0001/000001.jpg", "long"),
            ("celeste/0001/000002.jpg", "short"),
            ("celeste/0001/000002.jpg", "tags"),
            ("hades/0001/000001.jpg", "long"),
            ("hades/0001/000001.jpg", "short"),
            ("hades/0001/000001.jpg", "tags"),
        }
        plan = _loader(tmp_path).plan(lambda p, t: (p.as_posix(), t) in done, ["long", "short", "tags"])
        items = {p.as_posix(): types for p, types in plan.items}
        assert items["celeste/0001/000001.jpg"] == ["short", "tags"]
        assert items["celeste/0001/000002.jpg"] == ["long"]
        assert "hades/0001/000001.jpg" not in items
        assert items["celeste/0002/000001.jpg"] == ["long", "short", "tags"]
        assert plan.images_already_complete == 1
        assert plan.images_planned == 5
        assert plan.captions_planned == 2 + 1 + 3 * 3

    def test_all_complete_gives_empty_plan(self, tmp_path: Path):
        _dataset(tmp_path)
        plan = _loader(tmp_path).plan(lambda p, t: True, ["long"])
        assert plan.items == []
        assert (plan.images_already_complete, plan.images_planned, plan.captions_planned) == (6, 0, 0)

    def test_max_images_applies_after_resume(self, tmp_path: Path):
        # D4 / B4: the cap counts pending images, not discovered images.
        rels = sorted(_many(tmp_path, 10, sources=("s",)))
        done = set(rels[:4])
        plan = _loader(tmp_path, max_images=3).plan(lambda p, t: p.as_posix() in done, ["long"])
        assert _posix(p for p, _ in plan.items) == rels[4:7]
        assert plan.images_already_complete == 4
        assert plan.images_deferred == 3
        assert plan.images_planned == 3
        assert plan.images_in_shard == plan.images_already_complete + plan.images_deferred + plan.images_planned

    def test_repeated_limited_runs_advance(self, tmp_path: Path):
        # B4 regression: the second dry-run-sized run must not process zero images.
        rels = sorted(_many(tmp_path, 10, sources=("s",)))
        done: set[str] = set()
        batches = []
        while True:
            plan = _loader(tmp_path, max_images=4).plan(lambda p, t: p.as_posix() in done, ["long"])
            if not plan.items:
                break
            batch = _posix(p for p, _ in plan.items)
            batches.append(batch)
            done.update(batch)
        assert batches == [rels[0:4], rels[4:8], rels[8:10]]

    def test_max_images_counts_images_not_captions(self, tmp_path: Path):
        _many(tmp_path, 5)
        plan = _loader(tmp_path, max_images=2).plan(_never, ["long", "short", "tags"])
        assert plan.images_planned == 2
        assert plan.captions_planned == 6
        assert plan.images_deferred == 3

    def test_max_images_larger_than_pending(self, tmp_path: Path):
        _many(tmp_path, 5)
        plan = _loader(tmp_path, max_images=100).plan(_never, ["long"])
        assert (plan.images_planned, plan.images_deferred) == (5, 0)

    def test_max_images_applies_within_shard(self, tmp_path: Path):
        _many(tmp_path, 60)
        full = _loader(tmp_path, shard=(1, 3)).plan(_never, ["long"])
        capped = _loader(tmp_path, shard=(1, 3), max_images=5).plan(_never, ["long"])
        assert capped.items == full.items[:5]
        assert capped.images_in_shard == full.images_in_shard
        assert capped.images_deferred == full.images_in_shard - 5

    @pytest.mark.parametrize("count", [1, 2, 5])
    def test_summary_invariant_per_shard(self, tmp_path: Path, count):
        _many(tmp_path, 40)
        for index in range(count):
            plan = _loader(tmp_path, shard=(index, count), max_images=3).plan(
                lambda p, t: p.stem.endswith("3"), ["long"]
            )
            assert plan.images_in_shard == (
                plan.images_already_complete + plan.images_deferred + plan.images_planned
            )

    def test_new_predicate_each_call(self, tmp_path: Path):
        _many(tmp_path, 6)
        loader = _loader(tmp_path)
        assert loader.plan(_never, ["long"]).images_planned == 6
        assert loader.plan(lambda p, t: True, ["long"]).images_planned == 0

    def test_caption_types_are_copied(self, tmp_path: Path):
        _many(tmp_path, 2)
        types = ["long", "short"]
        plan = _loader(tmp_path).plan(_never, types)
        types.append("tags")
        assert plan.caption_types == ["long", "short"]
        assert all(t == ["long", "short"] for _, t in plan.items)

    @pytest.mark.parametrize("types", [[], ["long", "long"], "long", [1], None])
    def test_invalid_caption_types(self, tmp_path: Path, types):
        with pytest.raises(ValueError, match="caption_types"):
            _loader(tmp_path).plan(_never, types)  # type: ignore[arg-type]

    def test_predicate_exception_propagates(self, tmp_path: Path):
        _many(tmp_path, 2)

        def broken(p, t):
            raise RuntimeError("resume index broken")

        with pytest.raises(RuntimeError, match="resume index broken"):
            _loader(tmp_path).plan(broken, ["long"])

    def test_plan_is_logged(self, tmp_path: Path, caplog):
        _many(tmp_path, 6)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _loader(tmp_path, max_images=2).plan(lambda p, t: p.stem == "000000", ["long", "short"])
        assert "Plan for shard 0/1: 6 discovered, 6 in shard, 1 already complete, 3 deferred" in caplog.text
        assert "2 images / 4 captions planned" in caplog.text

    def test_captions_by_source(self, tmp_path: Path):
        _dataset(tmp_path)
        _file(tmp_path / "root.jpg", b"")
        plan = _loader(tmp_path).plan(
            lambda p, t: p.parts[0] == "hades" and t == "short", ["short", "long"]
        )
        assert plan.captions_by_source() == {
            "": {"short": 1, "long": 1},
            "celeste": {"short": 3, "long": 3},
            "hades": {"long": 1},
            "stardew-valley": {"short": 2, "long": 2},
        }
        assert list(plan.captions_by_source()["celeste"]) == ["short", "long"]

    def test_captions_by_source_empty(self):
        plan = Plan(shard=(0, 1), caption_types=["long"], images_discovered=0, images_in_shard=0,
                    images_already_complete=0, images_deferred=0)
        assert plan.items == [] and plan.captions_by_source() == {}
        assert (plan.images_planned, plan.captions_planned) == (0, 0)


# ---------------------------------------------------------------------------
# iter_images()
# ---------------------------------------------------------------------------


WORKERS = [0, 1, 4]


class TestIterImages:
    @pytest.mark.parametrize("num_workers", WORKERS)
    def test_yields_records_in_plan_order(self, tmp_path: Path, num_workers):
        _dataset(tmp_path)
        loader = _loader(tmp_path, num_workers=num_workers, prefetch=2)
        plan = loader.plan(_never, ["long"])
        out = list(loader.iter_images(plan))
        assert len(out) == plan.images_planned
        assert all(isinstance(r, ImageRecord) for r in out)
        assert [r.rel_path for r in out] == [p for p, _ in plan.items]
        first = out[0]
        assert first.path == (tmp_path / "celeste/0001/000001.jpg").resolve()
        assert (first.source, first.stem) == ("celeste", "000001")

    @pytest.mark.parametrize("num_workers", [0, 3])
    def test_images_not_resized_and_rgb(self, tmp_path: Path, num_workers):
        _img(tmp_path / "s" / "a.png", size=(37, 21), mode="RGBA")
        _img(tmp_path / "s" / "b.png", size=(600, 400), mode="P")
        _img(tmp_path / "s" / "c.png", size=(5, 513), mode="L")
        loader = _loader(tmp_path, num_workers=num_workers)
        out = list(loader.iter_images(loader.plan(_never, ["long"])))
        assert [(r.width, r.height) for r in out] == [(37, 21), (600, 400), (5, 513)]
        assert [r.image.size for r in out] == [(37, 21), (600, 400), (5, 513)]
        assert all(r.image.mode == "RGB" for r in out)

    def test_load_image_called_without_target_size(self, tmp_path: Path, monkeypatch):
        _img(tmp_path / "s" / "a.png")
        calls: list = []
        real = dl_module.load_image
        monkeypatch.setattr(dl_module, "load_image", lambda *a, **k: calls.append((a, k)) or real(*a, **k))
        loader = _loader(tmp_path)
        list(loader.iter_images(loader.plan(_never, ["long"])))
        assert calls == [((loader.input_dir / "s" / "a.png", loader.input_dir), {})]

    def test_root_level_file_has_empty_source(self, tmp_path: Path):
        _img(tmp_path / "top.png")
        loader = _loader(tmp_path)
        (record,) = list(loader.iter_images(loader.plan(_never, ["long"])))
        assert (record.rel_path, record.source) == (Path("top.png"), "")

    @pytest.mark.parametrize("num_workers", WORKERS)
    def test_undecodable_files_yield_load_failure(self, tmp_path: Path, num_workers, caplog):
        _img(tmp_path / "s" / "0_ok.jpg", fmt="JPEG")
        _file(tmp_path / "s" / "1_text.jpg", b"this is not a jpeg")
        _file(tmp_path / "s" / "2_empty.png", b"")
        good = _img(tmp_path / "s" / "3_trunc.jpg", size=(64, 64), fmt="JPEG")
        good.write_bytes(good.read_bytes()[:200])
        _img(tmp_path / "s" / "4_ok.png", size=(9, 7))
        loader = _loader(tmp_path, num_workers=num_workers, prefetch=2)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = list(loader.iter_images(loader.plan(_never, ["long"])))
        assert [type(r).__name__ for r in out] == [
            "ImageRecord", "LoadFailure", "LoadFailure", "LoadFailure", "ImageRecord"
        ]
        failures = [r for r in out if isinstance(r, LoadFailure)]
        assert failures[0].rel_path == Path("s/1_text.jpg")
        assert failures[0].path == loader.input_dir / "s" / "1_text.jpg"
        assert failures[0].source == "s"
        assert failures[0].error.startswith("UnidentifiedImageError: ")
        assert failures[1].error.startswith("UnidentifiedImageError: ")
        assert failures[2].error.startswith("OSError: ")
        assert out[-1].image.size == (9, 7)
        assert caplog.text.count("Could not decode") == 3
        assert "s/1_text.jpg" in caplog.text

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_any_exception_becomes_load_failure(self, tmp_path: Path, monkeypatch, num_workers):
        _many(tmp_path, 3, sources=("s",))

        def boom(path, root):
            raise MemoryError("decompression bomb") if path.stem.endswith("1") else RuntimeError("weird")

        monkeypatch.setattr(dl_module, "load_image", boom)
        loader = _loader(tmp_path, num_workers=num_workers)
        out = list(loader.iter_images(loader.plan(_never, ["long"])))
        assert [r.error for r in out] == [
            "RuntimeError: weird", "MemoryError: decompression bomb", "RuntimeError: weird"
        ]

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_keyboard_interrupt_propagates_and_threads_joined(self, tmp_path: Path, monkeypatch, num_workers):
        _many(tmp_path, 6, sources=("s",))

        def interrupt(path, root):
            if path.stem == "000002":
                raise KeyboardInterrupt
            raise ValueError("not an image")

        monkeypatch.setattr(dl_module, "load_image", interrupt)
        loader = _loader(tmp_path, num_workers=num_workers, prefetch=2)
        gen = loader.iter_images(loader.plan(_never, ["long"]))
        assert isinstance(next(gen), LoadFailure)
        assert isinstance(next(gen), LoadFailure)
        with pytest.raises(KeyboardInterrupt):
            next(gen)
        assert _decoder_threads() == []

    @pytest.mark.parametrize("num_workers", [1, 3, 8])
    def test_order_is_deterministic_with_uneven_decode_times(self, tmp_path: Path, monkeypatch, num_workers):
        rels = sorted(_many(tmp_path, 24, sources=("s",)))

        def slow_fake(path, root):
            i = int(path.stem)
            time.sleep(0.001 * (3 - i % 4))  # later items often finish first
            return LoadFailure(path=path, rel_path=path.relative_to(root), source="s", error=str(i))

        monkeypatch.setattr(dl_module, "load_image", slow_fake)
        loader = _loader(tmp_path, num_workers=num_workers, prefetch=6)
        plan = loader.plan(_never, ["long"])
        for _ in range(2):
            out = list(loader.iter_images(plan))
            assert _posix(r.rel_path for r in out) == rels

    @pytest.mark.parametrize(("num_workers", "prefetch"), [(1, 1), (2, 3), (4, 8), (8, 2)])
    def test_bounded_look_ahead(self, tmp_path: Path, monkeypatch, num_workers, prefetch):
        n = 20
        _many(tmp_path, n, sources=("s",))
        submitted: list[int] = []
        decoded: list[int] = []
        real_submit = ThreadPoolExecutor.submit

        class CountingExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                submitted.append(1)
                return real_submit(self, fn, *args, **kwargs)

        monkeypatch.setattr(dl_module, "ThreadPoolExecutor", CountingExecutor)
        monkeypatch.setattr(
            dl_module, "load_image",
            lambda path, root: decoded.append(1) or LoadFailure(path, path.relative_to(root), "s", "x"),
        )
        loader = _loader(tmp_path, num_workers=num_workers, prefetch=prefetch)
        gen = loader.iter_images(loader.plan(_never, ["long"]))
        for k in range(1, n + 1):
            next(gen)
            # while the consumer holds item k, at most `prefetch` items are queued ahead of it
            assert len(submitted) == min(prefetch + k - 1, n)
            assert len(decoded) <= len(submitted)
        with pytest.raises(StopIteration):
            next(gen)
        assert len(decoded) == n
        assert _decoder_threads() == []

    def test_early_close_cancels_and_joins(self, tmp_path: Path, monkeypatch):
        _many(tmp_path, 40, sources=("s",))
        decoded: list[str] = []
        lock = threading.Lock()

        def fake(path, root):
            time.sleep(0.001)
            with lock:
                decoded.append(path.stem)
            return LoadFailure(path, path.relative_to(root), "s", "x")

        monkeypatch.setattr(dl_module, "load_image", fake)
        loader = _loader(tmp_path, num_workers=2, prefetch=5)
        gen = loader.iter_images(loader.plan(_never, ["long"]))
        next(gen)
        next(gen)
        gen.close()
        assert _decoder_threads() == []
        assert len(decoded) <= 5 + 1  # never more than submitted

    def test_break_out_of_for_loop_joins_threads(self, tmp_path: Path):
        for i in range(10):
            _img(tmp_path / "s" / f"{i}.png")
        loader = _loader(tmp_path, num_workers=3, prefetch=4)
        plan = loader.plan(_never, ["long"])
        for i, record in enumerate(loader.iter_images(plan)):
            if i == 1:
                break
        assert _decoder_threads() == []

    def test_threads_joined_after_full_iteration(self, tmp_path: Path):
        _dataset(tmp_path)
        loader = _loader(tmp_path, num_workers=4, prefetch=3)
        list(loader.iter_images(loader.plan(_never, ["long"])))
        assert _decoder_threads() == []

    def test_empty_plan_yields_nothing_and_starts_no_threads(self, tmp_path: Path, monkeypatch):
        _dataset(tmp_path)
        monkeypatch.setattr(dl_module, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("pool created"))
        loader = _loader(tmp_path, num_workers=4)
        assert list(loader.iter_images(loader.plan(lambda p, t: True, ["long"]))) == []

    def test_num_workers_zero_decodes_inline_and_lazily(self, tmp_path: Path, monkeypatch):
        _many(tmp_path, 5, sources=("s",))
        monkeypatch.setattr(dl_module, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("pool created"))
        calls: list[str] = []

        def fake(path, root):
            calls.append(threading.current_thread().name)
            return LoadFailure(path, path.relative_to(root), "s", "x")

        monkeypatch.setattr(dl_module, "load_image", fake)
        loader = _loader(tmp_path, num_workers=0)
        gen = loader.iter_images(loader.plan(_never, ["long"]))
        assert calls == []
        next(gen)
        next(gen)
        assert calls == [threading.current_thread().name] * 2

    def test_decoding_runs_on_worker_threads(self, tmp_path: Path, monkeypatch):
        _many(tmp_path, 4, sources=("s",))
        names: list[str] = []
        monkeypatch.setattr(
            dl_module, "load_image",
            lambda path, root: names.append(threading.current_thread().name)
            or LoadFailure(path, path.relative_to(root), "s", "x"),
        )
        loader = _loader(tmp_path, num_workers=2)
        list(loader.iter_images(loader.plan(_never, ["long"])))
        assert len(names) == 4
        assert all(n.startswith(dl_module._THREAD_NAME_PREFIX) for n in names)

    def test_pairs_with_plan_items(self, tmp_path: Path):
        _dataset(tmp_path)
        _file(tmp_path / "celeste" / "0001" / "000003.jpg")  # corrupt one in the middle
        loader = _loader(tmp_path, num_workers=2, max_images=4)
        plan = loader.plan(lambda p, t: p.parts[0] == "celeste" and t == "long", ["long", "short"])
        pairs = list(zip(loader.iter_images(plan), plan.items))
        assert len(pairs) == 4
        assert all(item.rel_path == rel for item, (rel, _) in pairs)
        assert [types for _, (_, types) in pairs] == [["short"]] * 4

    def test_symlinked_file_resolving_elsewhere_is_a_failure(self, tmp_path: Path):
        _img(tmp_path / "s" / "real.png")
        try:
            os.symlink(tmp_path / "s" / "real.png", tmp_path / "s" / "link.png")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        loader = _loader(tmp_path)
        out = list(loader.iter_images(loader.plan(_never, ["long"])))
        assert [type(r).__name__ for r in out] == ["LoadFailure", "ImageRecord"]
        assert out[0].rel_path == Path("s/link.png")
        assert "s/real.png" in out[0].error


# ---------------------------------------------------------------------------
# Package boundary
# ---------------------------------------------------------------------------


def test_dataloader_import_pulls_in_no_heavy_or_sibling_modules():
    # ingestion may import common only (SPEC §2.2, AGENTS.md §4).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.ingestion.dataloader; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm', 'transformers') "
        "or m.startswith(('pixel_art_auto_captioner.captioning', 'pixel_art_auto_captioner.batch'))); "
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_ingestion_package_exports_loader():
    import pixel_art_auto_captioner.ingestion as ingestion

    assert ingestion.ImageDataLoader is ImageDataLoader
