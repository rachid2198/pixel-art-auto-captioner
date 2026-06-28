"""Unit tests for ``pixel_art_auto_captioner.ingestion.dataloader``."""

from pathlib import Path

import pytest
from PIL import Image

from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader
from pixel_art_auto_captioner.common.types import ImageRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Write a small synthetic RGB PNG to *path*."""
    img = Image.new("RGB", size, color=(128, 64, 200))
    img.save(path)


def _write_corrupt_file(path: Path) -> None:
    """Write a file with a .png extension but invalid image content."""
    path.write_text("this is not a valid png file")


def _make_config(source_dirs: list[Path], **overrides) -> dict:
    """Return a minimal valid config dict with overrides applied."""
    cfg: dict = {"source_dirs": [str(d) for d in source_dirs]}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_finds_images(self, tmp_path: Path):
        """Discover finds all PNG images in the source directory."""
        _write_png(tmp_path / "a.png")
        _write_png(tmp_path / "b.png")
        _write_png(tmp_path / "c.png")
        (tmp_path / "not_an_image.txt").write_text("hello")

        loader = ImageDataLoader(_make_config([tmp_path], recursive=False))
        paths = loader.discover()

        assert len(paths) == 3
        stems = {p.stem for p in paths}
        assert stems == {"a", "b", "c"}

    def test_filters_by_extension(self, tmp_path: Path):
        """discover() only returns files with configured extensions."""
        _write_png(tmp_path / "img.png")
        _write_png(tmp_path / "img.jpg")
        (tmp_path / "notes.txt").write_text("not an image")

        loader = ImageDataLoader(
            _make_config([tmp_path], extensions=[".png"], recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 1
        assert paths[0].suffix == ".png"

    def test_recursive_enabled(self, tmp_path: Path):
        """recursive=True finds images in subdirectories."""
        _write_png(tmp_path / "top.png")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_png(sub / "nested.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=True)
        )
        paths = loader.discover()

        assert len(paths) == 2
        stems = {p.stem for p in paths}
        assert stems == {"top", "nested"}

    def test_recursive_disabled(self, tmp_path: Path):
        """recursive=False only finds images at the top level."""
        _write_png(tmp_path / "top.png")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_png(sub / "nested.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 1
        assert paths[0].stem == "top"

    def test_max_images(self, tmp_path: Path):
        """discover() respects max_images limit."""
        for i in range(5):
            _write_png(tmp_path / f"img_{i}.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], max_images=3, recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 3

    def test_max_images_none(self, tmp_path: Path):
        """When max_images is None, all images are returned."""
        for i in range(5):
            _write_png(tmp_path / f"img_{i}.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], max_images=None, recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 5

    def test_empty_directory(self, tmp_path: Path):
        """discover() returns empty list when there are no images."""
        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False)
        )
        paths = loader.discover()
        assert paths == []

    def test_nonexistent_directory(self, tmp_path: Path):
        """discover() warns and continues when a source_dir is missing."""
        missing = tmp_path / "does_not_exist"
        _write_png(tmp_path / "img.png")

        loader = ImageDataLoader(
            _make_config([missing, tmp_path], recursive=False)
        )
        paths = loader.discover()

        # Should still find the image in the valid directory.
        assert len(paths) == 1

    def test_deduplicates_overlapping_source_dirs(self, tmp_path: Path):
        """Same resolved file discovered from two source dirs pointing
        to the same physical directory is counted only once."""
        _write_png(tmp_path / "img.png")

        # Two source dirs that resolve to the same physical location
        # produce the same resolved path for img.png.
        loader = ImageDataLoader(
            _make_config([tmp_path, tmp_path], recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 1, (
            f"Expected 1 unique path after dedup, got {len(paths)}: {paths}"
        )
        assert paths[0].stem == "img"

    def test_distinct_files_with_same_name_are_not_deduped(self, tmp_path: Path):
        """Two files with the same stem in different subdirectories are
        treated as separate images, not deduplicated."""
        _write_png(tmp_path / "img.png")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_png(sub / "img.png")

        loader = ImageDataLoader(
            _make_config([tmp_path, sub], recursive=False)
        )
        paths = loader.discover()

        assert len(paths) == 2
        stems = {p.stem for p in paths}
        assert stems == {"img"}

    def test_sorts_alphabetically(self, tmp_path: Path):
        """discover() returns paths in sorted order."""
        _write_png(tmp_path / "c.png")
        _write_png(tmp_path / "a.png")
        _write_png(tmp_path / "b.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False)
        )
        paths = loader.discover()

        stems = [p.stem for p in paths]
        assert stems == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_returns_image_record(self, tmp_path: Path):
        """load() returns an ImageRecord with correct fields."""
        _write_png(tmp_path / "sprite.png", size=(32, 64))

        loader = ImageDataLoader(
            _make_config([tmp_path])
        )
        record = loader.load(tmp_path / "sprite.png")

        assert isinstance(record, ImageRecord)
        assert record.stem == "sprite"
        assert record.width == 32
        assert record.height == 64
        assert record.path == (tmp_path / "sprite.png").resolve()
        assert record.rel_path == Path("sprite.png")
        assert record.image is not None

    def test_resize(self, tmp_path: Path):
        """load() resizes the image when image_size is configured."""
        _write_png(tmp_path / "img.png", size=(64, 64))

        loader = ImageDataLoader(
            _make_config([tmp_path], image_size=(32, 32))
        )
        record = loader.load(tmp_path / "img.png")

        assert record.width == 32
        assert record.height == 32

    def test_no_resize_when_none(self, tmp_path: Path):
        """load() keeps original size when image_size is None."""
        _write_png(tmp_path / "img.png", size=(48, 96))

        loader = ImageDataLoader(
            _make_config([tmp_path], image_size=None)
        )
        record = loader.load(tmp_path / "img.png")

        assert record.width == 48
        assert record.height == 96

    def test_raises_on_path_outside_source_dirs(self, tmp_path: Path):
        """load() raises ValueError when path is not under any source_dir."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        _write_png(outside_dir / "orphan.png")

        loader = ImageDataLoader(
            _make_config([source_dir])
        )
        with pytest.raises(ValueError, match="not under any configured source"):
            loader.load(outside_dir / "orphan.png")

    def test_raises_on_nonexistent_file(self, tmp_path: Path):
        """load() propagates FileNotFoundError for missing files."""
        loader = ImageDataLoader(
            _make_config([tmp_path])
        )
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "ghost.png")


# ---------------------------------------------------------------------------
# __iter__ / skip_existing / __len__
# ---------------------------------------------------------------------------

class TestIteration:
    def test_len(self, tmp_path: Path):
        """__len__ returns correct image count."""
        for i in range(3):
            _write_png(tmp_path / f"img_{i}.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False, skip_existing=False)
        )
        assert len(loader) == 3

    def test_iter_yields_all(self, tmp_path: Path):
        """__iter__ yields all discovered images."""
        for i in range(3):
            _write_png(tmp_path / f"img_{i}.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False, skip_existing=False)
        )
        records = list(loader)

        assert len(records) == 3
        stems = {r.stem for r in records}
        assert stems == {"img_0", "img_1", "img_2"}

    def test_skip_existing(self, tmp_path: Path):
        """Images with existing .txt sidecars are skipped during iteration."""
        _write_png(tmp_path / "keep.png")
        _write_png(tmp_path / "skip_me.png")
        _write_png(tmp_path / "also_keep.png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Pre-create a sidecar for skip_me.png
        (output_dir / "skip_me.txt").write_text("pre-existing caption")

        loader = ImageDataLoader(
            _make_config(
                [tmp_path],
                recursive=False,
                skip_existing=True,
                output_dir=str(output_dir),
            )
        )

        records = list(loader)

        assert len(records) == 2
        stems = {r.stem for r in records}
        assert stems == {"keep", "also_keep"}
        assert "skip_me" not in stems

    def test_len_reflects_skip_existing(self, tmp_path: Path):
        """__len__ returns count after skip_existing filtering."""
        _write_png(tmp_path / "a.png")
        _write_png(tmp_path / "b.png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "a.txt").write_text("existing")

        loader = ImageDataLoader(
            _make_config(
                [tmp_path],
                recursive=False,
                skip_existing=True,
                output_dir=str(output_dir),
            )
        )

        assert len(loader) == 1

    def test_skip_existing_disabled(self, tmp_path: Path):
        """When skip_existing=False, all images are yielded even with
        pre-existing sidecars."""
        _write_png(tmp_path / "img.png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "img.txt").write_text("existing")

        loader = ImageDataLoader(
            _make_config(
                [tmp_path],
                recursive=False,
                skip_existing=False,
                output_dir=str(output_dir),
            )
        )

        records = list(loader)
        assert len(records) == 1

    def test_skip_existing_preserves_directory_structure(self, tmp_path: Path):
        """Nested-directory skip_existing uses directory-preserving
        sidecar paths (output_dir / rel_path.with_suffix('.txt'))."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_png(tmp_path / "top.png")
        _write_png(sub / "nested.png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Pre-create sidecar at the nested output path
        nested_output = output_dir / "sub" / "nested.txt"
        nested_output.parent.mkdir(parents=True)
        nested_output.write_text("pre-existing caption")

        loader = ImageDataLoader(
            _make_config(
                [tmp_path],
                recursive=True,
                skip_existing=True,
                output_dir=str(output_dir),
            )
        )

        records = list(loader)

        assert len(records) == 1
        assert records[0].stem == "top"

    def test_skip_existing_flat_sidecar_wrong_location(self, tmp_path: Path):
        """A flat sidecar at output_dir/nested.txt does NOT match
        an image at sub/nested.png — the sidecar must be at the correct
        directory-preserving path to trigger a skip."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_png(sub / "nested.png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Put sidecar at the WRONG (flat) location
        (output_dir / "nested.txt").write_text("flat sidecar — wrong path")

        loader = ImageDataLoader(
            _make_config(
                [tmp_path],
                recursive=True,
                skip_existing=True,
                output_dir=str(output_dir),
            )
        )

        records = list(loader)

        # The flat sidecar should NOT match the nested image.
        assert len(records) == 1
        assert records[0].stem == "nested"

    def test_corrupt_image_excluded_from_len_and_iter(self, tmp_path: Path):
        """Corrupt images are excluded during filtering so __len__
        and __iter__ agree (Critique 1 fix)."""
        _write_png(tmp_path / "good.png")
        _write_corrupt_file(tmp_path / "bad.png")

        loader = ImageDataLoader(
            _make_config([tmp_path], recursive=False)
        )

        assert len(loader) == 1
        records = list(loader)
        assert len(records) == 1
        assert records[0].stem == "good"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_missing_source_dirs_raises(self):
        """__init__ raises ValueError when source_dirs is missing."""
        with pytest.raises(ValueError, match="source_dirs"):
            ImageDataLoader({})

    def test_empty_source_dirs_raises(self):
        """__init__ raises ValueError when source_dirs is empty."""
        with pytest.raises(ValueError, match="source_dirs"):
            ImageDataLoader({"source_dirs": []})

    def test_source_dirs_string_raises(self):
        """Passing a plain string as source_dirs raises ValueError
        instead of iterating characters (Critique 3 fix)."""
        with pytest.raises(ValueError, match="must be a list"):
            ImageDataLoader({"source_dirs": "/some/path"})

    def test_source_dirs_elements_must_be_strings(self):
        """Each element of source_dirs must be a string."""
        with pytest.raises(ValueError, match="must be a string"):
            ImageDataLoader({"source_dirs": ["/valid", 123]})

    def test_extensions_must_be_list(self):
        """extensions must be a list, not a string."""
        with pytest.raises(ValueError, match="must be a list"):
            ImageDataLoader(
                {"source_dirs": ["/tmp"], "extensions": ".png"}
            )