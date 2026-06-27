"""Tests for ``common/image_utils.py`` — load_image and validate_image."""

from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from pixel_art_auto_captioner.common.image_utils import load_image, validate_image
from pixel_art_auto_captioner.common.types import ImageRecord


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def test_load_image_returns_image_record(tmp_path: Path):
    """load_image opens a real PNG and returns a properly populated ImageRecord."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    file_path = input_root / "sprite.png"
    img = Image.new("RGB", (64, 32), color="red")
    img.save(file_path)

    record = load_image(file_path, input_root)

    assert isinstance(record, ImageRecord)
    assert record.path == file_path.resolve()
    assert record.stem == "sprite"
    assert record.image.size == (64, 32)
    assert record.image.mode == "RGB"
    assert record.width == 64
    assert record.height == 32


def test_load_image_resize(tmp_path: Path):
    """load_image resizes the image when target_size is provided."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    file_path = input_root / "big.png"
    img = Image.new("RGB", (200, 100))
    img.save(file_path)

    record = load_image(file_path, input_root, target_size=(32, 32))

    assert record.image.size == (32, 32)
    assert record.width == 32
    assert record.height == 32


def test_load_image_no_resize(tmp_path: Path):
    """load_image preserves original dimensions when target_size is None."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    file_path = input_root / "icon.png"
    img = Image.new("RGB", (16, 48))
    img.save(file_path)

    record = load_image(file_path, input_root)

    assert record.image.size == (16, 48)
    assert record.width == 16
    assert record.height == 48


def test_load_image_converts_rgba_to_rgb(tmp_path: Path):
    """load_image converts RGBA images to RGB, dropping the alpha channel."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    file_path = input_root / "rgba.png"
    img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    img.save(file_path)

    record = load_image(file_path, input_root)

    assert record.image.mode == "RGB"
    assert record.image.size == (32, 32)


def test_load_image_converts_grayscale_to_rgb(tmp_path: Path):
    """load_image converts greyscale (L) images to RGB."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    file_path = input_root / "grey.png"
    img = Image.new("L", (16, 16), color=128)
    img.save(file_path)

    record = load_image(file_path, input_root)

    assert record.image.mode == "RGB"
    assert record.image.size == (16, 16)


def test_load_image_file_not_found(tmp_path: Path):
    """load_image raises FileNotFoundError for a non-existent path."""
    input_root = tmp_path / "input"
    missing = input_root / "nonexistent.png"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(missing, input_root)


def test_load_image_corrupt_file(tmp_path: Path):
    """load_image raises UnidentifiedImageError for a corrupt/bogus file."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    bad_file = input_root / "corrupt.png"
    bad_file.write_bytes(b"this is not a valid image file")

    with pytest.raises(UnidentifiedImageError):
        load_image(bad_file, input_root)


def test_load_image_empty_file(tmp_path: Path):
    """load_image raises UnidentifiedImageError for an empty file."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    empty_file = input_root / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(UnidentifiedImageError):
        load_image(empty_file, input_root)


def test_load_image_path_is_directory(tmp_path: Path):
    """load_image raises UnidentifiedImageError when path is a directory."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    with pytest.raises((UnidentifiedImageError, IsADirectoryError, PermissionError)):
        load_image(input_root, input_root)


def test_load_image_relative_stem_in_subdir(tmp_path: Path):
    """load_image computes stem as underscored relative path when image
    is in a subdirectory of input_root."""
    input_root = tmp_path / "input"
    sub_dir = input_root / "folder1"
    sub_dir.mkdir(parents=True)
    file_path = sub_dir / "sprite.png"
    Image.new("RGB", (16, 16)).save(file_path)

    record = load_image(file_path, input_root)

    assert record.stem == "folder1_sprite"


def test_load_image_relative_stem_deeply_nested(tmp_path: Path):
    """load_image handles deeply nested paths correctly."""
    input_root = tmp_path / "input"
    sub_dir = input_root / "a" / "b" / "c"
    sub_dir.mkdir(parents=True)
    file_path = sub_dir / "img.png"
    Image.new("RGB", (8, 8)).save(file_path)

    record = load_image(file_path, input_root)

    assert record.stem == "a_b_c_img"


def test_load_image_path_not_under_input_root(tmp_path: Path):
    """load_image raises ValueError when path is not under input_root."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    outside = tmp_path / "outside" / "img.png"
    outside.parent.mkdir()
    Image.new("RGB", (8, 8)).save(outside)

    with pytest.raises(ValueError, match="not under input_root"):
        load_image(outside, input_root)


def test_load_image_stem_unique_for_duplicate_filenames(tmp_path: Path):
    """load_image produces distinct stems for same filename in different
    subdirectories, preventing output file collisions."""
    input_root = tmp_path / "input"
    sub_a = input_root / "chars" / "heroes"
    sub_b = input_root / "chars" / "enemies"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)

    path_a = sub_a / "sprite.png"
    path_b = sub_b / "sprite.png"
    Image.new("RGB", (8, 8), color="red").save(path_a)
    Image.new("RGB", (8, 8), color="blue").save(path_b)

    record_a = load_image(path_a, input_root)
    record_b = load_image(path_b, input_root)

    assert record_a.stem == "chars_heroes_sprite"
    assert record_b.stem == "chars_enemies_sprite"
    assert record_a.stem != record_b.stem


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------


def test_validate_image_valid_png(tmp_path: Path):
    """validate_image returns True for a valid PNG file."""
    file_path = tmp_path / "valid.png"
    Image.new("RGB", (16, 16)).save(file_path)

    assert validate_image(file_path) is True


def test_validate_image_valid_jpg(tmp_path: Path):
    """validate_image returns True for a valid JPEG file."""
    file_path = tmp_path / "valid.jpg"
    Image.new("RGB", (16, 16)).save(file_path)

    assert validate_image(file_path) is True


def test_validate_image_valid_webp(tmp_path: Path):
    """validate_image returns True for a valid WEBP file."""
    file_path = tmp_path / "valid.webp"
    Image.new("RGB", (16, 16)).save(file_path)

    assert validate_image(file_path) is True


def test_validate_image_not_found():
    """validate_image returns False for a non-existent path."""
    assert validate_image(Path("/nonexistent/ghost.png")) is False


def test_validate_image_corrupt(tmp_path: Path):
    """validate_image returns False for a corrupt file."""
    bad_file = tmp_path / "fake.png"
    bad_file.write_bytes(b"not-an-image")

    assert validate_image(bad_file) is False


def test_validate_image_empty(tmp_path: Path):
    """validate_image returns False for an empty file."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    assert validate_image(empty_file) is False


def test_validate_image_txt_file(tmp_path: Path):
    """validate_image returns False for a plain text file."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello world")

    assert validate_image(txt_file) is False


def test_validate_image_directory(tmp_path: Path):
    """validate_image returns False when path points to a directory."""
    assert validate_image(tmp_path) is False