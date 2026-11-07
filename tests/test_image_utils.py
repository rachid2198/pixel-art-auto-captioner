"""Tests for ``common/image_utils.py`` (SPEC v1.1 §4.5, §13).

Required coverage (SPEC §13): original size, handle closed, errors. Also
covers the record fields (``rel_path``, ``source``, ``stem``), RGB
conversion, and ``validate_image``'s catch-all behaviour.
"""

import gc
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from pixel_art_auto_captioner.common import image_utils
from pixel_art_auto_captioner.common.image_utils import load_image, validate_image
from pixel_art_auto_captioner.common.types import ImageRecord

SRC = Path(__file__).resolve().parent.parent / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, size: tuple[int, int] = (64, 32), mode: str = "RGB",
                 color=None, fmt: str | None = None) -> Path:
    """Create parent dirs and save a synthetic image at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        color = {"RGB": (200, 10, 30), "RGBA": (200, 10, 30, 128), "L": 90,
                 "LA": (90, 200), "P": 3, "1": 1}[mode]
    Image.new(mode, size, color=color).save(path, format=fmt)
    return path


def _write_animated_gif(path: Path, size: tuple[int, int] = (24, 16)) -> Path:
    """Save a 3-frame GIF. Pillow keeps the file open after decoding a
    multi-frame image until ``close()``, so it exposes handle leaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.new("RGB", size, c) for c in ("red", "green", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path


def _write_truncated_jpeg(path: Path) -> Path:
    """Save a noisy JPEG and drop its last 30 %: the header is intact, so
    ``Image.open`` succeeds and the error comes from decoding (``OSError``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.effect_noise((64, 64), 64).convert("RGB").save(path, format="JPEG", quality=95)
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * 0.7)])
    return path


def _write_crc_corrupt_png(path: Path) -> Path:
    """Save a PNG with a flipped byte in the IDAT chunk. ``verify()`` raises
    ``SyntaxError`` on it, which is not an ``OSError``."""
    _write_image(path, size=(16, 16))
    data = bytearray(path.read_bytes())
    idat = data.find(b"IDAT")
    data[idat + 6] ^= 0xFF
    path.write_bytes(bytes(data))
    return path


class _OpenSpy:
    """Wraps ``Image.open`` inside ``image_utils`` and records each image it
    returns plus the underlying file object Pillow opened for it.

    Checking the file object (not whether ``Image.close()`` was called) works
    across Pillow versions: Pillow 12's ``__exit__`` closes the file without
    calling ``close()``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.opened: list[Image.Image] = []
        self.files: list = []
        real_open = Image.open

        def spy_open(*args, **kwargs):
            img = real_open(*args, **kwargs)
            self.opened.append(img)
            self.files.append(img.fp)
            return img

        monkeypatch.setattr(image_utils.Image, "open", spy_open)

    def all_closed(self) -> bool:
        return all(f is not None and f.closed for f in self.files)


def _resource_warnings(fn) -> list[str]:
    """Run *fn*, force garbage collection, return any ResourceWarning texts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn()
        except Exception:
            pass
        gc.collect()
    return [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)]


# ---------------------------------------------------------------------------
# load_image: record fields
# ---------------------------------------------------------------------------


def test_load_image_returns_populated_record(tmp_path: Path):
    root = tmp_path / "input"
    file_path = _write_image(root / "celeste" / "0001" / "000001.png", size=(64, 32))

    record = load_image(file_path, root)

    assert isinstance(record, ImageRecord)
    assert record.path == file_path.resolve()
    assert record.path.is_absolute()
    assert record.rel_path == Path("celeste/0001/000001.png")
    assert record.source == "celeste"
    assert record.stem == "000001"
    assert record.image.mode == "RGB"
    assert record.image.size == (64, 32)
    assert (record.width, record.height) == (64, 32)


def test_load_image_file_directly_in_root_has_empty_source(tmp_path: Path):
    root = tmp_path / "input"
    file_path = _write_image(root / "sprite.png")

    record = load_image(file_path, root)

    assert record.rel_path == Path("sprite.png")
    assert record.source == ""
    assert record.stem == "sprite"


def test_load_image_source_is_first_component_for_one_level(tmp_path: Path):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "stardew-valley" / "a.png"), root)
    assert record.source == "stardew-valley"
    assert record.rel_path == Path("stardew-valley/a.png")


def test_load_image_same_stem_in_different_sources(tmp_path: Path):
    """Stems repeat across sources; rel_path keeps them apart (AGENTS §6.4)."""
    root = tmp_path / "input"
    a = load_image(_write_image(root / "celeste" / "0001" / "000001.jpg", fmt="JPEG"), root)
    b = load_image(_write_image(root / "hades" / "0001" / "000001.jpg", fmt="JPEG"), root)

    assert a.stem == b.stem == "000001"
    assert a.rel_path != b.rel_path
    assert (a.source, b.source) == ("celeste", "hades")


def test_load_image_relative_paths_are_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "input"
    _write_image(root / "src" / "img.png")
    monkeypatch.chdir(tmp_path)

    record = load_image(Path("input/src/../src/img.png"), Path("input"))

    assert record.path == (root / "src" / "img.png").resolve()
    assert record.rel_path == Path("src/img.png")
    assert record.source == "src"


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P", "1"])
def test_load_image_converts_to_rgb(tmp_path: Path, mode: str):
    root = tmp_path / "input"
    record = load_image(_write_image(root / f"img_{mode}.png", size=(20, 10), mode=mode), root)
    assert record.image.mode == "RGB"
    assert (record.width, record.height) == (20, 10)


@pytest.mark.parametrize(
    "name, fmt",
    [("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.jpeg", "JPEG"), ("a.webp", "WEBP"),
     ("a.JPG", "JPEG"), ("a.PNG", "PNG")],
)
def test_load_image_formats_and_extension_case(tmp_path: Path, name: str, fmt: str):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "src" / name, size=(33, 17), fmt=fmt), root)
    assert record.image.mode == "RGB"
    assert (record.width, record.height) == (33, 17)
    assert record.rel_path == Path("src") / name


def test_load_image_preserves_pixels(tmp_path: Path):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "red.png", color=(200, 10, 30)), root)
    assert record.image.getpixel((0, 0)) == (200, 10, 30)


# ---------------------------------------------------------------------------
# load_image: original size (B5)
# ---------------------------------------------------------------------------


def test_load_image_without_target_size_keeps_image_size(tmp_path: Path):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "icon.png", size=(16, 48)), root)
    assert record.image.size == (16, 48)
    assert (record.width, record.height) == (16, 48)


def test_load_image_target_size_resizes_image_but_records_original_size(tmp_path: Path):
    """B5 regression: width/height are the original size, not the resized one."""
    root = tmp_path / "input"
    record = load_image(_write_image(root / "big.png", size=(200, 100)), root, target_size=(32, 24))

    assert record.image.size == (32, 24)
    assert record.image.mode == "RGB"
    assert (record.width, record.height) == (200, 100)


def test_load_image_target_size_upscale_records_original_size(tmp_path: Path):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "tiny.png", size=(8, 4)), root, target_size=(64, 64))
    assert record.image.size == (64, 64)
    assert (record.width, record.height) == (8, 4)


def test_load_image_original_size_of_non_rgb_source(tmp_path: Path):
    root = tmp_path / "input"
    record = load_image(_write_image(root / "p.png", size=(30, 70), mode="P"), root,
                        target_size=(10, 10))
    assert (record.width, record.height) == (30, 70)


# ---------------------------------------------------------------------------
# load_image: file handle closed (B10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multi_frame", [False, True])
def test_load_image_closes_file_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       multi_frame: bool):
    root = tmp_path / "input"
    path = (_write_animated_gif(root / "anim.gif") if multi_frame
            else _write_image(root / "still.png"))
    spy = _OpenSpy(monkeypatch)

    record = load_image(path, root)

    assert len(spy.opened) == 1
    assert spy.all_closed()
    assert spy.opened[0].fp is None
    # The returned image is a separate, fully decoded copy and stays usable.
    assert record.image is not spy.opened[0]
    expected = (255, 0, 0) if multi_frame else (200, 10, 30)
    assert record.image.getpixel((0, 0)) == expected


def test_load_image_multi_frame_gif_emits_no_resource_warning(tmp_path: Path):
    """Pillow keeps a multi-frame file open after decoding; v1.0's missing
    ``close()`` leaked it and triggered an 'unclosed file' ResourceWarning."""
    root = tmp_path / "input"
    path = _write_animated_gif(root / "anim.gif")
    assert _resource_warnings(lambda: load_image(path, root)) == []


def test_load_image_file_can_be_replaced_after_loading(tmp_path: Path):
    """On Windows an open handle blocks replacing the file; must work after load."""
    root = tmp_path / "input"
    path = _write_animated_gif(root / "anim.gif")
    load_image(path, root)

    replacement = _write_image(tmp_path / "other.png")
    os.replace(replacement, path)
    path.write_bytes(path.read_bytes())
    assert path.exists()


def test_load_image_closes_file_handle_on_decode_error(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "input"
    path = _write_truncated_jpeg(root / "src" / "broken.jpg")
    spy = _OpenSpy(monkeypatch)

    with pytest.raises(OSError):
        load_image(path, root)

    assert len(spy.opened) == 1
    assert spy.all_closed()


def test_load_image_decode_error_emits_no_resource_warning(tmp_path: Path):
    root = tmp_path / "input"
    path = _write_truncated_jpeg(root / "broken.jpg")
    assert _resource_warnings(lambda: load_image(path, root)) == []


# ---------------------------------------------------------------------------
# load_image: errors
# ---------------------------------------------------------------------------


def test_load_image_path_outside_root_raises_value_error(tmp_path: Path):
    """Sibling dirs, AGENTS §6.2."""
    root = tmp_path / "root"
    root.mkdir()
    outside = _write_image(tmp_path / "outside" / "orphan.png")

    with pytest.raises(ValueError, match="not under input_root"):
        load_image(outside, root)


def test_load_image_sibling_with_common_prefix_is_outside(tmp_path: Path):
    """``data2/x.png`` is not under ``data`` even though the string starts with it."""
    root = tmp_path / "data"
    root.mkdir()
    outside = _write_image(tmp_path / "data2" / "x.png")

    with pytest.raises(ValueError, match="not under input_root"):
        load_image(outside, root)


def test_load_image_outside_root_checked_before_opening(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    root.mkdir()
    spy = _OpenSpy(monkeypatch)

    # Missing file outside the root: the boundary error wins, nothing is opened.
    with pytest.raises(ValueError, match="not under input_root"):
        load_image(tmp_path / "outside" / "ghost.png", root)
    assert spy.opened == []


def test_load_image_file_not_found(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        load_image(root / "src" / "missing.png", root)


def test_load_image_not_an_image_raises_unidentified(tmp_path: Path):
    root = tmp_path / "input"
    bad = root / "corrupt.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not a valid image file")

    with pytest.raises(UnidentifiedImageError):
        load_image(bad, root)


def test_load_image_empty_file_raises_unidentified(tmp_path: Path):
    root = tmp_path / "input"
    empty = root / "empty.png"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")

    with pytest.raises(UnidentifiedImageError):
        load_image(empty, root)


def test_load_image_msgpack_file_raises_unidentified(tmp_path: Path):
    root = tmp_path / "input"
    blob = root / "src" / "000001_clip_h.msgpack"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x92\xa3abc\xcd\x01\x00")

    with pytest.raises(UnidentifiedImageError):
        load_image(blob, root)


def test_load_image_truncated_jpeg_raises_os_error(tmp_path: Path):
    root = tmp_path / "input"
    path = _write_truncated_jpeg(root / "src" / "broken.jpg")
    with pytest.raises(OSError):
        load_image(path, root)


def test_load_image_directory_raises_os_error(tmp_path: Path):
    """IsADirectoryError on POSIX, PermissionError on Windows: both OSError."""
    root = tmp_path / "input"
    (root / "src").mkdir(parents=True)
    with pytest.raises(OSError):
        load_image(root / "src", root)


def test_load_image_error_types_are_exceptions(tmp_path: Path):
    """The data loader turns any ``Exception`` into a LoadFailure (SPEC §4.4)."""
    root = tmp_path / "input"
    root.mkdir()
    bad = root / "x.png"
    bad.write_bytes(b"nope")
    for path in (bad, root / "missing.png", tmp_path / "elsewhere.png"):
        with pytest.raises(Exception):
            load_image(path, root)


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, fmt", [("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.webp", "WEBP"), ("a.JPG", "JPEG")]
)
def test_validate_image_valid_files(tmp_path: Path, name: str, fmt: str):
    assert validate_image(_write_image(tmp_path / name, fmt=fmt)) is True


def test_validate_image_valid_animated_gif(tmp_path: Path):
    assert validate_image(_write_animated_gif(tmp_path / "anim.gif")) is True


def test_validate_image_not_found(tmp_path: Path):
    assert validate_image(tmp_path / "ghost.png") is False


def test_validate_image_corrupt(tmp_path: Path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not a valid image file")
    assert validate_image(bad) is False


def test_validate_image_empty(tmp_path: Path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert validate_image(empty) is False


def test_validate_image_text_file(tmp_path: Path):
    txt = tmp_path / "readme.txt"
    txt.write_text("hello world")
    assert validate_image(txt) is False


def test_validate_image_directory(tmp_path: Path):
    assert validate_image(tmp_path) is False


def test_validate_image_crc_corrupt_png_raising_syntax_error(tmp_path: Path):
    """Pillow's ``verify()`` raises ``SyntaxError`` (not ``OSError``) for a
    broken PNG chunk; v1.0's narrow except let it escape (SPEC §4.5)."""
    path = _write_crc_corrupt_png(tmp_path / "crc.png")
    with pytest.raises(SyntaxError):
        with Image.open(path) as img:
            img.verify()
    assert validate_image(path) is False


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("boom"), ValueError("bad"), SyntaxError("broken PNG file"),
     MemoryError(), Image.DecompressionBombError("too big"), OSError("io")],
    ids=lambda e: type(e).__name__,
)
def test_validate_image_catches_any_exception_from_open(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       exc: Exception):
    path = _write_image(tmp_path / "ok.png")

    def raising_open(*args, **kwargs):
        raise exc

    monkeypatch.setattr(image_utils.Image, "open", raising_open)
    assert validate_image(path) is False


@pytest.mark.parametrize("exc", [RuntimeError("boom"), SyntaxError("broken"), IndexError("x")],
                         ids=lambda e: type(e).__name__)
def test_validate_image_catches_any_exception_from_verify(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         exc: Exception):
    path = _write_image(tmp_path / "ok.png")
    spy = _OpenSpy(monkeypatch)
    real_spy_open = image_utils.Image.open

    def open_with_failing_verify(*args, **kwargs):
        img = real_spy_open(*args, **kwargs)

        def verify():
            raise exc

        img.verify = verify  # type: ignore[method-assign]
        return img

    monkeypatch.setattr(image_utils.Image, "open", open_with_failing_verify)
    assert validate_image(path) is False
    assert spy.all_closed()


def test_validate_image_does_not_swallow_keyboard_interrupt(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch):
    """Only ``Exception`` subclasses are caught; Ctrl-C still stops the process."""
    path = _write_image(tmp_path / "ok.png")

    def interrupted_open(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(image_utils.Image, "open", interrupted_open)
    with pytest.raises(KeyboardInterrupt):
        validate_image(path)


@pytest.mark.parametrize("valid", [True, False])
def test_validate_image_closes_file_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           valid: bool):
    path = (_write_animated_gif(tmp_path / "anim.gif") if valid
            else _write_crc_corrupt_png(tmp_path / "crc.png"))
    spy = _OpenSpy(monkeypatch)

    assert validate_image(path) is valid
    assert len(spy.opened) == 1
    assert spy.all_closed()


def test_validate_image_does_not_modify_file(tmp_path: Path):
    path = _write_image(tmp_path / "ok.png")
    before = path.read_bytes()
    validate_image(path)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Package boundary
# ---------------------------------------------------------------------------


def test_image_utils_import_pulls_in_no_torch_or_vllm():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.common.image_utils; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
