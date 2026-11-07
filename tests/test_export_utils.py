"""Tests for ``common/export_utils.py`` (SPEC v1.1 §8, §10.3, §13).

Covers ``ShardWriter`` (JSONL before ``.txt``, flush per record, fsync
cadence, append, tail repair), the JSONL format (§8.3: dataclass key order,
POSIX paths, ``ensure_ascii=False``, ``Z`` timestamps), ``read_records``
(malformed final line skipped, elsewhere an error), ``ResumeIndex`` (all
``captions/*.jsonl`` files (D3), latest wins, ``ok``/``flagged`` complete,
never ``.txt`` (AGENTS.md §6.8), prompt-change WARNING), ``build_record``,
``save_txt`` (naming + collision rule, B2) and ``generate_qa_page``
(self-contained, sampling, filters).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath

import PIL.Image
import pytest

from pixel_art_auto_captioner.common import export_utils
from pixel_art_auto_captioner.common.export_utils import (
    ResumeIndex,
    ShardWriter,
    build_record,
    generate_qa_page,
    read_records,
    save_txt,
)
from pixel_art_auto_captioner.common.types import (
    CaptionRecord,
    CaptionResult,
    ImageRecord,
    LoadFailure,
)

SRC = Path(__file__).resolve().parent.parent / "src"
LOGGER = "pixel_art_auto_captioner.common.export_utils"
Z_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(base: CaptionRecord, **changes) -> CaptionRecord:
    """Copy of ``base`` with ``changes`` (rel_path changes also set image_path/source)."""
    if "image_rel_path" in changes:
        rel = Path(changes["image_rel_path"])
        changes["image_rel_path"] = rel
        changes.setdefault("image_path", Path("/data") / rel)
        changes.setdefault("source", rel.parts[0] if len(rel.parts) > 1 else "")
    return dataclasses.replace(base, **changes)


def _line(**fields) -> str:
    """A minimal JSONL record line for resume tests."""
    rec = {
        "image_rel_path": "celeste/0001/000001.jpg",
        "caption_type": "long",
        "status": "ok",
        "timestamp_utc": "2026-11-07T12:00:00Z",
        "category": "game_crop",
        "source": "celeste",
        "prompt": "P",
    }
    rec.update(fields)
    return json.dumps(rec, ensure_ascii=False)


def _write_jsonl(path: Path, lines: list[str], *, final_newline: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + ("\n" if final_newline and lines else "")
    path.write_bytes(text.encode("utf-8"))
    return path


def _shard_file(out: Path, name: str = "shard-000-of-001.jsonl") -> Path:
    return out / "captions" / name


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _result(text: str = "A caption.", finish_reason: str = "stop", tokens: int = 3, error=None):
    return CaptionResult(text=text, finish_reason=finish_reason, num_output_tokens=tokens, error=error)


def _build_kwargs(**overrides) -> dict:
    kwargs = dict(
        caption_type="long",
        category="game_crop",
        prompt="Write a long detailed description for this image.",
        generation_params={"max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": None},
        seed=42,
        attempt=1,
        status="ok",
        flags=[],
        model_name="joycaption-beta-one",
        model_path="fancyfeast/llama-joycaption-beta-one-hf-llava",
        model_revision=None,
        backend="vllm",
        run_id="20261107T120000Z-0-abcd",
        shard=(0, 1),
    )
    kwargs.update(overrides)
    return kwargs


class _Html(HTMLParser):
    """Collects tags, attributes and text of an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def _parse_html(path: Path) -> _Html:
    parser = _Html()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _rows(parser: _Html) -> list[str]:
    """``alt`` (= rel_path) of each embedded image, in page order."""
    return [a["alt"] for t, a in parser.tags if t == "img"]


@pytest.fixture
def qa_setup(tmp_path: Path, sample_caption_record: CaptionRecord):
    """An input dir with real images and a writer factory for an output dir."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    def add(rel: str, caption_type: str = "long", size=(40, 30), make_image=True, **changes):
        if make_image:
            p = input_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                PIL.Image.new("RGB", size, (10, 200, 30)).save(p)
        return _rec(
            sample_caption_record,
            image_rel_path=rel,
            image_path=(input_dir / rel),
            caption_type=caption_type,
            caption_text=changes.pop("caption_text", f"{caption_type} caption of {rel}"),
            **changes,
        )

    def write(records, shard=(0, 1)):
        with ShardWriter(output_dir, shard, False) as w:
            for r in records:
                w.write(r)

    return input_dir, output_dir, add, write


# ===========================================================================
# 1. JSONL format (SPEC §8.3)
# ===========================================================================


class TestJsonlFormat:
    def test_keys_in_dataclass_order(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        raw = _shard_file(tmp_path).read_text(encoding="utf-8").splitlines()[0]
        keys = [k for k, _ in json.loads(raw, object_pairs_hook=list)]
        assert keys == [f.name for f in dataclasses.fields(CaptionRecord)]

    def test_values_round_trip(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        (rec,) = _read_lines(_shard_file(tmp_path))
        expected = dataclasses.asdict(sample_caption_record)
        expected["image_rel_path"] = sample_caption_record.image_rel_path.as_posix()
        expected["image_path"] = sample_caption_record.image_path.as_posix()
        assert rec == expected

    def test_paths_are_posix_strings(self, tmp_path, sample_caption_record):
        record = _rec(
            sample_caption_record,
            image_rel_path=PureWindowsPath("celeste\\0001\\000001.jpg"),
            image_path=PureWindowsPath("F:\\data\\celeste\\0001\\000001.jpg"),
            extra={"nested": PureWindowsPath("a\\b.png")},
        )
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(record)
        (rec,) = _read_lines(_shard_file(tmp_path))
        assert rec["image_rel_path"] == "celeste/0001/000001.jpg"
        assert rec["image_path"] == "F:/data/celeste/0001/000001.jpg"
        assert rec["extra"]["nested"] == "a/b.png"
        assert "\\" not in _shard_file(tmp_path).read_text(encoding="utf-8")

    def test_ensure_ascii_false(self, tmp_path, sample_caption_record):
        text = "Café — ピクセル ✓ \U0001F47E"
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, caption_text=text))
        raw = _shard_file(tmp_path).read_bytes()
        assert text.encode("utf-8") in raw
        assert b"\\u" not in raw
        assert _read_lines(_shard_file(tmp_path))[0]["caption_text"] == text

    @pytest.mark.parametrize("text", ["line1\nline2", "a\u2028b\u2029c", "x\x85y\rz", "tab\tq\"uote"])
    def test_one_physical_line_per_record(self, tmp_path, sample_caption_record, text):
        r1 = _rec(sample_caption_record, caption_text=text)
        r2 = _rec(sample_caption_record, caption_type="short")
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(r1)
            w.write(r2)
        raw = _shard_file(tmp_path).read_bytes()
        assert raw.count(b"\n") == 2 and raw.endswith(b"\n")
        recs = list(read_records([_shard_file(tmp_path)]))
        assert [r["caption_text"] for r in recs] == [text, r2.caption_text]

    def test_no_crlf_written(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        assert b"\r" not in _shard_file(tmp_path).read_bytes()

    def test_unserializable_value_raises_and_writes_nothing(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
            with pytest.raises(TypeError):
                w.write(_rec(sample_caption_record, extra={"bad": {1, 2}}))
            assert w.records_written == 1
        assert len(_read_lines(_shard_file(tmp_path))) == 1


# ===========================================================================
# 2. ShardWriter
# ===========================================================================


class TestShardWriterBasics:
    @pytest.mark.parametrize(
        "shard, name",
        [
            ("0/1", "shard-000-of-001.jsonl"),
            ((3, 12), "shard-003-of-012.jsonl"),
            ([1, 2], "shard-001-of-002.jsonl"),
            ("5/1000", "shard-005-of-1000.jsonl"),
            (" 2 / 4 ", "shard-002-of-004.jsonl"),
        ],
    )
    def test_shard_file_name(self, tmp_path, shard, name):
        w = ShardWriter(tmp_path, shard, False)
        w.close()
        assert w.path == tmp_path / "captions" / name
        assert w.path.is_file()

    @pytest.mark.parametrize(
        "shard", ["1/1", "2/1", "-1/2", "a/b", "1", "", None, (1, 1), (True, 2), (0, 1, 2), (0.0, 1)]
    )
    def test_invalid_shard(self, tmp_path, shard):
        with pytest.raises(ValueError, match="shard"):
            ShardWriter(tmp_path, shard, False)
        assert not (tmp_path / "captions").exists()

    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "10", None])
    def test_invalid_fsync_every(self, tmp_path, value):
        with pytest.raises(ValueError, match="fsync_every"):
            ShardWriter(tmp_path, "0/1", False, fsync_every=value)

    def test_creates_captions_dir_and_attributes(self, tmp_path):
        out = tmp_path / "deep" / "out"
        w = ShardWriter(out, "1/2", True, fsync_every=7)
        try:
            assert (out / "captions").is_dir()
            assert w.output_dir == out and w.shard == (1, 2)
            assert w.write_txt is True and w.fsync_every == 7
            assert w.records_written == 0 and not w.closed
        finally:
            w.close()
        assert w.closed

    def test_each_write_is_flushed(self, tmp_path, sample_caption_record):
        w = ShardWriter(tmp_path, "0/1", False, fsync_every=1000)
        try:
            for i in range(3):
                w.write(_rec(sample_caption_record, seed=i))
                # Visible to a separate reader without close()/fsync.
                assert len(_read_lines(w.path)) == i + 1
            assert w.records_written == 3
        finally:
            w.close()

    def test_appends_to_existing_shard_file(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=1))
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=2))
        assert [r["seed"] for r in _read_lines(_shard_file(tmp_path))] == [1, 2]

    def test_other_shard_files_untouched(self, tmp_path, sample_caption_record):
        other = _write_jsonl(_shard_file(tmp_path, "shard-001-of-002.jsonl"), [_line()])
        before = other.read_bytes()
        with ShardWriter(tmp_path, "0/2", False) as w:
            w.write(sample_caption_record)
        assert other.read_bytes() == before
        assert len(_read_lines(_shard_file(tmp_path, "shard-000-of-002.jsonl"))) == 1

    def test_write_after_close_raises(self, tmp_path, sample_caption_record):
        w = ShardWriter(tmp_path, "0/1", False)
        w.close()
        with pytest.raises(ValueError, match="closed"):
            w.write(sample_caption_record)

    def test_close_is_idempotent(self, tmp_path):
        w = ShardWriter(tmp_path, "0/1", False)
        w.close()
        w.close()
        assert w.closed

    def test_context_manager_closes_on_exception(self, tmp_path, sample_caption_record):
        with pytest.raises(RuntimeError):
            with ShardWriter(tmp_path, "0/1", False) as w:
                w.write(sample_caption_record)
                raise RuntimeError("boom")
        assert w.closed
        assert len(_read_lines(w.path)) == 1


class TestWriterTxtAndOrder:
    def test_jsonl_written_and_flushed_before_txt(self, tmp_path, sample_caption_record, monkeypatch):
        seen: list[int] = []
        real = export_utils.save_txt

        def spy(record, output_dir, collision_set):
            # The record's JSON line is already on disk when the txt is written.
            lines = _read_lines(output_dir / "captions" / "shard-000-of-001.jsonl")
            seen.append(len(lines))
            assert lines[-1]["caption_type"] == record.caption_type
            assert lines[-1]["seed"] == record.seed
            return real(record, output_dir, collision_set)

        monkeypatch.setattr(export_utils, "save_txt", spy)
        with ShardWriter(tmp_path, "0/1", True) as w:
            w.write(_rec(sample_caption_record, seed=1))
            w.write(_rec(sample_caption_record, caption_type="short", seed=2))
        assert seen == [1, 2]

    def test_txt_failure_leaves_jsonl_record(self, tmp_path, sample_caption_record, monkeypatch):
        def boom(*args):
            raise OSError("disk full")

        monkeypatch.setattr(export_utils, "save_txt", boom)
        w = ShardWriter(tmp_path, "0/1", True)
        with pytest.raises(OSError, match="disk full"):
            w.write(sample_caption_record)
        w.close()
        assert len(_read_lines(w.path)) == 1

    def test_txt_written_for_ok_and_flagged(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", True) as w:
            w.write(_rec(sample_caption_record, status="ok", caption_text="okay"))
            w.write(
                _rec(sample_caption_record, caption_type="short", status="flagged",
                     flags=["truncated"], caption_text="cut")
            )
        base = tmp_path / "txt" / "celeste" / "0001"
        assert (base / "000001.long.txt").read_text(encoding="utf-8") == "okay"
        assert (base / "000001.short.txt").read_text(encoding="utf-8") == "cut"

    def test_no_txt_for_failed(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", True) as w:
            w.write(_rec(sample_caption_record, status="failed", flags=["empty"], caption_text=""))
        assert not (tmp_path / "txt").exists()
        assert _read_lines(_shard_file(tmp_path))[0]["status"] == "failed"

    def test_no_txt_when_disabled(self, tmp_path, sample_caption_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        assert not (tmp_path / "txt").exists()

    @pytest.mark.parametrize("as_str", [False, True])
    def test_writer_applies_collision_set(self, tmp_path, sample_caption_record, as_str):
        png, jpg = Path("pack/sprites/sprite.png"), Path("pack/sprites/sprite.jpg")
        collisions = {p.as_posix() for p in (png, jpg)} if as_str else {png, jpg}
        with ShardWriter(tmp_path, "0/1", True, collision_set=collisions) as w:
            w.write(_rec(sample_caption_record, image_rel_path=png, caption_text="from png"))
            w.write(_rec(sample_caption_record, image_rel_path=jpg, caption_text="from jpg"))
        d = tmp_path / "txt" / "pack" / "sprites"
        assert sorted(p.name for p in d.iterdir()) == ["sprite.jpg.long.txt", "sprite.png.long.txt"]
        assert (d / "sprite.png.long.txt").read_text(encoding="utf-8") == "from png"
        assert (d / "sprite.jpg.long.txt").read_text(encoding="utf-8") == "from jpg"


class TestFsyncCadence:
    @pytest.fixture
    def fsyncs(self, monkeypatch):
        calls: list[tuple[int, int]] = []  # (fd inode, file size at fsync time)
        real = os.fsync

        def spy(fd):
            calls.append((os.fstat(fd).st_ino, os.fstat(fd).st_size))
            real(fd)

        monkeypatch.setattr(export_utils.os, "fsync", spy)
        return calls

    def test_every_n_records_and_on_close(self, tmp_path, sample_caption_record, fsyncs):
        w = ShardWriter(tmp_path, "0/1", False, fsync_every=3)
        line_len = len((export_utils._dumps(sample_caption_record) + "\n").encode("utf-8"))
        counts = []
        for _ in range(7):
            w.write(sample_caption_record)
            counts.append(len(fsyncs))
        assert counts == [0, 0, 1, 1, 1, 2, 2]
        w.close()
        assert len(fsyncs) == 3
        ino = w.path.stat().st_ino
        assert all(i == ino for i, _ in fsyncs)
        # Data was flushed before each fsync.
        assert [size for _, size in fsyncs] == [3 * line_len, 6 * line_len, 7 * line_len]

    def test_default_cadence_is_100(self, tmp_path, sample_caption_record, fsyncs):
        w = ShardWriter(tmp_path, "0/1", False)
        assert w.fsync_every == 100
        for _ in range(250):
            w.write(sample_caption_record)
        assert len(fsyncs) == 2
        w.close()
        assert len(fsyncs) == 3

    def test_fsync_every_one(self, tmp_path, sample_caption_record, fsyncs):
        with ShardWriter(tmp_path, "0/1", False, fsync_every=1) as w:
            for i in range(4):
                w.write(sample_caption_record)
                assert len(fsyncs) == i + 1
        assert len(fsyncs) == 5

    def test_close_without_writes_fsyncs_once(self, tmp_path, fsyncs):
        w = ShardWriter(tmp_path, "0/1", False)
        w.close()
        w.close()
        assert len(fsyncs) == 1


# ===========================================================================
# 3. Malformed lines, read_records and tail repair
# ===========================================================================


class TestReadRecords:
    def test_reads_files_in_given_order(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_line(seed=1), _line(seed=2)])
        b = _write_jsonl(tmp_path / "b.jsonl", [_line(seed=3)])
        assert [r["seed"] for r in read_records([b, a])] == [3, 1, 2]

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text(_line(seed=1) + "\n\n   \n" + _line(seed=2) + "\n\n", encoding="utf-8")
        assert [r["seed"] for r in read_records([p])] == [1, 2]

    @pytest.mark.parametrize(
        "bad, newline",
        [
            ('{"image_rel_path": "celeste/0001/000002.jpg", "capt', False),  # crash mid-write
            ('{"image_rel_path": "celeste/0001/000002.jpg", "capt', True),
            ("[1, 2]", True),  # JSON, but not an object
            ("42", False),
            ("not json at all", True),
        ],
    )
    def test_malformed_final_line_skipped_with_warning(self, tmp_path, caplog, bad, newline):
        p = _write_jsonl(tmp_path / "a.jsonl", [_line(seed=1), _line(seed=2), bad], final_newline=newline)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            recs = list(read_records([p]))
        assert [r["seed"] for r in recs] == [1, 2]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "malformed final line" in warnings[0].getMessage()
        assert "a.jsonl:3" in warnings[0].getMessage()

    def test_truncated_utf8_final_line_skipped(self, tmp_path, caplog):
        good = _line(seed=1).encode("utf-8") + b"\n"
        partial = _line(caption_text="ピクセル").encode("utf-8")
        cut = partial[: partial.index("ピ".encode("utf-8")) + 2]  # cut inside a multi-byte char
        p = tmp_path / "a.jsonl"
        p.write_bytes(good + cut)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [r["seed"] for r in read_records([p])] == [1]
        assert "malformed final line" in caplog.text

    def test_malformed_final_line_followed_by_blank_lines(self, tmp_path, caplog):
        p = tmp_path / "a.jsonl"
        p.write_text(_line(seed=1) + "\n{broken\n\n\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [r["seed"] for r in read_records([p])] == [1]
        assert "a.jsonl:2" in caplog.text

    @pytest.mark.parametrize("bad", ["{broken", "[1]", '"str"'])
    def test_malformed_line_elsewhere_is_error(self, tmp_path, bad):
        p = _write_jsonl(tmp_path / "a.jsonl", [_line(seed=1), bad, _line(seed=3)])
        with pytest.raises(ValueError, match=r"a\.jsonl:2: malformed JSONL line"):
            list(read_records([p]))

    def test_malformed_final_line_of_each_file_skipped(self, tmp_path, caplog):
        a = _write_jsonl(tmp_path / "a.jsonl", [_line(seed=1), "{x"], final_newline=False)
        b = _write_jsonl(tmp_path / "b.jsonl", [_line(seed=2), "{y"], final_newline=False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [r["seed"] for r in read_records([a, b])] == [1, 2]
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    def test_only_line_malformed(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", ["{x"], final_newline=False)
        assert list(read_records([p])) == []

    def test_empty_file(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_bytes(b"")
        assert list(read_records([p])) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            list(read_records([tmp_path / "nope.jsonl"]))


class TestTailRepair:
    def test_partial_final_line_removed_before_append(self, tmp_path, sample_caption_record, caplog):
        path = _shard_file(tmp_path)
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=1))
        full = (export_utils._dumps(_rec(sample_caption_record, seed=2)) + "\n").encode("utf-8")
        with open(path, "ab") as fh:
            fh.write(full[: len(full) // 2])  # simulated crash mid-write
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with ShardWriter(tmp_path, "0/1", False) as w:
                w.write(_rec(sample_caption_record, seed=3))
        assert "removed malformed final line" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert [r["seed"] for r in read_records([path])] == [1, 3]
            index = ResumeIndex.from_output_dir(tmp_path)
        assert "malformed" not in caplog.text
        assert index.records_read == 2

    def test_terminated_malformed_final_line_removed(self, tmp_path, sample_caption_record):
        path = _write_jsonl(_shard_file(tmp_path), [_line(seed=1), "{oops"])
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=2))
        assert [r["seed"] for r in read_records([path])] == [1, 2]

    def test_valid_unterminated_final_line_kept(self, tmp_path, sample_caption_record, caplog):
        path = _write_jsonl(_shard_file(tmp_path), [_line(seed=1), _line(seed=2)], final_newline=False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with ShardWriter(tmp_path, "0/1", False) as w:
                w.write(_rec(sample_caption_record, seed=3))
        assert "terminated an unterminated final line" in caplog.text
        assert [r["seed"] for r in read_records([path])] == [1, 2, 3]

    def test_clean_file_untouched(self, tmp_path, caplog):
        path = _write_jsonl(_shard_file(tmp_path), [_line(seed=1)])
        before = path.read_bytes()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ShardWriter(tmp_path, "0/1", False).close()
        assert path.read_bytes() == before
        assert caplog.text == ""

    @pytest.mark.parametrize("content", [b"", b"\n\n", b"  \r\n"])
    def test_empty_or_blank_file(self, tmp_path, sample_caption_record, content):
        path = _shard_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        assert len(list(read_records([path]))) == 1

    def test_only_partial_line(self, tmp_path, sample_caption_record):
        path = _write_jsonl(_shard_file(tmp_path), ['{"image_rel'], final_newline=False)
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(sample_caption_record)
        assert path.read_bytes().startswith(b'{"image_rel_path"')
        assert len(list(read_records([path]))) == 1

    def test_long_partial_line_beyond_read_block(self, tmp_path, sample_caption_record):
        path = _shard_file(tmp_path)
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=1))
        long_line = export_utils._dumps(_rec(sample_caption_record, caption_text="x" * 200_000))
        with open(path, "ab") as fh:
            fh.write(long_line.encode("utf-8")[:150_000])
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, seed=3))
        assert [r["seed"] for r in read_records([path])] == [1, 3]


# ===========================================================================
# 4. ResumeIndex
# ===========================================================================


class TestResumeIndex:
    def test_empty_index(self):
        index = ResumeIndex()
        assert len(index) == 0 and index.num_complete == 0
        assert not index.is_complete(Path("celeste/0001/000001.jpg"), "long")

    def test_missing_output_dir(self, tmp_path):
        index = ResumeIndex.from_output_dir(tmp_path / "nope")
        assert len(index) == 0 and index.files_read == []

    def test_empty_captions_dir(self, tmp_path):
        (tmp_path / "captions").mkdir()
        assert len(ResumeIndex.from_output_dir(tmp_path)) == 0

    @pytest.mark.parametrize("status, complete", [("ok", True), ("flagged", True), ("failed", False)])
    def test_status_rule(self, tmp_path, status, complete):
        _write_jsonl(_shard_file(tmp_path), [_line(status=status)])
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete(Path("celeste/0001/000001.jpg"), "long") is complete
        assert len(index) == 1
        assert index.num_complete == int(complete)

    def test_unknown_status_not_complete(self, tmp_path):
        _write_jsonl(_shard_file(tmp_path), [_line(status="weird")])
        assert not ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")

    def test_keyed_by_caption_type(self, tmp_path):
        _write_jsonl(_shard_file(tmp_path), [_line(caption_type="long")])
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete("celeste/0001/000001.jpg", "long")
        assert not index.is_complete("celeste/0001/000001.jpg", "short")

    def test_keyed_by_full_rel_path_not_stem(self, tmp_path):
        # AGENTS.md §6.4: stems repeat across folders; extension is part of the key (B2).
        _write_jsonl(
            _shard_file(tmp_path),
            [_line(image_rel_path="celeste/0001/000001.jpg"), _line(image_rel_path="pack/sprite.png")],
        )
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete(Path("celeste/0001/000001.jpg"), "long")
        assert not index.is_complete(Path("hades/0001/000001.jpg"), "long")
        assert not index.is_complete(Path("celeste/0002/000001.jpg"), "long")
        assert index.is_complete(Path("pack/sprite.png"), "long")
        assert not index.is_complete(Path("pack/sprite.jpg"), "long")

    def test_accepts_path_str_and_windows_path(self, tmp_path):
        _write_jsonl(_shard_file(tmp_path), [_line()])
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete(Path("celeste/0001/000001.jpg"), "long")
        assert index.is_complete(PurePosixPath("celeste/0001/000001.jpg"), "long")
        assert index.is_complete("celeste/0001/000001.jpg", "long")
        assert index.is_complete(PureWindowsPath("celeste\\0001\\000001.jpg"), "long")

    def test_reads_all_jsonl_files_d3(self, tmp_path):
        cap = tmp_path / "captions"
        _write_jsonl(cap / "shard-000-of-002.jsonl", [_line(image_rel_path="a/1.jpg")])
        _write_jsonl(cap / "shard-001-of-002.jsonl", [_line(image_rel_path="a/2.jpg")])
        _write_jsonl(cap / "shard-003-of-004.jsonl", [_line(image_rel_path="a/3.jpg")])  # other shard count
        _write_jsonl(cap / "machine-b-merged.jsonl", [_line(image_rel_path="a/4.jpg")])  # copied in
        _write_jsonl(cap / "UPPER.JSONL", [_line(image_rel_path="a/5.jpg")])
        index = ResumeIndex.from_output_dir(tmp_path)
        assert all(index.is_complete(f"a/{i}.jpg", "long") for i in range(1, 6))
        assert len(index.files_read) == 5
        assert [p.name for p in index.files_read] == sorted(p.name for p in cap.iterdir())

    def test_ignores_non_jsonl_failures_and_subdirs(self, tmp_path):
        _write_jsonl(tmp_path / "failures" / "shard-000-of-001.jsonl", [_line(image_rel_path="f/1.jpg")])
        _write_jsonl(tmp_path / "captions" / "old" / "x.jsonl", [_line(image_rel_path="f/2.jpg")])
        (tmp_path / "captions" / "notes.txt").write_text(_line(image_rel_path="f/3.jpg"))
        (tmp_path / "captions" / "dir.jsonl").mkdir()
        _write_jsonl(tmp_path / "runs" / "r.jsonl", [_line(image_rel_path="f/4.jpg")])
        index = ResumeIndex.from_output_dir(tmp_path)
        assert len(index) == 0 and index.files_read == []

    def test_never_consults_txt(self, tmp_path, sample_caption_record, monkeypatch):
        # A txt without JSONL is not complete (AGENTS.md §6.8) ...
        txt = tmp_path / "txt" / "celeste" / "0001" / "000001.long.txt"
        txt.parent.mkdir(parents=True)
        txt.write_text("caption", encoding="utf-8")
        assert not ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")
        # ... and JSONL without a txt is complete.
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(_rec(sample_caption_record, caption_type="short"))
        opened: list[str] = []
        real_open = open

        def spy(file, *args, **kwargs):
            opened.append(Path(file).name)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(export_utils, "open", spy, raising=False)
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete("celeste/0001/000001.jpg", "short")
        assert not index.is_complete("celeste/0001/000001.jpg", "long")
        assert opened == ["shard-000-of-001.jsonl"]

    @pytest.mark.parametrize(
        "first, second, complete",
        [
            (("ok", "2026-11-07T12:00:00Z"), ("failed", "2026-11-07T13:00:00Z"), False),
            (("failed", "2026-11-07T12:00:00Z"), ("ok", "2026-11-07T13:00:00Z"), True),
            (("failed", "2026-11-07T13:00:00Z"), ("ok", "2026-11-07T12:00:00Z"), False),
            (("ok", "2026-11-07T13:00:00Z"), ("failed", "2026-11-07T12:00:00Z"), True),
            (("failed", "2026-11-07T12:00:00Z"), ("flagged", "2026-11-08T00:00:00Z"), True),
        ],
    )
    def test_latest_timestamp_wins_in_one_file(self, tmp_path, first, second, complete):
        _write_jsonl(
            _shard_file(tmp_path),
            [_line(status=first[0], timestamp_utc=first[1]), _line(status=second[0], timestamp_utc=second[1])],
        )
        index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete("celeste/0001/000001.jpg", "long") is complete
        assert len(index) == 1 and index.records_read == 2

    def test_latest_wins_across_files_regardless_of_file_order(self, tmp_path):
        cap = tmp_path / "captions"
        # The newer record is in the file that is read first.
        _write_jsonl(cap / "a.jsonl", [_line(status="failed", timestamp_utc="2026-11-08T00:00:00Z")])
        _write_jsonl(cap / "b.jsonl", [_line(status="ok", timestamp_utc="2026-11-07T00:00:00Z")])
        assert not ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")

    def test_equal_timestamps_last_read_wins(self, tmp_path):
        cap = tmp_path / "captions"
        _write_jsonl(cap / "a.jsonl", [_line(status="ok")])
        _write_jsonl(cap / "b.jsonl", [_line(status="failed")])
        assert not ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")
        _write_jsonl(cap / "c.jsonl", [_line(status="ok"), _line(status="failed"), _line(status="flagged")])
        assert ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")

    def test_timestamps_compared_as_times_not_strings(self, tmp_path):
        # 12:00+01:00 == 11:00Z, which is older than 11:30Z.
        _write_jsonl(
            _shard_file(tmp_path),
            [
                _line(status="failed", timestamp_utc="2026-11-07T11:30:00Z"),
                _line(status="ok", timestamp_utc="2026-11-07T12:00:00+01:00"),
            ],
        )
        assert not ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")

    def test_malformed_final_line_skipped(self, tmp_path, caplog):
        _write_jsonl(
            _shard_file(tmp_path),
            [_line(image_rel_path="a/1.jpg"), '{"image_rel_path": "a/2.jpg", "sta'],
            final_newline=False,
        )
        _write_jsonl(_shard_file(tmp_path, "shard-001-of-002.jsonl"), [_line(image_rel_path="a/3.jpg")])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            index = ResumeIndex.from_output_dir(tmp_path)
        assert index.is_complete("a/1.jpg", "long") and index.is_complete("a/3.jpg", "long")
        assert not index.is_complete("a/2.jpg", "long")
        assert "malformed final line" in caplog.text

    def test_malformed_middle_line_is_error(self, tmp_path):
        _write_jsonl(_shard_file(tmp_path), [_line(), "{oops", _line(caption_type="short")])
        with pytest.raises(ValueError, match="shard-000-of-001.jsonl:2"):
            ResumeIndex.from_output_dir(tmp_path)

    @pytest.mark.parametrize(
        "change",
        [
            {"image_rel_path": None},
            {"image_rel_path": ""},
            {"caption_type": 3},
            {"status": None},
            {"timestamp_utc": None},
            {"timestamp_utc": "yesterday"},
        ],
    )
    def test_invalid_record_is_error(self, tmp_path, change):
        _write_jsonl(_shard_file(tmp_path), [_line(), _line(**change), _line(caption_type="tags")])
        with pytest.raises(ValueError, match=":2"):
            ResumeIndex.from_output_dir(tmp_path)

    def test_missing_key_is_error(self, tmp_path):
        rec = json.loads(_line())
        del rec["status"]
        _write_jsonl(_shard_file(tmp_path), [json.dumps(rec)])
        with pytest.raises(ValueError, match="status"):
            ResumeIndex.from_output_dir(tmp_path)

    def test_v10_style_offset_timestamp_accepted(self, tmp_path):
        _write_jsonl(_shard_file(tmp_path), [_line(timestamp_utc="2026-07-16T10:00:00.123456+00:00")])
        assert ResumeIndex.from_output_dir(tmp_path).is_complete("celeste/0001/000001.jpg", "long")

    def test_counts_and_info_log(self, tmp_path, caplog):
        _write_jsonl(
            _shard_file(tmp_path),
            [
                _line(image_rel_path="a/1.jpg"),
                _line(image_rel_path="a/1.jpg", caption_type="short", status="failed"),
                _line(image_rel_path="a/2.jpg", status="flagged"),
                _line(image_rel_path="a/2.jpg", status="failed", timestamp_utc="2026-11-06T00:00:00Z"),
            ],
        )
        with caplog.at_level(logging.INFO, logger=LOGGER):
            index = ResumeIndex.from_output_dir(tmp_path)
        assert index.records_read == 4
        assert len(index) == 3
        assert index.num_complete == 2
        assert "4 records in 1 files, 2 of 3" in caplog.text

    def test_round_trip_with_writer(self, tmp_path, sample_caption_record, sample_image_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            for ct, status in [("long", "ok"), ("short", "flagged"), ("tags", "failed")]:
                w.write(
                    build_record(
                        sample_image_record, _result(),
                        **_build_kwargs(caption_type=ct, status=status,
                                        flags=["error"] if status == "failed" else []),
                    )
                )
        index = ResumeIndex.from_output_dir(tmp_path)
        rel = sample_image_record.rel_path
        assert index.is_complete(rel, "long") and index.is_complete(rel, "short")
        assert not index.is_complete(rel, "tags")


class TestPromptChangeWarning:
    def _warnings(self, caplog):
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_different_prompts_same_group_warns(self, tmp_path, caplog):
        _write_jsonl(
            _shard_file(tmp_path),
            [
                _line(image_rel_path="celeste/1.jpg", prompt="old prompt"),
                _line(image_rel_path="celeste/2.jpg", prompt="new prompt"),
                _line(image_rel_path="celeste/3.jpg", prompt="new prompt"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            index = ResumeIndex.from_output_dir(tmp_path)
        (msg,) = self._warnings(caplog)
        assert "different prompts" in msg and "long/game_crop/celeste: 2 prompts (2, 1 records)" in msg
        assert "--no-resume" in msg
        assert index.is_complete("celeste/1.jpg", "long")  # still complete

    def test_same_category_different_sources_no_warning(self, tmp_path, caplog):
        # game_crop prompts legitimately differ by {source_name}.
        _write_jsonl(
            _shard_file(tmp_path),
            [
                _line(image_rel_path="celeste/1.jpg", source="celeste", prompt="... Celeste."),
                _line(image_rel_path="hades/1.jpg", source="hades", prompt="... Hades."),
                _line(image_rel_path="celeste/1.jpg", caption_type="short", prompt="short P"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ResumeIndex.from_output_dir(tmp_path)
        assert self._warnings(caplog) == []

    def test_failed_and_superseded_records_ignored(self, tmp_path, caplog):
        _write_jsonl(
            _shard_file(tmp_path),
            [
                _line(image_rel_path="celeste/1.jpg", prompt="new"),
                _line(image_rel_path="celeste/2.jpg", prompt="old", status="failed"),
                _line(image_rel_path="celeste/3.jpg", prompt="old", timestamp_utc="2026-11-01T00:00:00Z"),
                _line(image_rel_path="celeste/3.jpg", prompt="new"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ResumeIndex.from_output_dir(tmp_path)
        assert self._warnings(caplog) == []

    def test_warning_lists_at_most_ten_groups(self, tmp_path, caplog):
        lines = []
        for i in range(12):
            lines += [
                _line(image_rel_path=f"s{i:02d}/1.jpg", source=f"s{i:02d}", prompt="a"),
                _line(image_rel_path=f"s{i:02d}/2.jpg", source=f"s{i:02d}", prompt="b"),
            ]
        _write_jsonl(_shard_file(tmp_path), lines)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ResumeIndex.from_output_dir(tmp_path)
        (msg,) = self._warnings(caplog)
        assert "for 12 " in msg and "and 2 more" in msg
        assert "s09" in msg and "s10" not in msg


# ===========================================================================
# 5. build_record
# ===========================================================================


class TestBuildRecord:
    def test_fields_from_image_result_and_kwargs(self, sample_image_record):
        result = _result(text="A sprite.", finish_reason="length", tokens=96)
        kwargs = _build_kwargs(status="flagged", flags=["truncated"], extra={"source_name": "Celeste"})
        rec = build_record(sample_image_record, result, **kwargs)
        assert isinstance(rec, CaptionRecord)
        assert rec.image_rel_path == sample_image_record.rel_path
        assert rec.image_path == sample_image_record.path
        assert rec.source == "celeste"
        assert (rec.image_width, rec.image_height) == (32, 32)
        assert rec.caption_text == "A sprite."
        assert rec.finish_reason == "length" and rec.num_output_tokens == 96
        for key in ("caption_type", "category", "prompt", "seed", "attempt", "status", "model_name",
                    "model_path", "model_revision", "backend", "run_id"):
            assert getattr(rec, key) == kwargs[key]
        assert rec.flags == ["truncated"]
        assert rec.generation_params == kwargs["generation_params"]
        assert rec.extra == {"source_name": "Celeste"}
        assert rec.shard == "0/1"

    def test_timestamp_is_utc_z_second_precision(self, sample_image_record):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        rec = build_record(sample_image_record, _result(), **_build_kwargs())
        after = datetime.now(timezone.utc)
        assert Z_TIMESTAMP.match(rec.timestamp_utc), rec.timestamp_utc
        assert "+00:00" not in rec.timestamp_utc
        stamp = datetime.strptime(rec.timestamp_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert before <= stamp <= after

    def test_written_timestamp_keeps_z(self, tmp_path, sample_image_record):
        with ShardWriter(tmp_path, "0/1", False) as w:
            w.write(build_record(sample_image_record, _result(), **_build_kwargs()))
        assert Z_TIMESTAMP.match(_read_lines(_shard_file(tmp_path))[0]["timestamp_utc"])

    @pytest.mark.parametrize("shard, expected", [((2, 8), "2/8"), ("3/4", "3/4"), (" 1/2", "1/2")])
    def test_shard_formatting(self, sample_image_record, shard, expected):
        assert build_record(sample_image_record, _result(), **_build_kwargs(shard=shard)).shard == expected

    def test_load_failure(self, tmp_path):
        failure = LoadFailure(
            path=tmp_path / "pack" / "bad.jpg", rel_path=Path("pack/bad.jpg"), source="pack",
            error="OSError: truncated",
        )
        result = _result(text="", finish_reason="error", tokens=0, error=failure.error)
        rec = build_record(failure, result, **_build_kwargs(status="failed", flags=["error", "empty"]))
        assert rec.image_rel_path == Path("pack/bad.jpg") and rec.source == "pack"
        assert (rec.image_width, rec.image_height) == (0, 0)
        assert rec.status == "failed" and rec.caption_text == ""
        assert rec.extra == {"error": "OSError: truncated"}

    def test_result_error_does_not_override_extra(self, sample_image_record):
        result = _result(finish_reason="error", error="ValueError: x")
        rec = build_record(
            sample_image_record, result, **_build_kwargs(status="failed", flags=["error"], extra={"error": "mine"})
        )
        assert rec.extra == {"error": "mine"}

    def test_copies_mutable_inputs(self, sample_image_record):
        flags, params, extra = ["truncated"], {"max_new_tokens": 5}, {"k": 1}
        rec = build_record(
            sample_image_record, _result(),
            **_build_kwargs(status="flagged", flags=flags, generation_params=params, extra=extra),
        )
        flags.append("repetitive")
        params["max_new_tokens"] = 99
        extra["k"] = 2
        assert rec.flags == ["truncated"]
        assert rec.generation_params == {"max_new_tokens": 5}
        assert rec.extra == {"k": 1}

    def test_flags_tuple_becomes_list(self, sample_image_record):
        rec = build_record(sample_image_record, _result(), **_build_kwargs(status="flagged", flags=("repetitive",)))
        assert rec.flags == ["repetitive"]

    @pytest.mark.parametrize("status", ["done", "", "OK", None])
    def test_invalid_status(self, sample_image_record, status):
        with pytest.raises(ValueError, match="status"):
            build_record(sample_image_record, _result(), **_build_kwargs(status=status))

    def test_invalid_flag(self, sample_image_record):
        with pytest.raises(ValueError, match="flags"):
            build_record(sample_image_record, _result(), **_build_kwargs(status="flagged", flags=["blurry"]))

    @pytest.mark.parametrize("shard", ["1/1", (2, 1), "x"])
    def test_invalid_shard(self, sample_image_record, shard):
        with pytest.raises(ValueError, match="shard"):
            build_record(sample_image_record, _result(), **_build_kwargs(shard=shard))

    def test_invalid_image_type(self):
        with pytest.raises(TypeError):
            build_record(object(), _result(), **_build_kwargs())


# ===========================================================================
# 6. save_txt (SPEC §8.2)
# ===========================================================================


class TestSaveTxt:
    def test_path_and_content(self, tmp_path, sample_caption_record):
        out = save_txt(sample_caption_record, tmp_path, None)
        assert out == tmp_path / "txt" / "celeste" / "0001" / "000001.long.txt"
        assert out.read_bytes() == sample_caption_record.caption_text.encode("utf-8")

    def test_no_trailing_newline_and_no_translation(self, tmp_path, sample_caption_record):
        text = "first line\nsecond line — ピクセル"
        out = save_txt(_rec(sample_caption_record, caption_text=text), tmp_path, set())
        assert out.read_bytes() == text.encode("utf-8")

    def test_flagged_written(self, tmp_path, sample_caption_record):
        out = save_txt(_rec(sample_caption_record, status="flagged", flags=["repetitive"]), tmp_path, None)
        assert out.is_file()

    @pytest.mark.parametrize("status", ["failed", "unknown"])
    def test_refuses_non_done_status(self, tmp_path, sample_caption_record, status):
        with pytest.raises(ValueError, match="ok/flagged"):
            save_txt(_rec(sample_caption_record, status=status), tmp_path, None)
        assert not (tmp_path / "txt").exists()

    def test_root_level_image(self, tmp_path, sample_caption_record):
        out = save_txt(_rec(sample_caption_record, image_rel_path="loose.png"), tmp_path, None)
        assert out == tmp_path / "txt" / "loose.long.txt"

    def test_caption_types_get_separate_files(self, tmp_path, sample_caption_record):
        a = save_txt(sample_caption_record, tmp_path, None)
        b = save_txt(_rec(sample_caption_record, caption_type="sd_prompt", caption_text="sd"), tmp_path, None)
        assert a.name == "000001.long.txt" and b.name == "000001.sd_prompt.txt"
        assert a.read_text(encoding="utf-8") != b.read_text(encoding="utf-8")

    def test_same_stem_in_different_dirs_no_collision_naming(self, tmp_path, sample_caption_record):
        a = save_txt(_rec(sample_caption_record, image_rel_path="celeste/0001/000001.jpg"), tmp_path, None)
        b = save_txt(_rec(sample_caption_record, image_rel_path="hades/0001/000001.jpg"), tmp_path, None)
        assert a != b and a.name == b.name == "000001.long.txt"

    def test_collision_rule_b2(self, tmp_path, sample_caption_record):
        png = _rec(sample_caption_record, image_rel_path="pack/sprite.png", caption_text="png caption")
        jpg = _rec(sample_caption_record, image_rel_path="pack/sprite.jpg", caption_text="jpg caption")
        other = _rec(sample_caption_record, image_rel_path="pack/hero.png", caption_text="hero")
        collisions = {Path("pack/sprite.png"), Path("pack/sprite.jpg")}
        p1, p2, p3 = (save_txt(r, tmp_path, collisions) for r in (png, jpg, other))
        assert p1.name == "sprite.png.long.txt" and p2.name == "sprite.jpg.long.txt"
        assert p3.name == "hero.long.txt"  # not colliding: normal name
        assert p1.read_text(encoding="utf-8") == "png caption"
        assert p2.read_text(encoding="utf-8") == "jpg caption"

    def test_without_collision_set_same_stem_overwrites(self, tmp_path, sample_caption_record):
        # Documents why the collision set is needed (the v1.0 bug B2).
        png = _rec(sample_caption_record, image_rel_path="pack/sprite.png", caption_text="png")
        jpg = _rec(sample_caption_record, image_rel_path="pack/sprite.jpg", caption_text="jpg")
        assert save_txt(png, tmp_path, None) == save_txt(jpg, tmp_path, None)

    def test_collision_set_of_posix_strings(self, tmp_path, sample_caption_record):
        rec = _rec(sample_caption_record, image_rel_path=PureWindowsPath("pack\\sprite.png"))
        out = save_txt(rec, tmp_path, frozenset({"pack/sprite.png", "pack/sprite.jpg"}))
        assert out == tmp_path / "txt" / "pack" / "sprite.png.long.txt"

    def test_upper_case_extension_kept(self, tmp_path, sample_caption_record):
        rec = _rec(sample_caption_record, image_rel_path="pack/Sprite.PNG")
        assert save_txt(rec, tmp_path, {Path("pack/Sprite.PNG")}).name == "Sprite.PNG.long.txt"
        assert save_txt(rec, tmp_path, None).name == "Sprite.long.txt"

    def test_overwrites_existing(self, tmp_path, sample_caption_record):
        save_txt(_rec(sample_caption_record, caption_text="old and longer"), tmp_path, None)
        out = save_txt(_rec(sample_caption_record, caption_text="new"), tmp_path, None)
        assert out.read_text(encoding="utf-8") == "new"

    @pytest.mark.parametrize(
        "rel", ["/abs/x.jpg", "../x.jpg", "a/../../x.jpg", "C:/x.jpg", "", ".", "a/.."]
    )
    def test_unsafe_rel_path(self, tmp_path, sample_caption_record, rel):
        rec = dataclasses.replace(sample_caption_record, image_rel_path=rel)
        with pytest.raises(ValueError):
            save_txt(rec, tmp_path / "out", None)
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize("caption_type", ["", "a/b", "a\\b", "..", "x:y"])
    def test_unsafe_caption_type(self, tmp_path, sample_caption_record, caption_type):
        with pytest.raises(ValueError, match="caption_type"):
            save_txt(_rec(sample_caption_record, caption_type=caption_type), tmp_path, None)


# ===========================================================================
# 7. QA page (SPEC §10.3)
# ===========================================================================


class TestQaPage:
    def test_self_contained(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add("celeste/0001/000001.jpg"), add("celeste/0001/000001.jpg", "short"),
               add("hades/0001/000001.png", size=(600, 300))])
        page = generate_qa_page(output_dir, input_dir, n=10, name="run1")
        assert page == output_dir / "qa" / "run1.html"
        text = page.read_text(encoding="utf-8")
        parser = _parse_html(page)
        tags = {t for t, _ in parser.tags}
        assert not tags & {"script", "link", "iframe", "object", "embed", "source", "base", "frame"}
        assert "style" in tags  # inline CSS only
        for tag, attrs in parser.tags:
            for key in ("src", "href", "srcset", "action", "data", "poster", "background"):
                if key in attrs:
                    assert attrs[key].startswith("data:image/jpeg;base64,"), (tag, key)
        assert "url(" not in text and "@import" not in text
        assert "http://" not in text and "https://" not in text
        imgs = [a for t, a in parser.tags if t == "img"]
        assert len(imgs) == 2
        for attrs in imgs:
            data = base64.b64decode(attrs["src"].split(",", 1)[1])
            with PIL.Image.open(io.BytesIO(data)) as im:
                assert im.format == "JPEG"
                assert max(im.size) <= 256
        assert text.startswith("<!DOCTYPE html>") and '<meta charset="utf-8">' in text

    def test_all_caption_types_side_by_side_with_details(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        rel = "celeste/0001/000001.jpg"
        write([
            add(rel, "long", caption_text="LONG TEXT", prompt="Long prompt", seed=111, num_output_tokens=7),
            add(rel, "short", caption_text="SHORT TEXT", status="flagged", flags=["truncated", "repetitive"],
                finish_reason="length"),
        ])
        page = generate_qa_page(output_dir, input_dir, n=5, name="q")
        parser = _parse_html(page)
        headers = [d.strip() for d in parser.text]
        assert "long" in headers and "short" in headers
        text = "".join(parser.text)
        assert "LONG TEXT" in text and "SHORT TEXT" in text
        cells = [a for t, a in parser.tags if t == "td" and "cap" in a.get("class", "").split()]
        assert len(cells) == 2
        long_cell = next(a for a in cells if "Long prompt" in a["title"])
        assert "seed: 111" in long_cell["title"] and "tokens: 7" in long_cell["title"]
        classes = [a.get("class", "") for _, a in parser.tags]
        assert "cap cap-flagged" in classes and "badge st-flagged" in classes and "badge st-ok" in classes
        assert sum(1 for c in classes if c == "badge flag") == 2
        assert "truncated" in headers and "repetitive" in headers

    def test_html_escaping(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        evil = '<script>alert("x")</script> & <img src=x onerror=1>'
        write([add("celeste/a.jpg", caption_text=evil, prompt='He said "hi" <b>')])
        page = generate_qa_page(output_dir, input_dir, n=5, name="q")
        parser = _parse_html(page)
        assert "script" not in {t for t, _ in parser.tags}
        assert len([1 for t, _ in parser.tags if t == "img"]) == 1
        assert evil in "".join(parser.text)
        (cell,) = [a for t, a in parser.tags if t == "td" and "cap" in a.get("class", "")]
        assert 'He said "hi" <b>' in cell["title"]

    def test_n_zero_disables(self, qa_setup, caplog):
        input_dir, output_dir, add, write = qa_setup
        write([add("celeste/a.jpg")])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert generate_qa_page(output_dir, input_dir, n=0) is None
        assert not (output_dir / "qa").exists()
        assert "disabled" in caplog.text

    @pytest.mark.parametrize("n", [-1, True, 1.5, "3", None])
    def test_invalid_n(self, qa_setup, n):
        input_dir, output_dir, _, _ = qa_setup
        with pytest.raises(ValueError, match="n must be"):
            generate_qa_page(output_dir, input_dir, n=n)

    def test_samples_n_images(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add(f"celeste/{i:03d}.jpg") for i in range(10)])
        page = generate_qa_page(output_dir, input_dir, n=4, name="q")
        rows = _rows(_parse_html(page))
        assert len(rows) == 4 and len(set(rows)) == 4
        assert "4 of 10 matching images" in page.read_text(encoding="utf-8")

    def test_fewer_images_than_n(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add(f"celeste/{i}.jpg") for i in range(3)])
        assert len(_rows(_parse_html(generate_qa_page(output_dir, input_dir, n=50, name="q")))) == 3

    def test_stratified_by_source(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        records = [add(f"big/{i:03d}.jpg") for i in range(30)]
        records += [add(f"mid/{i:03d}.jpg") for i in range(10)]
        records.append(add("tiny/only.jpg"))
        write(records)
        for seed in range(5):
            rows = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=3, seed=seed, name="q")))
            assert sorted(r.split("/")[0] for r in rows) == ["big", "mid", "tiny"]
        rows = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=9, seed=0, name="q")))
        sources = [r.split("/")[0] for r in rows]
        assert sources.count("tiny") == 1 and sources.count("big") == 4 and sources.count("mid") == 4

    def test_rows_sorted_by_source_then_path(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add(f"{s}/{i}.jpg") for s in ("b", "a", "c") for i in range(3)])
        rows = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=9, name="q")))
        assert rows == sorted(rows)

    def test_deterministic_for_same_seed(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add(f"{s}/{i:02d}.jpg") for s in ("a", "b") for i in range(20)])
        first = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=6, seed=7, name="x")))
        again = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=6, seed=7, name="y")))
        others = {
            tuple(_rows(_parse_html(generate_qa_page(output_dir, input_dir, n=6, seed=s, name="z"))))
            for s in range(8, 14)
        }
        assert first == again
        assert len(others | {tuple(first)}) > 1

    def test_status_filter(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([
            add("s/ok.jpg"),
            add("s/flagged.jpg", status="flagged", flags=["truncated"]),
            add("s/failed.jpg", status="failed", flags=["empty"], caption_text=""),
        ])
        default = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=10, name="a")))
        assert default == ["s/flagged.jpg", "s/ok.jpg"]
        failed = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=10, statuses=["failed"], name="b")))
        assert failed == ["s/failed.jpg"]
        everything = generate_qa_page(output_dir, input_dir, n=10, statuses={"ok", "flagged", "failed"}, name="c")
        assert len(_rows(_parse_html(everything))) == 3

    def test_image_eligible_if_any_type_matches_and_shows_all_types(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([
            add("s/a.jpg", "long", status="failed", flags=["error"], caption_text="FAILED LONG"),
            add("s/a.jpg", "short", caption_text="OK SHORT"),
        ])
        page = generate_qa_page(output_dir, input_dir, n=10, name="q")
        text = "".join(_parse_html(page).text)
        assert "FAILED LONG" in text and "OK SHORT" in text

    def test_source_filter(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add(f"{s}/{i}.jpg") for s in ("celeste", "hades", "prey") for i in range(2)])
        rows = _rows(_parse_html(generate_qa_page(output_dir, input_dir, n=10, sources=["hades", "prey"], name="q")))
        assert sorted({r.split("/")[0] for r in rows}) == ["hades", "prey"] and len(rows) == 4

    def test_latest_record_shown_and_all_shards_read(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add("s/a.jpg", caption_text="OLD TEXT", timestamp_utc="2026-11-01T00:00:00Z")], shard=(0, 2))
        write([add("s/a.jpg", caption_text="NEW TEXT", timestamp_utc="2026-11-02T00:00:00Z"),
               add("s/b.jpg", caption_text="B TEXT")], shard=(1, 2))
        text = "".join(_parse_html(generate_qa_page(output_dir, input_dir, n=10, name="q")).text)
        assert "NEW TEXT" in text and "OLD TEXT" not in text and "B TEXT" in text

    def test_missing_image_placeholder(self, qa_setup, caplog):
        input_dir, output_dir, add, write = qa_setup
        write([add("s/gone.jpg", make_image=False), add("s/here.jpg")])
        (input_dir / "s").mkdir(exist_ok=True)
        (input_dir / "s" / "broken.jpg").write_bytes(b"not an image")
        write([add("s/broken.jpg", make_image=False)])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            page = generate_qa_page(output_dir, input_dir, n=10, name="q")
        parser = _parse_html(page)
        assert len([1 for t, _ in parser.tags if t == "img"]) == 1
        text = "".join(parser.text)
        assert text.count("image not available") == 2
        assert "FileNotFoundError" in text and "UnidentifiedImageError" in text
        assert "cannot embed s/gone.jpg" in caplog.text

    def test_rel_path_escaping_input_dir_not_embedded(self, qa_setup, tmp_path):
        input_dir, output_dir, add, write = qa_setup
        PIL.Image.new("RGB", (8, 8)).save(tmp_path / "secret.jpg")
        write([add("../secret.jpg", make_image=False)])
        text = "".join(_parse_html(generate_qa_page(output_dir, input_dir, n=5, name="q")).text)
        assert "image not available" in text

    def test_no_records(self, qa_setup):
        input_dir, output_dir, _, _ = qa_setup
        page = generate_qa_page(output_dir, input_dir, n=10, name="empty")
        assert page.is_file() and "No records match" in page.read_text(encoding="utf-8")

    def test_default_name(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add("s/a.jpg")])
        page = generate_qa_page(output_dir, input_dir, n=1)
        assert page.parent == output_dir / "qa"
        assert re.fullmatch(r"qa-\d{8}T\d{6}Z\.html", page.name)

    @pytest.mark.parametrize("name", ["", "a/b", "a\\b", "..", "c:x"])
    def test_invalid_name(self, qa_setup, name):
        input_dir, output_dir, _, _ = qa_setup
        with pytest.raises(ValueError, match="name"):
            generate_qa_page(output_dir, input_dir, n=1, name=name)

    @pytest.mark.parametrize("statuses", [[], ["done"], ("ok", "bad")])
    def test_invalid_statuses(self, qa_setup, statuses):
        input_dir, output_dir, _, _ = qa_setup
        with pytest.raises(ValueError, match="statuses"):
            generate_qa_page(output_dir, input_dir, n=1, statuses=statuses)

    def test_missing_input_dir(self, qa_setup, tmp_path):
        _, output_dir, _, _ = qa_setup
        with pytest.raises(FileNotFoundError):
            generate_qa_page(output_dir, tmp_path / "nope", n=1)

    def test_does_not_write_to_input_dir(self, qa_setup):
        input_dir, output_dir, add, write = qa_setup
        write([add("s/a.jpg")])
        before = sorted((p.relative_to(input_dir), p.stat().st_mtime_ns) for p in input_dir.rglob("*"))
        generate_qa_page(output_dir, input_dir, n=5, name="q")
        after = sorted((p.relative_to(input_dir), p.stat().st_mtime_ns) for p in input_dir.rglob("*"))
        assert before == after

    def test_malformed_middle_line_is_error(self, qa_setup):
        input_dir, output_dir, _, _ = qa_setup
        _write_jsonl(_shard_file(output_dir), [_line(), "{bad", _line(caption_type="short")])
        with pytest.raises(ValueError, match="malformed"):
            generate_qa_page(output_dir, input_dir, n=1, name="q")


# ===========================================================================
# 8. Package boundaries and exports
# ===========================================================================


def test_common_exports_new_api():
    import pixel_art_auto_captioner.common as common

    for name in ("ShardWriter", "ResumeIndex", "build_record", "save_txt", "read_records", "generate_qa_page"):
        assert name in common.__all__ and getattr(common, name) is getattr(export_utils, name)
    for old in ("save_jsonl_entry", "save_txt_sidecar"):
        assert old not in common.__all__ and not hasattr(export_utils, old)


def test_import_pulls_in_no_heavy_or_upper_layer_modules():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    code = (
        "import sys, pixel_art_auto_captioner.common.export_utils; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm', 'transformers') "
        "or m.startswith(('pixel_art_auto_captioner.ingestion', 'pixel_art_auto_captioner.captioning', "
        "'pixel_art_auto_captioner.batch'))); "
        "assert not bad, bad"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_module_uses_logging_not_print():
    source = Path(export_utils.__file__).read_text(encoding="utf-8")
    assert "print(" not in source
