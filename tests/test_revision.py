"""CPU tests for ``captioning/revision.py`` (SPEC §6.2, R4).

No real ``huggingface_hub`` is used: the cache lookup gets a fake module via
``sys.modules``, and local folders are built under ``tmp_path``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pixel_art_auto_captioner.captioning import revision
from pixel_art_auto_captioner.captioning.revision import _hub_cache_lookup as real_hub_cache_lookup
from pixel_art_auto_captioner.captioning.revision import resolve_revision

HASH = "ebf414ea497a020da0f82df3913e5b6cb8e9663a"
OTHER = "551ffacae92f217ee670de703c764c750d79faab"
REPO_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def snapshot_dir(root: Path, commit: str = HASH) -> Path:
    path = root / "models--fancyfeast--llama-joycaption-beta-one-hf-llava" / "snapshots" / commit
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def local_dir_download(root: Path, first_line: str = HASH) -> Path:
    path = root / "joycaption"
    meta = path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (meta / "config.json.metadata").write_text(f"{first_line}\n{OTHER}\n1795304644.13\n", encoding="utf-8")
    return path


class TestCommitHashFromLibrary:
    def test_hash_is_used(self, tmp_path) -> None:
        assert resolve_revision(str(tmp_path), None, HASH) == HASH

    @pytest.mark.parametrize("value", [None, "", "abc", HASH.upper(), HASH + "0", 123, object()])
    def test_non_hash_is_ignored(self, tmp_path, value) -> None:
        assert resolve_revision(str(tmp_path), None, value) is None  # plain folder

    def test_hash_wins_over_folder(self, tmp_path) -> None:
        path = snapshot_dir(tmp_path, OTHER)
        assert resolve_revision(str(path), None, HASH) == HASH


class TestLocalFolder:
    def test_cache_snapshot(self, tmp_path) -> None:
        assert resolve_revision(str(snapshot_dir(tmp_path)), None) == HASH

    def test_local_dir_download_metadata(self, tmp_path) -> None:
        assert resolve_revision(str(local_dir_download(tmp_path)), None) == HASH

    def test_path_object(self, tmp_path) -> None:
        assert resolve_revision(snapshot_dir(tmp_path), None) == HASH  # type: ignore[arg-type]

    def test_plain_folder_is_none_without_warning(self, tmp_path, caplog) -> None:
        caplog.set_level(logging.WARNING)
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert resolve_revision(str(tmp_path), HASH) is None
        assert caplog.records == []

    def test_snapshots_parent_with_non_hash_name(self, tmp_path) -> None:
        path = tmp_path / "snapshots" / "main"
        path.mkdir(parents=True)
        assert resolve_revision(str(path), None) is None

    @pytest.mark.parametrize("first_line", ["", "not-a-hash", "main"])
    def test_bad_metadata_is_none(self, tmp_path, first_line) -> None:
        assert resolve_revision(str(local_dir_download(tmp_path, first_line)), None) is None

    def test_unreadable_metadata_warns_and_is_none(self, tmp_path, caplog) -> None:
        path = local_dir_download(tmp_path)
        (path / ".cache" / "huggingface" / "download" / "config.json.metadata").write_bytes(b"\xff\xfe\x00")
        caplog.set_level(logging.WARNING)
        assert resolve_revision(str(path), None) is None
        assert "could not resolve" in caplog.text


class TestRepoId:
    def test_uses_hub_lookup_with_revision(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(revision, "_hub_cache_lookup", lambda repo, rev: calls.append((repo, rev)) or HASH)
        assert resolve_revision(REPO_ID, HASH) == HASH
        assert calls == [(REPO_ID, HASH)]

    def test_not_cached_warns(self, caplog) -> None:  # conftest's lookup returns None
        caplog.set_level(logging.WARNING)
        assert resolve_revision(REPO_ID, None) is None
        assert "not found in the Hugging Face cache" in caplog.text

    def test_lookup_error_warns_and_never_raises(self, monkeypatch, caplog) -> None:
        def boom(repo, rev):
            raise ImportError("No module named 'huggingface_hub'")

        monkeypatch.setattr(revision, "_hub_cache_lookup", boom)
        caplog.set_level(logging.WARNING)
        assert resolve_revision(REPO_ID, None) is None
        assert "ImportError" in caplog.text and REPO_ID in caplog.text


class TestHubCacheLookup:
    def fake_hub(self, monkeypatch, result) -> list:
        calls: list = []

        def try_to_load_from_cache(repo_id, filename, revision=None):
            calls.append((repo_id, filename, revision))
            return result

        monkeypatch.setitem(sys.modules, "huggingface_hub",
                            SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache))
        return calls

    def test_snapshot_hash(self, monkeypatch, tmp_path) -> None:
        config = snapshot_dir(tmp_path) / "config.json"
        calls = self.fake_hub(monkeypatch, str(config))
        assert real_hub_cache_lookup(REPO_ID, HASH) == HASH
        assert calls == [(REPO_ID, "config.json", HASH)]

    @pytest.mark.parametrize("result", [None, object()])  # not cached / _CACHED_NO_EXIST
    def test_not_cached(self, monkeypatch, result) -> None:
        self.fake_hub(monkeypatch, result)
        assert real_hub_cache_lookup(REPO_ID, None) is None

    def test_path_outside_snapshots(self, monkeypatch, tmp_path) -> None:
        self.fake_hub(monkeypatch, str(tmp_path / "config.json"))
        assert real_hub_cache_lookup(REPO_ID, None) is None

    def test_missing_library_raises_import_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError):
            real_hub_cache_lookup(REPO_ID, None)


def test_module_imports_no_heavy_library() -> None:
    source = Path(revision.__file__).read_text(encoding="utf-8")
    for name in ("import torch", "import transformers", "import vllm", "import huggingface_hub"):
        assert name not in source
