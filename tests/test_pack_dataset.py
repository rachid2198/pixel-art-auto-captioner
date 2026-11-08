"""Tests for ``scripts/pack_dataset.py`` and ``scripts/unpack_dataset.sh`` (SPEC v1.1 §10.4, §13).

Covers every ``pack_dataset`` item of SPEC §13: only images are packed
(JPEGs by default, case-insensitive, ``.msgpack`` never opened), the tree is
preserved (member names = POSIX relative paths), size splitting, manifest
checksums, restart skips complete tars, ``out_dir`` inside ``input_dir`` is
refused, and ``input_dir`` is never written to. Read-only-ness is proven
three ways: a full before/after snapshot of ``input_dir`` (every entry's
type, size, mtime, mode and sha256, and every directory listing), a spy on
every file-system mutation / write-mode open during the run, and a run on a
``chmod``-read-only tree. ``unpack_dataset.sh`` is run with bash on a pack
produced by the script (POSIX only).

The script is driven in-process through ``main(argv)``; subprocess tests run
the real script with no ``PYTHONPATH``.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pack_dataset.py"
UNPACK = REPO / "scripts" / "unpack_dataset.sh"
LOGGER = "pixel_art_auto_captioner.scripts.pack_dataset"

_spec = importlib.util.spec_from_file_location("pack_dataset_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
pack_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pack_mod  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(pack_mod)

#: Images packed with the default extensions (.jpg/.jpeg, any case).
JPEGS = [
    "celeste/0001/000001.jpg",
    "celeste/0001/000002.jpg",
    "celeste/0002/000001.JPG",
    "fallout 2/0001/000001.jpeg",
    "café-game/0001/000001.Jpeg",
    "hades/0001/000001.jpg",
    "hades/0001/000002.jpg",
    "loose.jpg",
    "deep/" + "/".join(["a-rather-long-folder-name-for-pax"] * 4) + "/000001.jpg",
]
#: Files that are never packed with the default extensions.
OTHERS = [
    "celeste/0001/000001_clip_h.msgpack",
    "celeste/0001/000001_vae_latent_64.msgpack",
    "celeste/0001/000002_vae_latent_256.msgpack",
    "hades/0001/000003.png",
    "hades/0001/notes.txt",
    "hades/0001/000004.webp",
]

HAS_POSIX_TOOLS = (
    os.name != "nt"
    and all(shutil.which(t) for t in ("bash", "sha256sum", "tar"))
)
requires_bash = pytest.mark.skipif(
    not HAS_POSIX_TOOLS, reason="unpack_dataset.sh needs POSIX bash, sha256sum and tar"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content(rel: str, size: int | None = None) -> bytes:
    rng = random.Random(rel)
    return rng.randbytes(size if size is not None else rng.randint(500, 3000))


def make_dataset(root: Path, files: list[str] | None = None, sizes: dict[str, int] | None = None) -> Path:
    """Create ``root`` with ``JPEGS`` + ``OTHERS`` (or ``files``), deterministic bytes and mtimes."""
    root.mkdir(parents=True, exist_ok=True)
    for i, rel in enumerate(files if files is not None else JPEGS + OTHERS):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_content(rel, (sizes or {}).get(rel)))
        os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))
    return root


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Sibling ``input`` and ``out`` dirs (AGENTS.md §6.2); the dataset is created."""
    input_dir = make_dataset(tmp_path / "root" / "input")
    return input_dir, tmp_path / "root" / "out"


def run(input_dir: Path, out_dir: Path, *extra: str) -> int:
    return pack_mod.main(["--input_dir", str(input_dir), "--out_dir", str(out_dir), *extra])


def snapshot(root: Path) -> dict[str, tuple]:
    """Every entry under ``root`` (and ``root`` itself): type, size, mtime_ns, mode, sha256/listing."""
    snap: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        st = os.lstat(d)
        snap[d.relative_to(root).as_posix() + "/"] = (
            "dir", st.st_mtime_ns, stat.S_IMODE(st.st_mode), tuple(sorted(os.listdir(d))),
        )
        for name in dirnames + filenames:
            p = d / name
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                snap[p.relative_to(root).as_posix()] = ("link", os.readlink(p))
            elif stat.S_ISREG(st.st_mode):
                snap[p.relative_to(root).as_posix()] = (
                    "file", st.st_size, st.st_mtime_ns, stat.S_IMODE(st.st_mode),
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
    return snap


def manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def members(tar_path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(tar_path) as tar:
        return tar.getmembers()


def all_member_names(out_dir: Path) -> list[str]:
    return [m.name for e in manifest(out_dir)["tars"] for m in members(out_dir / e["name"])]


def tar_paths(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("pixel-art-*.tar"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path, root: Path) -> bool:
    if isinstance(path, int):
        return False
    try:
        p = Path(os.fsdecode(path))
    except TypeError:
        return False
    p = p if p.is_absolute() else Path.cwd() / p
    try:
        p = Path(os.path.normpath(p))
        return p == root or root in p.parents
    except (OSError, ValueError):
        return False


@pytest.fixture
def write_guard(monkeypatch: pytest.MonkeyPatch):
    """Record every mutating file-system call and write-mode open under a root."""

    def install(root: Path) -> list[str]:
        root = Path(os.path.normpath(root.resolve()))
        violations: list[str] = []
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

        real_os_open = os.open

        def os_open(path, flags, *a, **k):
            if flags & write_flags and _under(path, root):
                violations.append(f"os.open({path!r}, {flags:#o})")
            return real_os_open(path, flags, *a, **k)

        monkeypatch.setattr(os, "open", os_open)

        for opener_owner, attr in ((builtins, "open"), (io, "open")):
            real = getattr(opener_owner, attr)

            def make(real_open):
                def guarded(file, mode="r", *a, **k):
                    if any(c in mode for c in "wax+") and _under(file, root):
                        violations.append(f"open({file!r}, {mode!r})")
                    return real_open(file, mode, *a, **k)
                return guarded

            monkeypatch.setattr(opener_owner, attr, make(real))

        for name in ("replace", "rename", "unlink", "remove", "rmdir", "mkdir", "makedirs",
                     "utime", "chmod", "symlink", "link", "truncate"):
            real_fn = getattr(os, name, None)
            if real_fn is None:
                continue

            def make_fn(fn_name, fn):
                def guarded(*args, **kwargs):
                    if any(_under(a, root) for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))):
                        violations.append(f"os.{fn_name}{args!r}")
                    return fn(*args, **kwargs)
                return guarded

            monkeypatch.setattr(os, name, make_fn(name, real_fn))
        return violations

    return install


# ===========================================================================
# 1. Command line
# ===========================================================================


class TestCommandLine:
    def test_defaults(self):
        args = pack_mod.parse_args(["--input_dir", "a", "--out_dir", "b"])
        assert args.input_dir == Path("a") and args.out_dir == Path("b")
        assert args.target_size_gb == 1.5
        assert args.extensions == [".jpg", ".jpeg"]
        assert args.include_sources is None and args.exclude_sources == []

    def test_all_flags_parse(self):
        args = pack_mod.parse_args([
            "--input_dir", "a", "--out_dir", "b", "--target_size_gb", "2",
            "--extensions", ".png", ".JPG", "--include_sources", "x", "y",
            "--exclude_sources", "z",
        ])
        assert args.target_size_gb == 2.0
        assert args.extensions == [".png", ".JPG"]
        assert args.include_sources == ["x", "y"] and args.exclude_sources == ["z"]

    def test_help_lists_every_flag(self, capsys):
        assert pack_mod.main(["--help"]) == 0
        out = capsys.readouterr().out
        for flag in ("--input_dir", "--out_dir", "--target_size_gb", "--extensions",
                     "--include_sources", "--exclude_sources"):
            assert flag in out

    @pytest.mark.parametrize("argv", [
        [],
        ["--input_dir", "a"],
        ["--out_dir", "b"],
        ["--input_dir", "a", "--out_dir", "b", "--target_size_gb", "big"],
        ["--input_dir", "a", "--out_dir", "b", "--bogus"],
    ])
    def test_usage_errors_exit_1(self, argv, capsys):
        assert pack_mod.main(argv) == 1
        assert "error" in capsys.readouterr().err

    @pytest.mark.parametrize("size", ["0", "-1", "nan", "inf"])
    def test_bad_target_size_exit_1(self, dirs, size):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", size) == 1
        assert not out_dir.exists()

    def test_missing_input_dir_exit_1(self, tmp_path):
        assert run(tmp_path / "nope", tmp_path / "out") == 1
        assert not (tmp_path / "out").exists()

    def test_input_dir_is_a_file_exit_1(self, tmp_path):
        f = tmp_path / "f.jpg"
        f.write_bytes(b"x")
        assert run(f, tmp_path / "out") == 1

    def test_bad_extension_exit_1(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--extensions", ".") == 1

    def test_logs_to_stderr_in_spec_format_stdout_empty(self, dirs, capfd):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        out, err = capfd.readouterr()
        assert out == ""
        assert "  INFO      pixel_art_auto_captioner.scripts.pack_dataset  Done:" in err

    def test_log_handlers_removed_after_main(self, dirs):
        input_dir, out_dir = dirs
        before = list(logging.getLogger().handlers)
        run(input_dir, out_dir)
        assert logging.getLogger().handlers == before


# ===========================================================================
# 2. Only images are packed (SPEC §10.4, §4.2)
# ===========================================================================


class TestOnlyImages:
    def test_default_packs_only_jpegs_case_insensitive(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        assert sorted(all_member_names(out_dir)) == sorted(JPEGS)

    def test_msgpack_and_other_files_are_never_opened(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        opened: list[str] = []
        real_os_open, real_open = os.open, builtins.open

        def spy_os_open(path, *a, **k):
            opened.append(os.fsdecode(path))
            return real_os_open(path, *a, **k)

        def spy_open(file, *a, **k):
            if not isinstance(file, int):
                opened.append(os.fsdecode(file))
            return real_open(file, *a, **k)

        monkeypatch.setattr(os, "open", spy_os_open)
        monkeypatch.setattr(builtins, "open", spy_open)
        assert run(input_dir, out_dir) == 0
        inside = [p for p in opened if _under(p, input_dir.resolve())]
        assert sorted(Path(p).relative_to(input_dir.resolve()).as_posix() for p in inside) == sorted(JPEGS)
        assert not any(p.endswith((".msgpack", ".png", ".txt", ".webp")) for p in opened)

    def test_custom_extensions(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--extensions", "PNG", ".WEBP") == 0
        assert sorted(all_member_names(out_dir)) == ["hades/0001/000003.png", "hades/0001/000004.webp"]
        assert manifest(out_dir)["extensions"] == [".png", ".webp"]

    def test_include_sources(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--include_sources", "celeste", "fallout 2") == 0
        names = all_member_names(out_dir)
        assert sorted(names) == sorted(r for r in JPEGS if r.split("/")[0] in ("celeste", "fallout 2"))
        assert "loose.jpg" not in names
        assert manifest(out_dir)["include_sources"] == ["celeste", "fallout 2"]

    def test_exclude_sources(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--exclude_sources", "hades", "deep") == 0
        assert sorted(all_member_names(out_dir)) == sorted(
            r for r in JPEGS if r.split("/")[0] not in ("hades", "deep")
        )

    def test_uses_the_dataloader_discovery(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        calls = []
        real = pack_mod.ImageDataLoader.discover

        def spy(self):
            calls.append(self.input_dir)
            return real(self)

        monkeypatch.setattr(pack_mod.ImageDataLoader, "discover", spy)
        assert run(input_dir, out_dir) == 0
        assert calls == [input_dir.resolve()]

    def test_empty_dataset(self, tmp_path):
        input_dir = make_dataset(tmp_path / "in", files=OTHERS)
        out_dir = tmp_path / "out"
        assert run(input_dir, out_dir) == 0
        m = manifest(out_dir)
        assert m["complete"] is True and m["tars"] == [] and m["files_total"] == 0
        assert tar_paths(out_dir) == []
        assert (out_dir / "SHA256SUMS").read_text() == ""


# ===========================================================================
# 3. Tree preserved
# ===========================================================================


class TestTreePreserved:
    def test_extracted_tree_equals_the_images(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000005") == 0
        target = tmp_path / "extracted"
        for tar_path in tar_paths(out_dir):
            with tarfile.open(tar_path) as tar:
                tar.extractall(target, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        got = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        assert got == sorted(JPEGS)
        for rel in JPEGS:
            assert (target / rel).read_bytes() == (input_dir / rel).read_bytes()

    def test_member_headers(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        (tar_path,) = tar_paths(out_dir)
        infos = members(tar_path)
        assert [m.name for m in infos] == sorted(JPEGS)  # sorted POSIX order, files only
        for m in infos:
            st = os.stat(input_dir / m.name)
            assert m.isreg() and m.size == st.st_size and m.mtime == int(st.st_mtime)
            assert m.mode == 0o644 and m.uid == 0 and m.gid == 0 and m.uname == "" and m.gname == ""
            assert not m.name.startswith("/") and ".." not in m.name.split("/") and "\\" not in m.name

    def test_unicode_space_and_long_names_round_trip(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        names = all_member_names(out_dir)
        assert "café-game/0001/000001.Jpeg" in names and "fallout 2/0001/000001.jpeg" in names
        long_name = next(r for r in JPEGS if r.startswith("deep/"))
        assert len(long_name) > 100 and long_name in names

    def test_tar_is_uncompressed_pax(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        (tar_path,) = tar_paths(out_dir)
        assert tar_path.read_bytes()[257:262] == b"ustar"  # plain tar, no gzip/bz2/xz magic
        with tarfile.open(tar_path, "r:") as tar:
            assert tar.getmembers()


# ===========================================================================
# 4. Size splitting
# ===========================================================================


class TestSplitting:
    def test_splits_into_consecutive_tars_of_about_the_target(self, dirs):
        input_dir, out_dir = dirs
        target_gb = 0.000006  # 6000 bytes
        assert run(input_dir, out_dir, "--target_size_gb", str(target_gb)) == 0
        m = manifest(out_dir)
        assert m["target_size_bytes"] == 6000
        assert len(m["tars"]) > 2
        assert [e["name"] for e in m["tars"]] == [f"pixel-art-{i:03d}.tar" for i in range(len(m["tars"]))]
        assert [p.name for p in tar_paths(out_dir)] == [e["name"] for e in m["tars"]]
        assert all_member_names(out_dir) == sorted(JPEGS)  # each file once, in order
        for e in m["tars"]:
            infos = members(out_dir / e["name"])
            files = [pack_mod.PackFile(i.name, i.size, i.mtime) for i in infos]
            content = sum(pack_mod.member_size(f) for f in files) + 1024
            assert content <= 6000 or len(infos) == 1

    def test_tar_size_estimate_is_exact(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        for e in manifest(out_dir)["tars"]:
            infos = members(out_dir / e["name"])
            files = [pack_mod.PackFile(i.name, i.size, i.mtime) for i in infos]
            assert pack_mod.tar_size(files) == (out_dir / e["name"]).stat().st_size == e["bytes"]

    def test_greedy_split_unit(self):
        F = pack_mod.PackFile
        files = [F(f"s/{i}.jpg", 1000, 0) for i in range(6)]  # each member = 512 + 1024 bytes
        groups = pack_mod.split_into_tars(files, 1024 + 2 * 1536)
        assert [len(g) for g in groups] == [2, 2, 2]
        assert [f for g in groups for f in g] == files
        assert pack_mod.split_into_tars(files, 10**9) == [files]
        assert pack_mod.split_into_tars([], 100) == []

    def test_oversized_file_gets_its_own_tar(self):
        F = pack_mod.PackFile
        files = [F("a/1.jpg", 100, 0), F("a/2.jpg", 50_000, 0), F("a/3.jpg", 100, 0)]
        groups = pack_mod.split_into_tars(files, 5000)
        assert [[f.rel for f in g] for g in groups] == [["a/1.jpg"], ["a/2.jpg"], ["a/3.jpg"]]

    def test_one_tar_with_the_default_target(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        assert [p.name for p in tar_paths(out_dir)] == ["pixel-art-000.tar"]
        assert manifest(out_dir)["target_size_bytes"] == 1_500_000_000

    def test_deterministic_output(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        out2 = tmp_path / "root" / "out2"
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        assert run(input_dir, out2, "--target_size_gb", "0.000006") == 0
        assert [e["sha256"] for e in manifest(out_dir)["tars"]] == [
            e["sha256"] for e in manifest(out2)["tars"]
        ]
        assert (out_dir / "SHA256SUMS").read_bytes() == (out2 / "SHA256SUMS").read_bytes()


# ===========================================================================
# 5. Manifest + checksums
# ===========================================================================


class TestManifest:
    def test_per_tar_entries_and_checksums(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        m = manifest(out_dir)
        assert m["complete"] is True and m["manifest_version"] == 1
        assert m["tars_total"] == len(m["tars"])
        for e in m["tars"]:
            path = out_dir / e["name"]
            infos = members(path)
            assert e["sha256"] == sha256_file(path)
            assert e["bytes"] == path.stat().st_size
            assert e["files"] == len(infos)
            assert e["content_bytes"] == sum(i.size for i in infos)
            assert e["first_member"] == infos[0].name and e["last_member"] == infos[-1].name
            files = [pack_mod.PackFile(i.name, i.size, i.mtime) for i in infos]
            assert e["members_sha256"] == pack_mod.members_digest(files)

    def test_totals_per_source(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        m = manifest(out_dir)
        expected: dict[str, dict[str, int]] = {}
        for rel in JPEGS:
            src = rel.split("/")[0] if "/" in rel else ""
            t = expected.setdefault(src, {"files": 0, "bytes": 0})
            t["files"] += 1
            t["bytes"] += (input_dir / rel).stat().st_size
        assert m["sources"] == expected
        assert m["files_total"] == len(JPEGS)
        assert m["bytes_total"] == sum(t["bytes"] for t in expected.values())
        assert m["input_dir"] == input_dir.resolve().as_posix()

    def test_sha256sums_file(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        raw = (out_dir / "SHA256SUMS").read_bytes()
        assert b"\r" not in raw
        lines = raw.decode().splitlines()
        assert lines == [f"{e['sha256']}  {e['name']}" for e in manifest(out_dir)["tars"]]

    @requires_bash
    def test_sha256sum_c_accepts_the_file(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        proc = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out_dir, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_manifest_is_lf_utf8_json(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        raw = (out_dir / "manifest.json").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        assert '"complete": true' in raw.decode("utf-8")  # unpack_dataset.sh greps for this
        assert "café-game" in raw.decode("utf-8")

    def test_no_temp_files_left(self, dirs):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        assert not [p for p in out_dir.iterdir() if p.name.endswith((".partial", ".tmp"))]


# ===========================================================================
# 6. Restart
# ===========================================================================


def _mtimes(out_dir: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in tar_paths(out_dir)}


class TestRestart:
    SIZE = "0.000006"

    def _count_writes(self, monkeypatch) -> list[str]:
        written: list[str] = []
        real = pack_mod.write_tar

        def spy(input_dir, members, path):
            written.append(path.name)
            return real(input_dir, members, path)

        monkeypatch.setattr(pack_mod, "write_tar", spy)
        return written

    def test_second_run_rewrites_nothing(self, dirs, monkeypatch, caplog):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        before, first = _mtimes(out_dir), manifest(out_dir)
        written = self._count_writes(monkeypatch)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert written == []
        assert _mtimes(out_dir) == before
        assert manifest(out_dir)["tars"] == first["tars"]
        assert "not rewritten" in caplog.text

    def test_missing_tar_is_rewritten_only(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        first = manifest(out_dir)
        (out_dir / "pixel-art-001.tar").unlink()
        written = self._count_writes(monkeypatch)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert written == ["pixel-art-001.tar"]
        assert manifest(out_dir)["tars"] == first["tars"]

    def test_truncated_tar_is_rewritten(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        first = manifest(out_dir)
        p = out_dir / "pixel-art-000.tar"
        p.write_bytes(p.read_bytes()[:1000])
        written = self._count_writes(monkeypatch)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert written == ["pixel-art-000.tar"]
        assert sha256_file(p) == first["tars"][0]["sha256"]

    def test_changed_input_file_rewrites_its_tar(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        m = manifest(out_dir)
        target = m["tars"][1]["first_member"]
        os.utime(input_dir / target, (1_800_000_000, 1_800_000_000))
        written = self._count_writes(monkeypatch)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert written == ["pixel-art-001.tar"]
        (info,) = [i for i in members(out_dir / "pixel-art-001.tar") if i.name == target]
        assert info.mtime == 1_800_000_000

    def test_interrupted_run_resumes_to_the_same_pack(self, dirs, tmp_path, monkeypatch):
        input_dir, out_dir = dirs
        reference = tmp_path / "root" / "reference"
        assert run(input_dir, reference, "--target_size_gb", self.SIZE) == 0

        real = pack_mod.write_tar
        calls = {"n": 0}

        def interrupt_second(input_dir_, members_, path):
            calls["n"] += 1
            if calls["n"] == 2:
                # Simulate Ctrl-C in the middle of the second tar.
                partial = path.with_name(path.name + ".partial")
                partial.write_bytes(b"half a tar")
                try:
                    raise KeyboardInterrupt
                finally:
                    partial.unlink()
            return real(input_dir_, members_, path)

        monkeypatch.setattr(pack_mod, "write_tar", interrupt_second)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 2
        m = manifest(out_dir)
        assert m["complete"] is False
        assert [e["name"] for e in m["tars"]] == ["pixel-art-000.tar"]
        assert not (out_dir / "SHA256SUMS").exists()
        assert [p.name for p in tar_paths(out_dir)] == ["pixel-art-000.tar"]

        monkeypatch.setattr(pack_mod, "write_tar", real)
        written = self._count_writes(monkeypatch)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert "pixel-art-000.tar" not in written and written
        assert manifest(out_dir)["tars"] == manifest(reference)["tars"]
        assert (out_dir / "SHA256SUMS").read_bytes() == (reference / "SHA256SUMS").read_bytes()

    def test_real_partial_file_removed_on_interrupt(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        real_open = pack_mod._open_readonly
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return real_open(path)

        monkeypatch.setattr(pack_mod, "_open_readonly", flaky)
        assert run(input_dir, out_dir) == 2
        assert list(out_dir.glob("*.partial")) == [] and tar_paths(out_dir) == []

    def test_read_error_exit_1_and_no_partial(self, dirs, monkeypatch):
        input_dir, out_dir = dirs

        def broken(path):
            raise PermissionError(13, "denied", str(path))

        monkeypatch.setattr(pack_mod, "_open_readonly", broken)
        assert run(input_dir, out_dir) == 1
        assert list(out_dir.glob("*.partial")) == []
        assert not (out_dir / "SHA256SUMS").exists()

    def test_file_shrinking_while_packed_is_an_error(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        real_open = pack_mod._open_readonly

        def short(path):
            fh = real_open(path)
            data = fh.read()[:10]
            fh.close()
            return io.BytesIO(data)

        monkeypatch.setattr(pack_mod, "_open_readonly", short)
        assert run(input_dir, out_dir) == 1
        assert tar_paths(out_dir) == []

    def test_pack_change_drops_stale_sha256sums(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        new = input_dir / "zzz" / "0001" / "000001.jpg"
        new.parent.mkdir(parents=True)
        new.write_bytes(b"new image")

        def boom(*a):
            raise KeyboardInterrupt

        monkeypatch.setattr(pack_mod, "write_tar", boom)
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 2
        assert not (out_dir / "SHA256SUMS").exists()
        assert manifest(out_dir)["complete"] is True  # untouched old manifest...
        # ...but unpack refuses without SHA256SUMS, so a half-updated pack is never unpacked.

    def test_corrupt_manifest_rewrites_everything(self, dirs, monkeypatch, caplog):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        n = len(manifest(out_dir)["tars"])
        (out_dir / "manifest.json").write_text("{not json", encoding="utf-8")
        written = self._count_writes(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        assert len(written) == n
        assert "Ignoring unreadable" in caplog.text
        assert manifest(out_dir)["complete"] is True

    def test_stale_tars_are_warned_about_and_left_out(self, dirs, caplog):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", self.SIZE) == 0
        n = len(manifest(out_dir)["tars"])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert run(input_dir, out_dir) == 0  # default size -> one tar
        m = manifest(out_dir)
        assert [e["name"] for e in m["tars"]] == ["pixel-art-000.tar"]
        assert (out_dir / "SHA256SUMS").read_text().count("\n") == 1
        assert "not part of this pack" in caplog.text and f"pixel-art-{n - 1:03d}.tar" in caplog.text
        assert all_member_names(out_dir) == sorted(JPEGS)


# ===========================================================================
# 7. out_dir placement
# ===========================================================================


class TestOutDirPlacement:
    def test_out_dir_equal_to_input_dir_refused(self, dirs):
        input_dir, _ = dirs
        before = snapshot(input_dir)
        assert run(input_dir, input_dir) == 1
        assert snapshot(input_dir) == before

    @pytest.mark.parametrize("sub", ["tars", "celeste/tars", "new/deeper/tars", "celeste"])
    def test_out_dir_inside_input_dir_refused(self, dirs, sub, caplog):
        input_dir, _ = dirs
        before = snapshot(input_dir)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert run(input_dir, input_dir / sub) == 1
        assert snapshot(input_dir) == before
        assert "must not be input_dir or inside it" in caplog.text

    def test_relative_dotdot_spelling_refused(self, dirs, monkeypatch):
        input_dir, _ = dirs
        monkeypatch.chdir(input_dir / "celeste")
        before = snapshot(input_dir)
        assert pack_mod.main(["--input_dir", "..", "--out_dir", "../hades/../tars"]) == 1
        assert snapshot(input_dir) == before

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_out_dir_symlink_into_input_dir_refused(self, dirs, tmp_path):
        input_dir, _ = dirs
        link = tmp_path / "root" / "sneaky"
        link.symlink_to(input_dir / "celeste", target_is_directory=True)
        before = snapshot(input_dir)
        assert run(input_dir, link / "tars") == 1
        assert snapshot(input_dir) == before

    def test_out_dir_is_a_file_refused(self, dirs):
        input_dir, out_dir = dirs
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        out_dir.write_text("x")
        assert run(input_dir, out_dir) == 1

    def test_input_dir_inside_out_dir_is_allowed_and_untouched(self, dirs):
        input_dir, _ = dirs
        parent = input_dir.parent
        before = snapshot(input_dir)
        assert run(input_dir, parent) == 0
        assert snapshot(input_dir) == before
        assert (parent / "pixel-art-000.tar").is_file()

    def test_check_dirs_unit(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        assert pack_mod.check_dirs(a, tmp_path / "b") == (a.resolve(), (tmp_path / "b").resolve())
        with pytest.raises(ValueError):
            pack_mod.check_dirs(a, a / "x")
        with pytest.raises(ValueError):
            pack_mod.check_dirs(tmp_path / "missing", tmp_path / "b")


# ===========================================================================
# 8. input_dir is never written to
# ===========================================================================


class TestInputReadOnly:
    def test_snapshot_unchanged_across_pack_restart_and_interrupt(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        before = snapshot(input_dir)
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        (out_dir / "pixel-art-000.tar").unlink()

        def boom(*a):
            raise KeyboardInterrupt

        monkeypatch.setattr(pack_mod, "write_tar", boom)
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 2
        assert run(input_dir, input_dir / "x") == 1
        assert snapshot(input_dir) == before

    def test_no_mutating_call_or_write_open_under_input_dir(self, dirs, write_guard):
        input_dir, out_dir = dirs
        violations = write_guard(input_dir)
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        assert run(input_dir, input_dir / "tars") == 1
        assert violations == []

    def test_write_guard_catches_writes(self, dirs, write_guard):
        input_dir, _ = dirs
        violations = write_guard(input_dir)
        (input_dir / "probe.txt").write_text("x")
        os.replace(input_dir / "probe.txt", input_dir / "probe2.txt")
        fd = os.open(input_dir / "probe3", os.O_WRONLY | os.O_CREAT)
        os.close(fd)
        assert len(violations) == 3

    def test_input_files_opened_read_only_without_following_symlinks(self, dirs, monkeypatch):
        input_dir, out_dir = dirs
        flags_seen: list[int] = []
        real = os.open

        def spy(path, flags, *a, **k):
            if _under(path, input_dir.resolve()):
                flags_seen.append(flags)
            return real(path, flags, *a, **k)

        monkeypatch.setattr(os, "open", spy)
        assert run(input_dir, out_dir) == 0
        assert len(flags_seen) == len(JPEGS)
        for flags in flags_seen:
            assert flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) == 0
            if hasattr(os, "O_NOFOLLOW"):
                assert flags & os.O_NOFOLLOW

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_works_on_a_read_only_tree(self, dirs):
        input_dir, out_dir = dirs
        before = snapshot(input_dir)
        paths = sorted(input_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        try:
            for p in paths:
                p.chmod(0o444 if p.is_file() else 0o555)
            input_dir.chmod(0o555)
            if os.geteuid() != 0:  # root ignores permission bits
                with pytest.raises(PermissionError):
                    (input_dir / "probe").write_text("x")
            ro_snapshot = snapshot(input_dir)
            assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
            assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
            assert snapshot(input_dir) == ro_snapshot
            assert all_member_names(out_dir) == sorted(JPEGS)
        finally:
            input_dir.chmod(0o755)
            for p in reversed(paths):
                p.chmod(0o644 if p.is_file() else 0o755)
        assert snapshot(input_dir) == before

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlinks_are_never_followed(self, dirs, tmp_path, caplog):
        input_dir, out_dir = dirs
        outside = tmp_path / "root" / "outside"
        (outside / "sub").mkdir(parents=True)
        (outside / "secret.jpg").write_bytes(b"SECRET-FILE")
        (outside / "sub" / "000001.jpg").write_bytes(b"SECRET-DIR")
        (input_dir / "celeste" / "0001" / "link.jpg").symlink_to(outside / "secret.jpg")
        (input_dir / "linked-source").symlink_to(outside / "sub", target_is_directory=True)
        before = snapshot(input_dir)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert run(input_dir, out_dir) == 0
        names = all_member_names(out_dir)
        assert sorted(names) == sorted(JPEGS)
        blob = b"".join(p.read_bytes() for p in tar_paths(out_dir))
        assert b"SECRET" not in blob
        assert "celeste/0001/link.jpg" in caplog.text
        assert snapshot(input_dir) == before


# ===========================================================================
# 9. unpack_dataset.sh
# ===========================================================================


def unpack(tar_dir: Path, target: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(UNPACK), *([str(tar_dir), str(target)] if not extra else extra)],
        capture_output=True, text=True, timeout=120,
    )


def tree_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".unpacked" not in p.relative_to(root).parts
    }


class TestUnpackScript:
    def test_file_is_lf_bash_with_shebang(self):
        raw = UNPACK.read_bytes()
        assert raw.startswith(b"#!/usr/bin/env bash\n")
        assert b"\r" not in raw
        assert b"set -euo pipefail" in raw

    @requires_bash
    def test_bash_syntax(self):
        proc = subprocess.run(["bash", "-n", str(UNPACK)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    @requires_bash
    def test_pack_then_unpack_round_trip_and_idempotent(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        n = len(manifest(out_dir)["tars"])
        target = tmp_path / "volume" / "pixel-art"
        proc = unpack(out_dir, target)
        assert proc.returncode == 0, proc.stderr
        assert f"{n} tars extracted, 0 already extracted" in proc.stderr
        assert tree_files(target) == {rel: (input_dir / rel).read_bytes() for rel in JPEGS}
        assert sorted(p.name for p in (target / ".unpacked").iterdir()) == [
            f"pixel-art-{i:03d}.tar.sha256" for i in range(n)
        ]
        again = unpack(out_dir, target)
        assert again.returncode == 0, again.stderr
        assert f"0 tars extracted, {n} already extracted" in again.stderr
        assert "verifying" not in again.stderr
        assert tree_files(target) == {rel: (input_dir / rel).read_bytes() for rel in JPEGS}

    @requires_bash
    def test_rerun_after_interruption_extracts_only_the_rest(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        target = tmp_path / "pixel-art"
        assert unpack(out_dir, target).returncode == 0
        (target / ".unpacked" / "pixel-art-001.tar.sha256").unlink()
        victim = manifest(out_dir)["tars"][1]["first_member"]
        (target / victim).write_bytes(b"half-written")
        proc = unpack(out_dir, target)
        assert proc.returncode == 0, proc.stderr
        assert "extracting pixel-art-001.tar" in proc.stderr
        assert "extracting pixel-art-000.tar" not in proc.stderr
        assert (target / victim).read_bytes() == (input_dir / victim).read_bytes()

    @requires_bash
    def test_corrupt_tar_extracts_nothing(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        p = out_dir / "pixel-art-002.tar"
        data = bytearray(p.read_bytes())
        data[600] ^= 0xFF
        p.write_bytes(bytes(data))
        target = tmp_path / "pixel-art"
        proc = unpack(out_dir, target)
        assert proc.returncode == 1
        assert "checksum mismatch for pixel-art-002.tar" in proc.stderr
        assert not target.exists()

    @requires_bash
    def test_missing_tar(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 0
        (out_dir / "pixel-art-001.tar").unlink()
        proc = unpack(out_dir, tmp_path / "t")
        assert proc.returncode == 1 and "missing tar" in proc.stderr
        assert not (tmp_path / "t").exists()

    @requires_bash
    def test_incomplete_pack_refused(self, dirs, tmp_path, monkeypatch):
        input_dir, out_dir = dirs
        calls = {"n": 0}
        real = pack_mod.write_tar

        def stop_second(*a):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
            return real(*a)

        monkeypatch.setattr(pack_mod, "write_tar", stop_second)
        assert run(input_dir, out_dir, "--target_size_gb", "0.000006") == 2
        proc = unpack(out_dir, tmp_path / "t")
        assert proc.returncode == 1 and "does not mark the pack complete" in proc.stderr

    @requires_bash
    def test_missing_sha256sums_refused(self, dirs, tmp_path):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        (out_dir / "SHA256SUMS").unlink()
        proc = unpack(out_dir, tmp_path / "t")
        assert proc.returncode == 1 and "SHA256SUMS" in proc.stderr

    @requires_bash
    @pytest.mark.parametrize("line", [
        "0" * 64 + "  ../evil.tar",
        "0" * 64 + "  /etc/passwd",
        "xyz  pixel-art-000.tar",
        "0" * 64 + " pixel-art-000.tar",
    ])
    def test_malformed_sha256sums_line_refused(self, dirs, tmp_path, line):
        input_dir, out_dir = dirs
        assert run(input_dir, out_dir) == 0
        (out_dir / "SHA256SUMS").write_text(line + "\n")
        proc = unpack(out_dir, tmp_path / "t")
        assert proc.returncode == 1 and "malformed line" in proc.stderr
        assert not (tmp_path / "t").exists()

    @requires_bash
    def test_missing_tar_dir_and_manifest(self, tmp_path):
        assert unpack(tmp_path / "nope", tmp_path / "t").returncode == 1
        (tmp_path / "empty").mkdir()
        proc = unpack(tmp_path / "empty", tmp_path / "t")
        assert proc.returncode == 1 and "manifest.json" in proc.stderr

    @requires_bash
    def test_help_and_usage_errors(self, tmp_path):
        proc = unpack(tmp_path, tmp_path, "--help")
        assert proc.returncode == 0
        assert "/workspace/dataset-tars" in proc.stdout and "/workspace/pixel-art" in proc.stdout
        assert unpack(tmp_path, tmp_path, "--bogus").returncode == 1
        assert unpack(tmp_path, tmp_path, "a", "b", "c").returncode == 1


# ===========================================================================
# 10. Script standards
# ===========================================================================


class TestScriptStandards:
    def test_no_print_lf_and_docstrings(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "print(" not in source
        assert "\r" not in SCRIPT.read_bytes().decode("utf-8")
        tree = ast.parse(source)
        assert ast.get_docstring(tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), node.name
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                assert node.returns is not None, node.name

    def test_stdlib_only_and_no_heavy_imports(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots - {"__future__", "pixel_art_auto_captioner"} <= set(sys.stdlib_module_names)

    def test_subprocess_run_without_pythonpath(self, dirs):
        input_dir, out_dir = dirs
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        code = (
            "import runpy, sys\n"
            f"sys.argv = ['pack_dataset.py', '--input_dir', {str(input_dir)!r}, '--out_dir', {str(out_dir)!r}]\n"
            "try:\n"
            f"    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    code = e.code\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm')"
            " or m.startswith(('pixel_art_auto_captioner.captioning', 'pixel_art_auto_captioner.batch')))\n"
            "assert not bad, bad\n"
            "sys.exit(code)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert manifest(out_dir)["complete"] is True
        assert proc.stdout == ""

    def test_subprocess_config_error_exit_1(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--input_dir", str(tmp_path / "nope"), "--out_dir", str(tmp_path / "o")],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 1
        assert "Configuration error" in proc.stderr
