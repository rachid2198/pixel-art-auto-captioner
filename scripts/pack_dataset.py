"""Pack the dataset's image files into ~1-2 GB tar files for upload (SPEC v1.1 §10.4, Q15).

Thin wiring only (AGENTS.md §4): discovery is the dataloader's
(:meth:`ImageDataLoader.discover`, SPEC §4.2: one walk, no symlink following,
case-insensitive extensions, ``include_sources`` / ``exclude_sources``), so
only image files are packed and the ``.msgpack`` leftovers are skipped
without being opened. By default only JPEGs are packed (Q15).

**Read-only on ``input_dir``.** Image files are opened read-only (with
``O_NOFOLLOW`` where the OS has it); nothing is created, modified, renamed or
deleted under ``input_dir``. ``out_dir`` must not be ``input_dir`` or inside
it (checked on resolved paths, so a symlink cannot sneak it in). Symlinked
files are skipped with a WARNING (never followed out of ``input_dir``).

Output in ``out_dir``:

- ``pixel-art-<NNN>.tar`` (uncompressed, PAX format; ``NNN`` from ``000``),
  each about ``--target_size_gb`` (10^9 bytes) and split between files.
  Member names are the POSIX paths relative to ``input_dir``, so unpacking
  recreates the folder tree. Files are assigned to tars in sorted order,
  greedily, so the same files give the same tars (byte-identical).
- ``manifest.json``: per tar the file count, size in bytes and sha256 (plus
  a digest of its member list); total counts per source; ``complete``.
- ``SHA256SUMS``: ``sha256sum -c`` format, written only when the pack is
  complete; ``scripts/unpack_dataset.sh`` verifies the tars with it.

Restartable: each tar is written to ``<name>.partial`` and renamed when
complete, then the manifest is updated. On a re-run, a tar listed in the
manifest with the same member list (paths, sizes, mtimes) and present on disk
with the recorded size is not rewritten.

Exit codes (same convention as ``run_caption.py``):

- ``0`` done (all tars complete);
- ``1`` usage / config error (bad flag, missing ``input_dir``, ``out_dir``
  inside ``input_dir``) or I/O error;
- ``2`` interrupted (Ctrl-C); complete tars are kept and listed in the
  manifest, the next run continues.

Logging goes to stderr with the SPEC §12 format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import stat
import sys
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, NoReturn

# Until the package is installed (pyproject.toml, step 13), make ``src/``
# importable when the script is run from a checkout. Harmless once installed:
# the import succeeds and ``sys.path`` is left alone.
try:  # pragma: no cover - depends on how the package is installed
    import pixel_art_auto_captioner  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixel_art_auto_captioner import __version__ as _PACKAGE_VERSION  # noqa: E402
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader  # noqa: E402

logger = logging.getLogger("pixel_art_auto_captioner.scripts.pack_dataset")

#: Process exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 2

#: Log line format (SPEC §12).
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

#: Extensions packed by default: JPEGs only (Q15). Matched case-insensitively.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg")

#: Default approximate size of one tar, in GB (10^9 bytes).
DEFAULT_TARGET_SIZE_GB = 1.5

#: File names written into ``out_dir``.
TAR_PREFIX = "pixel-art-"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"
PARTIAL_SUFFIX = ".partial"

#: Version of the ``manifest.json`` layout.
MANIFEST_VERSION = 1

_GB = 10**9
_TAR_FORMAT = tarfile.PAX_FORMAT
_TAR_ENCODING = "utf-8"
_COPY_CHUNK = 1 << 20

# Handlers installed by _configure_logging(); removed when main() returns.
_cli_handlers: list[logging.Handler] = []


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackFile:
    """One image file to pack.

    Attributes:
        rel: Path relative to ``input_dir``, POSIX (= the tar member name).
        size: File size in bytes.
        mtime: Modification time, whole seconds (stored in the tar header).
    """

    rel: str
    size: int
    mtime: int

    @property
    def source(self) -> str:
        """First component of ``rel`` (the source folder), ``""`` for top-level files."""
        head, sep, _ = self.rel.partition("/")
        return head if sep else ""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class _ArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose usage errors exit with code 1 (config error)."""

    def error(self, message: str) -> NoReturn:
        """Print the usage and the message to stderr, then exit with ``EXIT_ERROR``."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SPEC §10.4 command line.

    Returns:
        The configured parser.
    """
    parser = _ArgumentParser(
        prog="pack_dataset.py",
        description="Pack the dataset's image files (JPEGs by default) into ~1-2 GB tar "
        "files plus manifest.json and SHA256SUMS. Never writes inside input_dir.",
        epilog="Exit codes: 0 done; 1 config or I/O error; 2 interrupted (re-run to continue).",
    )
    parser.add_argument("--input_dir", type=Path, required=True, metavar="DIR",
                        help="dataset root (read-only)")
    parser.add_argument("--out_dir", type=Path, required=True, metavar="DIR",
                        help="where the tars and manifest go; must not be inside input_dir")
    parser.add_argument("--target_size_gb", type=float, default=DEFAULT_TARGET_SIZE_GB,
                        metavar="GB", help="approximate size of each tar in GB (10^9 bytes; "
                        f"default {DEFAULT_TARGET_SIZE_GB})")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS),
                        metavar="EXT", help="file extensions to pack, case-insensitive "
                        f"(default: {' '.join(DEFAULT_EXTENSIONS)})")
    parser.add_argument("--include_sources", nargs="+", default=None, metavar="SOURCE",
                        help="only these top-level folders")
    parser.add_argument("--exclude_sources", nargs="+", default=[], metavar="SOURCE",
                        help="skip these top-level folders")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: ``--help`` (code 0) or a usage error (code 1).
    """
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def check_dirs(input_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Resolve both directories and refuse an ``out_dir`` inside ``input_dir``.

    Args:
        input_dir: Dataset root; must be an existing directory.
        out_dir: Output directory (need not exist yet).

    Returns:
        ``(input_dir, out_dir)``, both resolved (symlinks followed).

    Raises:
        ValueError: ``input_dir`` is not a directory, or ``out_dir`` is
            ``input_dir`` or inside it, or ``out_dir`` exists and is not a
            directory.
    """
    if not Path(input_dir).is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")
    src = Path(input_dir).resolve()
    dst = Path(out_dir).resolve()
    if dst == src or src in dst.parents:
        raise ValueError(
            f"out_dir must not be input_dir or inside it (input_dir is read-only): "
            f"out_dir={dst.as_posix()}, input_dir={src.as_posix()}"
        )
    if dst.exists() and not dst.is_dir():
        raise ValueError(f"out_dir exists and is not a directory: {dst.as_posix()}")
    return src, dst


def collect_files(loader: ImageDataLoader) -> list[PackFile]:
    """Stat the discovered image files; skip symlinks and non-regular files.

    Uses ``os.lstat`` (never follows a symlink) and never opens a file.

    Args:
        loader: Dataloader built for ``input_dir`` with the pack filters.

    Returns:
        One :class:`PackFile` per regular image file, in discovery order
        (sorted by POSIX relative path).

    Raises:
        OSError: A file cannot be stat'ed.
    """
    root = loader.input_dir
    files: list[PackFile] = []
    skipped: list[str] = []
    for rel in loader.discover():
        st = os.lstat(root / rel)
        if not stat.S_ISREG(st.st_mode):
            skipped.append(rel.as_posix())
            continue
        files.append(PackFile(rel.as_posix(), st.st_size, int(st.st_mtime)))
    if skipped:
        logger.warning(
            "Skipping %d symlinks / non-regular files (never followed): %s%s",
            len(skipped), skipped[:10], " ..." if len(skipped) > 10 else "",
        )
    return files


def member_size(f: PackFile) -> int:
    """Exact number of bytes ``f`` occupies in a tar written by :func:`write_tar`."""
    header = len(_tarinfo(f).tobuf(_TAR_FORMAT, _TAR_ENCODING, "surrogateescape"))
    return header + math.ceil(f.size / tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE


def tar_size(members: Sequence[PackFile]) -> int:
    """Exact size of the tar file :func:`write_tar` produces for ``members``."""
    body = sum(member_size(f) for f in members) + 2 * tarfile.BLOCKSIZE
    return math.ceil(body / tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def split_into_tars(files: Sequence[PackFile], target_bytes: int) -> list[list[PackFile]]:
    """Split ``files`` (in order) into consecutive groups of about ``target_bytes``.

    Greedy: a new tar starts when adding the next file would exceed
    ``target_bytes``. A file larger than the target gets a tar of its own.
    Deterministic: the same files give the same groups.

    Args:
        files: Files in packing order.
        target_bytes: Approximate tar size in bytes (> 0).

    Returns:
        Non-empty groups covering ``files`` exactly once, in order.
    """
    overhead = 2 * tarfile.BLOCKSIZE  # end-of-archive blocks
    groups: list[list[PackFile]] = []
    current: list[PackFile] = []
    size = overhead
    for f in files:
        m = member_size(f)
        if current and size + m > target_bytes:
            groups.append(current)
            current, size = [], overhead
        current.append(f)
        size += m
    if current:
        groups.append(current)
    return groups


def tar_name(index: int) -> str:
    """``pixel-art-<NNN>.tar`` (zero-padded to at least 3 digits)."""
    return f"{TAR_PREFIX}{index:03d}.tar"


def members_digest(members: Sequence[PackFile]) -> str:
    """sha256 over the member list (path, size, mtime): detects a changed tar content plan."""
    h = hashlib.sha256()
    for f in members:
        h.update(f"{f.rel}\t{f.size}\t{f.mtime}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class _HashingWriter:
    """Write-only file wrapper that hashes and counts everything written."""

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh
        self.sha256 = hashlib.sha256()
        self.bytes = 0

    def write(self, data: bytes) -> int:
        self._fh.write(data)
        self.sha256.update(data)
        self.bytes += len(data)
        return len(data)

    def tell(self) -> int:
        return self.bytes

    def flush(self) -> None:
        self._fh.flush()


def _tarinfo(f: PackFile) -> tarfile.TarInfo:
    """Deterministic header: no owner names, uid/gid 0, mode 0644."""
    info = tarfile.TarInfo(f.rel)
    info.size = f.size
    info.mtime = f.mtime
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _open_readonly(path: Path) -> BinaryIO:
    """Open an input file read-only, refusing to follow a symlink where the OS allows."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    return os.fdopen(fd, "rb")


def write_tar(input_dir: Path, members: Sequence[PackFile], path: Path) -> tuple[str, int]:
    """Write one tar to ``<path>.partial``, fsync it and rename it to ``path``.

    Input files are only opened for reading. If anything fails, the partial
    file is removed and the exception re-raised.

    Args:
        input_dir: Resolved dataset root.
        members: Files to put in the tar, in order.
        path: Final tar path (in ``out_dir``).

    Returns:
        ``(sha256 hex digest, size in bytes)`` of the written tar.

    Raises:
        OSError: An input file cannot be read or changed size while being
            read, or the tar cannot be written.
    """
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    try:
        with open(partial, "wb") as raw:
            writer = _HashingWriter(raw)
            with tarfile.open(fileobj=writer, mode="w", format=_TAR_FORMAT,  # type: ignore[call-overload]
                              encoding=_TAR_ENCODING) as tar:
                for f in members:
                    with _open_readonly(input_dir / f.rel) as src:
                        tar.addfile(_tarinfo(f), src)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(partial, path)
    except BaseException:
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    return writer.sha256.hexdigest(), writer.bytes


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON (LF, UTF-8) to ``<path>.tmp``, fsync, then replace ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_checksums(path: Path, entries: Sequence[dict[str, Any]]) -> None:
    """Write ``SHA256SUMS`` (``<sha256>  <name>`` per line, LF) atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for entry in entries:
            fh.write(f"{entry['sha256']}  {entry['name']}\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Tar entries of an existing manifest by name; ``{}`` if missing or unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = {e["name"]: e for e in data["tars"] if isinstance(e, dict) and "name" in e}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable %s (%s); all tars will be rewritten", path, exc)
        return {}
    return entries


def _is_reusable(entry: dict[str, Any] | None, path: Path, members: Sequence[PackFile]) -> bool:
    """Whether a manifest entry describes ``members`` and its tar is on disk, complete."""
    if entry is None:
        return False
    if entry.get("members_sha256") != members_digest(members) or entry.get("files") != len(members):
        return False
    try:
        return path.is_file() and path.stat().st_size == entry.get("bytes")
    except OSError:
        return False


def _source_totals(files: Sequence[PackFile]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for f in files:
        t = totals.setdefault(f.source, {"files": 0, "bytes": 0})
        t["files"] += 1
        t["bytes"] += f.size
    return dict(sorted(totals.items()))


def pack(
    input_dir: Path,
    out_dir: Path,
    *,
    target_size_gb: float = DEFAULT_TARGET_SIZE_GB,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    include_sources: Sequence[str] | None = None,
    exclude_sources: Sequence[str] = (),
) -> dict[str, Any]:
    """Pack the image files of ``input_dir`` into tars in ``out_dir`` (SPEC §10.4).

    Args:
        input_dir: Dataset root (never written to).
        out_dir: Output directory, created if needed; not inside ``input_dir``.
        target_size_gb: Approximate tar size in GB (10^9 bytes), > 0.
        extensions: Extensions to pack (case-insensitive).
        include_sources: Only these top-level folders (``None`` = all).
        exclude_sources: Top-level folders to skip.

    Returns:
        The final manifest (also written to ``out_dir/manifest.json``).

    Raises:
        ValueError: Invalid arguments (see :func:`check_dirs`, and the
            dataloader's validation of the filters and extensions).
        OSError: Reading an input file or writing the output failed.
    """
    if not (isinstance(target_size_gb, (int, float)) and math.isfinite(target_size_gb)
            and target_size_gb > 0):
        raise ValueError(f"target_size_gb must be a finite number > 0, got {target_size_gb!r}")
    target_bytes = max(1, int(target_size_gb * _GB))
    src, dst = check_dirs(input_dir, out_dir)
    loader = ImageDataLoader({
        "input_dir": src,
        "extensions": list(extensions),
        "recursive": True,
        "include_sources": None if include_sources is None else list(include_sources),
        "exclude_sources": list(exclude_sources),
    })
    files = collect_files(loader)
    groups = split_into_tars(files, target_bytes)
    logger.info(
        "Packing %d files (%.2f GB) from %s into %d tars of about %.2f GB in %s",
        len(files), sum(f.size for f in files) / _GB, src.as_posix(), len(groups),
        target_bytes / _GB, dst.as_posix(),
    )

    dst.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / MANIFEST_NAME
    checksums_path = dst / CHECKSUMS_NAME
    old_entries = _load_manifest(manifest_path)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def manifest(entries: dict[str, dict[str, Any]], complete: bool) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "tool": "pack_dataset.py",
            "package_version": _PACKAGE_VERSION,
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "started_utc": created,
            "complete": complete,
            "input_dir": src.as_posix(),
            "extensions": sorted(loader.extensions),
            "include_sources": loader.include_sources,
            "exclude_sources": loader.exclude_sources,
            "target_size_bytes": target_bytes,
            "tars_total": len(groups),
            "files_total": len(files),
            "bytes_total": sum(f.size for f in files),
            "sources": _source_totals(files),
            "tars": [entries[k] for k in sorted(entries)],
        }

    plan = [(tar_name(i), members) for i, members in enumerate(groups)]
    names = {name for name, _ in plan}
    reusable = {
        name for name, members in plan
        if _is_reusable(old_entries.get(name), dst / name, members)
    }
    if len(reusable) < len(plan) and checksums_path.exists():
        # The pack is about to change: a stale SHA256SUMS must not mark it complete.
        checksums_path.unlink()
    # While working, keep old entries not yet re-checked (a re-run checks them again).
    entries: dict[str, dict[str, Any]] = {
        k: v for k, v in old_entries.items() if k in reusable or k not in names
    }
    written = 0
    for position, (name, members) in enumerate(plan, 1):
        path = dst / name
        if name in reusable:
            logger.info("%s: complete in the manifest, not rewritten (%d files)", name, len(members))
            continue
        sha256, nbytes = write_tar(src, members, path)
        entries[name] = {
            "name": name,
            "files": len(members),
            "bytes": nbytes,
            "sha256": sha256,
            "content_bytes": sum(f.size for f in members),
            "first_member": members[0].rel,
            "last_member": members[-1].rel,
            "members_sha256": members_digest(members),
        }
        written += 1
        _write_json_atomic(manifest_path, manifest(entries, complete=False))
        logger.info(
            "%s: %d files, %.3f GB, sha256 %s (%d/%d)",
            name, len(members), nbytes / _GB, sha256, position, len(plan),
        )

    final_entries = {name: entries[name] for name, _ in plan}
    final = manifest(final_entries, complete=True)
    _write_json_atomic(manifest_path, final)
    _write_checksums(checksums_path, final["tars"])

    stale = sorted(
        p.name for p in dst.iterdir()
        if p.name.startswith(TAR_PREFIX) and p.name.lower().endswith(".tar") and p.name not in names
    )
    if stale:
        logger.warning(
            "%d tar files in %s are not part of this pack (not in the manifest, ignored by "
            "unpack_dataset.sh); delete them before uploading: %s",
            len(stale), dst.as_posix(), stale,
        )
    logger.info(
        "Done: %d tars (%d written, %d already complete), %d files, %d sources; manifest %s",
        len(plan), written, len(plan) - written, len(files), len(final["sources"]),
        manifest_path.as_posix(),
    )
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """INFO logging to stderr with the SPEC §12 format (handlers removed by ``main``)."""
    _remove_cli_handlers()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _cli_handlers.append(handler)


def _remove_cli_handlers() -> None:
    root = logging.getLogger()
    while _cli_handlers:
        handler = _cli_handlers.pop()
        root.removeHandler(handler)
        handler.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code (never raises ``SystemExit``).

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv[1:]``.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_ERROR` or :data:`EXIT_INTERRUPTED`.
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # --help or a usage error
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    _configure_logging()
    try:
        pack(
            args.input_dir,
            args.out_dir,
            target_size_gb=args.target_size_gb,
            extensions=args.extensions,
            include_sources=args.include_sources,
            exclude_sources=args.exclude_sources,
        )
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted; complete tars are kept in the manifest, re-run to continue "
            "(exit code %d)", EXIT_INTERRUPTED,
        )
        return EXIT_INTERRUPTED
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_ERROR
    except OSError as exc:
        logger.error("I/O error: %s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    finally:
        _remove_cli_handlers()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
