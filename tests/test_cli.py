"""Tests for ``scripts/run_caption.py`` (SPEC v1.1 §10.1, §12, §13 CLI item).

Covers argument parsing (every §10.1 flag, shard parsing, usage errors),
override precedence (flag > file > default; unset flags never override),
``--dry_run`` (plan JSON shape on stdout, no model, no torch/vllm import,
nothing written), every exit code, B1 (the CLI never calls ``load()``),
logging setup, and the shipped ``configs/*.json`` files.

The CLI is driven in-process through ``main(argv)`` with a fake
``CaptionModel`` patched in for ``get_model``; subprocess tests run the real
script with no stub on ``PYTHONPATH``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.config import DEFAULT_CONFIG, load_config, validate_config
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_caption.py"
CONFIGS = REPO / "configs"
CATEGORIES = {"game_crop", "sprite_sheet", "ai_pixel_art", "mixed_collection"}
HEAVY = ("torch", "transformers", "vllm", "bitsandbytes")

_spec = importlib.util.spec_from_file_location("run_caption_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

IMAGES = [
    "celeste/0001/000001.png",
    "celeste/0001/000002.png",
    "hades/0001/000001.png",
    "pinterest/0001/000001.jpg",
    "pinterest/0001/000002.jpg",
]


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


def ok_result(request: CaptionRequest) -> CaptionResult:
    return CaptionResult(
        text=f"A caption of {request.image_rel_path.as_posix()} ({request.caption_type}).",
        finish_reason="stop",
        num_output_tokens=10,
    )


class FakeModel(CaptionModel):
    """Unloaded fake backend; ``responder`` gives a result or an exception to raise."""

    model_name = "joycaption-beta-one"
    backend = "hf"

    def __init__(self, responder=None, *, load_error: BaseException | None = None) -> None:
        self.responder = responder or ok_result
        self.load_error = load_error
        self.load_configs: list[Any] = []
        self.unload_calls = 0
        self.batches = 0
        self.loaded = False

    def load(self, config: dict) -> None:
        self.load_configs.append(config)
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        assert self.loaded, "caption_batch before load()"
        self.batches += 1
        results = []
        for request in requests:
            out = self.responder(request)
            if isinstance(out, BaseException):
                raise out
            results.append(out)
        return results

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


class ModelFactory:
    """Stands in for ``get_model``: records the backend, returns one fake."""

    def __init__(self, model: CaptionModel | None = None) -> None:
        self.model = model or FakeModel()
        self.calls: list[str] = []

    def __call__(self, backend: str, *args: Any, **kwargs: Any) -> CaptionModel:
        self.calls.append(backend)
        return self.model


class RecordingRunner:
    """Stands in for ``CaptionRunner``: records its arguments, never touches the model."""

    instances: list[RecordingRunner] = []

    def __init__(self, dataloader: Any, model: Any, config: dict) -> None:
        self.dataloader = dataloader
        self.model = model
        self.config = config
        RecordingRunner.instances.append(self)

    def run(self) -> dict[str, Any]:
        return {"captions_planned": 0, "captions_ok": 0, "captions_flagged": 0}


def make_dataset(root: Path, images: list[str] = IMAGES) -> Path:
    for rel in images:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.new("RGB", (8, 8), (40, 80, 120)).save(path)
        # legacy sidecar, must be ignored by discovery
        path.with_name(path.stem + "_clip_h.msgpack").write_bytes(b"\x00")
    return root


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "data")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def config_file(tmp_path: Path, dataset: Path, out_dir: Path) -> Path:
    cfg = {
        "input": {"input_dir": str(dataset)},
        "model": {"backend": "hf"},
        "prompts": {
            "sources_file": None,
            "sources": {"celeste": {"category": "game_crop", "name": "Celeste"}},
        },
        "output": {"output_dir": str(out_dir), "qa_sample": 0},
        "runtime": {"num_workers": 0, "batch_size": 2, "log_every_s": 3600},
    }
    return write_config(tmp_path / "config.json", cfg)


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> ModelFactory:
    fac = ModelFactory()
    monkeypatch.setattr(cli, "get_model", fac)
    return fac


@pytest.fixture
def recording_runner(monkeypatch: pytest.MonkeyPatch) -> type[RecordingRunner]:
    RecordingRunner.instances = []
    monkeypatch.setattr(cli, "CaptionRunner", RecordingRunner)
    return RecordingRunner


@pytest.fixture(autouse=True)
def _no_cli_handlers_left() -> Any:
    yield
    assert cli._cli_handlers == [], "main() must remove its log handlers"


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def dry_run(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    code, out, err = run_cli(capsys, "--dry_run", *argv)
    assert code == 0, err
    return json.loads(out)


def tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def subprocess_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_all_defaults_none(self) -> None:
        args = cli.parse_args([])
        for name in ("config", "input_dir", "output_dir", "backend", "model_path", "revision",
                     "caption_types", "shard", "max_images", "batch_size", "seed", "write_txt",
                     "resume", "gpu_hourly_cost", "log_level"):
            assert getattr(args, name) is None, name
        assert args.dry_run is False
        assert cli.build_overrides(args) == {}

    def test_every_flag_parsed_with_its_type(self) -> None:
        args = cli.parse_args([
            "--config", "c.json", "--input_dir", "in", "--output_dir", "out",
            "--backend", "hf", "--model_path", "./Models", "--revision", "abc123",
            "--caption_types", "long", "tags", "--shard", "2/8", "--max_images", "20",
            "--batch_size", "4", "--seed", "7", "--write_txt", "--no-resume",
            "--gpu_hourly_cost", "0.69", "--dry_run", "--log_level", "DEBUG",
        ])
        assert args.config == Path("c.json")
        assert (args.input_dir, args.output_dir) == ("in", "out")
        assert (args.backend, args.model_path, args.revision) == ("hf", "./Models", "abc123")
        assert args.caption_types == ["long", "tags"]
        assert args.shard == "2/8"
        assert (args.max_images, args.batch_size, args.seed) == (20, 4, 7)
        assert args.write_txt is True and args.resume is False
        assert args.gpu_hourly_cost == pytest.approx(0.69)
        assert args.dry_run is True and args.log_level == "DEBUG"

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [([], None), (["--write_txt"], True), (["--no-write_txt"], False),
         (["--write_txt", "--no-write_txt"], False), (["--no-write_txt", "--write_txt"], True)],
    )
    def test_write_txt_pair(self, argv: list[str], expected: bool | None) -> None:
        assert cli.parse_args(argv).write_txt is expected

    @pytest.mark.parametrize("shard", ["0/1", "3/4", "15/16", " 1/2", "x", "4/4", "1-4"])
    def test_shard_kept_as_string_for_validate_config(self, shard: str) -> None:
        args = cli.parse_args(["--shard", shard])
        assert args.shard == shard
        assert cli.build_overrides(args) == {"runtime": {"shard": shard}}

    def test_help_lists_every_spec_flag_and_exits_0(self, capsys) -> None:
        code, out, _ = run_cli(capsys, "--help")
        assert code == 0
        for flag in ("--config", "--input_dir", "--output_dir", "--backend", "--model_path",
                     "--revision", "--caption_types", "--shard", "--max_images", "--batch_size",
                     "--seed", "--write_txt", "--no-write_txt", "--no-resume",
                     "--gpu_hourly_cost", "--dry_run", "--log_level"):
            assert flag in out, flag
        assert "{vllm,hf}" in out and "{DEBUG,INFO,WARNING,ERROR}" in out

    @pytest.mark.parametrize(
        "argv",
        [["--bogus"], ["--max_images", "abc"], ["--batch_size", "1.5"], ["--seed", "x"],
         ["--backend", "onnx"], ["--log_level", "TRACE"], ["--gpu_hourly_cost", "cheap"],
         ["--caption_types"], ["positional"]],
    )
    def test_usage_error_exits_1_not_2(self, capsys, argv: list[str]) -> None:
        code, out, err = run_cli(capsys, *argv)
        assert code == cli.EXIT_ERROR == 1
        assert "usage:" in err and "error:" in err
        assert out == ""


class TestBuildOverrides:
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["--input_dir", "d"], {"input": {"input_dir": "d"}}),
            (["--output_dir", "o"], {"output": {"output_dir": "o"}}),
            (["--backend", "vllm"], {"model": {"backend": "vllm"}}),
            (["--model_path", "m"], {"model": {"model_path": "m"}}),
            (["--revision", "r"], {"model": {"revision": "r"}}),
            (["--caption_types", "short"], {"generation": {"enabled_caption_types": ["short"]}}),
            (["--shard", "1/4"], {"runtime": {"shard": "1/4"}}),
            (["--max_images", "5"], {"runtime": {"max_images": 5}}),
            (["--batch_size", "8"], {"runtime": {"batch_size": 8}}),
            (["--seed", "0"], {"runtime": {"seed": 0}}),
            (["--write_txt"], {"output": {"write_txt": True}}),
            (["--no-write_txt"], {"output": {"write_txt": False}}),
            (["--no-resume"], {"output": {"resume": False}}),
            (["--gpu_hourly_cost", "1"], {"runtime": {"gpu_hourly_cost": 1.0}}),
            (["--log_level", "WARNING"], {"runtime": {"log_level": "WARNING"}}),
            (["--dry_run"], {}),
            (["--config", "x.json"], {}),
        ],
    )
    def test_each_flag_maps_to_its_config_key(self, argv, expected) -> None:
        assert cli.build_overrides(cli.parse_args(argv)) == expected

    def test_falsy_values_are_kept(self) -> None:
        overrides = cli.build_overrides(cli.parse_args(["--seed", "0", "--no-write_txt",
                                                        "--max_images", "0"]))
        assert overrides == {"runtime": {"seed": 0, "max_images": 0},
                             "output": {"write_txt": False}}

    def test_sections_merge(self) -> None:
        overrides = cli.build_overrides(cli.parse_args(
            ["--shard", "0/2", "--batch_size", "3", "--backend", "hf", "--revision", "r"]))
        assert overrides == {"runtime": {"shard": "0/2", "batch_size": 3},
                             "model": {"backend": "hf", "revision": "r"}}


# ---------------------------------------------------------------------------
# Precedence (flag > file > default)
# ---------------------------------------------------------------------------


class TestPrecedence:
    def make_file(self, tmp_path: Path, dataset: Path, **extra: Any) -> Path:
        cfg = {
            "input": {"input_dir": str(dataset)},
            "model": {"backend": "hf", "revision": "file-rev"},
            "prompts": {"sources_file": None},
            "output": {"output_dir": str(tmp_path / "file-out"), "write_txt": True},
            "runtime": {"batch_size": 7, "seed": 11, "max_images": 3, "shard": "1/2",
                        "log_level": "WARNING", "num_workers": 0},
        }
        for section, values in extra.items():
            cfg.setdefault(section, {}).update(values)
        return write_config(tmp_path / "prec.json", cfg)

    def test_file_beats_default_and_unset_flags_do_not_override(
        self, capsys, tmp_path, dataset, factory, recording_runner
    ) -> None:
        path = self.make_file(tmp_path, dataset)
        code, _, err = run_cli(capsys, "--config", str(path))
        assert code == 0, err
        cfg = recording_runner.instances[0].config
        assert cfg["runtime"]["batch_size"] == 7            # file
        assert cfg["runtime"]["seed"] == 11
        assert cfg["runtime"]["max_images"] == 3
        assert cfg["runtime"]["shard"] == "1/2"
        assert cfg["model"]["revision"] == "file-rev"
        assert cfg["output"]["write_txt"] is True
        assert cfg["output"]["resume"] is True              # default
        assert cfg["runtime"]["prefetch"] == DEFAULT_CONFIG["runtime"]["prefetch"]
        assert cfg["model"]["model_path"] == DEFAULT_CONFIG["model"]["model_path"]
        assert cfg["generation"]["enabled_caption_types"] == ["long", "short"]

    def test_flags_beat_file(self, capsys, tmp_path, dataset, factory, recording_runner) -> None:
        path = self.make_file(tmp_path, dataset)
        code, _, err = run_cli(
            capsys, "--config", str(path), "--batch_size", "3", "--seed", "0",
            "--max_images", "1", "--shard", "0/3", "--revision", "flag-rev", "--no-write_txt",
            "--no-resume", "--caption_types", "tags", "--output_dir", str(tmp_path / "flag-out"),
            "--model_path", "./Models", "--gpu_hourly_cost", "0.5", "--log_level", "ERROR",
        )
        assert code == 0, err
        runner = recording_runner.instances[0]
        cfg = runner.config
        assert cfg["runtime"]["batch_size"] == 3
        assert cfg["runtime"]["seed"] == 0
        assert cfg["runtime"]["max_images"] == 1
        assert cfg["runtime"]["shard"] == "0/3"
        assert cfg["runtime"]["gpu_hourly_cost"] == 0.5
        assert cfg["runtime"]["log_level"] == "ERROR"
        assert cfg["model"]["revision"] == "flag-rev"
        assert cfg["model"]["model_path"] == "./Models"
        assert cfg["output"]["write_txt"] is False
        assert cfg["output"]["resume"] is False
        assert cfg["output"]["output_dir"] == str(tmp_path / "flag-out")
        assert cfg["generation"]["enabled_caption_types"] == ["tags"]
        # the loader was built from the same merged config
        assert runner.dataloader.shard == (0, 3)
        assert runner.dataloader.max_images == 1

    def test_flag_beats_default_from_repo_root(self, capsys, monkeypatch, dataset, factory,
                                               recording_runner, tmp_path) -> None:
        monkeypatch.chdir(REPO)
        code, _, err = run_cli(capsys, "--input_dir", str(dataset), "--output_dir",
                               str(tmp_path / "o"), "--batch_size", "2", "--backend", "hf")
        assert code == 0, err
        cfg = recording_runner.instances[0].config
        assert cfg["runtime"]["batch_size"] == 2
        assert cfg["model"]["backend"] == "hf"
        assert cfg["runtime"]["seed"] == DEFAULT_CONFIG["runtime"]["seed"]
        assert cfg["prompts"]["sources"]["celeste"] == {"category": "game_crop",
                                                         "name": "Celeste"}
        assert factory.calls == ["hf"]

    def test_dry_run_reflects_precedence(self, capsys, tmp_path, dataset) -> None:
        path = self.make_file(tmp_path, dataset, runtime={"shard": "0/1"})
        report = dry_run(capsys, "--config", str(path))
        assert report["max_images"] == 3 and report["images_planned"] == 3
        report = dry_run(capsys, "--config", str(path), "--max_images", "1")
        assert report["max_images"] == 1 and report["images_planned"] == 1
        assert report["images_deferred"] == len(IMAGES) - 1


# ---------------------------------------------------------------------------
# --dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_plan_json_shape(self, capsys, config_file, dataset, out_dir) -> None:
        code, out, err = run_cli(capsys, "--config", str(config_file), "--dry_run")
        assert code == 0, err
        report = json.loads(out)
        assert list(report) == list(cli.PLAN_KEYS)
        assert report["dry_run"] is True
        assert report["shard"] == "0/1"
        assert report["input_dir"] == dataset.resolve().as_posix()
        assert report["output_dir"] == out_dir.absolute().as_posix()
        assert report["resume"] is True and report["max_images"] is None
        assert report["caption_types"] == ["long", "short"]
        assert report["images_discovered"] == report["images_in_shard"] == 5
        assert report["images_already_complete"] == report["images_deferred"] == 0
        assert report["images_planned"] == 5 and report["captions_planned"] == 10
        assert report["captions_by_type"] == {"long": 5, "short": 5}
        assert report["captions_by_source"] == {
            "celeste": {"long": 2, "short": 2},
            "hades": {"long": 1, "short": 1},
            "pinterest": {"long": 2, "short": 2},
        }
        assert out.endswith("}\n") and out.count("\n{") == 0  # one JSON document only
        assert "Dry run" in err  # log lines go to stderr

    def test_summary_invariant(self, capsys, config_file) -> None:
        for shard in ("0/2", "1/2"):
            r = dry_run(capsys, "--config", str(config_file), "--shard", shard, "--max_images", "1")
            assert r["images_in_shard"] == (r["images_already_complete"] + r["images_deferred"]
                                            + r["images_planned"])

    def test_never_builds_a_model_or_runner(self, capsys, monkeypatch, config_file) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("dry run must not build a model or runner")

        monkeypatch.setattr(cli, "get_model", boom)
        monkeypatch.setattr(cli, "CaptionRunner", boom)
        report = dry_run(capsys, "--config", str(config_file), "--backend", "vllm")
        assert report["images_planned"] == 5

    def test_writes_nothing(self, capsys, tmp_path, config_file, dataset, out_dir) -> None:
        before = tree_snapshot(tmp_path)
        dry_run(capsys, "--config", str(config_file))
        assert not out_dir.exists()
        assert tree_snapshot(tmp_path) == before

    def test_caption_types_and_max_images(self, capsys, config_file) -> None:
        r = dry_run(capsys, "--config", str(config_file), "--caption_types", "tags",
                    "sd_prompt", "long", "--max_images", "2")
        assert r["caption_types"] == ["tags", "sd_prompt", "long"]
        assert r["captions_by_type"] == {"tags": 2, "sd_prompt": 2, "long": 2}
        assert (r["images_planned"], r["images_deferred"]) == (2, 3)
        assert r["captions_by_source"] == {"celeste": {"tags": 2, "sd_prompt": 2, "long": 2}}

    def test_shards_partition_the_dataset(self, capsys, config_file) -> None:
        reports = [dry_run(capsys, "--config", str(config_file), "--shard", f"{i}/3")
                   for i in range(3)]
        assert [r["shard"] for r in reports] == ["0/3", "1/3", "2/3"]
        assert sum(r["images_in_shard"] for r in reports) == 5
        assert all(r["images_discovered"] == 5 for r in reports)

    def test_resume_and_no_resume(self, capsys, config_file, factory, out_dir) -> None:
        code, _, err = run_cli(capsys, "--config", str(config_file), "--caption_types", "long")
        assert code == 0, err
        r = dry_run(capsys, "--config", str(config_file))
        assert r["images_already_complete"] == 0
        assert r["captions_by_type"] == {"long": 0, "short": 5}
        assert r["captions_planned"] == 5
        r = dry_run(capsys, "--config", str(config_file), "--caption_types", "long")
        assert (r["images_already_complete"], r["images_planned"]) == (5, 0)
        assert r["captions_by_source"] == {}
        r = dry_run(capsys, "--config", str(config_file), "--caption_types", "long",
                    "--no-resume")
        assert r["resume"] is False and r["images_planned"] == 5

    def test_malformed_caption_file_exits_1(self, capsys, config_file, out_dir) -> None:
        captions = out_dir / "captions"
        captions.mkdir(parents=True)
        (captions / "shard-000-of-001.jsonl").write_text('{"broken\n{"also": "bad"}\n',
                                                         encoding="utf-8")
        code, out, err = run_cli(capsys, "--config", str(config_file), "--dry_run")
        assert code == 1 and out == ""
        assert "Planning failed" in err

    def test_config_error_exits_1(self, capsys, config_file) -> None:
        code, out, err = run_cli(capsys, "--config", str(config_file), "--dry_run",
                                 "--shard", "3/3")
        assert code == 1 and out == ""
        assert "runtime.shard" in err

    def test_interrupt_exits_2(self, capsys, monkeypatch, config_file) -> None:
        def interrupted(*args: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "make_plan", interrupted)
        code, out, _ = run_cli(capsys, "--config", str(config_file), "--dry_run")
        assert code == 2 and out == ""

    def test_stdout_is_ascii(self, capsys, tmp_path, out_dir) -> None:
        data = make_dataset(tmp_path / "data3", ["jeu-été/0001/000001.png"])
        cfg = write_config(tmp_path / "c3.json", {"prompts": {"sources_file": None}})
        code, out, err = run_cli(capsys, "--dry_run", "--config", str(cfg), "--input_dir",
                                 str(data), "--output_dir", str(out_dir))
        assert code == 0, err
        assert out.isascii()
        assert "jeu-été" in json.loads(out)["captions_by_source"]


class TestSubprocess:
    """The real script, no stub and no PYTHONPATH: exercises the ``src`` path guard."""

    def test_dry_run_script(self, tmp_path, config_file) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(config_file), "--dry_run"],
            capture_output=True, text=True, encoding="utf-8", env=subprocess_env(),
            cwd=tmp_path, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert report["images_planned"] == 5
        assert re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}  INFO {6}\S+  ", proc.stderr)

    def test_dry_run_imports_no_heavy_module(self, tmp_path, config_file) -> None:
        marker = tmp_path / "modules.json"
        code = (
            "import json, runpy, sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            f"        if name.split('.')[0] in {HEAVY!r}:\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            f"sys.argv = [{str(SCRIPT)!r}, '--config', {str(config_file)!r}, '--dry_run']\n"
            "try:\n"
            f"    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    rc = exc.code\n"
            "mods = sorted(m for m in sys.modules if m.split('.')[0] in "
            f"{HEAVY!r} or 'joycaption' in m)\n"
            f"open({str(marker)!r}, 'w').write(json.dumps({{'rc': rc, 'mods': mods}}))\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              encoding="utf-8", env=subprocess_env(), cwd=tmp_path, timeout=120)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(marker.read_text())
        assert result == {"rc": 0, "mods": []}
        assert json.loads(proc.stdout)["captions_planned"] == 10

    def test_config_error_exit_code(self, tmp_path) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--input_dir", str(tmp_path / "missing"), "--dry_run"],
            capture_output=True, text=True, encoding="utf-8", env=subprocess_env(),
            cwd=tmp_path, timeout=120,
        )
        assert proc.returncode == 1
        assert proc.stdout == ""
        assert "ERROR" in proc.stderr and "input.input_dir" in proc.stderr

    def test_usage_error_exit_code(self, tmp_path) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--shard"], capture_output=True,
                              text=True, encoding="utf-8", env=subprocess_env(), cwd=tmp_path,
                              timeout=120)
        assert proc.returncode == 1
        assert "usage:" in proc.stderr


# ---------------------------------------------------------------------------
# Exit codes (SPEC §10.1, §12)
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_constants(self) -> None:
        assert (cli.EXIT_OK, cli.EXIT_ERROR, cli.EXIT_INTERRUPTED, cli.EXIT_NO_SUCCESS) == (
            0, 1, 2, 3)

    @pytest.mark.parametrize(
        ("summary", "code"),
        [
            ({"captions_planned": 0, "captions_ok": 0, "captions_flagged": 0}, 0),
            ({"captions_planned": 4, "captions_ok": 4, "captions_flagged": 0}, 0),
            ({"captions_planned": 4, "captions_ok": 0, "captions_flagged": 1}, 0),
            ({"captions_planned": 4, "captions_ok": 1, "captions_flagged": 0}, 0),
            ({"captions_planned": 4, "captions_ok": 0, "captions_flagged": 0}, 3),
            ({"captions_planned": 1, "captions_ok": 0, "captions_flagged": 0}, 3),
        ],
    )
    def test_exit_code_for(self, summary, code) -> None:
        assert cli.exit_code_for(summary) == code

    def test_success_0(self, capsys, config_file, factory, out_dir) -> None:
        code, out, err = run_cli(capsys, "--config", str(config_file))
        assert code == 0, err
        assert out == ""  # stdout is reserved for --dry_run
        records = [json.loads(line) for f in (out_dir / "captions").glob("*.jsonl")
                   for line in f.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 10 and {r["status"] for r in records} == {"ok"}
        assert "Summary:" in err
        model = factory.model
        assert model.unload_calls == 1 and not model.loaded

    def test_empty_plan_0_without_load(self, capsys, config_file, factory) -> None:
        assert run_cli(capsys, "--config", str(config_file))[0] == 0
        factory.model.load_configs.clear()
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 0, err
        assert factory.model.load_configs == []

    def test_partial_failure_0(self, capsys, config_file, factory) -> None:
        def some_fail(request: CaptionRequest) -> CaptionResult:
            if request.image_rel_path.as_posix().startswith("hades"):
                return CaptionResult("", "error", 0, error="ValueError: x")
            return ok_result(request)

        factory.model.responder = some_fail
        assert run_cli(capsys, "--config", str(config_file))[0] == 0

    def test_flagged_only_0(self, capsys, config_file, factory) -> None:
        factory.model.responder = lambda r: CaptionResult("A cut caption", "length", 512)
        assert run_cli(capsys, "--config", str(config_file))[0] == 0

    def test_zero_succeeded_3(self, capsys, config_file, factory) -> None:
        factory.model.responder = lambda r: CaptionResult("", "error", 0, error="RuntimeError: x")
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 3
        assert "none succeeded" in err

    def test_all_images_undecodable_3(self, capsys, tmp_path, config_file, factory,
                                      dataset) -> None:
        for rel in IMAGES:
            (dataset / rel).write_bytes(b"not an image")
        code, _, _ = run_cli(capsys, "--config", str(config_file))
        assert code == 3

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("weights missing"), ImportError("No module named 'vllm'"),
         OSError("disk"), MemoryError("host OOM")],
    )
    def test_model_load_error_1(self, capsys, config_file, factory, out_dir, error) -> None:
        factory.model.load_error = error
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 1
        assert "Run failed" in err and type(error).__name__ in err
        assert factory.model.unload_calls == 1
        manifest = json.loads(next((out_dir / "runs").glob("*.json")).read_text())
        assert manifest["error"].startswith(type(error).__name__)

    def test_cuda_oom_on_single_request_1(self, capsys, config_file, factory) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        factory.model.responder = lambda r: OutOfMemoryError("CUDA out of memory")
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 1
        assert "OutOfMemoryError" in err
        assert factory.model.unload_calls == 1

    def test_write_error_1(self, capsys, monkeypatch, config_file, factory) -> None:
        from pixel_art_auto_captioner.common import export_utils

        def disk_full(self: Any, record: Any) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(export_utils.ShardWriter, "write", disk_full)
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 1
        assert "No space left" in err
        assert factory.model.unload_calls == 1

    def test_get_model_import_error_1(self, capsys, monkeypatch, config_file) -> None:
        def missing(backend: str) -> Any:
            raise ImportError(f"cannot import the {backend!r} backend module")

        monkeypatch.setattr(cli, "get_model", missing)
        code, _, err = run_cli(capsys, "--config", str(config_file), "--backend", "vllm")
        assert code == 1
        assert "Cannot create the 'vllm' backend" in err

    def test_keyboard_interrupt_2(self, capsys, config_file, factory, out_dir) -> None:
        factory.model.responder = lambda r: KeyboardInterrupt()
        code, out, err = run_cli(capsys, "--config", str(config_file))
        assert code == 2 and out == ""
        assert "interrupted" in err
        manifest = json.loads(next((out_dir / "runs").glob("*.json")).read_text())
        assert manifest["interrupted"] is True
        assert factory.model.unload_calls == 1

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM") or sys.platform == "win32",
                        reason="POSIX SIGTERM delivery")
    def test_sigterm_2_and_handler_restored(self, capsys, config_file, factory,
                                            out_dir) -> None:
        sent: list[bool] = []

        def term_once(request: CaptionRequest) -> CaptionResult:
            if not sent:
                sent.append(True)
                signal.raise_signal(signal.SIGTERM)
            return ok_result(request)

        factory.model.responder = term_once
        previous = signal.getsignal(signal.SIGTERM)
        code, _, _ = run_cli(capsys, "--config", str(config_file))
        assert code == 2
        assert signal.getsignal(signal.SIGTERM) is previous
        manifest = json.loads(next((out_dir / "runs").glob("*.json")).read_text())
        assert manifest["interrupted"] is True

    def test_interrupt_during_config_2(self, capsys, monkeypatch, config_file) -> None:
        def interrupted(*args: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "load_config", interrupted)
        assert run_cli(capsys, "--config", str(config_file))[0] == 2

    def test_unexpected_run_error_1(self, capsys, monkeypatch, config_file, factory) -> None:
        class Exploding(RecordingRunner):
            def run(self) -> dict[str, Any]:
                raise RuntimeError("boom")

        monkeypatch.setattr(cli, "CaptionRunner", Exploding)
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 1 and "RuntimeError: boom" in err


class TestConfigErrors:
    """Config problems exit 1 before any model is created (SPEC §12)."""

    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("no model on a config error")

        monkeypatch.setattr(cli, "get_model", boom)

    def check(self, capsys, *argv: str, expect: str) -> None:
        code, out, err = run_cli(capsys, *argv)
        assert code == 1, err
        assert out == ""
        assert "ERROR" in err and expect in err

    def test_missing_config_file(self, capsys, tmp_path) -> None:
        self.check(capsys, "--config", str(tmp_path / "nope.json"), expect="nope.json")

    def test_invalid_json(self, capsys, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        self.check(capsys, "--config", str(path), expect="not valid JSON")

    def test_non_object_json(self, capsys, tmp_path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        self.check(capsys, "--config", str(path), expect="JSON object")

    def test_missing_input_dir(self, capsys, config_file, tmp_path) -> None:
        self.check(capsys, "--config", str(config_file), "--input_dir",
                   str(tmp_path / "missing"), expect="input.input_dir")

    @pytest.mark.parametrize("shard", ["3/3", "x", "1-4", "-1/2", "0/0"])
    def test_bad_shard(self, capsys, config_file, shard) -> None:
        self.check(capsys, "--config", str(config_file), f"--shard={shard}",
                   expect="runtime.shard")

    def test_unknown_caption_type(self, capsys, config_file) -> None:
        self.check(capsys, "--config", str(config_file), "--caption_types", "poem",
                   expect="enabled_caption_types")

    @pytest.mark.parametrize(("flag", "value", "key"),
                             [("--batch_size", "0", "runtime.batch_size"),
                              ("--max_images", "0", "runtime.max_images"),
                              ("--max_images", "-3", "runtime.max_images")])
    def test_bad_numbers(self, capsys, config_file, flag, value, key) -> None:
        self.check(capsys, "--config", str(config_file), flag, value, expect=key)

    def test_all_problems_listed_together(self, capsys, config_file, tmp_path) -> None:
        code, _, err = run_cli(capsys, "--config", str(config_file), "--shard", "9/2",
                               "--batch_size", "0", "--input_dir", str(tmp_path / "missing"))
        assert code == 1
        assert "3 problem(s)" in err

    def test_missing_sources_file(self, capsys, tmp_path, dataset) -> None:
        cfg = write_config(tmp_path / "c.json", {
            "input": {"input_dir": str(dataset)},
            "prompts": {"sources_file": str(tmp_path / "no-sources.json")}})
        self.check(capsys, "--config", str(cfg), expect="prompts.sources_file")

    def test_bad_log_level_in_file(self, capsys, tmp_path, dataset) -> None:
        cfg = write_config(tmp_path / "c.json", {
            "input": {"input_dir": str(dataset)}, "prompts": {"sources_file": None},
            "runtime": {"log_level": "LOUD"}})
        self.check(capsys, "--config", str(cfg), expect="runtime.log_level")

    def test_bad_loader_value(self, capsys, tmp_path, dataset) -> None:
        cfg = write_config(tmp_path / "c.json", {
            "input": {"input_dir": str(dataset)}, "prompts": {"sources_file": None},
            "runtime": {"prefetch": 0}})
        self.check(capsys, "--config", str(cfg), expect="prefetch")

    @pytest.mark.parametrize(("runtime", "expect"),
                             [({"gpu_hourly_cost": -1}, "gpu_hourly_cost"),
                              ({"seed": "x"}, "runtime.seed")])
    def test_runner_validation(self, capsys, monkeypatch, tmp_path, dataset, runtime,
                               expect) -> None:
        monkeypatch.setattr(cli, "get_model", ModelFactory())
        cfg = write_config(tmp_path / "c.json", {
            "input": {"input_dir": str(dataset)}, "prompts": {"sources_file": None},
            "runtime": runtime})
        self.check(capsys, "--config", str(cfg), expect=expect)

    def test_negative_gpu_cost_flag(self, capsys, monkeypatch, config_file) -> None:
        monkeypatch.setattr(cli, "get_model", ModelFactory())
        self.check(capsys, "--config", str(config_file), "--gpu_hourly_cost", "-2",
                   expect="gpu_hourly_cost")


# ---------------------------------------------------------------------------
# B1: the CLI never loads the model
# ---------------------------------------------------------------------------


class TestB1:
    def test_cli_never_calls_load(self, capsys, config_file, factory, recording_runner) -> None:
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 0, err
        assert factory.calls == ["hf"]
        runner = recording_runner.instances[0]
        assert runner.model is factory.model
        assert factory.model.load_configs == []  # only the runner may call load()
        assert factory.model.unload_calls == 0

    def test_backend_flag_selects_get_model(self, capsys, config_file, factory,
                                            recording_runner) -> None:
        assert run_cli(capsys, "--config", str(config_file), "--backend", "vllm")[0] == 0
        assert factory.calls == ["vllm"]

    def test_runner_loads_once_with_model_section_only(self, capsys, config_file,
                                                       factory) -> None:
        code, _, err = run_cli(capsys, "--config", str(config_file), "--model_path", "./M",
                               "--revision", "r1")
        assert code == 0, err
        (received,) = factory.model.load_configs
        expected = load_config(config_file, {"model": {"model_path": "./M", "revision": "r1"}})
        assert received == expected["model"]
        assert set(received) == {"backend", "model_path", "revision", "dtype", "hf", "vllm"}

    def test_source_has_no_load_call(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert "load" not in calls and "unload" not in calls


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_log_file_written_with_spec_format(self, capsys, tmp_path, config_file,
                                               factory) -> None:
        log_file = tmp_path / "logs" / "nested" / "run.log"
        cfg = json.loads(config_file.read_text())
        cfg["runtime"]["log_file"] = str(log_file)
        write_config(config_file, cfg)
        code, _, err = run_cli(capsys, "--config", str(config_file))
        assert code == 0, err
        text = log_file.read_text(encoding="utf-8")
        assert "Summary:" in text
        assert re.search(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}  INFO {6}"
                         r"pixel_art_auto_captioner\.\S+  ", text, re.M)
        # the file handler was closed: appending again from a second run works
        assert run_cli(capsys, "--config", str(config_file))[0] == 0
        assert log_file.read_text(encoding="utf-8").count("Summary:") == 2

    def test_log_level_flag(self, capsys, config_file, monkeypatch) -> None:
        levels: list[int] = []
        original = cli.configure_logging

        def spy(level: str, log_file: Any = None) -> None:
            original(level, log_file)
            levels.append(logging.getLogger().level)

        monkeypatch.setattr(cli, "configure_logging", spy)
        code, _, err = run_cli(capsys, "--config", str(config_file), "--dry_run",
                               "--log_level", "WARNING")
        assert code == 0
        assert levels == [logging.WARNING, logging.WARNING]
        assert "INFO" not in err

    def test_debug_level(self, capsys, config_file, monkeypatch) -> None:
        seen: list[int] = []
        original = cli.make_plan

        def spy(cfg: dict, loader: Any) -> Any:
            seen.append(logging.getLogger().level)
            logging.getLogger("pixel_art_auto_captioner.test").debug("debug line visible")
            return original(cfg, loader)

        monkeypatch.setattr(cli, "make_plan", spy)
        code, _, err = run_cli(capsys, "--config", str(config_file), "--dry_run",
                               "--log_level", "DEBUG")
        assert code == 0 and seen == [logging.DEBUG]
        assert re.search(r"  DEBUG {5}pixel_art_auto_captioner\.test  debug line visible", err)

    def test_lowercase_level_in_file(self, capsys, tmp_path, dataset) -> None:
        cfg = write_config(tmp_path / "c.json", {
            "input": {"input_dir": str(dataset)}, "prompts": {"sources_file": None},
            "runtime": {"log_level": "warning"}})
        code, _, err = run_cli(capsys, "--config", str(cfg), "--dry_run")
        assert code == 0 and "Dry run" not in err  # INFO suppressed once configured

    def test_configure_logging_replaces_its_handlers(self, tmp_path) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        try:
            cli.configure_logging("INFO")
            cli.configure_logging("DEBUG", tmp_path / "x.log")
            added = [h for h in root.handlers if h not in before]
            assert len(added) == 2 and root.level == logging.DEBUG
            with pytest.raises(ValueError, match="log_level"):
                cli.configure_logging("VERBOSE")
            with pytest.raises(ValueError, match="log_file"):
                cli.configure_logging("INFO", 42)  # type: ignore[arg-type]
        finally:
            cli._remove_cli_handlers()
        assert list(root.handlers) == before


# ---------------------------------------------------------------------------
# Shipped config files
# ---------------------------------------------------------------------------


def read_json(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


class TestShippedConfigs:
    @pytest.fixture(autouse=True)
    def _repo_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO)  # sources_file is relative to the working directory

    @pytest.mark.parametrize("name", ["example_config.json", "local_hf_config.json"])
    def test_loads_and_validates_with_tmp_input_dir(self, name, tmp_path) -> None:
        cfg = load_config(CONFIGS / name, {"input": {"input_dir": str(tmp_path)}})
        validate_config(cfg)
        assert len(cfg["prompts"]["sources"]) == len(read_json("sources.json"))

    @pytest.mark.parametrize("name", ["example_config.json", "local_hf_config.json"])
    def test_only_input_dir_fails_without_override(self, name) -> None:
        cfg = load_config(CONFIGS / name, {})
        cfg["input"]["input_dir"] = str(Path(__file__).resolve().parent / "does-not-exist")
        with pytest.raises(ValueError, match=r"1 problem\(s\)"):
            validate_config(cfg)

    @pytest.mark.parametrize("name", ["example_config.json", "local_hf_config.json"])
    def test_dry_run_with_shipped_config(self, name, capsys, dataset, tmp_path) -> None:
        r = dry_run(capsys, "--config", str(CONFIGS / name), "--input_dir", str(dataset),
                    "--output_dir", str(tmp_path / "o"))
        assert r["images_planned"] == 5

    def test_example_is_spec_9_2_with_runpod_paths(self) -> None:
        example = read_json("example_config.json")
        expected = json.loads(json.dumps(DEFAULT_CONFIG))
        expected["input"]["input_dir"] = "/workspace/pixel-art"
        expected["output"]["output_dir"] = "/workspace/captions"
        assert example == expected
        assert list(example) == list(DEFAULT_CONFIG)
        assert example["model"]["backend"] == "vllm"
        assert example["prompts"]["sources_file"] == "configs/sources.json"

    def test_local_hf_config(self) -> None:
        local = read_json("local_hf_config.json")
        assert local["model"]["backend"] == "hf"
        assert local["model"]["hf"]["quantization"] == "nf4"
        assert local["runtime"]["batch_size"] <= 8
        for value in (local["input"]["input_dir"], local["output"]["output_dir"],
                      local["model"]["model_path"]):
            assert not Path(value).is_absolute() and ":" not in value and "\\" not in value
        text = (CONFIGS / "local_hf_config.json").read_text(encoding="utf-8")
        for personal in ("F:", "Users", "Research", "datasets", "/home/", "rachid"):
            assert personal not in text

    def test_configs_are_lf_utf8(self) -> None:
        for name in ("example_config.json", "local_hf_config.json", "sources.json"):
            raw = (CONFIGS / name).read_bytes()
            assert b"\r" not in raw and raw.endswith(b"\n")
            raw.decode("utf-8")


class TestSourcesJson:
    def test_shape_and_categories(self) -> None:
        sources = read_json("sources.json")
        assert len(sources) >= 100
        assert list(sources) == sorted(sources)
        for folder, entry in sources.items():
            assert set(entry) == {"category", "name"}, folder
            assert entry["category"] in CATEGORIES, folder
            assert isinstance(entry["name"], str) and entry["name"].strip() == entry["name"]
            assert entry["name"], folder

    def test_categories_exist_in_default_config(self) -> None:
        assert CATEGORIES == set(DEFAULT_CONFIG["prompts"]["categories"])

    @pytest.mark.parametrize(
        ("folder", "category", "name"),
        [
            ("katanna-zero", "game_crop", "Katana ZERO"),
            ("crypt-necrodancer", "game_crop", "Crypt of the NecroDancer"),
            ("fallout 2", "game_crop", "Fallout 2"),
            ("axiom-verge 2", "game_crop", "Axiom Verge 2"),
            ("celeste", "game_crop", "Celeste"),
            ("stardew-valley", "game_crop", "Stardew Valley"),
            ("midjourney-dataset", "ai_pixel_art", "Midjourney"),
        ],
    )
    def test_known_entries(self, folder, category, name) -> None:
        assert read_json("sources.json")[folder] == {"category": category, "name": name}

    @pytest.mark.parametrize(
        "folder",
        ["pinterest-dataset-game", "pinterest-dataset-video-game",
         "pinterest-pixel-art-2023-07-01-v3", "pixel-art-dataset", "metroid-fusion-dataset",
         "super-metroid-redesign-dataset", "midjourney-dataset"],
    )
    def test_collections_are_not_game_crops(self, folder) -> None:
        assert read_json("sources.json")[folder]["category"] != "game_crop"

    def test_renders_a_prompt_for_every_source(self) -> None:
        from pixel_art_auto_captioner.common.prompts import render_prompt

        prompt_config = {**DEFAULT_CONFIG["prompts"], "sources": read_json("sources.json"),
                         "caption_types": DEFAULT_CONFIG["generation"]["caption_types"]}
        for folder in prompt_config["sources"]:
            assert render_prompt("long", folder, prompt_config)
        assert "Katana ZERO" in render_prompt("short", "katanna-zero", prompt_config)


# ---------------------------------------------------------------------------
# Source standards
# ---------------------------------------------------------------------------


class TestSourceStandards:
    def test_no_print_and_no_heavy_imports(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        names = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "print" not in names
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & set(HEAVY)
        assert "signal" not in imported  # the runner owns SIGTERM handling

    def test_public_functions_documented_and_typed(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node), node.name
            if isinstance(node, ast.FunctionDef):
                assert node.returns is not None, node.name
                for arg in node.args.args:
                    assert arg.annotation is not None, (node.name, arg.arg)

    def test_lf_line_endings(self) -> None:
        assert b"\r\n" not in SCRIPT.read_bytes()
        assert b"\r\n" not in Path(__file__).read_bytes()

    def test_importing_the_script_loads_no_backend(self) -> None:
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('rc', {str(SCRIPT)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "bad = [n for n in sys.modules if n.split('.')[0] in "
            f"{HEAVY!r} or 'joycaption' in n]\n"
            "assert not bad, bad\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=subprocess_env(), timeout=120)
        assert proc.returncode == 0, proc.stderr
