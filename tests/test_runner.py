"""Tests for ``batch/runner.py`` (SPEC v1.1 §7, §8.1, §11.4; §13 runner items).

A fake ``CaptionModel`` stands in for the backends; the dataloader, resume
index and shard writer are the real ones, on ``tmp_path``. No torch, no
transformers, no vllm, no real sleeps (the progress clock is replaced).
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

from pixel_art_auto_captioner import batch
from pixel_art_auto_captioner.batch import runner as runner_mod
from pixel_art_auto_captioner.batch.runner import GEN_PARAM_KEYS, SUMMARY_KEYS, CaptionRunner
from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.config import load_config
from pixel_art_auto_captioner.common.export_utils import ShardWriter
from pixel_art_auto_captioner.common.prompts import prompt_config_hash, render_prompt
from pixel_art_auto_captioner.common.seeding import derive_seed
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader, Plan

SRC = Path(runner_mod.__file__).resolve().parents[2]
TYPES = ["long", "short"]
IMAGES = [
    "celeste/0001/000001.png",
    "celeste/0001/000002.png",
    "hades/0001/000001.png",
    "pinterest/0001/000001.jpg",
    "pinterest/0001/000002.jpg",
]
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-(\d+)-[0-9a-f]{4}$")
MANIFEST_KEYS = {
    "run_id", "started_utc", "finished_utc", "interrupted", "error", "summary", "throughput",
    "model", "versions", "git_sha", "prompt_config_hash", "files", "config",
}


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class OutOfMemoryError(RuntimeError):
    """Stand-in for ``torch.cuda.OutOfMemoryError`` (detected by class name)."""


Responder = Callable[[CaptionRequest], "CaptionResult | BaseException"]


def ok_responder(request: CaptionRequest) -> CaptionResult:
    text = (
        f"Caption of {request.image_rel_path.as_posix()} for {request.caption_type}, "
        f"attempt {request.attempt}."
    )
    return CaptionResult(text=text, finish_reason="stop", num_output_tokens=12)


def scripted(rules: dict[tuple[str, str], Callable[[CaptionRequest], Any]]) -> Responder:
    """Per (rel_path, caption_type) behaviour; everything else is ok."""

    def respond(request: CaptionRequest) -> Any:
        rule = rules.get((request.image_rel_path.as_posix(), request.caption_type))
        return rule(request) if rule is not None else ok_responder(request)

    return respond


def truncated(request: CaptionRequest) -> CaptionResult:
    return CaptionResult(text="A long caption that was cut", finish_reason="length",
                         num_output_tokens=512)


def errored(request: CaptionRequest) -> CaptionResult:
    return CaptionResult(text="", finish_reason="error", num_output_tokens=0,
                         error="ValueError: bad input")


class FakeModel(CaptionModel):
    """Records every call; results come from ``responder`` (an exception is raised)."""

    model_name = "joycaption-beta-one"
    backend = "hf"

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        load_error: BaseException | None = None,
        batch_hook: Callable[[FakeModel, list[CaptionRequest]], None] | None = None,
    ) -> None:
        self.responder = responder or ok_responder
        self.load_error = load_error
        self.batch_hook = batch_hook
        self.load_configs: list[Any] = []
        self.batches: list[list[CaptionRequest]] = []
        self.events: list[str] = []
        self.unload_calls = 0
        self.loaded = False

    def load(self, config: dict) -> None:
        self.events.append("load")
        self.load_configs.append(config)
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        assert self.loaded, "caption_batch before load()"
        self.events.append("batch")
        self.batches.append(list(requests))
        if self.batch_hook is not None:
            self.batch_hook(self, list(requests))
        results = []
        for request in requests:
            out = self.responder(request)
            if isinstance(out, BaseException):
                raise out
            results.append(out)
        return results

    def unload(self) -> None:
        self.events.append("unload")
        self.unload_calls += 1
        self.loaded = False

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model_path": "fake/joycaption", "revision": "rev123",
                "dtype": "bfloat16", "fake_version": "9.9"}

    @property
    def requests(self) -> list[CaptionRequest]:
        return [r for b in self.batches for r in b]


def make_image(path: Path, size: tuple[int, int] = (8, 8), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.new("RGB", size, color).save(path)


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_config(tmp_path: Path, input_dir: Path, overrides: dict | None = None) -> dict:
    base = {
        "input": {"input_dir": str(input_dir)},
        "model": {"backend": "hf"},
        "prompts": {
            "sources_file": None,
            "sources": {
                "celeste": {"category": "game_crop", "name": "Celeste"},
                "hades": {"category": "game_crop"},
            },
        },
        "output": {"output_dir": str(tmp_path / "out"), "qa_sample": 0},
        "runtime": {"num_workers": 0, "batch_size": 2, "log_every_s": 3600},
    }
    return load_config(None, merge(base, overrides or {}))


def make_loader(cfg: dict) -> ImageDataLoader:
    return ImageDataLoader({**cfg["input"], **cfg["runtime"]})


def make_runner(cfg: dict, model: FakeModel | None = None) -> tuple[CaptionRunner, FakeModel]:
    model = model or FakeModel()
    return CaptionRunner(make_loader(cfg), model, cfg), model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def captions_path(cfg: dict, shard: str = "000-of-001") -> Path:
    return Path(cfg["output"]["output_dir"]) / "captions" / f"shard-{shard}.jsonl"


def failures_path(cfg: dict, shard: str = "000-of-001") -> Path:
    return Path(cfg["output"]["output_dir"]) / "failures" / f"shard-{shard}.jsonl"


def manifests(cfg: dict) -> list[dict]:
    runs = Path(cfg["output"]["output_dir"]) / "runs"
    if not runs.is_dir():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs.glob("*.json"))]


def manifest_for(cfg: dict, run_id: str) -> dict:
    path = Path(cfg["output"]["output_dir"]) / "runs" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def only_manifest(cfg: dict) -> dict:
    found = manifests(cfg)
    assert len(found) == 1, found
    return found[0]


def assert_invariants(summary: dict) -> None:
    assert summary["images_in_shard"] == (
        summary["images_already_complete"] + summary["images_deferred"] + summary["images_planned"]
    )
    if not summary["interrupted"]:
        assert summary["captions_planned"] == (
            summary["captions_ok"] + summary["captions_flagged"] + summary["captions_failed"]
        )


def prompt_config(cfg: dict) -> dict:
    return {**cfg["prompts"], "caption_types": cfg["generation"]["caption_types"]}


def nth_batch_hook(n: int, action: Callable[[], None]) -> Callable[[FakeModel, list], None]:
    """Run ``action`` on the n-th (1-based) caption_batch call."""

    def hook(model: FakeModel, requests: list) -> None:
        if len(model.batches) == n:
            action()

    return hook


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for i, rel in enumerate(IMAGES):
        make_image(root / rel, size=(8 + i, 6 + i))
    return root


@pytest.fixture
def cfg(tmp_path: Path, dataset: Path) -> dict:
    return build_config(tmp_path, dataset)


@pytest.fixture
def restore_sigterm():
    original = signal.getsignal(signal.SIGTERM)
    yield original
    signal.signal(signal.SIGTERM, original)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_config_raises_value_error(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"batch_size": 0}})
        with pytest.raises(ValueError, match="batch_size"):
            CaptionRunner(make_loader(build_config(tmp_path, dataset)), FakeModel(), cfg)

    def test_all_validation_problems_reported(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset)
        bad = merge(cfg, {"runtime": {"batch_size": 0, "shard": "3/2"},
                          "model": {"backend": "tpu"}})
        with pytest.raises(ValueError) as info:
            CaptionRunner(make_loader(cfg), FakeModel(), bad)
        for key in ("batch_size", "runtime.shard", "model.backend"):
            assert key in str(info.value)

    def test_missing_input_dir_is_a_config_error(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset)
        loader = make_loader(cfg)
        cfg["input"]["input_dir"] = str(tmp_path / "missing")
        with pytest.raises(ValueError, match="input_dir"):
            CaptionRunner(loader, FakeModel(), cfg)

    def test_shard_must_match_the_dataloader(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"shard": "1/2"}})
        loader = make_loader(build_config(tmp_path, dataset, {"runtime": {"shard": "0/2"}}))
        with pytest.raises(ValueError, match="does not match the dataloader"):
            CaptionRunner(loader, FakeModel(), cfg)

    @pytest.mark.parametrize(
        "override, key",
        [
            ({"quality": {"max_retries": -1}}, "max_retries"),
            ({"quality": {"max_retries": True}}, "max_retries"),
            ({"quality": {"max_retries": 1.5}}, "max_retries"),
            ({"runtime": {"seed": "0"}}, "seed"),
            ({"runtime": {"seed": None}}, "seed"),
            ({"runtime": {"gpu_hourly_cost": 0}}, "gpu_hourly_cost"),
            ({"runtime": {"gpu_hourly_cost": -1.0}}, "gpu_hourly_cost"),
            ({"runtime": {"gpu_hourly_cost": "0.69"}}, "gpu_hourly_cost"),
            ({"runtime": {"gpu_hourly_cost": float("nan")}}, "gpu_hourly_cost"),
        ],
    )
    def test_runner_specific_values(self, tmp_path, dataset, override, key) -> None:
        cfg = build_config(tmp_path, dataset, override)
        with pytest.raises(ValueError, match=key):
            CaptionRunner(make_loader(cfg), FakeModel(), cfg)

    def test_constructor_touches_neither_model_nor_disk(self, cfg) -> None:
        _, model = make_runner(cfg)
        assert model.events == []
        assert not Path(cfg["output"]["output_dir"]).exists()

    def test_config_is_copied_and_never_modified(self, cfg) -> None:
        before = copy.deepcopy(cfg)
        runner, _ = make_runner(cfg)
        cfg["runtime"]["batch_size"] = 99
        assert runner.config["runtime"]["batch_size"] == 2
        cfg["runtime"]["batch_size"] = 2
        runner.run()
        assert cfg == before

    def test_attributes(self, cfg) -> None:
        runner, _ = make_runner(cfg)
        assert runner.shard == (0, 1)
        assert runner.output_dir == Path(cfg["output"]["output_dir"]).absolute()
        assert runner.config == cfg


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRun:
    def test_summary_keys_counts_and_invariants(self, cfg) -> None:
        summary = make_runner(cfg)[0].run()
        assert list(summary) == list(SUMMARY_KEYS)
        assert summary["shard"] == "0/1"
        assert summary["images_discovered"] == summary["images_in_shard"] == 5
        assert summary["images_planned"] == 5
        assert summary["images_already_complete"] == summary["images_deferred"] == 0
        assert summary["captions_planned"] == summary["captions_ok"] == 10
        assert summary["captions_flagged"] == summary["captions_failed"] == 0
        assert summary["images_load_failed"] == summary["retries"] == 0
        assert summary["interrupted"] is False
        assert summary["output_dir"] == Path(cfg["output"]["output_dir"]).absolute().as_posix()
        assert RUN_ID_RE.match(summary["run_id"])
        assert_invariants(summary)

    def test_b1_load_receives_only_the_model_section(self, cfg) -> None:
        runner, model = make_runner(cfg)
        runner.run()
        assert len(model.load_configs) == 1
        received = model.load_configs[0]
        assert received == cfg["model"]
        assert set(received) == set(cfg["model"])
        for section in ("input", "generation", "prompts", "quality", "output", "runtime"):
            assert section not in received
        assert received["model_path"] == "fancyfeast/llama-joycaption-beta-one-hf-llava"
        assert received["hf"] == {"quantization": "nf4", "device_map": 0}
        received["hf"]["quantization"] = "changed"  # a copy: the runner's config is untouched
        assert runner.config["model"]["hf"]["quantization"] == "nf4"

    def test_lifecycle_load_batches_unload(self, cfg) -> None:
        runner, model = make_runner(cfg)
        runner.run()
        assert model.events[0] == "load"
        assert model.events[-1] == "unload"
        assert model.events.count("load") == 1 and model.unload_calls == 1
        assert set(model.events[1:-1]) == {"batch"}

    def test_batches_of_batch_size_images_in_plan_order(self, cfg) -> None:
        runner, model = make_runner(cfg)
        runner.run()
        assert [len(b) for b in model.batches] == [4, 4, 2]
        keys = [(r.image_rel_path.as_posix(), r.caption_type) for r in model.requests]
        assert keys == [(rel, t) for rel in IMAGES for t in TYPES]

    def test_requests_carry_prompt_seed_params_and_image(self, cfg) -> None:
        runner, model = make_runner(cfg)
        runner.run()
        pc = prompt_config(cfg)
        for request in model.requests:
            rel = request.image_rel_path
            source = rel.parts[0]
            assert request.attempt == 1
            assert request.seed == derive_seed(0, rel, request.caption_type, 1)
            assert request.prompt == render_prompt(request.caption_type, source, pc)
            spec = cfg["generation"]["caption_types"][request.caption_type]
            assert request.gen_params == {k: spec[k] for k in GEN_PARAM_KEYS}
            assert request.image.mode == "RGB"
            index = IMAGES.index(rel.as_posix())
            assert request.image.size == (8 + index, 6 + index)  # not resized

    def test_prompt_uses_the_source_mapping(self, cfg) -> None:
        runner, model = make_runner(cfg)
        runner.run()
        prompts = {(r.image_rel_path.parts[0], r.caption_type): r.prompt for r in model.requests}
        assert "video game Celeste." in prompts[("celeste", "long")]
        assert "video game Hades." in prompts[("hades", "long")]
        assert "video game" not in prompts[("pinterest", "long")]
        assert prompts[("pinterest", "short")].startswith("Write a short description")

    def test_seed_and_generation_params_follow_the_config(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {
            "runtime": {"seed": 1234},
            "generation": {"enabled_caption_types": ["tags"],
                           "caption_types": {"tags": {"max_new_tokens": 50, "top_k": 40}}},
        })
        runner, model = make_runner(cfg)
        runner.run()
        assert {r.caption_type for r in model.requests} == {"tags"}
        for request in model.requests:
            assert request.seed == derive_seed(1234, request.image_rel_path, "tags", 1)
            assert request.gen_params == {"max_new_tokens": 50, "temperature": 0.6,
                                          "top_p": 0.9, "top_k": 40}

    def test_records_written(self, cfg) -> None:
        summary = make_runner(cfg)[0].run()
        records = read_jsonl(captions_path(cfg))
        assert len(records) == 10
        assert {(r["image_rel_path"], r["caption_type"]) for r in records} == {
            (rel, t) for rel in IMAGES for t in TYPES
        }
        pc = prompt_config(cfg)
        dataset = Path(cfg["input"]["input_dir"])
        for record in records:
            rel = record["image_rel_path"]
            index = IMAGES.index(rel)
            source = rel.split("/")[0]
            assert record["status"] == "ok" and record["flags"] == []
            assert record["source"] == source
            assert record["image_path"] == (dataset / rel).resolve().as_posix()
            assert record["prompt"] == render_prompt(record["caption_type"], source, pc)
            assert record["seed"] == derive_seed(0, Path(rel), record["caption_type"], 1)
            assert record["attempt"] == 1
            assert record["caption_text"].startswith(f"Caption of {rel}")
            assert record["finish_reason"] == "stop" and record["num_output_tokens"] == 12
            assert (record["image_width"], record["image_height"]) == (8 + index, 6 + index)
            assert record["model_name"] == "joycaption-beta-one"
            assert record["model_path"] == "fake/joycaption"
            assert record["model_revision"] == "rev123"
            assert record["backend"] == "hf"
            assert record["run_id"] == summary["run_id"]
            assert record["shard"] == "0/1"
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["timestamp_utc"])
            spec = cfg["generation"]["caption_types"][record["caption_type"]]
            assert record["generation_params"] == {k: spec[k] for k in GEN_PARAM_KEYS}
        by_source = {r["source"]: (r["category"], r["extra"]["source_name"]) for r in records}
        assert by_source == {
            "celeste": ("game_crop", "Celeste"),
            "hades": ("game_crop", "Hades"),
            "pinterest": ("mixed_collection", "Pinterest"),
        }
        assert not failures_path(cfg).exists()

    def test_model_path_falls_back_to_the_config(self, cfg) -> None:
        class Bare(FakeModel):
            def describe(self) -> dict[str, Any]:
                return {"model_name": self.model_name, "backend": self.backend}

        cfg["model"]["revision"] = "abc"
        runner, _ = make_runner(cfg, Bare())
        runner.run()
        record = read_jsonl(captions_path(cfg))[0]
        assert record["model_path"] == cfg["model"]["model_path"]
        assert record["model_revision"] == "abc"

    def test_other_shard(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"shard": "1/2"}})
        summary = make_runner(cfg)[0].run()
        assert summary["shard"] == "1/2"
        assert RUN_ID_RE.match(summary["run_id"]).group(1) == "1"
        records = read_jsonl(captions_path(cfg, "001-of-002"))
        assert len(records) == summary["captions_ok"] == 2 * summary["images_planned"]
        assert {r["shard"] for r in records} == {"1/2"}
        assert not captions_path(cfg).exists()
        other = make_runner(build_config(tmp_path, dataset, {"runtime": {"shard": "0/2"}}))[0].run()
        assert summary["images_in_shard"] + other["images_in_shard"] == 5

    def test_decode_threads_give_the_same_records(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"num_workers": 2, "prefetch": 2}})
        runner, model = make_runner(cfg)
        summary = runner.run()
        assert summary["captions_ok"] == 10
        keys = [(r.image_rel_path.as_posix(), r.caption_type) for r in model.requests]
        assert keys == [(rel, t) for rel in IMAGES for t in TYPES]

    def test_run_ids_are_unique(self, cfg) -> None:
        runner, _ = make_runner(cfg)
        first = runner.run()["run_id"]
        cfg2 = merge(cfg, {"output": {"resume": False}})
        second = make_runner(cfg2)[0].run()["run_id"]
        assert first != second

    def test_run_id_uses_the_utc_start_time(self, cfg, monkeypatch) -> None:
        from datetime import datetime, timezone
        monkeypatch.setattr(runner_mod, "_utc_now",
                            lambda: datetime(2026, 11, 8, 1, 2, 3, tzinfo=timezone.utc))
        summary = make_runner(cfg)[0].run()
        assert re.fullmatch(r"20261108T010203Z-0-[0-9a-f]{4}", summary["run_id"])
        assert only_manifest(cfg)["started_utc"] == "2026-11-08T01:02:03Z"


# ---------------------------------------------------------------------------
# Retry flow (§5.3)
# ---------------------------------------------------------------------------


class TestRetry:
    KEY = ("celeste/0001/000001.png", "long")

    def test_flagged_then_ok(self, cfg) -> None:
        def first_truncated(request: CaptionRequest) -> CaptionResult:
            return truncated(request) if request.attempt == 1 else ok_responder(request)

        runner, model = make_runner(cfg, FakeModel(scripted({self.KEY: first_truncated})))
        summary = runner.run()
        assert summary["retries"] == 1
        assert summary["captions_ok"] == 10
        assert_invariants(summary)
        # The follow-up batch holds only the retried request, attempt 2, new seed.
        assert [len(b) for b in model.batches] == [4, 1, 4, 2]
        retried = model.batches[1][0]
        original = model.batches[0][0]
        assert (retried.image_rel_path.as_posix(), retried.caption_type) == self.KEY
        assert retried.attempt == 2
        assert retried.seed == derive_seed(0, Path(self.KEY[0]), "long", 2) != original.seed
        assert retried.prompt == original.prompt and retried.gen_params == original.gen_params
        assert retried.image is original.image
        record = next(r for r in read_jsonl(captions_path(cfg))
                      if (r["image_rel_path"], r["caption_type"]) == self.KEY)
        assert (record["status"], record["attempt"], record["seed"]) == ("ok", 2, retried.seed)
        assert len(read_jsonl(captions_path(cfg))) == 10  # one record per final result

    @pytest.mark.parametrize(
        "behaviour, status, flags, finish_reason",
        [
            (truncated, "flagged", ["truncated"], "length"),
            (errored, "failed", ["error", "empty"], "error"),
            (lambda r: CaptionResult("   ", "stop", 1), "failed", ["empty"], "stop"),
            (lambda r: CaptionResult("pixel art " * 40, "stop", 80), "flagged", ["repetitive"],
             "stop"),
            (lambda r: CaptionResult("pixel art " * 400, "length", 512), "flagged",
             ["truncated", "repetitive"], "length"),
        ],
    )
    def test_still_bad_after_the_last_attempt(self, cfg, behaviour, status, flags,
                                              finish_reason) -> None:
        runner, model = make_runner(cfg, FakeModel(scripted({self.KEY: behaviour})))
        summary = runner.run()
        attempts = [r.attempt for r in model.requests
                    if (r.image_rel_path.as_posix(), r.caption_type) == self.KEY]
        assert attempts == [1, 2]
        assert summary["retries"] == 1
        record = next(r for r in read_jsonl(captions_path(cfg))
                      if (r["image_rel_path"], r["caption_type"]) == self.KEY)
        assert record["status"] == status
        assert record["flags"] == flags
        assert record["attempt"] == 2
        assert record["finish_reason"] == finish_reason
        assert record["seed"] == derive_seed(0, Path(self.KEY[0]), "long", 2)
        counts = {"ok": 9, "flagged": 0, "failed": 0}
        counts[status] += 1
        assert (summary["captions_ok"], summary["captions_flagged"],
                summary["captions_failed"]) == (counts["ok"], counts["flagged"], counts["failed"])
        assert_invariants(summary)
        if status == "failed" and "error" in flags:
            assert record["extra"]["error"] == "ValueError: bad input"

    def test_no_retry_when_max_retries_is_zero(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"quality": {"max_retries": 0}})
        runner, model = make_runner(cfg, FakeModel(scripted({self.KEY: truncated})))
        summary = runner.run()
        assert summary["retries"] == 0
        assert len(model.requests) == 10
        record = next(r for r in read_jsonl(captions_path(cfg))
                      if (r["image_rel_path"], r["caption_type"]) == self.KEY)
        assert (record["status"], record["attempt"]) == ("flagged", 1)

    def test_more_retries(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"quality": {"max_retries": 2}})
        runner, model = make_runner(cfg, FakeModel(scripted({self.KEY: errored})))
        summary = runner.run()
        attempts = [(r.attempt, r.seed) for r in model.requests
                    if (r.image_rel_path.as_posix(), r.caption_type) == self.KEY]
        assert [a for a, _ in attempts] == [1, 2, 3]
        assert len({s for _, s in attempts}) == 3
        assert summary["retries"] == 2
        assert summary["captions_failed"] == 1
        record = next(r for r in read_jsonl(captions_path(cfg))
                      if (r["image_rel_path"], r["caption_type"]) == self.KEY)
        assert record["attempt"] == 3

    def test_retries_of_one_group_go_out_together(self, cfg) -> None:
        rules = {("celeste/0001/000001.png", "long"): truncated,
                 ("celeste/0001/000002.png", "short"): truncated}
        runner, model = make_runner(cfg, FakeModel(scripted(rules)))
        summary = runner.run()
        assert [len(b) for b in model.batches] == [4, 2, 4, 2]
        assert {r.attempt for r in model.batches[1]} == {2}
        assert summary["retries"] == 2 and summary["captions_flagged"] == 2

    @pytest.mark.parametrize(
        "flags, status",
        [([], "ok"), (["truncated"], "flagged"), (["repetitive"], "flagged"),
         (["truncated", "repetitive"], "flagged"), (["empty"], "failed"), (["error"], "failed"),
         (["error", "empty"], "failed"), (["empty", "repetitive"], "failed")],
    )
    def test_status_rule(self, flags, status) -> None:
        assert runner_mod._status(flags) == status


# ---------------------------------------------------------------------------
# Batch exceptions (§7.2)
# ---------------------------------------------------------------------------


class TestBatchFallback:
    def test_batch_exception_retries_requests_one_at_a_time(self, cfg) -> None:
        def hook(model: FakeModel, requests: list) -> None:
            if len(requests) > 1:
                raise RuntimeError("transient engine error")

        runner, model = make_runner(cfg, FakeModel(batch_hook=hook))
        summary = runner.run()
        assert summary["captions_ok"] == 10 and summary["captions_failed"] == 0
        assert summary["retries"] == 0
        assert [len(b) for b in model.batches] == [4, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1]
        assert {r.attempt for r in model.requests} == {1}
        events = read_jsonl(failures_path(cfg))
        assert [e["event"] for e in events] == ["batch_exception"] * 3
        assert events[0]["error"] == "RuntimeError: transient engine error"
        assert events[0]["requests"][0] == {"image_rel_path": IMAGES[0], "caption_type": "long",
                                            "attempt": 1}
        assert len(events[0]["requests"]) == 4
        assert only_manifest(cfg)["files"]["failures"] == "failures/shard-000-of-001.jsonl"

    def test_request_that_still_raises_gets_an_error_result(self, cfg, caplog) -> None:
        bad = ("hades/0001/000001.png", "short")
        runner, model = make_runner(
            cfg, FakeModel(scripted({bad: lambda r: ValueError("cannot process")})))
        with caplog.at_level(logging.WARNING, logger=runner_mod.logger.name):
            summary = runner.run()
        assert summary["captions_failed"] == 1 and summary["captions_ok"] == 9
        assert summary["retries"] == 1  # the error result goes through the retry policy
        assert_invariants(summary)
        record = next(r for r in read_jsonl(captions_path(cfg))
                      if (r["image_rel_path"], r["caption_type"]) == bad)
        assert record["status"] == "failed"
        assert record["flags"] == ["error", "empty"]
        assert record["finish_reason"] == "error" and record["caption_text"] == ""
        assert record["extra"]["error"] == "ValueError: cannot process"
        assert record["attempt"] == 2
        events = [e["event"] for e in read_jsonl(failures_path(cfg))]
        assert events.count("request_exception") == 2
        assert events.count("batch_exception") == 2  # attempt 1 group + attempt 2 retry batch
        assert "cannot process" in caplog.text

    def test_wrong_result_count_is_treated_as_a_batch_exception(self, cfg) -> None:
        class Short(FakeModel):
            def caption_batch(self, requests):
                results = super().caption_batch(requests)
                return results[:-1] if len(requests) > 1 else results

        runner, model = make_runner(cfg, Short())
        summary = runner.run()
        assert summary["captions_ok"] == 10
        event = read_jsonl(failures_path(cfg))[0]
        assert event["event"] == "batch_exception"
        assert "returned 3 results for 4 requests" in event["error"]

    def test_wrong_result_count_on_a_single_request_is_an_error_result(self, cfg) -> None:
        class Empty(FakeModel):
            def caption_batch(self, requests):
                super().caption_batch(requests)
                return []

        cfg = merge(cfg, {"quality": {"max_retries": 0}})
        runner, _ = make_runner(cfg, Empty())
        summary = runner.run()
        assert summary["captions_failed"] == 10
        record = read_jsonl(captions_path(cfg))[0]
        assert record["extra"]["error"].startswith("RuntimeError")

    def test_oom_on_a_batch_that_works_singly_does_not_abort(self, cfg) -> None:
        def hook(model: FakeModel, requests: list) -> None:
            if len(requests) > 1:
                raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

        runner, _ = make_runner(cfg, FakeModel(batch_hook=hook))
        summary = runner.run()
        assert summary["captions_ok"] == 10


# ---------------------------------------------------------------------------
# CUDA OOM on a single request (§7.2, §12)
# ---------------------------------------------------------------------------


class TestCudaOom:
    @pytest.mark.parametrize(
        "exc",
        [OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
         RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")],
    )
    def test_aborts_after_flushing(self, cfg, caplog, exc) -> None:
        bad = ("hades/0001/000001.png", "long")
        runner, model = make_runner(cfg, FakeModel(scripted({bad: lambda r: exc})))
        with caplog.at_level(logging.INFO, logger=runner_mod.logger.name):
            with pytest.raises(type(exc)) as info:
                runner.run()
        assert info.value is exc
        assert any(r.levelno == logging.CRITICAL and "CUDA out of memory" in r.getMessage()
                   for r in caplog.records)
        # The first group's records are on disk and readable.
        records = read_jsonl(captions_path(cfg))
        assert {r["image_rel_path"] for r in records} == set(IMAGES[:2])
        assert len(records) == 4
        assert runner._writer is not None and runner._writer.closed
        assert model.unload_calls == 1 and model.events[-1] == "unload"
        manifest = only_manifest(cfg)
        assert manifest["interrupted"] is False
        assert manifest["error"].startswith(type(exc).__name__)
        assert manifest["summary"]["captions_ok"] == 4
        assert manifest["files"]["qa_page"] is None
        events = [e["event"] for e in read_jsonl(failures_path(cfg))]
        assert events == ["batch_exception", "cuda_oom"]
        # No request after the OOM.
        assert model.requests[-1].image_rel_path.as_posix() == bad[0]

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (OutOfMemoryError("x"), True),
            (type("OutOfMemoryError", (Exception,), {})("x"), True),
            (type("CudaOOM", (OutOfMemoryError,), {})("x"), True),
            (RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"), True),
            (RuntimeError("CUDA error: out of memory"), True),
            (RuntimeError("boom"), False),
            (MemoryError("out of memory"), False),
            (ValueError("bad input"), False),
        ],
    )
    def test_detection_without_torch(self, exc, expected) -> None:
        assert runner_mod._is_cuda_oom(exc) is expected


# ---------------------------------------------------------------------------
# Empty plan and resume
# ---------------------------------------------------------------------------


class TestEmptyPlanAndResume:
    def test_second_run_plans_nothing_and_never_loads(self, cfg) -> None:
        first = make_runner(cfg)[0].run()
        runner, model = make_runner(cfg)
        summary = runner.run()
        assert model.events == []  # never loaded, nothing to unload
        assert summary["images_planned"] == summary["captions_planned"] == 0
        assert summary["images_already_complete"] == 5
        assert summary["run_id"] != first["run_id"]
        assert_invariants(summary)
        assert len(read_jsonl(captions_path(cfg))) == 10  # nothing appended
        assert len(manifests(cfg)) == 1  # no manifest for an empty plan

    def test_empty_dataset_never_loads_and_writes_nothing(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        cfg = build_config(tmp_path, empty)
        runner, model = make_runner(cfg)
        summary = runner.run()
        assert model.events == []
        assert summary["images_discovered"] == 0
        assert_invariants(summary)
        assert not Path(cfg["output"]["output_dir"]).exists()

    def test_max_images_advances_across_runs(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"max_images": 2}})
        planned = []
        for _ in range(4):
            summary = make_runner(cfg)[0].run()
            assert_invariants(summary)
            planned.append((summary["images_planned"], summary["images_deferred"],
                            summary["images_already_complete"]))
        assert planned == [(2, 3, 0), (2, 1, 2), (1, 0, 4), (0, 0, 5)]
        records = read_jsonl(captions_path(cfg))
        assert len(records) == 10
        assert len({(r["image_rel_path"], r["caption_type"]) for r in records}) == 10

    def test_failed_captions_are_planned_again(self, cfg) -> None:
        bad = ("hades/0001/000001.png", "short")
        make_runner(cfg, FakeModel(scripted({bad: errored})))[0].run()
        runner, model = make_runner(cfg)
        summary = runner.run()
        assert summary["images_planned"] == 1 and summary["captions_planned"] == 1
        assert [(r.image_rel_path.as_posix(), r.caption_type) for r in model.requests] == [bad]
        assert summary["captions_ok"] == 1

    def test_resume_off_recaptions_everything(self, cfg) -> None:
        make_runner(cfg)[0].run()
        cfg2 = merge(cfg, {"output": {"resume": False}})
        summary = make_runner(cfg2)[0].run()
        assert summary["images_already_complete"] == 0
        assert summary["captions_planned"] == summary["captions_ok"] == 10
        assert len(read_jsonl(captions_path(cfg))) == 20

    def test_new_caption_type_plans_only_the_missing_type(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset)
        make_runner(cfg)[0].run()
        cfg2 = build_config(tmp_path, dataset,
                            {"generation": {"enabled_caption_types": ["long", "short", "tags"]}})
        runner, model = make_runner(cfg2)
        summary = runner.run()
        assert summary["captions_planned"] == 5
        assert {r.caption_type for r in model.requests} == {"tags"}

    def test_resume_reads_other_shard_files(self, cfg) -> None:
        make_runner(cfg)[0].run()
        out = Path(cfg["output"]["output_dir"])
        captions_path(cfg).rename(out / "captions" / "merged-from-elsewhere.jsonl")
        runner, model = make_runner(cfg)
        assert runner.run()["images_planned"] == 0
        assert model.events == []


# ---------------------------------------------------------------------------
# Model load errors (§7.2 step 3, §12)
# ---------------------------------------------------------------------------


class TestLoadError:
    @pytest.mark.parametrize("exc", [RuntimeError("weights missing"), ImportError("no torch")])
    def test_load_error_is_logged_reraised_and_unloads(self, cfg, caplog, exc) -> None:
        runner, model = make_runner(cfg, FakeModel(load_error=exc))
        with caplog.at_level(logging.ERROR, logger=runner_mod.logger.name):
            with pytest.raises(type(exc)) as info:
                runner.run()
        assert info.value is exc
        assert model.events == ["load", "unload"]
        assert any("Model load failed" in r.getMessage() and r.exc_info for r in caplog.records)
        assert not captions_path(cfg).exists()
        manifest = only_manifest(cfg)
        assert manifest["error"] == f"{type(exc).__name__}: {exc}"
        assert manifest["files"]["captions"] is None
        assert manifest["summary"]["captions_ok"] == 0

    def test_unload_error_does_not_fail_a_finished_run(self, cfg, caplog) -> None:
        class BadUnload(FakeModel):
            def unload(self) -> None:
                super().unload()
                raise RuntimeError("driver hiccup")

        runner, _ = make_runner(cfg, BadUnload())
        summary = runner.run()
        assert summary["captions_ok"] == 10
        assert "Model unload failed" in caplog.text

    def test_describe_failure_gives_an_empty_model_entry(self, cfg) -> None:
        class NoDescribe(FakeModel):
            def describe(self):
                raise RuntimeError("nope")

        runner, _ = make_runner(cfg, NoDescribe())
        runner.run()
        assert only_manifest(cfg)["model"] == {}
        record = read_jsonl(captions_path(cfg))[0]
        assert record["model_path"] == cfg["model"]["model_path"]


# ---------------------------------------------------------------------------
# Interrupts (§7.2)
# ---------------------------------------------------------------------------


def _raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


def _send_sigterm() -> None:
    signal.raise_signal(signal.SIGTERM)


class TestInterrupt:
    @pytest.mark.parametrize("action", [_raise_keyboard_interrupt, _send_sigterm],
                             ids=["ctrl-c", "sigterm"])
    def test_interrupt_flushes_unloads_writes_manifest_and_reraises(
        self, tmp_path, dataset, restore_sigterm, action
    ) -> None:
        cfg = build_config(tmp_path, dataset, {"output": {"qa_sample": 10}})
        runner, model = make_runner(cfg, FakeModel(batch_hook=nth_batch_hook(2, action)))
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        records = read_jsonl(captions_path(cfg))
        assert len(records) == 4  # the first batch; the in-flight one is lost
        assert runner._writer is not None and runner._writer.closed
        assert model.events[-1] == "unload" and model.unload_calls == 1
        manifest = only_manifest(cfg)
        assert manifest["interrupted"] is True
        assert manifest["summary"]["interrupted"] is True
        assert manifest["summary"]["captions_ok"] == 4
        assert manifest["error"].startswith("KeyboardInterrupt")
        assert manifest["files"]["qa_page"] is None
        assert not (Path(cfg["output"]["output_dir"]) / "qa").exists()
        assert_invariants(manifest["summary"])
        assert signal.getsignal(signal.SIGTERM) == restore_sigterm

    def test_sigterm_handler_installed_during_the_run_and_restored(self, cfg,
                                                                   restore_sigterm) -> None:
        def previous(signum, frame):  # pragma: no cover - never called
            pass

        signal.signal(signal.SIGTERM, previous)
        seen = []
        runner, _ = make_runner(cfg, FakeModel(
            batch_hook=lambda m, r: seen.append(signal.getsignal(signal.SIGTERM))))
        runner.run()
        assert seen and all(h == runner._on_sigterm for h in seen)
        assert signal.getsignal(signal.SIGTERM) is previous

    def test_handler_restored_after_a_load_error(self, cfg, restore_sigterm) -> None:
        runner, _ = make_runner(cfg, FakeModel(load_error=RuntimeError("x")))
        with pytest.raises(RuntimeError):
            runner.run()
        assert signal.getsignal(signal.SIGTERM) == restore_sigterm

    def test_default_handler_restored_when_previous_was_not_from_python(
        self, cfg, monkeypatch, restore_sigterm
    ) -> None:
        real_getsignal = signal.getsignal
        monkeypatch.setattr(runner_mod.signal, "getsignal",
                            lambda s: None if s == signal.SIGTERM else real_getsignal(s))
        runner, _ = make_runner(cfg)
        runner.run()
        assert real_getsignal(signal.SIGTERM) == signal.SIG_DFL

    def test_second_sigterm_during_shutdown_is_ignored(self, cfg, caplog) -> None:
        runner, _ = make_runner(cfg)
        runner._stopping = True
        assert runner._on_sigterm(signal.SIGTERM, None) is None
        assert "while shutting down" in caplog.text

    def test_sigterm_handler_raises_keyboard_interrupt(self, cfg) -> None:
        runner, _ = make_runner(cfg)
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            runner._on_sigterm(signal.SIGTERM, None)
        assert runner._stopping is True

    def test_run_off_the_main_thread(self, cfg, restore_sigterm) -> None:
        seen: list = []
        result: dict = {}

        def target() -> None:
            runner, _ = make_runner(cfg, FakeModel(
                batch_hook=lambda m, r: seen.append(signal.getsignal(signal.SIGTERM))))
            try:
                result["summary"] = runner.run()
            except BaseException as exc:  # pragma: no cover - reported below
                result["error"] = exc

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=60)
        assert "error" not in result, result.get("error")
        assert result["summary"]["captions_ok"] == 10
        assert seen and all(h == restore_sigterm for h in seen)
        assert signal.getsignal(signal.SIGTERM) == restore_sigterm

    def test_decode_threads_are_joined_after_an_interrupt(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"num_workers": 2, "prefetch": 4,
                                                           "batch_size": 1}})
        runner, _ = make_runner(cfg, FakeModel(
            batch_hook=nth_batch_hook(1, _raise_keyboard_interrupt)))
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        assert not [t for t in threading.enumerate() if t.name.startswith("pac-decode")]

    def test_resume_after_an_interrupt_completes_without_duplicates(self, cfg) -> None:
        runner, _ = make_runner(cfg, FakeModel(
            batch_hook=nth_batch_hook(2, _raise_keyboard_interrupt)))
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        summary = make_runner(cfg)[0].run()
        assert summary["images_already_complete"] == 2 and summary["images_planned"] == 3
        records = read_jsonl(captions_path(cfg))
        assert len(records) == 10
        assert len({(r["image_rel_path"], r["caption_type"]) for r in records}) == 10
        assert len(manifests(cfg)) == 2


# ---------------------------------------------------------------------------
# Image load failures (§7.2, D5, B8)
# ---------------------------------------------------------------------------


class TestLoadFailure:
    BROKEN = "broken/0001/000001.jpg"

    @pytest.fixture
    def broken_cfg(self, tmp_path, dataset) -> dict:
        path = dataset / self.BROKEN
        path.parent.mkdir(parents=True)
        path.write_bytes(b"this is not an image")
        return build_config(tmp_path, dataset)

    def test_load_failure_writes_failed_records_to_failures(self, broken_cfg) -> None:
        runner, model = make_runner(broken_cfg)
        summary = runner.run()
        assert summary["images_planned"] == 6
        assert summary["images_load_failed"] == 1
        assert summary["captions_planned"] == 12
        assert summary["captions_ok"] == 10 and summary["captions_failed"] == 2
        assert_invariants(summary)
        assert self.BROKEN not in {r.image_rel_path.as_posix() for r in model.requests}
        assert self.BROKEN not in {r["image_rel_path"] for r in read_jsonl(captions_path(broken_cfg))}
        failed = read_jsonl(failures_path(broken_cfg))
        assert [(r["image_rel_path"], r["caption_type"]) for r in failed] == [
            (self.BROKEN, "long"), (self.BROKEN, "short")]
        pc = prompt_config(broken_cfg)
        for record in failed:
            assert record["status"] == "failed"
            assert record["flags"] == ["error", "empty"]
            assert record["caption_text"] == "" and record["finish_reason"] == "error"
            assert record["num_output_tokens"] == 0
            assert record["image_width"] == record["image_height"] == 0
            assert record["attempt"] == 1
            assert record["seed"] == derive_seed(0, Path(self.BROKEN), record["caption_type"], 1)
            assert record["prompt"] == render_prompt(record["caption_type"], "broken", pc)
            assert record["category"] == "mixed_collection"
            assert record["source"] == "broken"
            assert record["extra"]["error"].startswith("UnidentifiedImageError")
            assert record["run_id"] == summary["run_id"]
        assert only_manifest(broken_cfg)["files"]["failures"] == "failures/shard-000-of-001.jsonl"

    def test_load_failure_is_planned_again_next_run(self, broken_cfg) -> None:
        make_runner(broken_cfg)[0].run()
        runner, model = make_runner(broken_cfg)
        summary = runner.run()
        assert summary["images_planned"] == 1 and summary["images_load_failed"] == 1
        assert model.requests == []
        assert len(read_jsonl(failures_path(broken_cfg))) == 4

    def test_only_pending_types_are_recorded(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset)
        make_runner(cfg)[0].run()  # long + short complete for the good images
        (dataset / self.BROKEN).parent.mkdir(parents=True)
        (dataset / self.BROKEN).write_bytes(b"\x00\x01")
        cfg2 = build_config(tmp_path, dataset,
                            {"generation": {"enabled_caption_types": ["long", "short", "tags"]}})
        summary = make_runner(cfg2)[0].run()
        failed = read_jsonl(failures_path(cfg2))
        assert [r["caption_type"] for r in failed] == ["long", "short", "tags"]
        assert summary["captions_planned"] == 5 + 3
        assert summary["captions_failed"] == 3
        assert_invariants(summary)


# ---------------------------------------------------------------------------
# .txt output and collisions (§8.2)
# ---------------------------------------------------------------------------


class TestTxt:
    def test_txt_files_and_collision_naming(self, tmp_path, dataset) -> None:
        make_image(dataset / "sprites/0001/sprite.png")
        make_image(dataset / "sprites/0001/sprite.jpg")
        cfg = build_config(tmp_path, dataset, {"output": {"write_txt": True}})
        bad = ("hades/0001/000001.png", "short")
        runner, _ = make_runner(cfg, FakeModel(scripted({bad: errored})))
        runner.run()
        txt = Path(cfg["output"]["output_dir"]) / "txt"
        assert (txt / "sprites/0001/sprite.png.long.txt").is_file()
        assert (txt / "sprites/0001/sprite.jpg.short.txt").is_file()
        assert not (txt / "sprites/0001/sprite.long.txt").exists()
        normal = txt / "celeste/0001/000001.long.txt"
        assert normal.read_text(encoding="utf-8") == (
            "Caption of celeste/0001/000001.png for long, attempt 1.")
        assert (txt / "hades/0001/000001.long.txt").is_file()
        assert not (txt / "hades/0001/000001.short.txt").exists()  # failed: no txt

    def test_no_txt_by_default(self, cfg) -> None:
        make_runner(cfg)[0].run()
        assert not (Path(cfg["output"]["output_dir"]) / "txt").exists()

    def test_collision_set_matches_the_dataloader_rule(self, tmp_path) -> None:
        root = tmp_path / "data"
        for rel in ["a/x/Sprite.PNG", "a/x/sprite.jpg", "a/y/sprite.png", "b/x/sprite.png",
                    "a/x/other.png"]:
            make_image(root / rel)
        cfg = build_config(tmp_path, root)
        runner, _ = make_runner(cfg)
        assert runner._collision_set() == {Path("a/x/Sprite.PNG"), Path("a/x/sprite.jpg")}


# ---------------------------------------------------------------------------
# Write errors (§12)
# ---------------------------------------------------------------------------


class TestWriteError:
    def test_write_error_is_critical_reraised_and_unloads(self, cfg, caplog,
                                                          monkeypatch) -> None:
        calls = {"n": 0}
        real_write = ShardWriter.write

        def failing_write(self, record):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError(28, "No space left on device")
            real_write(self, record)

        monkeypatch.setattr(ShardWriter, "write", failing_write)
        runner, model = make_runner(cfg)
        with pytest.raises(OSError, match="No space left"):
            runner.run()
        assert any(r.levelno == logging.CRITICAL and "Write error" in r.getMessage()
                   for r in caplog.records)
        assert model.unload_calls == 1
        assert len(read_jsonl(captions_path(cfg))) == 2
        assert only_manifest(cfg)["error"].startswith("OSError")


# ---------------------------------------------------------------------------
# Manifest, QA page, git sha (§8.1, D8, §10.3)
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_contents(self, cfg) -> None:
        runner, _ = make_runner(cfg)
        summary = runner.run()
        out = Path(cfg["output"]["output_dir"])
        manifest = manifest_for(cfg, summary["run_id"])
        assert set(manifest) == MANIFEST_KEYS
        assert manifest["run_id"] == summary["run_id"]
        assert manifest["summary"] == summary
        assert manifest["interrupted"] is False and manifest["error"] is None
        assert manifest["config"] == cfg
        assert manifest["model"] == FakeModel().describe()
        versions = manifest["versions"]
        assert versions["python"] == ".".join(map(str, sys.version_info[:3]))
        assert versions["pillow"] == PIL.__version__
        assert set(versions) == {"pixel_art_auto_captioner", "python", "pillow", "platform"}
        assert manifest["git_sha"] is None or re.fullmatch(r"[0-9a-f]{40}", manifest["git_sha"])
        assert manifest["prompt_config_hash"] == prompt_config_hash(prompt_config(cfg))
        assert manifest["files"] == {"captions": "captions/shard-000-of-001.jsonl",
                                     "failures": None, "qa_page": None}
        for key in ("started_utc", "finished_utc"):
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", manifest[key])
        throughput = manifest["throughput"]
        assert throughput["images_done"] == 5 and throughput["images_remaining"] == 0
        assert throughput["gpu_hourly_cost"] is None and throughput["cost_usd"] is None
        assert [p.name for p in (out / "runs").iterdir()] == [f"{summary['run_id']}.json"]

    def test_qa_page_written_with_the_run_id(self, tmp_path, dataset) -> None:
        cfg = build_config(tmp_path, dataset, {"output": {"qa_sample": 3}})
        summary = make_runner(cfg)[0].run()
        page = Path(cfg["output"]["output_dir"]) / "qa" / f"{summary['run_id']}.html"
        assert page.is_file()
        assert "data:image/jpeg;base64" in page.read_text(encoding="utf-8")
        assert only_manifest(cfg)["files"]["qa_page"] == f"qa/{summary['run_id']}.html"

    def test_qa_page_failure_is_logged_not_raised(self, tmp_path, dataset, monkeypatch,
                                                  caplog) -> None:
        cfg = build_config(tmp_path, dataset, {"output": {"qa_sample": 3}})

        def boom(*args, **kwargs):
            raise OSError("cannot write page")

        monkeypatch.setattr(runner_mod, "generate_qa_page", boom)
        summary = make_runner(cfg)[0].run()
        assert summary["captions_ok"] == 10
        assert "Could not write the QA page" in caplog.text
        assert only_manifest(cfg)["files"]["qa_page"] is None

    def test_manifest_write_error_fails_a_finished_run(self, cfg, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(runner_mod.os, "replace", boom)
        runner, model = make_runner(cfg)
        with pytest.raises(OSError, match="disk full"):
            runner.run()
        assert model.unload_calls == 1

    def test_manifest_write_error_does_not_mask_the_run_error(self, cfg, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(runner_mod.os, "replace", boom)
        runner, _ = make_runner(cfg, FakeModel(load_error=RuntimeError("weights")))
        with pytest.raises(RuntimeError, match="weights"):
            runner.run()

    def test_git_sha_from_a_branch_ref(self, tmp_path) -> None:
        sha = "a" * 40
        git = tmp_path / "repo" / ".git"
        (git / "refs/heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/v1.1\n", encoding="utf-8")
        (git / "refs/heads/v1.1").write_text(sha + "\n", encoding="utf-8")
        nested = tmp_path / "repo" / "src" / "pkg"
        nested.mkdir(parents=True)
        assert runner_mod._git_sha([nested]) == sha

    def test_git_sha_from_packed_refs(self, tmp_path) -> None:
        sha = "b" * 40
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{'c' * 40} refs/heads/other\n"
            f"{sha} refs/heads/main\n", encoding="utf-8")
        assert runner_mod._git_sha([tmp_path]) == sha

    def test_git_sha_detached_head(self, tmp_path) -> None:
        sha = "0123456789abcdef0123456789abcdef01234567"
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text(sha + "\n", encoding="utf-8")
        assert runner_mod._git_sha([tmp_path]) == sha

    def test_git_sha_gitdir_file(self, tmp_path) -> None:
        sha = "d" * 40
        real = tmp_path / "real-git"
        (real / "refs/heads").mkdir(parents=True)
        (real / "HEAD").write_text("ref: refs/heads/x\n", encoding="utf-8")
        (real / "refs/heads/x").write_text(sha, encoding="utf-8")
        work = tmp_path / "work"
        work.mkdir()
        (work / ".git").write_text("gitdir: ../real-git\n", encoding="utf-8")
        assert runner_mod._git_sha([work]) == sha

    def test_git_sha_none_without_checkout_or_with_garbage(self, tmp_path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert runner_mod._read_git_sha(plain) is None
        bad = tmp_path / "bad"
        (bad / ".git").mkdir(parents=True)
        (bad / ".git" / "HEAD").write_text("ref: refs/heads/missing\n", encoding="utf-8")
        assert runner_mod._git_sha([bad]) is None
        broken = tmp_path / "broken"
        (broken / ".git").mkdir(parents=True)  # no HEAD file -> OSError -> None
        assert runner_mod._git_sha([broken]) is None


# ---------------------------------------------------------------------------
# Progress and cost logging (§7.2 step 5, §11.4)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class TestProgress:
    def test_progress_and_cost_lines(self, tmp_path, dataset, monkeypatch, caplog) -> None:
        cfg = build_config(tmp_path, dataset, {"runtime": {"log_every_s": 5,
                                                           "gpu_hourly_cost": 0.69}})
        monkeypatch.setattr(runner_mod, "_monotonic", FakeClock(step=10.0))
        with caplog.at_level(logging.INFO, logger=runner_mod.logger.name):
            make_runner(cfg)[0].run()
        progress = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Progress:")]
        done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Done:")]
        assert progress and len(done) == 1
        assert "captions/s" in progress[0] and "images/h" in progress[0]
        assert "flagged" in progress[0] and "failed" in progress[0] and "ETA" in progress[0]
        assert "per 1k images" in progress[0] and "projected $" in progress[0]
        assert done[0].startswith("Done: 5/5 images")
        throughput = only_manifest(cfg)["throughput"]
        assert throughput["gpu_hourly_cost"] == 0.69
        assert throughput["usd_per_1k_images"] == pytest.approx(
            0.69 / throughput["images_per_hour"] * 1000, rel=1e-3)
        assert throughput["projected_usd_remaining"] == 0

    def test_no_cost_without_hourly_cost_and_no_progress_before_interval(
        self, cfg, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(runner_mod, "_monotonic", FakeClock(step=0.001))
        with caplog.at_level(logging.INFO, logger=runner_mod.logger.name):
            make_runner(cfg)[0].run()
        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if m.startswith("Progress:")]
        done = [m for m in messages if m.startswith("Done:")]
        assert len(done) == 1 and "$" not in done[0]

    def test_throughput_figures(self, cfg, monkeypatch) -> None:
        runner, _ = make_runner(cfg)
        runner._plan = Plan(
            shard=(0, 1), caption_types=["long"], images_discovered=160, images_in_shard=160,
            images_already_complete=0, images_deferred=10,
            items=[(Path(f"s/{i}.png"), ["long"]) for i in range(150)],
        )
        runner._counts.images_done = 100
        runner._counts.captions_ok = 100
        runner._t_start = 0.0
        runner._t_processing = 100.0
        runner._gpu_hourly_cost = 2.0
        monkeypatch.setattr(runner_mod, "_monotonic", lambda: 3700.0)
        stats = runner._throughput()
        assert stats["processing_s"] == 3600.0 and stats["elapsed_s"] == 3700.0
        assert stats["images_per_hour"] == 100.0
        assert stats["captions_per_s"] == pytest.approx(100 / 3600, abs=1e-4)
        assert stats["images_remaining"] == 50
        assert stats["eta_s"] == 1800.0
        assert stats["cost_usd"] == pytest.approx(2.0 * 3700 / 3600, abs=1e-4)
        assert stats["usd_per_1k_images"] == 20.0
        assert stats["projected_usd_remaining"] == 1.0
        assert stats["projected_usd_remaining_incl_deferred"] == 1.2

    def test_duration_format(self) -> None:
        assert runner_mod._fmt_duration(3725.4) == "1:02:05"
        assert runner_mod._fmt_duration(0) == "0:00:00"


# ---------------------------------------------------------------------------
# Package boundaries and standards
# ---------------------------------------------------------------------------


class TestPackage:
    def test_exports(self) -> None:
        assert batch.__all__ == ["CaptionRunner"]
        assert batch.CaptionRunner is CaptionRunner
        assert SUMMARY_KEYS == (
            "run_id", "shard", "images_discovered", "images_in_shard",
            "images_already_complete", "images_deferred", "images_planned",
            "images_load_failed", "captions_planned", "captions_ok", "captions_flagged",
            "captions_failed", "retries", "interrupted", "output_dir",
        )

    def test_import_and_run_need_no_heavy_libraries(self, tmp_path) -> None:
        data = tmp_path / "data"
        make_image(data / "a/0001/000001.png")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "from pixel_art_auto_captioner.batch import CaptionRunner\n"
            "from pixel_art_auto_captioner.captioning.base import CaptionModel\n"
            "from pixel_art_auto_captioner.common.config import load_config\n"
            "from pixel_art_auto_captioner.common.types import CaptionResult\n"
            "from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader\n"
            "class M(CaptionModel):\n"
            "    model_name = 'joycaption-beta-one'; backend = 'vllm'\n"
            "    def load(self, config): pass\n"
            "    def caption_batch(self, rs): return [CaptionResult('A caption.', 'stop', 3) for r in rs]\n"
            "    def unload(self): pass\n"
            f"cfg = load_config(None, {{'input': {{'input_dir': {str(data)!r}}},\n"
            f"    'output': {{'output_dir': {str(tmp_path / 'out')!r}, 'qa_sample': 0}},\n"
            "    'prompts': {'sources_file': None}, 'runtime': {'num_workers': 0}})\n"
            "s = CaptionRunner(ImageDataLoader({**cfg['input'], **cfg['runtime']}), M(), cfg).run()\n"
            "assert s['captions_ok'] == 2, s\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm'))\n"
            "assert not bad, bad\n"
            "assert not [m for m in sys.modules if 'joycaption' in m]\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                              text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr

    def test_source_standards(self) -> None:
        source = Path(runner_mod.__file__).read_text(encoding="utf-8")
        assert "print(" not in source
        assert "\r\n" not in source
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for name in imported:
            assert name.split(".")[0] not in {"torch", "transformers", "vllm", "subprocess"}, name
            if name.startswith("pixel_art_auto_captioner."):
                assert name.split(".")[1] in {"common", "ingestion", "captioning"}, name
            assert "joycaption" not in name
        assert runner_mod.logger.name == "pixel_art_auto_captioner.batch.runner"

    def test_public_callables_have_docstrings(self) -> None:
        assert CaptionRunner.__doc__ and CaptionRunner.__init__.__doc__
        assert CaptionRunner.run.__doc__
        assert runner_mod.__doc__ and runner_mod._git_sha.__doc__
