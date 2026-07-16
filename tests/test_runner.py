"""Tests for CaptionRunner — integration tests with a mock model.

Uses a ``MockCaptionModel`` that returns deterministic captions
without requiring a GPU, so the full orchestration pipeline can be
exercised on CPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import PIL.Image
import pytest

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader


# ---------------------------------------------------------------------------
# Mock model — no GPU required
# ---------------------------------------------------------------------------


class MockCaptionModel(CaptionModel):
    """A deterministic fake captioner for integration tests.

    Returns ``"Caption for {stem}"`` for every image and tracks
    load/unload calls.
    """

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name = model_name
        self.load_called = False
        self.unload_called = False
        self.caption_call_count = 0

    def load(self, config: dict) -> None:
        self.load_called = True
        # Accept but ignore config; no real loading needed.

    def caption(
        self, image: PIL.Image.Image, prompt: str, **gen_kwargs
    ) -> tuple[str, dict]:
        self.caption_call_count += 1
        # Use image dimensions to generate a deterministic-but-unique caption
        caption = f"Mock caption (size={image.size[0]}x{image.size[1]})"
        metadata = {"gen_params": gen_kwargs}
        return caption, metadata

    def unload(self) -> None:
        self.unload_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """A fresh MockCaptionModel instance."""
    return MockCaptionModel()


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 3 synthetic PNG images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        img = PIL.Image.new("RGB", (32 + i * 8, 32 + i * 8), color=(i * 80, 0, 0))
        img.save(img_dir / f"image_{i}.png")
    return img_dir


@pytest.fixture
def dataloader_config(temp_image_dir: Path, tmp_path: Path) -> dict:
    """Minimal dataloader config for a temp image set."""
    return {
        "source_dirs": [str(temp_image_dir)],
        "extensions": [".png"],
        "recursive": True,
        "max_images": None,
        "skip_existing": False,
        "output_dir": str(tmp_path / "output"),
        "image_size": None,
    }


@pytest.fixture
def dataloader(dataloader_config: dict) -> ImageDataLoader:
    """An ImageDataLoader pointing at the temp image directory."""
    return ImageDataLoader(dataloader_config)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Output directory for runner tests."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def runner_config(output_dir: Path) -> dict:
    """Minimal runner config."""
    return {
        "output_dir": str(output_dir),
        "output_formats": ["txt", "jsonl"],
        "prompt_template": "Describe this pixel art image.",
        "generation_params": {"temperature": 0.6, "max_new_tokens": 100},
        "resume": False,
    }


@pytest.fixture
def runner(dataloader, mock_model, runner_config):
    """A fully wired CaptionRunner ready for testing."""
    from pixel_art_auto_captioner.batch.runner import CaptionRunner

    return CaptionRunner(dataloader, mock_model, runner_config)


# ---------------------------------------------------------------------------
# Config validation (no model needed)
# ---------------------------------------------------------------------------


class TestCaptionRunnerConfig:
    """Config validation tests — no runner execution required."""

    def test_missing_output_dir_raises(self, dataloader, mock_model):
        """Runner raises ValueError when output_dir is missing."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="output_dir"):
            CaptionRunner(
                dataloader,
                mock_model,
                {"prompt_template": "test"},
            )

    def test_empty_output_dir_raises(self, dataloader, mock_model):
        """Runner raises ValueError when output_dir is empty string."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="output_dir"):
            CaptionRunner(
                dataloader,
                mock_model,
                {"output_dir": "", "prompt_template": "test"},
            )

    def test_missing_prompt_template_raises(self, dataloader, mock_model):
        """Runner raises ValueError when prompt_template is missing."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="prompt_template"):
            CaptionRunner(
                dataloader,
                mock_model,
                {"output_dir": "/tmp"},
            )

    def test_empty_prompt_template_raises(self, dataloader, mock_model):
        """Runner raises ValueError when prompt_template is empty."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="prompt_template"):
            CaptionRunner(
                dataloader,
                mock_model,
                {"output_dir": "/tmp", "prompt_template": ""},
            )

    # -- Critique 4: output_formats validation --------------------------------

    def test_output_formats_empty_list_raises(self, dataloader, mock_model, output_dir):
        """Runner raises ValueError when output_formats is an empty list."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="output_formats"):
            CaptionRunner(
                dataloader,
                mock_model,
                {
                    "output_dir": str(output_dir),
                    "prompt_template": "Test.",
                    "output_formats": [],
                },
            )

    def test_output_formats_invalid_value_raises(self, dataloader, mock_model, output_dir):
        """Runner raises ValueError when output_formats contains an invalid value."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        with pytest.raises(ValueError, match="Unsupported output format"):
            CaptionRunner(
                dataloader,
                mock_model,
                {
                    "output_dir": str(output_dir),
                    "prompt_template": "Test.",
                    "output_formats": ["txt", "yaml"],
                },
            )

    # -- Critique 3: resume consistency warning -------------------------------

    def test_resume_true_warns_on_skip_existing_false(
        self, temp_image_dir, tmp_path, output_dir, caplog
    ):
        """Runner logs a warning when resume=True but dataloader.skip_existing=False."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        dl_cfg = {
            "source_dirs": [str(temp_image_dir)],
            "extensions": [".png"],
            "recursive": True,
            "max_images": None,
            "skip_existing": False,
            "output_dir": str(tmp_path / "output"),
            "image_size": None,
        }
        dl = ImageDataLoader(dl_cfg)

        with caplog.at_level(logging.WARNING):
            CaptionRunner(
                dl,
                MockCaptionModel(),
                {
                    "output_dir": str(output_dir),
                    "prompt_template": "Test.",
                    "resume": True,
                },
            )

        assert "resume=True" in caplog.text
        assert "skip_existing=False" in caplog.text

    def test_resume_true_no_warning_when_skip_existing_true(
        self, temp_image_dir, tmp_path, output_dir, caplog
    ):
        """No warning when resume=True and dataloader.skip_existing=True."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        dl_cfg = {
            "source_dirs": [str(temp_image_dir)],
            "extensions": [".png"],
            "recursive": True,
            "max_images": None,
            "skip_existing": True,
            "output_dir": str(tmp_path / "output"),
            "image_size": None,
        }
        dl = ImageDataLoader(dl_cfg)

        with caplog.at_level(logging.WARNING):
            CaptionRunner(
                dl,
                MockCaptionModel(),
                {
                    "output_dir": str(output_dir),
                    "prompt_template": "Test.",
                    "resume": True,
                },
            )

        assert "resume=True" not in caplog.text

    def test_default_output_formats(self, dataloader, mock_model, output_dir):
        """output_formats defaults to ['txt', 'jsonl'] when not specified."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        runner = CaptionRunner(
            dataloader,
            mock_model,
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test prompt.",
            },
        )
        assert runner.output_formats == ["txt", "jsonl"]

    def test_resume_defaults_to_true(self, dataloader, mock_model, output_dir):
        """resume defaults to True when not specified."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        runner = CaptionRunner(
            dataloader,
            mock_model,
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
            },
        )
        assert runner.resume is True

    def test_generation_params_defaults_to_empty_dict(
        self, dataloader, mock_model, output_dir
    ):
        """generation_params defaults to {} when not specified."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        runner = CaptionRunner(
            dataloader,
            mock_model,
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
            },
        )
        assert runner.generation_params == {}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCaptionRunnerIntegration:
    """End-to-end tests using MockCaptionModel."""

    def test_runner_integration(self, runner, output_dir, mock_model):
        """Full run processes all images, writes txt + jsonl output."""
        summary = runner.run()

        # -- summary counts -----------------------------------------------
        assert summary["total"] == 3
        assert summary["succeeded"] == 3
        assert summary["failed"] == 0
        assert summary["skipped"] == 0
        assert summary["output_dir"] == str(output_dir)

        # -- model lifecycle -----------------------------------------------
        assert mock_model.load_called is True
        assert mock_model.unload_called is True
        assert mock_model.caption_call_count == 3

        # -- output files --------------------------------------------------
        for i in range(3):
            txt_path = output_dir / f"image_{i}.txt"
            assert txt_path.exists(), f"Expected {txt_path} to exist"
            content = txt_path.read_text(encoding="utf-8")
            assert "Mock caption" in content

        jsonl_path = output_dir / "captions.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record["model_name"] == "mock-model"
            assert record["prompt_template"] == "Describe this pixel art image."
            assert "image_path" in record
            assert "timestamp_utc" in record

    def test_runner_summary_counts(self, runner):
        """Summary dict has correct structure and counts."""
        summary = runner.run()
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {
            "total",
            "succeeded",
            "failed",
            "skipped",
            "output_dir",
        }
        assert summary["succeeded"] + summary["failed"] + summary["skipped"] == summary["total"]

    def test_runner_resume(
        self, temp_image_dir, dataloader_config, mock_model, output_dir, tmp_path
    ):
        """Second run with resume=True skips images that already have output."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        # -- first run: disable resume, process everything -----------------
        cfg1 = dict(dataloader_config)
        cfg1["output_dir"] = str(output_dir)
        cfg1["skip_existing"] = False
        dl1 = ImageDataLoader(cfg1)

        runner1_config = {
            "output_dir": str(output_dir),
            "output_formats": ["txt"],
            "prompt_template": "Test.",
            "resume": False,
        }
        runner1 = CaptionRunner(dl1, MockCaptionModel(), runner1_config)
        summary1 = runner1.run()
        assert summary1["succeeded"] == 3

        # -- second run: enable resume/skip_existing -----------------------
        cfg2 = dict(dataloader_config)
        cfg2["output_dir"] = str(output_dir)
        cfg2["skip_existing"] = True
        dl2 = ImageDataLoader(cfg2)

        runner2_config = {
            "output_dir": str(output_dir),
            "output_formats": ["txt"],
            "prompt_template": "Test.",
            "resume": True,
        }
        runner2 = CaptionRunner(dl2, MockCaptionModel(model_name="mock-model-v2"), runner2_config)
        summary2 = runner2.run()
        # All 3 images should be skipped (already have .txt sidecars)
        assert summary2["skipped"] == 3
        assert summary2["succeeded"] == 0
        assert summary2["total"] == 3

    def test_runner_respects_output_formats(
        self, temp_image_dir, dataloader_config, mock_model, tmp_path
    ):
        """Only txt output is written when output_formats=['txt']."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        out = tmp_path / "txt_only_output"
        out.mkdir()

        cfg = dict(dataloader_config)
        cfg["output_dir"] = str(out)
        cfg["skip_existing"] = False
        dl = ImageDataLoader(cfg)

        runner_cfg = {
            "output_dir": str(out),
            "output_formats": ["txt"],
            "prompt_template": "Test prompt.",
            "resume": False,
        }
        r = CaptionRunner(dl, MockCaptionModel(), runner_cfg)
        summary = r.run()

        assert summary["succeeded"] == 3
        # txt sidecars should exist
        for i in range(3):
            assert (out / f"image_{i}.txt").exists()
        # JSONL should NOT exist
        assert not (out / "captions.jsonl").exists()

    def test_runner_model_unload_guaranteed(
        self, dataloader, mock_model, output_dir
    ):
        """Model.unload() is called even when a caption() raises."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        class FailingModel(MockCaptionModel):
            def caption(self, image, prompt, **gen_kwargs):
                self.caption_call_count += 1
                raise RuntimeError("Simulated GPU error")

        runner = CaptionRunner(
            dataloader,
            FailingModel(),
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
                "resume": False,
            },
        )
        summary = runner.run()

        assert summary["total"] == 3
        assert summary["succeeded"] == 0
        assert summary["failed"] == 3
        # unload must have been called despite the failures
        assert runner._model.unload_called is True

    # -- Critique 1 & 2: model.load() failure must propagate (not swallow) --

    def test_runner_model_load_failure_propagates(
        self, dataloader, output_dir
    ):
        """model.load() failure propagates exception (fail-fast, SPEC §12.1)."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        class LoadFailingModel(MockCaptionModel):
            def load(self, config):
                raise RuntimeError("OOM during load")

        runner = CaptionRunner(
            dataloader,
            LoadFailingModel(),
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
                "resume": False,
            },
        )
        with pytest.raises(RuntimeError, match="OOM during load"):
            runner.run()

    def test_runner_model_load_failure_still_unloads(
        self, dataloader, output_dir
    ):
        """unload() is called even when model.load() raises (partial alloc)."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        class PartialLoadModel(MockCaptionModel):
            def load(self, config):
                self.load_called = True
                raise RuntimeError("OOM after partial alloc")

        model = PartialLoadModel()
        runner = CaptionRunner(
            dataloader,
            model,
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
                "resume": False,
            },
        )
        with pytest.raises(RuntimeError):
            runner.run()

        # unload must STILL be called (Critique 2 fix)
        assert model.unload_called is True

    # -- Critique 5: iterator load-drop invariant ---------------------------

    def test_runner_iterator_load_drop_tracks_failed(
        self, temp_image_dir, output_dir
    ):
        """Shortfall from iterator load drops is counted as failed."""
        from pixel_art_auto_captioner.batch.runner import CaptionRunner

        # Dataloader whose iterator drops every other image
        class DroppingDataLoader(ImageDataLoader):
            def __iter__(self):
                paths = self._get_filtered_paths()
                total = len(paths)
                for idx, p in enumerate(paths, start=1):
                    if idx % 2 == 0:
                        # Simulate load failure — skip without yielding
                        continue
                    yield self.load(p)

        dl = DroppingDataLoader(
            {
                "source_dirs": [str(temp_image_dir)],
                "extensions": [".png"],
                "recursive": True,
                "max_images": None,
                "skip_existing": False,
            }
        )
        # 3 images, iterator yields only images 1 and 3 → 1 dropped
        runner = CaptionRunner(
            dl,
            MockCaptionModel(),
            {
                "output_dir": str(output_dir),
                "prompt_template": "Test.",
                "resume": False,
            },
        )
        summary = runner.run()

        assert summary["total"] == 3
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1  # the dropped image
        assert summary["succeeded"] + summary["failed"] + summary["skipped"] == summary["total"]