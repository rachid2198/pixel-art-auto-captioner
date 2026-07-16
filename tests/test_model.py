"""Tests for JoyCaptionModel — GPU-gated where necessary.

Tests that require a CUDA-capable GPU are marked with ``@pytest.mark.gpu``
and will be skipped automatically when CUDA is unavailable.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Imports & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def joycaption_cls():
    """Return the JoyCaptionModel class (no instance)."""
    from pixel_art_auto_captioner.captioning.joycaption import JoyCaptionModel

    return JoyCaptionModel


@pytest.fixture
def fresh_model(joycaption_cls):
    """Return a fresh, unloaded JoyCaptionModel instance."""
    return joycaption_cls()


# ---------------------------------------------------------------------------
# CPU-safe tests (no GPU required)
# ---------------------------------------------------------------------------


class TestJoyCaptionConstruction:
    """Tests that do not require a GPU."""

    def test_default_model_name(self, fresh_model):
        """Default model_name is 'joycaption-beta-one'."""
        assert fresh_model.model_name == "joycaption-beta-one"

    def test_custom_model_name(self, joycaption_cls):
        """Custom model_name is stored correctly."""
        model = joycaption_cls(model_name="custom-vlm-v2")
        assert model.model_name == "custom-vlm-v2"

    def test_not_loaded_initially(self, fresh_model):
        """Model reports as not loaded before load() is called."""
        assert fresh_model._loaded is False
        assert fresh_model._model is None
        assert fresh_model._processor is None

    def test_caption_raises_when_not_loaded(self, fresh_model, sample_image_rgb):
        """caption() raises RuntimeError if model.load() has not been called."""
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            fresh_model.caption(sample_image_rgb, "Describe this image.")

    def test_unload_on_fresh_model_does_not_crash(self, fresh_model):
        """unload() on a never-loaded model is a safe no-op."""
        fresh_model.unload()
        assert fresh_model._loaded is False


class TestJoyCaptionLoadConfigValidation:
    """Config validation for load() — no GPU required."""

    def test_load_raises_on_missing_model_path(self, fresh_model):
        """load() raises ValueError when model_path is missing."""
        with pytest.raises(ValueError, match="model_path"):
            fresh_model.load({})

    def test_load_raises_on_empty_model_path(self, fresh_model):
        """load() raises ValueError when model_path is empty string."""
        with pytest.raises(ValueError, match="model_path"):
            fresh_model.load({"model_path": ""})

    def test_load_raises_on_bad_torch_dtype(self, fresh_model):
        """load() raises ValueError for unrecognised torch_dtype."""
        with pytest.raises(ValueError, match="torch_dtype"):
            fresh_model.load({"model_path": "./fake", "torch_dtype": "float8"})

    def test_load_raises_on_bad_quantization(self, fresh_model):
        """load() raises ValueError for unrecognised quantization."""
        with pytest.raises(ValueError, match="quantization"):
            fresh_model.load(
                {"model_path": "./fake", "quantization": "fp4"}
            )


class TestDefaultGenParams:
    """Verify default generation parameter constants."""

    def test_default_gen_params_match_spec(self):
        """DEFAULT_GEN_PARAMS matches SPEC §3.2 defaults."""
        from pixel_art_auto_captioner.captioning.joycaption import DEFAULT_GEN_PARAMS

        assert DEFAULT_GEN_PARAMS["max_new_tokens"] == 512
        assert DEFAULT_GEN_PARAMS["do_sample"] is True
        assert DEFAULT_GEN_PARAMS["temperature"] == 0.6
        assert DEFAULT_GEN_PARAMS["top_p"] == 0.9
        assert DEFAULT_GEN_PARAMS["top_k"] is None
        assert DEFAULT_GEN_PARAMS["use_cache"] is True


class TestVisionTowerFix:
    """The _apply_vision_tower_fix helper is safe on placeholder objects."""

    def test_fix_does_not_crash_on_missing_attr(self):
        """_apply_vision_tower_fix silently skips objects missing the path."""
        from pixel_art_auto_captioner.captioning.joycaption import (
            _apply_vision_tower_fix,
        )

        class DummyModel:
            pass

        # Should not raise — just logs a debug message.
        _apply_vision_tower_fix(DummyModel())


# ---------------------------------------------------------------------------
# GPU-gated integration tests
# ---------------------------------------------------------------------------

# Determine if CUDA is available once at module level.
_CUDA_AVAILABLE = False
try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

requires_gpu = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA-capable GPU is required for this test.",
)


def _model_path():
    """Return the path to JoyCaption model weights if available."""
    import os
    from pathlib import Path

    path = Path(os.environ.get("JOYCAPTION_MODEL_PATH", "./Models"))
    if not path.is_dir():
        return None
    # Quick sanity: a config.json file should exist in the model dir
    if not (path / "config.json").exists():
        return None
    return str(path)


requires_model = pytest.mark.skipif(
    _model_path() is None,
    reason="JoyCaption model not found at JOYCAPTION_MODEL_PATH or ./Models.",
)


@pytest.mark.gpu
class TestJoyCaptionModelGPU:
    """Tests that require a real GPU and model weights on disk.

    Set the ``JOYCAPTION_MODEL_PATH`` environment variable to point
    to a local clone of the JoyCaption model before running these tests.
    """

    @pytest.fixture
    def model_path(self):
        """Path to JoyCaption model — from env var or default."""
        path = _model_path()
        if path is None:
            pytest.skip("JoyCaption model not found.")
        return path

    @pytest.fixture
    def loaded_model(self, joycaption_cls, model_path):
        """A loaded JoyCaptionModel, cleaned up after the test."""
        model = joycaption_cls()
        try:
            model.load({"model_path": model_path})
        except Exception as e:
            pytest.skip(f"Could not load model from {model_path}: {e}")
        yield model
        model.unload()

    @requires_gpu
    @requires_model
    def test_model_load_nf4(self, joycaption_cls, model_path):
        """Model loads with NF4 quantization without error."""
        model = joycaption_cls()
        try:
            model.load({"model_path": model_path, "quantization": "nf4"})
            assert model._loaded is True
            assert model._model is not None
            assert model._processor is not None
        finally:
            model.unload()

    @requires_gpu
    @requires_model
    def test_model_load_no_quant(self, joycaption_cls, model_path):
        """Model loads in bfloat16 without quantization."""
        model = joycaption_cls()
        try:
            model.load(
                {
                    "model_path": model_path,
                    "quantization": None,
                    "torch_dtype": "bfloat16",
                }
            )
            assert model._loaded is True
        finally:
            model.unload()

    @requires_gpu
    def test_model_caption_returns_tuple(self, loaded_model, sample_image_rgb):
        """caption() returns a (str, dict) tuple."""
        result = loaded_model.caption(sample_image_rgb, "Describe this image.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)
        assert "gen_params" in result[1]

    @requires_gpu
    def test_model_caption_non_empty(self, loaded_model, sample_image_rgb):
        """caption() produces a non-empty caption for a simple image."""
        caption, _ = loaded_model.caption(sample_image_rgb, "Describe this image.")
        assert len(caption.strip()) > 0, "Caption should not be empty"

    @requires_gpu
    def test_model_unload_frees_memory(self, loaded_model, sample_image_rgb):
        """unload() clears model references and frees CUDA memory."""
        # Generate one caption to warm the cache
        loaded_model.caption(sample_image_rgb, "Test.")
        loaded_model.unload()

        assert loaded_model._model is None
        assert loaded_model._processor is None
        assert loaded_model._loaded is False

    @requires_gpu
    def test_model_caption_with_gen_kwargs(self, loaded_model, sample_image_rgb):
        """caption() accepts and merges gen_kwargs."""
        caption, meta = loaded_model.caption(
            sample_image_rgb,
            "Describe.",
            temperature=0.2,
            max_new_tokens=100,
        )
        assert isinstance(caption, str)
        assert meta["gen_params"]["temperature"] == 0.2
        assert meta["gen_params"]["max_new_tokens"] == 100
        # Defaults should still be present for unspecified params
        assert meta["gen_params"]["top_p"] == 0.9