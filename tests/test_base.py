"""Tests for ``captioning.base.CaptionModel`` ABC.

Verifies that the abstract base class enforces its interface contract:
- Direct instantiation is forbidden.
- Concrete subclasses must implement all three abstract methods.
- Well-formed subclasses behave as expected.
"""

import PIL.Image
import pytest

from pixel_art_auto_captioner.captioning.base import CaptionModel


# ---------------------------------------------------------------------------
# Minimal concrete subclass for interface testing
# ---------------------------------------------------------------------------


class _MinimalCaptionModel(CaptionModel):
    """A fully-implemented concrete subclass used only for testing."""

    def __init__(self, model_name: str = "minimal-test-model") -> None:
        self.model_name = model_name

    def load(self, config: dict) -> None:
        pass

    def caption(
        self, image: PIL.Image.Image, prompt: str, **gen_kwargs
    ) -> tuple[str, dict]:
        return ("a synthetic caption", {"gen_params": gen_kwargs})

    def unload(self) -> None:
        pass


class _IncompleteModel(CaptionModel):
    """A subclass missing ``unload`` — should fail to instantiate."""

    model_name = "incomplete"

    def load(self, config: dict) -> None:
        pass

    def caption(
        self, image: PIL.Image.Image, prompt: str, **gen_kwargs
    ) -> tuple[str, dict]:
        return ("incomplete", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptionModelABC:
    """Verify that ``CaptionModel`` enforces its abstract contract."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Instantiating the ABC itself must raise ``TypeError``."""
        with pytest.raises(TypeError):
            CaptionModel()  # type: ignore[abstract]

    def test_can_instantiate_concrete_subclass(self) -> None:
        """A subclass implementing all abstract methods is instantiable."""
        model = _MinimalCaptionModel()
        assert isinstance(model, CaptionModel)
        assert model.model_name == "minimal-test-model"

    def test_missing_abstract_method_raises(self) -> None:
        """Subclass missing *any* abstract method cannot be instantiated."""
        with pytest.raises(TypeError):
            _IncompleteModel()  # type: ignore[abstract]

    def test_model_name_is_accessible(self) -> None:
        """``model_name`` is a readable attribute on concrete instances."""
        model = _MinimalCaptionModel(model_name="custom-vlm-42")
        assert model.model_name == "custom-vlm-42"

    def test_caption_signature_accepts_kwargs(self) -> None:
        """``caption()`` must accept and forward ``**gen_kwargs``."""
        model = _MinimalCaptionModel()
        caption, meta = model.caption(
            PIL.Image.new("RGB", (16, 16)),
            "Describe this.",
            temperature=0.5,
            max_new_tokens=256,
        )
        assert isinstance(caption, str)
        assert caption == "a synthetic caption"
        assert meta["gen_params"] == {"temperature": 0.5, "max_new_tokens": 256}

    def test_caption_returns_tuple_of_str_and_dict(self) -> None:
        """Return type must be ``tuple[str, dict]``."""
        model = _MinimalCaptionModel()
        result = model.caption(PIL.Image.new("RGB", (8, 8)), "prompt")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_load_and_unload_are_callable(self) -> None:
        """``load()`` and ``unload()`` execute without error on a stub."""
        model = _MinimalCaptionModel()
        model.load({"model_path": "/fake/path"})
        model.unload()
        # If we got here without exception, the interface is satisfied.

    def test_custom_model_name_persists_after_load(self) -> None:
        """``model_name`` must survive ``load()`` / ``unload()`` cycles."""
        model = _MinimalCaptionModel(model_name="persistent-vlm")
        model.load({})
        assert model.model_name == "persistent-vlm"
        model.unload()
        assert model.model_name == "persistent-vlm"