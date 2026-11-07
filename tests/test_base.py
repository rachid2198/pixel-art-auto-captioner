"""Tests for ``captioning.base``: the ``CaptionModel`` interface and ``get_model``.

Covers SPEC §6.1 / §2.2 / D6:

- ``CaptionModel`` is abstract in exactly ``load``, ``caption_batch``, ``unload``;
  ``caption`` and ``describe`` have defaults.
- ``caption(request)`` is ``caption_batch([request])[0]``: one call, a
  one-element list, the same result object; systemic exceptions propagate;
  a wrong result count is a contract violation (``RuntimeError``).
- The batch contract (``len(results) == len(requests)``, same order, per-request
  errors returned not raised) on a reference fake.
- ``describe()`` defaults.
- ``get_model``: unknown backend / model name, lazy import of only the selected
  backend module (fake modules injected into ``sys.modules``), import errors,
  non-``CaptionModel`` classes, fresh unloaded instances.
- Importing ``captioning`` / ``captioning.base`` pulls in no torch,
  transformers, vllm, backend module, ``ingestion`` or ``batch`` (subprocess
  without any test stub on ``PYTHONPATH``).

No real model library is imported.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import PIL.Image
import pytest

import pixel_art_auto_captioner.captioning as captioning
from pixel_art_auto_captioner.captioning import base
from pixel_art_auto_captioner.captioning.base import CaptionModel, get_model
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

SRC = Path(__file__).resolve().parent.parent / "src"
HF_MODULE = "pixel_art_auto_captioner.captioning.joycaption_hf"
VLLM_MODULE = "pixel_art_auto_captioner.captioning.joycaption_vllm"
ABSTRACT_METHODS = {"load", "caption_batch", "unload"}


# ---------------------------------------------------------------------------
# Helpers and fakes
# ---------------------------------------------------------------------------


def _request(i: int = 0, caption_type: str = "long") -> CaptionRequest:
    return CaptionRequest(
        image_rel_path=Path(f"celeste/0001/{i:06d}.jpg"),
        caption_type=caption_type,
        image=PIL.Image.new("RGB", (8, 8), color=(i % 256, 0, 0)),
        prompt=f"prompt {i}",
        gen_params={"max_new_tokens": 16, "temperature": 0.6, "top_p": 0.9, "top_k": None},
        seed=1000 + i,
    )


class EchoModel(CaptionModel):
    """Reference fake: one result per request, in order; records every call.

    A request whose prompt contains ``"bad"`` gets an error result (per-request
    problem, returned not raised).
    """

    model_name = "joycaption-beta-one"
    backend = "hf"

    def __init__(self) -> None:
        self.calls: list[list[CaptionRequest]] = []
        self.loaded_with: dict | None = None
        self.unload_calls = 0

    def load(self, config: dict) -> None:
        self.loaded_with = config

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        self.calls.append(requests)
        results = []
        for r in requests:
            if "bad" in r.prompt:
                results.append(CaptionResult("", "error", 0, error="ValueError: bad input"))
            else:
                results.append(CaptionResult(f"{r.image_rel_path.as_posix()}|{r.caption_type}|{r.seed}", "stop", 3))
        return results

    def unload(self) -> None:
        self.unload_calls += 1


class WrongCountModel(EchoModel):
    """Violates the contract by returning ``n`` results whatever the input."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        self.calls.append(requests)
        return [CaptionResult("x", "stop", 1) for _ in range(self.n)]


class SystemicFailureModel(EchoModel):
    """``caption_batch`` raises (e.g. CUDA OOM for the whole batch)."""

    class FakeOOM(RuntimeError):
        pass

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        self.calls.append(requests)
        raise self.FakeOOM("CUDA out of memory")


def _subclass_without(method: str) -> type:
    """A ``CaptionModel`` subclass implementing every abstract method except one."""
    ns = {name: (lambda self, *a, **k: None) for name in ABSTRACT_METHODS if name != method}
    ns.update(model_name="m", backend="hf")
    return type(f"Missing_{method}", (CaptionModel,), ns)


def _fake_backend_module(name: str, class_name: str, *, backend: str) -> tuple[types.ModuleType, list]:
    """Build a fake backend module whose class records constructor calls."""
    constructed: list = []

    class FakeBackend(EchoModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            constructed.append((args, kwargs))

    FakeBackend.backend = backend
    FakeBackend.__name__ = class_name
    module = types.ModuleType(name)
    setattr(module, class_name, FakeBackend)
    return module, constructed


@pytest.fixture
def fake_hf(monkeypatch):
    module, constructed = _fake_backend_module(HF_MODULE, "JoyCaptionHFModel", backend="hf")
    monkeypatch.setitem(sys.modules, HF_MODULE, module)
    return module, constructed


@pytest.fixture
def fake_vllm(monkeypatch):
    module, constructed = _fake_backend_module(VLLM_MODULE, "JoyCaptionVLLMModel", backend="vllm")
    monkeypatch.setitem(sys.modules, VLLM_MODULE, module)
    return module, constructed


# ---------------------------------------------------------------------------
# 1. Abstractness
# ---------------------------------------------------------------------------


class TestAbstractInterface:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            CaptionModel()  # type: ignore[abstract]

    def test_abstract_methods_are_exactly_load_caption_batch_unload(self) -> None:
        assert set(CaptionModel.__abstractmethods__) == ABSTRACT_METHODS

    def test_caption_and_describe_are_concrete(self) -> None:
        assert not getattr(CaptionModel.caption, "__isabstractmethod__", False)
        assert not getattr(CaptionModel.describe, "__isabstractmethod__", False)

    @pytest.mark.parametrize("method", sorted(ABSTRACT_METHODS))
    def test_missing_any_abstract_method_cannot_instantiate(self, method: str) -> None:
        cls = _subclass_without(method)
        with pytest.raises(TypeError, match=method):
            cls()

    def test_implementing_the_three_abstract_methods_is_enough(self) -> None:
        model = EchoModel()
        assert isinstance(model, CaptionModel)
        assert model.model_name == "joycaption-beta-one"
        assert model.backend == "hf"

    def test_v1_caption_signature_no_longer_satisfies_the_interface(self) -> None:
        """A v1.0-style backend (``caption(image, prompt)``, no ``caption_batch``) is abstract."""

        class V10Style(CaptionModel):
            model_name = "old"

            def load(self, config: dict) -> None: ...

            def caption(self, image, prompt, **gen_kwargs):  # type: ignore[override]
                return "x", {}

            def unload(self) -> None: ...

        with pytest.raises(TypeError, match="caption_batch"):
            V10Style()

    def test_signatures_match_spec(self) -> None:
        def params(fn):
            return list(inspect.signature(fn).parameters)

        assert params(CaptionModel.load) == ["self", "config"]
        assert params(CaptionModel.caption_batch) == ["self", "requests"]
        assert params(CaptionModel.unload) == ["self"]
        assert params(CaptionModel.caption) == ["self", "request"]
        assert params(CaptionModel.describe) == ["self"]
        sig = inspect.signature(get_model)
        assert list(sig.parameters) == ["backend", "model_name"]
        assert sig.parameters["model_name"].default == "joycaption-beta-one"

    def test_class_attributes_are_declared(self) -> None:
        assert CaptionModel.__annotations__["model_name"] in (str, "str")
        assert CaptionModel.__annotations__["backend"] in (str, "str")

    def test_load_and_unload_on_a_fake(self) -> None:
        model = EchoModel()
        model.unload()  # before load: allowed by the contract
        model.load({"backend": "hf", "model_path": "/fake"})
        assert model.loaded_with == {"backend": "hf", "model_path": "/fake"}
        model.unload()
        model.unload()
        assert model.unload_calls == 3


# ---------------------------------------------------------------------------
# 2. caption() delegates to caption_batch()
# ---------------------------------------------------------------------------


class TestCaptionDelegation:
    def test_caption_calls_caption_batch_once_with_a_one_element_list(self) -> None:
        model = EchoModel()
        req = _request(7)
        model.caption(req)
        assert len(model.calls) == 1
        assert isinstance(model.calls[0], list)
        assert len(model.calls[0]) == 1 and model.calls[0][0] is req

    def test_caption_returns_the_batch_result_object(self) -> None:
        sentinel = CaptionResult("hello", "stop", 1)

        class Fixed(EchoModel):
            def caption_batch(self, requests):
                return [sentinel]

        assert Fixed().caption(_request()) is sentinel

    def test_caption_returns_error_results_without_raising(self) -> None:
        req = _request()
        req.prompt = "bad prompt"
        result = EchoModel().caption(req)
        assert result.finish_reason == "error"
        assert result.error == "ValueError: bad input"

    def test_systemic_exception_propagates_unchanged(self) -> None:
        model = SystemicFailureModel()
        with pytest.raises(SystemicFailureModel.FakeOOM, match="out of memory"):
            model.caption(_request())
        assert len(model.calls) == 1

    @pytest.mark.parametrize("n", [0, 2, 5])
    def test_wrong_result_count_is_a_contract_violation(self, n: int) -> None:
        with pytest.raises(RuntimeError, match=f"returned {n} results for 1 request"):
            WrongCountModel(n).caption(_request())

    def test_caption_uses_an_overridden_caption_batch(self) -> None:
        """The default ``caption`` is built on ``caption_batch`` (no separate path)."""

        class Upper(EchoModel):
            def caption_batch(self, requests):
                return [CaptionResult(r.prompt.upper(), "length", 2) for r in requests]

        result = Upper().caption(_request(3))
        assert (result.text, result.finish_reason, result.num_output_tokens) == ("PROMPT 3", "length", 2)

    def test_subclass_may_override_caption(self) -> None:
        class Own(EchoModel):
            def caption(self, request):
                return CaptionResult("own", "stop", 0)

        model = Own()
        assert model.caption(_request()).text == "own"
        assert model.calls == []


# ---------------------------------------------------------------------------
# 3. Batch contract on the reference fake
# ---------------------------------------------------------------------------


class TestBatchContract:
    @pytest.mark.parametrize("n", [0, 1, 4, 17])
    def test_one_result_per_request_in_order(self, n: int) -> None:
        requests = [_request(i, "long" if i % 2 else "short") for i in range(n)]
        results = EchoModel().caption_batch(requests)
        assert len(results) == len(requests)
        for req, res in zip(requests, results):
            assert res.text == f"{req.image_rel_path.as_posix()}|{req.caption_type}|{req.seed}"

    def test_caption_equals_the_matching_batch_element(self) -> None:
        requests = [_request(i) for i in range(5)]
        model = EchoModel()
        batch = model.caption_batch(requests)
        assert [model.caption(r) for r in requests] == batch

    def test_per_request_error_does_not_affect_neighbours(self) -> None:
        requests = [_request(i) for i in range(3)]
        requests[1].prompt = "bad"
        results = EchoModel().caption_batch(requests)
        assert [r.finish_reason for r in results] == ["stop", "error", "stop"]
        assert results[1].error and results[0].error is None and results[2].error is None


# ---------------------------------------------------------------------------
# 4. describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_default_describe(self) -> None:
        assert EchoModel().describe() == {"model_name": "joycaption-beta-one", "backend": "hf"}

    def test_describe_returns_a_new_dict_each_call(self) -> None:
        model = EchoModel()
        first = model.describe()
        first["model_name"] = "changed"
        assert model.describe()["model_name"] == "joycaption-beta-one"
        assert model.describe() is not model.describe()

    def test_describe_without_attributes_gives_none(self) -> None:
        class Bare(CaptionModel):
            def load(self, config): ...
            def caption_batch(self, requests): return []
            def unload(self): ...

        assert Bare().describe() == {"model_name": None, "backend": None}

    def test_describe_reflects_instance_attributes(self) -> None:
        model = EchoModel()
        model.backend = "vllm"
        assert model.describe()["backend"] == "vllm"

    def test_subclass_extends_describe(self) -> None:
        class Ext(EchoModel):
            def describe(self):
                return {**super().describe(), "model_path": "/m", "revision": None, "dtype": "bfloat16"}

        assert Ext().describe() == {
            "model_name": "joycaption-beta-one", "backend": "hf",
            "model_path": "/m", "revision": None, "dtype": "bfloat16",
        }


# ---------------------------------------------------------------------------
# 5. get_model()
# ---------------------------------------------------------------------------


class TestGetModel:
    @pytest.mark.parametrize("bad", ["", "HF", "Vllm", "openai", "joycaption", " hf", None, 1, ["hf"]])
    def test_unknown_backend_raises_value_error(self, bad, monkeypatch) -> None:
        imported: list[str] = []
        monkeypatch.setattr(base.importlib, "import_module", lambda name: imported.append(name))
        with pytest.raises(ValueError, match=r"unknown backend .*\['hf', 'vllm'\]"):
            get_model(bad)
        assert imported == []

    @pytest.mark.parametrize("bad", ["", "qwen3-vl-8b", "JoyCaption-Beta-One", None])
    def test_unknown_model_name_raises_value_error(self, bad, monkeypatch) -> None:
        imported: list[str] = []
        monkeypatch.setattr(base.importlib, "import_module", lambda name: imported.append(name))
        with pytest.raises(ValueError, match=r"unknown model_name .*joycaption-beta-one"):
            get_model("hf", bad)
        assert imported == []

    def test_hf_returns_fresh_unloaded_instance(self, fake_hf) -> None:
        module, constructed = fake_hf
        model = get_model("hf")
        assert isinstance(model, module.JoyCaptionHFModel)
        assert isinstance(model, CaptionModel)
        assert model.backend == "hf" and model.model_name == "joycaption-beta-one"
        assert model.loaded_with is None  # get_model never calls load()
        assert constructed == [((), {})]  # constructed with no arguments

    def test_vllm_returns_fresh_unloaded_instance(self, fake_vllm) -> None:
        module, constructed = fake_vllm
        model = get_model("vllm", "joycaption-beta-one")
        assert isinstance(model, module.JoyCaptionVLLMModel)
        assert model.backend == "vllm" and model.loaded_with is None
        assert constructed == [((), {})]

    def test_each_call_returns_a_new_instance(self, fake_hf) -> None:
        assert get_model("hf") is not get_model("hf")

    def test_imports_only_the_selected_backend_module(self, fake_hf, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, VLLM_MODULE, raising=False)
        imported: list[str] = []
        real = base.importlib.import_module

        def spy(name, *a, **k):
            imported.append(name)
            return real(name, *a, **k)

        monkeypatch.setattr(base.importlib, "import_module", spy)
        get_model("hf")
        assert imported == [HF_MODULE]
        assert VLLM_MODULE not in sys.modules

    @pytest.mark.parametrize("backend, module_name", [("hf", HF_MODULE), ("vllm", VLLM_MODULE)])
    def test_import_failure_becomes_import_error_naming_the_backend(self, backend, module_name, monkeypatch) -> None:
        def fail(name, *a, **k):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        monkeypatch.setattr(base.importlib, "import_module", fail)
        with pytest.raises(ImportError, match=rf"'{backend}' backend module '{module_name}'.*torch") as info:
            get_model(backend)
        assert isinstance(info.value.__cause__, ModuleNotFoundError)

    def test_blocked_module_in_sys_modules_gives_import_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, HF_MODULE, None)  # import of this name raises ImportError
        with pytest.raises(ImportError, match="'hf' backend"):
            get_model("hf")

    def test_module_without_the_class_gives_import_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, VLLM_MODULE, types.ModuleType(VLLM_MODULE))
        with pytest.raises(ImportError, match="has no class 'JoyCaptionVLLMModel'"):
            get_model("vllm")

    @pytest.mark.parametrize("obj", [object, 42, lambda: None])
    def test_non_caption_model_class_gives_type_error(self, obj, monkeypatch) -> None:
        module = types.ModuleType(HF_MODULE)
        module.JoyCaptionHFModel = obj
        monkeypatch.setitem(sys.modules, HF_MODULE, module)
        with pytest.raises(TypeError, match="not a CaptionModel subclass"):
            get_model("hf")

    def test_constructor_errors_propagate(self, monkeypatch) -> None:
        class Broken(EchoModel):
            def __init__(self):
                raise RuntimeError("boom")

        module = types.ModuleType(HF_MODULE)
        module.JoyCaptionHFModel = Broken
        monkeypatch.setitem(sys.modules, HF_MODULE, module)
        with pytest.raises(RuntimeError, match="boom"):
            get_model("hf")


# ---------------------------------------------------------------------------
# 6. Package surface, laziness, boundaries
# ---------------------------------------------------------------------------


class TestPackage:
    def test_package_exports_only_caption_model_and_get_model(self) -> None:
        assert captioning.__all__ == ["CaptionModel", "get_model"]
        assert captioning.CaptionModel is CaptionModel
        assert captioning.get_model is get_model
        assert not hasattr(captioning, "JoyCaptionModel")

    @pytest.mark.parametrize(
        "module", ["pixel_art_auto_captioner.captioning", "pixel_art_auto_captioner.captioning.base"]
    )
    def test_import_pulls_in_no_heavy_backend_or_upper_layer_modules(self, module: str) -> None:
        """Clean interpreter, ``PYTHONPATH`` replaced so no torch stub is visible."""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(SRC)
        code = (
            f"import sys, {module}; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm') "
            "or m.startswith(('pixel_art_auto_captioner.captioning.joycaption', "
            "'pixel_art_auto_captioner.ingestion', 'pixel_art_auto_captioner.batch'))); "
            "assert not bad, bad"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_get_model_with_a_fake_backend_needs_no_torch(self, tmp_path: Path) -> None:
        """End to end in a clean interpreter: a fake backend module, no torch anywhere."""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys, types\n"
            "from pixel_art_auto_captioner.captioning import CaptionModel, get_model\n"
            "class Fake(CaptionModel):\n"
            "    model_name = 'joycaption-beta-one'; backend = 'vllm'\n"
            "    def load(self, config): pass\n"
            "    def caption_batch(self, requests): return []\n"
            "    def unload(self): pass\n"
            f"m = types.ModuleType({VLLM_MODULE!r}); m.JoyCaptionVLLMModel = Fake\n"
            f"sys.modules[{VLLM_MODULE!r}] = m\n"
            "model = get_model('vllm')\n"
            "assert type(model) is Fake, type(model)\n"
            "bad = sorted(x for x in sys.modules if x.split('.')[0] in ('torch', 'transformers', 'vllm'))\n"
            "assert not bad, bad\n"
            f"assert {HF_MODULE!r} not in sys.modules\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize("path", [base.__file__, captioning.__file__])
    def test_no_heavy_top_level_imports_or_print_in_source(self, path: str) -> None:
        source = Path(path).read_text(encoding="utf-8")
        assert "print(" not in source
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert not any(lib in stripped for lib in ("torch", "transformers", "vllm")), stripped
                assert "ingestion" not in stripped and "batch" not in stripped, stripped
                assert "joycaption" not in stripped, stripped

    def test_base_uses_module_logger(self) -> None:
        assert base.logger.name == "pixel_art_auto_captioner.captioning.base"
