"""CPU tests for ``captioning/joycaption_vllm.py`` (SPEC §6.3, §13) using a fake vLLM.

No real ``vllm`` or ``torch`` is imported: a fake ``vllm`` module (``LLM``,
``SamplingParams``, ``RequestOutput``-like outputs) and a fake ``torch`` (only
``cuda.is_available`` / ``cuda.empty_cache``) are injected through the
constructor's keyword parameters (AGENTS.md §5). The fake ``LLM`` records the
constructor kwargs and every ``chat`` call, so the tests check the engine
config, the conversations, one ``SamplingParams`` per request, ordering,
``finish_reason`` mapping, error isolation and the unload lifecycle.

The last class holds GPU tests (``pytest.mark.gpu``) that need vLLM, CUDA and
the model weights (AGENTS.md §6.3 dual guard + vLLM installed); they are
skipped in the CPU suite.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import PIL.Image
import pytest

from pixel_art_auto_captioner.captioning import CaptionModel, get_model
from pixel_art_auto_captioner.captioning import joycaption_hf, joycaption_vllm
from pixel_art_auto_captioner.captioning.joycaption_vllm import (
    DEFAULT_GEN_PARAMS,
    DEFAULT_LOAD_CONFIG,
    LIMIT_MM_PER_PROMPT,
    SYSTEM_PROMPT,
    TOP_K_DISABLED,
    JoyCaptionVLLMModel,
    build_conversation,
)
from pixel_art_auto_captioner.common.config import DEFAULT_CONFIG
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

SRC = Path(__file__).resolve().parent.parent / "src"
REPO_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
HEAVY = ("torch", "transformers", "bitsandbytes", "vllm", "accelerate")
MODULE = "pixel_art_auto_captioner.captioning.joycaption_vllm"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSamplingParams:
    """Records its keyword arguments (vLLM's ``SamplingParams`` is keyword-based)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        error = FakeVLLM.sampling_error
        if error is not None and error[0](kwargs):
            raise error[1]

    def __repr__(self) -> str:
        return f"FakeSamplingParams({self.kwargs})"


class FakeCompletion:
    def __init__(self, text: Any, token_ids: Any, finish_reason: Any) -> None:
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class FakeRequestOutput:
    def __init__(self, request_id: str, outputs: list) -> None:
        self.request_id = request_id
        self.outputs = outputs


def _default_responder(conversation: list, params: FakeSamplingParams) -> FakeCompletion:
    """Echo the user prompt so each output identifies its conversation."""
    prompt = conversation[1]["content"][1]["text"]
    n = min(3, params.kwargs.get("max_tokens", 3))
    return FakeCompletion(f"  caption for: {prompt}\n", list(range(n)), "stop")


class FakeLLM:
    """Fake ``vllm.LLM``: records kwargs and ``chat`` calls, answers per conversation."""

    def __init__(self, owner: "FakeVLLM", **kwargs: Any) -> None:
        if owner.llm_error is not None:
            raise owner.llm_error
        self.owner = owner
        self.kwargs = kwargs
        self.chat_calls: list[dict] = []
        owner.engines.append(self)

    def chat(self, messages: Any, sampling_params: Any = None, use_tqdm: Any = True, **kwargs: Any) -> list:
        self.chat_calls.append(
            {"messages": messages, "sampling_params": sampling_params, "use_tqdm": use_tqdm, "kwargs": kwargs}
        )
        if self.owner.chat_error is not None:
            raise self.owner.chat_error
        if self.owner.raw_outputs is not None:
            return self.owner.raw_outputs(messages, sampling_params)
        outputs = []
        for i, (conversation, params) in enumerate(zip(messages, sampling_params)):
            reply = self.owner.responder(conversation, params)
            completions = reply if isinstance(reply, list) else [reply]
            outputs.append(FakeRequestOutput(str(len(self.owner.engines) * 1000 + i), completions))
        return outputs


class FakeVLLM:
    """Fake ``vllm`` module."""

    __version__ = "0.33.0+fake"
    sampling_error: tuple[Callable[[dict], bool], Exception] | None = None

    def __init__(self) -> None:
        self.engines: list[FakeLLM] = []
        self.llm_error: BaseException | None = None
        self.chat_error: BaseException | None = None
        self.responder: Callable[[list, FakeSamplingParams], Any] = _default_responder
        self.raw_outputs: Callable[[list, list], list] | None = None
        self.SamplingParams = FakeSamplingParams
        FakeVLLM.sampling_error = None

    def LLM(self, **kwargs: Any) -> FakeLLM:  # noqa: N802 - mirrors vllm.LLM
        return FakeLLM(self, **kwargs)

    @property
    def engine(self) -> FakeLLM:
        return self.engines[-1]


class FakeCuda:
    def __init__(self, log: list) -> None:
        self.log = log
        self.available = True
        self.empty_cache_error: Exception | None = None

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.log.append("empty_cache")
        if self.empty_cache_error is not None:
            raise self.empty_cache_error


class FakeTorch:
    __version__ = "2.12.0+fake"

    def __init__(self) -> None:
        self.log: list = []
        self.cuda = FakeCuda(self.log)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fakes() -> SimpleNamespace:
    vllm = FakeVLLM()
    torch = FakeTorch()
    yield SimpleNamespace(vllm=vllm, torch=torch, log=torch.log)
    FakeVLLM.sampling_error = None


@pytest.fixture
def make(fakes):
    def _make() -> JoyCaptionVLLMModel:
        return JoyCaptionVLLMModel(vllm_module=fakes.vllm, torch_module=fakes.torch)

    return _make


@pytest.fixture
def loaded(make, fakes) -> JoyCaptionVLLMModel:
    model = make()
    model.load({"model_path": "./Models"})
    fakes.log.clear()
    return model


def _request(i: int = 0, *, prompt: str | None = None, seed: int | None = None, **gen: Any) -> CaptionRequest:
    params = {"max_new_tokens": 16, "temperature": 0.6, "top_p": 0.9, "top_k": None}
    params.update(gen)
    return CaptionRequest(
        image_rel_path=Path(f"celeste/0001/{i:06d}.jpg"),
        caption_type="long",
        image=PIL.Image.new("RGB", (8, 8), color=(i % 256, 0, 0)),
        prompt=prompt if prompt is not None else f"Describe image {i}.",
        gen_params=params,
        seed=seed if seed is not None else 1000 + i,
    )


def _block_heavy_imports(monkeypatch) -> None:
    for name in ("vllm", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    return env


def _expected_engine_kwargs(**overrides: Any) -> dict:
    kwargs = {
        "model": REPO_ID,
        "revision": None,
        "tokenizer_revision": None,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 128,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "limit_mm_per_prompt": {"image": 1},
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 1. Construction, registry, constants
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructible_with_no_arguments(self) -> None:
        model = JoyCaptionVLLMModel()
        assert isinstance(model, CaptionModel)
        assert model.loaded is False

    def test_class_attributes(self) -> None:
        assert JoyCaptionVLLMModel.model_name == "joycaption-beta-one"
        assert JoyCaptionVLLMModel.backend == "vllm"

    def test_constructor_parameters_are_optional_keyword_only(self) -> None:
        params = inspect.signature(JoyCaptionVLLMModel).parameters
        assert list(params) == ["vllm_module", "torch_module"]
        for param in params.values():
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is None

    def test_interface_signatures(self) -> None:
        assert list(inspect.signature(JoyCaptionVLLMModel.load).parameters) == ["self", "config"]
        assert list(inspect.signature(JoyCaptionVLLMModel.caption_batch).parameters) == ["self", "requests"]
        assert list(inspect.signature(JoyCaptionVLLMModel.unload).parameters) == ["self"]
        assert not getattr(JoyCaptionVLLMModel, "__abstractmethods__", frozenset())

    def test_get_model_vllm_resolves_to_this_class(self) -> None:
        model = get_model("vllm")
        assert type(model) is JoyCaptionVLLMModel
        assert model.loaded is False

    def test_get_model_vllm_needs_no_vllm_or_torch(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        model = get_model("vllm", "joycaption-beta-one")
        assert type(model) is JoyCaptionVLLMModel
        model.unload()

    def test_system_prompt_same_as_hf_backend(self) -> None:
        assert SYSTEM_PROMPT == "You are a helpful image captioner."
        assert SYSTEM_PROMPT == joycaption_hf.SYSTEM_PROMPT

    def test_load_defaults_match_spec_table(self) -> None:
        assert DEFAULT_LOAD_CONFIG == {
            "model_path": REPO_ID,
            "revision": None,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 128,
            "enable_prefix_caching": True,
            "enforce_eager": False,
        }

    def test_load_defaults_match_default_config(self) -> None:
        model_cfg = DEFAULT_CONFIG["model"]
        for key in ("model_path", "revision", "dtype"):
            assert DEFAULT_LOAD_CONFIG[key] == model_cfg[key]
        for key, value in model_cfg["vllm"].items():
            assert DEFAULT_LOAD_CONFIG[key] == value

    def test_gen_defaults_same_as_hf_backend(self) -> None:
        assert DEFAULT_GEN_PARAMS == joycaption_hf.DEFAULT_GEN_PARAMS
        assert DEFAULT_GEN_PARAMS == {"max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": None}

    def test_version_sensitive_constants(self) -> None:
        assert TOP_K_DISABLED == 0  # vLLM 0.33: "Set to 0 (or -1) to consider all tokens", prefers 0
        assert LIMIT_MM_PER_PROMPT == {"image": 1}

    def test_module_logger(self) -> None:
        assert joycaption_vllm.logger.name == MODULE


class TestImports:
    def test_module_imports_without_heavy_libraries(self) -> None:
        """Clean interpreter, nothing on ``PYTHONPATH`` but ``src``: nothing heavy is loaded."""
        code = (
            f"import sys, {MODULE}\n"
            f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {HEAVY!r}"
            " or m.startswith(('pixel_art_auto_captioner.ingestion', 'pixel_art_auto_captioner.batch',"
            " 'pixel_art_auto_captioner.captioning.joycaption_hf')))\n"
            "assert not bad, bad\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=_clean_env(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_get_model_describe_unload_and_load_error_without_vllm(self) -> None:
        code = (
            "import sys\n"
            "for name in ('vllm', 'torch'):\n"
            "    sys.modules[name] = None  # make sure neither can be imported\n"
            "from pixel_art_auto_captioner.captioning import get_model\n"
            "m = get_model('vllm'); d = m.describe(); m.unload(); m.unload()\n"
            "assert type(m).__name__ == 'JoyCaptionVLLMModel', type(m)\n"
            "assert d['backend'] == 'vllm' and d['model_path'] is None, d\n"
            "try:\n"
            "    m.load({})\n"
            "except ImportError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('load() without vllm must raise ImportError')\n"
            "m.unload(); m.unload()\n"
            "assert not m.loaded\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=_clean_env(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_vllm_imported_only_inside_load_and_torch_never(self) -> None:
        source = Path(joycaption_vllm.__file__).read_text(encoding="utf-8")
        assert "print(" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in HEAVY, alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] not in HEAVY, module
                if module.startswith("pixel_art_auto_captioner"):
                    assert module.startswith(
                        ("pixel_art_auto_captioner.common", "pixel_art_auto_captioner.captioning.base")
                    ), module

        # Dynamic imports: only import_module("vllm"), and only inside JoyCaptionVLLMModel.load().
        dynamic: list[tuple[str, str]] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in ("import_module", "__import__"):
                    arg = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                    dynamic.append((".".join(self.stack), arg))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert dynamic == [("load", "vllm")]


# ---------------------------------------------------------------------------
# 2. load(): engine kwargs from config, validation
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults(self, make, fakes) -> None:
        model = make()
        model.load({})
        assert model.loaded
        assert len(fakes.vllm.engines) == 1
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs()

    def test_full_default_model_section(self, make, fakes) -> None:
        model = make()
        model.load(json.loads(json.dumps(DEFAULT_CONFIG["model"])))
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs()

    def test_limit_mm_per_prompt_is_one_image(self, make, fakes) -> None:
        make().load({})
        assert fakes.vllm.engine.kwargs["limit_mm_per_prompt"] == {"image": 1}
        fakes.vllm.engine.kwargs["limit_mm_per_prompt"]["image"] = 5  # a copy, not the constant
        assert LIMIT_MM_PER_PROMPT == {"image": 1}

    def test_vllm_subsection(self, make, fakes) -> None:
        config = {
            "backend": "vllm",
            "model_path": "/workspace/models/joycaption",
            "revision": "abc123",
            "dtype": "float16",
            "hf": {"quantization": "nf4", "device_map": 0},
            "vllm": {
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.8,
                "max_num_seqs": 32,
                "enable_prefix_caching": False,
                "enforce_eager": True,
            },
        }
        make().load(config)
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs(
            model="/workspace/models/joycaption",
            revision="abc123",
            tokenizer_revision="abc123",
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            max_num_seqs=32,
            enable_prefix_caching=False,
            enforce_eager=True,
        )

    def test_partial_vllm_subsection_uses_defaults_for_the_rest(self, make, fakes) -> None:
        make().load({"vllm": {"max_num_seqs": 16}})
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs(max_num_seqs=16)

    def test_top_level_fallback_without_vllm_section(self, make, fakes) -> None:
        make().load({"max_num_seqs": 8, "enforce_eager": True})
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs(max_num_seqs=8, enforce_eager=True)

    def test_vllm_subsection_wins_over_top_level(self, make, fakes) -> None:
        make().load({"max_num_seqs": 8, "vllm": {"max_num_seqs": 64}})
        assert fakes.vllm.engine.kwargs["max_num_seqs"] == 64

    def test_vllm_null_uses_defaults(self, make, fakes) -> None:
        make().load({"vllm": None})
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs()

    def test_hf_section_ignored(self, make, fakes) -> None:
        make().load({"hf": {"quantization": "int8", "device_map": "auto"}})
        assert fakes.vllm.engine.kwargs == _expected_engine_kwargs()

    def test_revision_used_for_model_and_tokenizer(self, make, fakes) -> None:
        make().load({"revision": "deadbeef"})
        assert fakes.vllm.engine.kwargs["revision"] == "deadbeef"
        assert fakes.vllm.engine.kwargs["tokenizer_revision"] == "deadbeef"

    @pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
    def test_dtype_passed_as_string(self, make, fakes, dtype: str) -> None:
        make().load({"dtype": dtype})
        assert fakes.vllm.engine.kwargs["dtype"] == dtype

    def test_path_model_path_becomes_string(self, make, fakes, tmp_path: Path) -> None:
        model = make()
        model.load({"model_path": tmp_path})
        assert fakes.vllm.engine.kwargs["model"] == str(tmp_path)
        assert model.describe()["model_path"] == str(tmp_path)

    def test_integer_gpu_memory_utilization_one(self, make, fakes) -> None:
        make().load({"vllm": {"gpu_memory_utilization": 1}})
        assert fakes.vllm.engine.kwargs["gpu_memory_utilization"] == 1

    @pytest.mark.parametrize(
        "config, match",
        [
            ({"model_path": ""}, "model_path"),
            ({"model_path": None}, "model_path"),
            ({"model_path": 3}, "model_path"),
            ({"revision": ""}, "revision"),
            ({"revision": 5}, "revision"),
            ({"dtype": "bf16"}, "dtype"),
            ({"dtype": None}, "dtype"),
            ({"vllm": []}, "model.vllm"),
            ({"vllm": {"max_model_len": 0}}, "max_model_len"),
            ({"vllm": {"max_model_len": 4096.0}}, "max_model_len"),
            ({"vllm": {"max_model_len": True}}, "max_model_len"),
            ({"vllm": {"max_num_seqs": -1}}, "max_num_seqs"),
            ({"vllm": {"max_num_seqs": None}}, "max_num_seqs"),
            ({"vllm": {"gpu_memory_utilization": 0}}, "gpu_memory_utilization"),
            ({"vllm": {"gpu_memory_utilization": 1.5}}, "gpu_memory_utilization"),
            ({"vllm": {"gpu_memory_utilization": "0.9"}}, "gpu_memory_utilization"),
            ({"vllm": {"gpu_memory_utilization": True}}, "gpu_memory_utilization"),
            ({"vllm": {"enable_prefix_caching": 1}}, "enable_prefix_caching"),
            ({"vllm": {"enable_prefix_caching": None}}, "enable_prefix_caching"),
            ({"vllm": {"enforce_eager": "false"}}, "enforce_eager"),
        ],
    )
    def test_invalid_config_raises_before_any_engine(self, make, fakes, monkeypatch, config, match) -> None:
        model = make()
        with pytest.raises(ValueError, match=match):
            model.load(config)
        assert fakes.vllm.engines == []
        assert not model.loaded

    def test_invalid_config_without_vllm_is_value_error_not_import_error(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        with pytest.raises(ValueError, match="dtype"):
            JoyCaptionVLLMModel().load({"dtype": "int4"})

    @pytest.mark.parametrize("config", [None, [], "model"])
    def test_non_dict_config(self, make, config) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            make().load(config)

    def test_all_problems_reported_together(self, make) -> None:
        with pytest.raises(ValueError) as info:
            make().load(
                {
                    "model_path": "",
                    "dtype": "x",
                    "vllm": {"max_model_len": 0, "gpu_memory_utilization": 2, "enforce_eager": None},
                }
            )
        message = str(info.value)
        for key in ("model_path", "dtype", "max_model_len", "gpu_memory_utilization", "enforce_eager"):
            assert key in message


# ---------------------------------------------------------------------------
# 3. Lifecycle: load failures, reload, unload
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_engine_construction_failure_reraises_and_releases(self, make, fakes) -> None:
        fakes.vllm.llm_error = MemoryError("CUDA out of memory while profiling")
        model = make()
        with pytest.raises(MemoryError):
            model.load({})
        assert not model.loaded and model._llm is None
        assert "empty_cache" in fakes.log
        model.unload()
        model.unload()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.caption_batch([_request()])

    def test_load_failure_is_logged(self, make, fakes, caplog) -> None:
        fakes.vllm.llm_error = RuntimeError("bad weights")
        with caplog.at_level(logging.ERROR, logger=joycaption_vllm.logger.name):
            with pytest.raises(RuntimeError, match="bad weights"):
                make().load({"model_path": "./Models"})
        assert "./Models" in caplog.text

    def test_load_without_vllm_installed(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        model = JoyCaptionVLLMModel()
        with pytest.raises(ImportError):
            model.load({})
        assert not model.loaded
        model.unload()
        model.unload()

    def test_load_twice_unloads_first(self, make, fakes, caplog) -> None:
        model = make()
        model.load({})
        first = fakes.vllm.engine
        fakes.log.clear()
        with caplog.at_level(logging.WARNING, logger=joycaption_vllm.logger.name):
            model.load({"vllm": {"max_num_seqs": 4}})
        assert "unloading it first" in caplog.text
        assert "empty_cache" in fakes.log
        assert len(fakes.vllm.engines) == 2 and model._llm is not first
        assert model.loaded and model.describe()["max_num_seqs"] == 4

    def test_unload_before_load_never_imports_anything(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        model = JoyCaptionVLLMModel()
        model.unload()
        model.unload()
        assert not model.loaded

    def test_unload_before_load_does_not_touch_cuda(self, make, fakes) -> None:
        model = make()
        model.unload()
        assert fakes.log == []

    def test_unload_after_load_twice(self, loaded, fakes) -> None:
        loaded.unload()
        assert not loaded.loaded and loaded._llm is None
        assert fakes.log.count("empty_cache") == 1
        loaded.unload()
        assert fakes.log.count("empty_cache") == 2
        with pytest.raises(RuntimeError, match="not loaded"):
            loaded.caption_batch([_request()])

    def test_unload_collects_garbage(self, loaded, monkeypatch) -> None:
        calls: list[int] = []
        monkeypatch.setattr(joycaption_vllm.gc, "collect", lambda *a: calls.append(1) or 0)
        loaded.unload()
        assert calls == [1]

    def test_unload_skips_empty_cache_without_cuda(self, loaded, fakes) -> None:
        fakes.torch.cuda.available = False
        loaded.unload()
        assert "empty_cache" not in fakes.log

    def test_unload_never_raises_when_empty_cache_fails(self, loaded, fakes, caplog) -> None:
        fakes.torch.cuda.empty_cache_error = RuntimeError("driver gone")
        with caplog.at_level(logging.WARNING, logger=joycaption_vllm.logger.name):
            loaded.unload()
        assert "empty_cache" in caplog.text
        assert not loaded.loaded

    def test_unload_uses_the_torch_that_vllm_loaded(self, fakes, monkeypatch) -> None:
        torch = FakeTorch()
        monkeypatch.setitem(sys.modules, "torch", torch)
        model = JoyCaptionVLLMModel(vllm_module=fakes.vllm)
        model.load({})
        model.unload()
        assert torch.log == ["empty_cache"]

    def test_unload_without_torch_in_sys_modules(self, fakes, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        model = JoyCaptionVLLMModel(vllm_module=fakes.vllm)
        model.load({})
        model.unload()
        model.unload()
        assert not model.loaded

    def test_caption_batch_before_load(self, make) -> None:
        with pytest.raises(RuntimeError, match="not loaded"):
            make().caption_batch([_request()])


# ---------------------------------------------------------------------------
# 4. Conversations and the llm.chat call (SPEC §6.3.2)
# ---------------------------------------------------------------------------


class TestChatCall:
    def test_build_conversation_structure(self) -> None:
        image = PIL.Image.new("RGB", (4, 4))
        conversation = build_conversation("Describe this.", image)
        assert conversation == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": "Describe this."},
                ],
            },
        ]
        assert conversation[1]["content"][0]["image_pil"] is image

    def test_same_system_and_user_text_as_hf_backend(self) -> None:
        conversation = build_conversation("Prompt text.", PIL.Image.new("RGB", (4, 4)))
        hf_messages = [
            {"role": "system", "content": joycaption_hf.SYSTEM_PROMPT},
            {"role": "user", "content": "Prompt text."},
        ]
        assert [m["role"] for m in conversation] == [m["role"] for m in hf_messages]
        assert conversation[0]["content"] == hf_messages[0]["content"]
        texts = [part["text"] for part in conversation[1]["content"] if part["type"] == "text"]
        assert texts == [hf_messages[1]["content"]]

    def test_one_chat_call_with_one_conversation_per_request(self, loaded, fakes) -> None:
        requests = [_request(i) for i in range(3)]
        loaded.caption_batch(requests)
        calls = fakes.vllm.engine.chat_calls
        assert len(calls) == 1
        call = calls[0]
        assert call["use_tqdm"] is False
        assert call["kwargs"] == {}
        assert len(call["messages"]) == 3
        for request, conversation in zip(requests, call["messages"]):
            assert conversation[0] == {"role": "system", "content": SYSTEM_PROMPT}
            assert conversation[1]["role"] == "user"
            image_part, text_part = conversation[1]["content"]
            assert image_part["type"] == "image_pil" and image_part["image_pil"] is request.image
            assert text_part == {"type": "text", "text": request.prompt}

    def test_one_sampling_params_per_request(self, loaded, fakes) -> None:
        requests = [_request(i) for i in range(4)]
        loaded.caption_batch(requests)
        params = fakes.vllm.engine.chat_calls[0]["sampling_params"]
        assert isinstance(params, list) and len(params) == 4
        assert all(isinstance(p, FakeSamplingParams) for p in params)
        assert len({id(p) for p in params}) == 4

    def test_messages_passed_positionally(self, loaded, fakes, monkeypatch) -> None:
        seen: list = []
        engine = fakes.vllm.engine
        original = engine.chat

        def spy(*args, **kwargs):
            seen.append((len(args), sorted(kwargs)))
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "chat", spy)
        loaded.caption_batch([_request()])
        assert seen == [(1, ["sampling_params", "use_tqdm"])]

    def test_empty_batch_makes_no_call(self, loaded, fakes) -> None:
        assert loaded.caption_batch([]) == []
        assert fakes.vllm.engine.chat_calls == []

    def test_each_batch_is_one_call(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(0), _request(1)])
        loaded.caption_batch([_request(2)])
        assert [len(c["messages"]) for c in fakes.vllm.engine.chat_calls] == [2, 1]

    def test_caption_wrapper(self, loaded) -> None:
        result = loaded.caption(_request(7, prompt="Hello."))
        assert isinstance(result, CaptionResult)
        assert result.text == "caption for: Hello."


# ---------------------------------------------------------------------------
# 5. SamplingParams construction
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_exact_kwargs(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(0, max_new_tokens=96, temperature=0.7, top_p=0.95, top_k=None, seed=42)])
        (params,) = fakes.vllm.engine.chat_calls[0]["sampling_params"]
        assert params.kwargs == {"temperature": 0.7, "top_p": 0.95, "top_k": TOP_K_DISABLED, "max_tokens": 96, "seed": 42}

    def test_seed_is_each_requests_seed(self, loaded, fakes) -> None:
        requests = [_request(i, seed=s) for i, s in enumerate([5, 0, 2**31 - 1, 5])]
        loaded.caption_batch(requests)
        seeds = [p.kwargs["seed"] for p in fakes.vllm.engine.chat_calls[0]["sampling_params"]]
        assert seeds == [5, 0, 2**31 - 1, 5]

    @pytest.mark.parametrize("top_k, expected", [(None, 0), (0, 0), (1, 1), (40, 40)])
    def test_top_k(self, loaded, fakes, top_k, expected) -> None:
        loaded.caption_batch([_request(top_k=top_k)])
        assert fakes.vllm.engine.chat_calls[0]["sampling_params"][0].kwargs["top_k"] == expected

    def test_max_new_tokens_becomes_max_tokens(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(max_new_tokens=160)])
        kwargs = fakes.vllm.engine.chat_calls[0]["sampling_params"][0].kwargs
        assert kwargs["max_tokens"] == 160 and "max_new_tokens" not in kwargs

    @pytest.mark.parametrize("key", ["max_new_tokens", "temperature", "top_p", "top_k"])
    def test_none_or_missing_gen_params_use_backend_defaults(self, loaded, fakes, key: str) -> None:
        none_request = _request(**{key: None})
        missing_request = _request()
        del missing_request.gen_params[key]
        loaded.caption_batch([none_request, missing_request])
        expected = {"max_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": 0}
        renamed = "max_tokens" if key == "max_new_tokens" else key
        for params in fakes.vllm.engine.chat_calls[0]["sampling_params"]:
            assert params.kwargs[renamed] == expected[renamed]

    @pytest.mark.parametrize("gen_params", [{}, None])
    def test_empty_gen_params(self, loaded, fakes, gen_params) -> None:
        request = _request(seed=9)
        request.gen_params = gen_params
        loaded.caption_batch([request])
        assert fakes.vllm.engine.chat_calls[0]["sampling_params"][0].kwargs == {
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 0,
            "max_tokens": 512,
            "seed": 9,
        }

    def test_unknown_keys_not_forwarded(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(repetition_penalty=1.2, do_sample=True)])
        kwargs = fakes.vllm.engine.chat_calls[0]["sampling_params"][0].kwargs
        assert set(kwargs) == {"temperature", "top_p", "top_k", "max_tokens", "seed"}

    def test_temperature_zero_passed_through(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(temperature=0)])
        assert fakes.vllm.engine.chat_calls[0]["sampling_params"][0].kwargs["temperature"] == 0

    def test_gen_params_not_mutated(self, loaded) -> None:
        request = _request(top_k=None)
        before = dict(request.gen_params)
        loaded.caption_batch([request])
        assert request.gen_params == before

    @pytest.mark.parametrize(
        "gen, match",
        [
            ({"max_new_tokens": 0}, "max_new_tokens"),
            ({"max_new_tokens": 1.5}, "max_new_tokens"),
            ({"max_new_tokens": True}, "max_new_tokens"),
            ({"temperature": -0.1}, "temperature"),
            ({"temperature": "hot"}, "temperature"),
            ({"top_p": 0}, "top_p"),
            ({"top_p": 1.01}, "top_p"),
            ({"top_k": -1}, "top_k"),
            ({"top_k": 2.0}, "top_k"),
            ({"top_k": False}, "top_k"),
        ],
    )
    def test_invalid_gen_params_give_error_result_without_sending(self, loaded, fakes, gen, match) -> None:
        (result,) = loaded.caption_batch([_request(**gen)])
        assert result.finish_reason == "error"
        assert result.text == "" and result.num_output_tokens == 0
        assert result.error.startswith("ValueError: ") and match in result.error
        assert fakes.vllm.engine.chat_calls == []

    def test_non_dict_gen_params_give_error_result(self, loaded) -> None:
        request = _request()
        request.gen_params = [("max_new_tokens", 3)]
        (result,) = loaded.caption_batch([request])
        assert result.finish_reason == "error" and "gen_params" in result.error

    @pytest.mark.parametrize("seed", [None, 1.0, "7", True])
    def test_invalid_seed_gives_error_result(self, loaded, seed) -> None:
        request = _request()
        request.seed = seed
        (result,) = loaded.caption_batch([request])
        assert result.finish_reason == "error" and "seed" in result.error

    def test_sampling_params_constructor_error_isolated(self, loaded, fakes) -> None:
        FakeVLLM.sampling_error = (lambda kw: kw["seed"] == 13, ValueError("seed rejected"))
        results = loaded.caption_batch([_request(0, seed=12), _request(1, seed=13), _request(2, seed=14)])
        assert [r.finish_reason for r in results] == ["stop", "error", "stop"]
        assert results[1].error == "ValueError: seed rejected"
        assert len(fakes.vllm.engine.chat_calls[0]["messages"]) == 2

    def test_invalid_request_logged_with_its_key(self, loaded, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=joycaption_vllm.logger.name):
            loaded.caption_batch([_request(3, max_new_tokens=0)])
        assert "celeste/0001/000003.jpg" in caplog.text and "long" in caplog.text


# ---------------------------------------------------------------------------
# 6. Output mapping, ordering, errors
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_text_stripped_only(self, loaded, fakes) -> None:
        fakes.vllm.responder = lambda conv, params: FakeCompletion("\n  A  red  square.  \t\n", [1, 2, 3], "stop")
        (result,) = loaded.caption_batch([_request()])
        assert result == CaptionResult(text="A  red  square.", finish_reason="stop", num_output_tokens=3)

    @pytest.mark.parametrize("token_ids", [[], [7], list(range(17)), (4, 5, 6)])
    def test_num_output_tokens_is_len_token_ids(self, loaded, fakes, token_ids) -> None:
        fakes.vllm.responder = lambda conv, params: FakeCompletion("x", token_ids, "length")
        (result,) = loaded.caption_batch([_request()])
        assert result.num_output_tokens == len(token_ids)

    @pytest.mark.parametrize("reason", ["stop", "length"])
    def test_finish_reason_passed_through(self, loaded, fakes, reason) -> None:
        fakes.vllm.responder = lambda conv, params: FakeCompletion("text", [1, 2], reason)
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == reason and result.error is None

    @pytest.mark.parametrize("reason", ["abort", None, "error", "STOP", "", 1])
    def test_other_finish_reasons_become_error_results(self, loaded, fakes, reason, caplog) -> None:
        fakes.vllm.responder = lambda conv, params: FakeCompletion(" partial ", [1, 2], reason)
        with caplog.at_level(logging.WARNING, logger=joycaption_vllm.logger.name):
            (result,) = loaded.caption_batch([_request(5)])
        assert result.finish_reason == "error"
        assert repr(reason) in result.error
        assert result.text == "partial" and result.num_output_tokens == 2
        assert "celeste/0001/000005.jpg" in caplog.text

    def test_only_first_completion_used(self, loaded, fakes) -> None:
        fakes.vllm.responder = lambda conv, params: [
            FakeCompletion("first", [1], "stop"),
            FakeCompletion("second", [1, 2], "length"),
        ]
        (result,) = loaded.caption_batch([_request()])
        assert result.text == "first" and result.num_output_tokens == 1

    def test_no_completion_becomes_error_result(self, loaded, fakes) -> None:
        fakes.vllm.responder = lambda conv, params: []
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error" and "no completion" in result.error

    @pytest.mark.parametrize(
        "completion",
        [
            FakeCompletion(None, [1], "stop"),
            FakeCompletion("x", None, "stop"),
            SimpleNamespace(text="x"),
        ],
    )
    def test_malformed_completion_becomes_error_result(self, loaded, fakes, completion) -> None:
        fakes.vllm.responder = lambda conv, params: completion
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error" and result.error

    def test_results_follow_request_order(self, loaded, fakes) -> None:
        requests = [_request(i, prompt=f"prompt {i}") for i in range(6)]
        results = loaded.caption_batch(requests)
        assert [r.text for r in results] == [f"caption for: prompt {i}" for i in range(6)]

    def test_outputs_mapped_by_position_not_by_request_id(self, loaded, fakes) -> None:
        """Request ids are engine-internal; LLM.chat returns outputs in input order."""

        def outputs(messages, params):
            return [
                FakeRequestOutput(str(99 - i), [FakeCompletion(conv[1]["content"][1]["text"], [1], "stop")])
                for i, conv in enumerate(messages)
            ]

        fakes.vllm.raw_outputs = outputs
        results = loaded.caption_batch([_request(i, prompt=f"p{i}") for i in range(3)])
        assert [r.text for r in results] == ["p0", "p1", "p2"]

    def test_mixed_invalid_requests_keep_positions(self, loaded, fakes) -> None:
        requests = [
            _request(0, prompt="a"),
            _request(1, max_new_tokens=0),
            _request(2, prompt="c"),
            _request(3, top_p=5),
            _request(4, prompt="e"),
        ]
        results = loaded.caption_batch(requests)
        assert [r.finish_reason for r in results] == ["stop", "error", "stop", "error", "stop"]
        assert [r.text for r in results] == ["caption for: a", "", "caption for: c", "", "caption for: e"]
        call = fakes.vllm.engine.chat_calls[0]
        assert [c[1]["content"][1]["text"] for c in call["messages"]] == ["a", "c", "e"]
        assert [p.kwargs["seed"] for p in call["sampling_params"]] == [1000, 1002, 1004]

    def test_all_requests_invalid_makes_no_call(self, loaded, fakes) -> None:
        results = loaded.caption_batch([_request(0, top_p=0), _request(1, max_new_tokens=-3)])
        assert [r.finish_reason for r in results] == ["error", "error"]
        assert fakes.vllm.engine.chat_calls == []

    def test_per_request_seeds_and_results_align(self, loaded, fakes) -> None:
        fakes.vllm.responder = lambda conv, params: FakeCompletion(f"seed={params.kwargs['seed']}", [1], "stop")
        requests = [_request(i, seed=s) for i, s in enumerate([31, 7, 19])]
        assert [r.text for r in loaded.caption_batch(requests)] == ["seed=31", "seed=7", "seed=19"]

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_wrong_number_of_outputs_raises(self, loaded, fakes, count) -> None:
        fakes.vllm.raw_outputs = lambda messages, params: [
            FakeRequestOutput(str(i), [FakeCompletion("x", [1], "stop")]) for i in range(count)
        ]
        with pytest.raises(RuntimeError, match=rf"{count} outputs for 2 conversations"):
            loaded.caption_batch([_request(0), _request(1)])

    def test_chat_exception_propagates_and_model_stays_usable(self, loaded, fakes, caplog) -> None:
        fakes.vllm.chat_error = RuntimeError("engine dead")
        with caplog.at_level(logging.ERROR, logger=joycaption_vllm.logger.name):
            with pytest.raises(RuntimeError, match="engine dead"):
                loaded.caption_batch([_request(0), _request(1)])
        assert "batch of 2" in caplog.text
        assert loaded.loaded
        fakes.vllm.chat_error = None
        assert [r.finish_reason for r in loaded.caption_batch([_request(2)])] == ["stop"]

    def test_chat_exception_propagates_through_caption(self, loaded, fakes) -> None:
        fakes.vllm.chat_error = ValueError("prompt too long")
        with pytest.raises(ValueError, match="prompt too long"):
            loaded.caption(_request())

    def test_keyboard_interrupt_propagates(self, loaded, fakes) -> None:
        fakes.vllm.chat_error = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            loaded.caption_batch([_request()])

    def test_results_are_caption_results_with_valid_finish_reasons(self, loaded, fakes) -> None:
        reasons = iter(["stop", "length", "abort"])
        fakes.vllm.responder = lambda conv, params: FakeCompletion("t", [1], next(reasons))
        results = loaded.caption_batch([_request(i) for i in range(3)] + [_request(3, top_p=0)])
        assert all(isinstance(r, CaptionResult) for r in results)
        assert [r.finish_reason for r in results] == ["stop", "length", "error", "error"]
        assert [r.error is None for r in results] == [True, True, False, False]


# ---------------------------------------------------------------------------
# 7. describe() (SPEC §6.1, D11; proposal P4 not implemented)
# ---------------------------------------------------------------------------

DESCRIBE_KEYS = {
    "model_name",
    "backend",
    "model_path",
    "revision",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "enable_prefix_caching",
    "enforce_eager",
    "limit_mm_per_prompt",
    "vllm_version",
    "torch_version",
    "transformers_version",
}


class TestDescribe:
    def test_before_load(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        info = JoyCaptionVLLMModel().describe()
        assert set(info) == DESCRIBE_KEYS
        assert info["model_name"] == "joycaption-beta-one" and info["backend"] == "vllm"
        for key in DESCRIBE_KEYS - {"model_name", "backend", "vllm_version", "torch_version", "transformers_version"}:
            assert info[key] is None, key

    def test_after_load(self, make) -> None:
        model = make()
        model.load({"model_path": "/m", "revision": "abc", "dtype": "float16", "vllm": {"max_num_seqs": 64}})
        info = model.describe()
        assert set(info) == DESCRIBE_KEYS
        assert info["model_path"] == "/m"
        assert info["revision"] == "abc"
        assert info["dtype"] == "float16"
        assert info["max_model_len"] == 4096
        assert info["gpu_memory_utilization"] == 0.9
        assert info["max_num_seqs"] == 64
        assert info["enable_prefix_caching"] is True
        assert info["enforce_eager"] is False
        assert info["limit_mm_per_prompt"] == {"image": 1}
        assert info["vllm_version"] == "0.33.0+fake"
        assert info["torch_version"] == "2.12.0+fake"

    def test_revision_as_configured_none(self, make) -> None:
        model = make()
        model.load({})
        assert model.describe()["revision"] is None

    def test_no_resolved_commit_hash(self, make) -> None:
        """Proposal P4 is not implemented: only the configured revision is recorded."""
        model = make()
        model.load({})
        assert not any("commit" in key or "resolved" in key for key in model.describe())

    def test_json_serialisable(self, make, tmp_path: Path) -> None:
        model = make()
        model.load({"model_path": tmp_path})
        json.dumps(model.describe())
        json.dumps(JoyCaptionVLLMModel().describe())

    def test_new_dict_per_call(self, loaded) -> None:
        first, second = loaded.describe(), loaded.describe()
        assert first == second and first is not second
        first["limit_mm_per_prompt"]["image"] = 9
        assert loaded.describe()["limit_mm_per_prompt"] == {"image": 1}

    def test_kept_after_unload(self, loaded) -> None:
        loaded.unload()
        info = loaded.describe()
        assert info["model_path"] == "./Models" and info["vllm_version"] == "0.33.0+fake"

    def test_versions_from_installed_metadata(self, monkeypatch) -> None:
        versions = {"vllm": "0.33.0", "torch": "2.9.0", "transformers": "4.99.0"}

        def fake_version(name: str) -> str:
            if name in versions:
                return versions[name]
            raise joycaption_vllm.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(joycaption_vllm.importlib.metadata, "version", fake_version)
        monkeypatch.setitem(sys.modules, "torch", None)
        info = JoyCaptionVLLMModel().describe()
        assert info["vllm_version"] == "0.33.0"
        assert info["torch_version"] == "2.9.0"
        assert info["transformers_version"] == "4.99.0"

    def test_versions_none_when_not_installed(self, monkeypatch) -> None:
        def missing(name: str) -> str:
            raise joycaption_vllm.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(joycaption_vllm.importlib.metadata, "version", missing)
        monkeypatch.setitem(sys.modules, "torch", None)
        info = JoyCaptionVLLMModel().describe()
        assert info["vllm_version"] is None
        assert info["torch_version"] is None
        assert info["transformers_version"] is None


# ---------------------------------------------------------------------------
# 8. GPU tests (vLLM + CUDA + weights; skipped in the CPU suite)
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(os.environ.get("JOYCAPTION_MODEL_PATH", "./Models"))
_MODEL_AVAILABLE = (_MODEL_PATH / "config.json").exists()
_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None


def _cuda_available() -> bool:
    """Import torch lazily; ``False`` if it is missing or has no CUDA (AGENTS.md §6.3)."""
    if not _VLLM_AVAILABLE:
        return False  # avoid importing torch for a test that is skipped anyway
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


requires_vllm = pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vLLM not installed")
requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
requires_model = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="JoyCaption weights not found")


@pytest.fixture(scope="module")
def gpu_model():
    model = get_model("vllm")
    model.load({"model_path": str(_MODEL_PATH), "vllm": {"max_num_seqs": 8, "gpu_memory_utilization": 0.85}})
    try:
        yield model
    finally:
        model.unload()


@pytest.mark.gpu
@requires_vllm
@requires_cuda
@requires_model
class TestGPU:
    def test_batch_in_order_with_valid_finish_reasons(self, gpu_model) -> None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        requests = [
            CaptionRequest(
                image_rel_path=Path(f"gpu/{i}.png"),
                caption_type="short",
                image=PIL.Image.new("RGB", (64, 64), color=color),
                prompt="Write a short description for this image in one or two sentences.",
                gen_params={"max_new_tokens": 64, "temperature": 0.6, "top_p": 0.9, "top_k": None},
                seed=i,
            )
            for i, color in enumerate(colors)
        ]
        results = gpu_model.caption_batch(requests)
        assert len(results) == 3
        for result in results:
            assert result.finish_reason in ("stop", "length"), result
            assert result.text and result.num_output_tokens > 0

    def test_length_limit(self, gpu_model) -> None:
        request = CaptionRequest(
            image_rel_path=Path("gpu/limit.png"),
            caption_type="long",
            image=PIL.Image.new("RGB", (64, 64), color=(10, 200, 30)),
            prompt="Write a long detailed description for this image.",
            gen_params={"max_new_tokens": 4, "temperature": 0.6, "top_p": 0.9, "top_k": None},
            seed=1,
        )
        result = gpu_model.caption(request)
        assert result.finish_reason == "length" and result.num_output_tokens == 4

    def test_describe_versions(self, gpu_model) -> None:
        info = gpu_model.describe()
        assert info["vllm_version"] and info["torch_version"]
        json.dumps(info)
