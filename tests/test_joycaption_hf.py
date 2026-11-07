"""CPU tests for ``captioning/joycaption_hf.py`` (SPEC §6.2, §13) using fakes.

No real ``torch``/``transformers``/``bitsandbytes`` is imported: a fake torch
and a fake transformers (processor, tokenizer, LLaVA model,
``BitsAndBytesConfig``) are injected through the constructor's keyword
parameters (AGENTS.md §5). The fakes record every call so the tests can check
the exact HF call pattern (AGENTS.md §6.6), the dtype cast, prompt trimming,
``finish_reason``, per-request seeding, error mapping, the quantization config
and the unload lifecycle.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import PIL.Image
import pytest

from pixel_art_auto_captioner.captioning import CaptionModel, get_model
from pixel_art_auto_captioner.captioning import joycaption_hf
from pixel_art_auto_captioner.captioning.joycaption_hf import (
    DEFAULT_GEN_PARAMS,
    DEFAULT_LOAD_CONFIG,
    QUANT_SKIP_MODULES,
    SYSTEM_PROMPT,
    JoyCaptionHFModel,
)
from pixel_art_auto_captioner.common.config import DEFAULT_CONFIG
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

SRC = Path(__file__).resolve().parent.parent / "src"
REPO_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
EOS = 99  # tokenizer EOS
EOT = 98  # second EOS id, only in the model's generation_config
HEAVY = ("torch", "transformers", "bitsandbytes", "vllm", "accelerate")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"fake.{self.name}"


class FakeTensor:
    """Minimal tensor: nested lists + dtype + device, with ``to``/slicing/``shape``/``tolist``."""

    def __init__(self, data: list, dtype: Any = "int64", device: Any = "cpu") -> None:
        self.data = data
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def to(self, target: Any) -> "FakeTensor":
        if isinstance(target, FakeDtype):
            return FakeTensor(self.data, target, self.device)
        return FakeTensor(self.data, self.dtype, target)

    def __getitem__(self, idx: Any) -> Any:
        item = self.data[idx]
        return FakeTensor(item, self.dtype, self.device) if isinstance(item, list) else item

    def __len__(self) -> int:
        return len(self.data)

    def tolist(self) -> list:
        return copy.deepcopy(self.data)


class FakeOOM(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError``."""


class FakeCuda:
    OutOfMemoryError = FakeOOM

    def __init__(self, log: list) -> None:
        self.log = log
        self.available = True
        self.empty_cache_error: Exception | None = None

    def manual_seed_all(self, seed: int) -> None:
        self.log.append(("cuda_seed", seed))

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.log.append(("empty_cache",))
        if self.empty_cache_error is not None:
            raise self.empty_cache_error


class FakeTorch:
    __version__ = "2.12.0+fake"

    def __init__(self) -> None:
        self.log: list = []
        self.bfloat16 = FakeDtype("bfloat16")
        self.float16 = FakeDtype("float16")
        self.float32 = FakeDtype("float32")
        self.cuda = FakeCuda(self.log)
        self.in_inference = False
        self.current_seed: int | None = None
        self.linear_calls: list = []
        self.nn = SimpleNamespace(Linear=lambda *a, **k: self.linear_calls.append((a, k)))

    def manual_seed(self, seed: int) -> None:
        self.log.append(("seed", seed))
        self.current_seed = seed

    @contextmanager
    def inference_mode(self):
        self.in_inference = True
        try:
            yield
        finally:
            self.in_inference = False


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id: Any = EOS
        self.calls: list = []
        self.error: Exception | None = None

    def decode(self, ids: Any, **kwargs: Any) -> str:
        tokens = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        self.calls.append((tokens, kwargs))
        if self.error is not None:
            raise self.error
        special = {EOS, EOT} if kwargs.get("skip_special_tokens") else set()
        return "  " + " ".join(f"w{t}" for t in tokens if t not in special) + " \n"


class FakeProcessor:
    """Fake LLaVA processor. Prompt length depends on the prompt text."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(size={"height": 384, "width": 384}, do_resize=True)
        self.template_calls: list = []
        self.calls: list = []
        self.template_result: Any = None  # None = default string
        self.fail_on_prompt: dict[str, Exception] = {}
        self.call_error: Exception | None = None

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        self.template_calls.append((copy.deepcopy(args), dict(kwargs)))
        self.log.append(("template",))
        user = args[0][1]["content"]
        if user in self.fail_on_prompt:
            raise self.fail_on_prompt[user]
        if self.template_result is not None:
            return self.template_result
        return f"<convo>{user}</convo>"

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        self.calls.append((args, dict(kwargs)))
        self.log.append(("process",))
        if self.call_error is not None:
            raise self.call_error
        text = kwargs["text"][0]
        prompt_len = 3 + len(text) % 5
        return {
            "input_ids": FakeTensor([[100 + i for i in range(prompt_len)]], "int64"),
            "attention_mask": FakeTensor([[1] * prompt_len], "int64"),
            "pixel_values": FakeTensor([[0.5, 0.25]], FakeDtype("float32")),
        }


OUT_PROJ = object()


class FakeModel:
    """Fake ``LlavaForConditionalGeneration``. ``script`` pops one behaviour per generate()."""

    def __init__(self, torch: FakeTorch, log: list) -> None:
        self.torch = torch
        self.log = log
        self.device = "cuda:0"
        self.generation_config = SimpleNamespace(eos_token_id=[EOT, EOS])
        self.eval_calls = 0
        self.calls: list = []
        self.script: list = []
        self.model = SimpleNamespace(
            vision_tower=SimpleNamespace(head=SimpleNamespace(attention=SimpleNamespace(out_proj=OUT_PROJ, embed_dim=8)))
        )

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.calls.append(
            {"kwargs": kwargs, "seed": self.torch.current_seed, "inference": self.torch.in_inference}
        )
        self.log.append(("generate",))
        behaviour: Any = self.script.pop(0) if self.script else [5, 6, EOS]
        if isinstance(behaviour, BaseException):
            raise behaviour
        if callable(behaviour):
            behaviour = behaviour(kwargs)
        prompt = kwargs["input_ids"].data[0]
        return FakeTensor([list(prompt) + list(behaviour)], "int64", self.device)


class FakeBnBConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeTransformers:
    __version__ = "5.12.0+fake"

    def __init__(self, torch: FakeTorch) -> None:
        self.torch = torch
        self.log = torch.log
        self.calls: list = []
        self.bnb_configs: list[FakeBnBConfig] = []
        self.processor = FakeProcessor(self.log)
        self.model = FakeModel(torch, self.log)
        self.processor_error: Exception | None = None
        self.model_error: BaseException | None = None
        outer = self

        class AutoProcessor:
            @staticmethod
            def from_pretrained(path: Any, *args: Any, **kwargs: Any) -> FakeProcessor:
                outer.calls.append(("processor", path, args, dict(kwargs)))
                if outer.processor_error is not None:
                    raise outer.processor_error
                return outer.processor

        class LlavaForConditionalGeneration:
            @staticmethod
            def from_pretrained(path: Any, *args: Any, **kwargs: Any) -> FakeModel:
                outer.calls.append(("model", path, args, dict(kwargs)))
                if outer.model_error is not None:
                    raise outer.model_error
                return outer.model

        def bnb(**kwargs: Any) -> FakeBnBConfig:
            config = FakeBnBConfig(**kwargs)
            outer.bnb_configs.append(config)
            return config

        self.AutoProcessor = AutoProcessor
        self.LlavaForConditionalGeneration = LlavaForConditionalGeneration
        self.BitsAndBytesConfig = bnb

    def model_kwargs(self) -> dict:
        return [c for c in self.calls if c[0] == "model"][-1][3]

    def processor_kwargs(self) -> dict:
        return [c for c in self.calls if c[0] == "processor"][-1][3]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fakes() -> SimpleNamespace:
    torch = FakeTorch()
    tf = FakeTransformers(torch)
    return SimpleNamespace(torch=torch, tf=tf, processor=tf.processor, model=tf.model, log=torch.log)


@pytest.fixture
def make(fakes):
    def _make() -> JoyCaptionHFModel:
        return JoyCaptionHFModel(torch_module=fakes.torch, transformers_module=fakes.tf)

    return _make


@pytest.fixture
def loaded(make, fakes) -> JoyCaptionHFModel:
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
    for name in ("torch", "transformers", "bitsandbytes"):
        monkeypatch.setitem(sys.modules, name, None)


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    return env


# ---------------------------------------------------------------------------
# 1. Construction, alias, registry, imports
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructible_with_no_arguments(self) -> None:
        model = JoyCaptionHFModel()
        assert isinstance(model, CaptionModel)
        assert model.loaded is False

    def test_class_attributes(self) -> None:
        assert JoyCaptionHFModel.model_name == "joycaption-beta-one"
        assert JoyCaptionHFModel.backend == "hf"

    def test_constructor_parameters_are_optional_keyword_only(self) -> None:
        params = inspect.signature(JoyCaptionHFModel).parameters
        assert list(params) == ["torch_module", "transformers_module"]
        for param in params.values():
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is None

    def test_interface_signatures(self) -> None:
        assert list(inspect.signature(JoyCaptionHFModel.load).parameters) == ["self", "config"]
        assert list(inspect.signature(JoyCaptionHFModel.caption_batch).parameters) == ["self", "requests"]
        assert list(inspect.signature(JoyCaptionHFModel.unload).parameters) == ["self"]
        assert not getattr(JoyCaptionHFModel, "__abstractmethods__", frozenset())

    def test_alias_in_joycaption_hf(self) -> None:
        assert joycaption_hf.JoyCaptionModel is JoyCaptionHFModel

    def test_v1_module_is_a_shim_for_the_alias(self) -> None:
        from pixel_art_auto_captioner.captioning import joycaption

        assert joycaption.JoyCaptionModel is JoyCaptionHFModel
        source = Path(joycaption.__file__).read_text(encoding="utf-8")
        code_lines = [line for line in source.splitlines() if line.strip() and not line.startswith('"""')]
        assert len(code_lines) <= 2

    def test_get_model_hf_resolves_to_this_class(self) -> None:
        model = get_model("hf")
        assert type(model) is JoyCaptionHFModel
        assert model.loaded is False

    def test_get_model_hf_needs_no_torch(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        assert type(get_model("hf")) is JoyCaptionHFModel

    def test_system_prompt_unchanged_from_main_py(self) -> None:
        assert SYSTEM_PROMPT == "You are a helpful image captioner."

    def test_default_tables_match_spec(self) -> None:
        assert DEFAULT_LOAD_CONFIG == {
            "model_path": REPO_ID,
            "revision": None,
            "dtype": "bfloat16",
            "quantization": "nf4",
            "device_map": 0,
        }
        assert DEFAULT_GEN_PARAMS == {"max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": None}
        assert {"vision_tower", "multi_modal_projector"} <= set(QUANT_SKIP_MODULES)

    def test_module_logger(self) -> None:
        assert joycaption_hf.logger.name == "pixel_art_auto_captioner.captioning.joycaption_hf"


class TestImports:
    @pytest.mark.parametrize(
        "module",
        ["pixel_art_auto_captioner.captioning.joycaption_hf", "pixel_art_auto_captioner.captioning.joycaption"],
    )
    def test_module_imports_without_heavy_libraries(self, module: str) -> None:
        """Clean interpreter, no torch stub on ``PYTHONPATH``: nothing heavy is loaded."""
        code = (
            f"import sys, {module}\n"
            f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {HEAVY!r}"
            " or m.startswith(('pixel_art_auto_captioner.ingestion', 'pixel_art_auto_captioner.batch')))\n"
            "assert not bad, bad\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=_clean_env(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_get_model_and_describe_without_torch_in_clean_interpreter(self) -> None:
        code = (
            "import sys\n"
            "from pixel_art_auto_captioner.captioning import get_model\n"
            "m = get_model('hf'); d = m.describe(); m.unload(); m.unload()\n"
            "assert type(m).__name__ == 'JoyCaptionHFModel', type(m)\n"
            "assert d['backend'] == 'hf' and d['model_path'] is None, d\n"
            f"bad = sorted(x for x in sys.modules if x.split('.')[0] in {HEAVY!r})\n"
            "assert not bad, bad\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=_clean_env(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_source_has_no_top_level_heavy_imports_print_or_upper_layers(self) -> None:
        source = Path(joycaption_hf.__file__).read_text(encoding="utf-8")
        assert "print(" not in source
        tree = ast.parse(source)
        for node in tree.body:  # module level only; load() imports lazily
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in HEAVY, name
                assert not name.startswith(("pixel_art_auto_captioner.ingestion", "pixel_art_auto_captioner.batch"))
        for node in ast.walk(tree):  # anywhere: only captioning.base and common from the package
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pixel_art_auto_captioner"):
                assert node.module.startswith(
                    ("pixel_art_auto_captioner.common", "pixel_art_auto_captioner.captioning.base")
                ), node.module


# ---------------------------------------------------------------------------
# 2. load(): config keys, defaults, validation
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults(self, make, fakes) -> None:
        model = make()
        model.load({})
        assert model.loaded
        assert [c[0] for c in fakes.tf.calls] == ["processor", "model"]
        assert fakes.tf.calls[0][1] == REPO_ID and fakes.tf.calls[1][1] == REPO_ID
        assert fakes.tf.processor_kwargs() == {}
        kwargs = fakes.tf.model_kwargs()
        assert kwargs["dtype"] is fakes.torch.bfloat16
        assert kwargs["device_map"] == 0
        assert kwargs["quantization_config"] is fakes.tf.bnb_configs[0]
        assert fakes.tf.bnb_configs[0].kwargs["load_in_4bit"] is True
        assert "revision" not in kwargs and "torch_dtype" not in kwargs

    def test_full_default_model_section(self, make, fakes) -> None:
        """The §9.2 ``model`` section, as the runner passes it (backend/vllm keys ignored)."""
        section = copy.deepcopy(DEFAULT_CONFIG["model"])
        section["backend"] = "hf"
        model = make()
        model.load(section)
        kwargs = fakes.tf.model_kwargs()
        assert fakes.tf.calls[1][1] == section["model_path"]
        assert kwargs["device_map"] == section["hf"]["device_map"]
        assert fakes.tf.bnb_configs[0].kwargs["bnb_4bit_quant_type"] == "nf4"
        assert set(kwargs) == {"dtype", "device_map", "quantization_config"}

    def test_hf_subsection(self, make, fakes) -> None:
        make().load({"model_path": "./Models", "hf": {"quantization": "int8", "device_map": "auto"}})
        kwargs = fakes.tf.model_kwargs()
        assert kwargs["device_map"] == "auto"
        assert fakes.tf.bnb_configs[0].kwargs["load_in_8bit"] is True

    def test_top_level_fallback_without_hf_section(self, make, fakes) -> None:
        make().load({"model_path": "./Models", "quantization": None, "device_map": "cuda:1"})
        assert fakes.tf.model_kwargs()["device_map"] == "cuda:1"
        assert fakes.tf.bnb_configs == []

    def test_hf_subsection_wins_over_top_level(self, make, fakes) -> None:
        make().load({"quantization": "int8", "device_map": 1, "hf": {"quantization": None, "device_map": 0}})
        assert fakes.tf.bnb_configs == []
        assert fakes.tf.model_kwargs()["device_map"] == 0

    def test_hf_null_uses_defaults(self, make, fakes) -> None:
        make().load({"hf": None})
        assert fakes.tf.bnb_configs[0].kwargs["load_in_4bit"] is True

    def test_revision_passed_to_both_from_pretrained_calls(self, make, fakes) -> None:
        make().load({"revision": "abc123"})
        assert fakes.tf.processor_kwargs() == {"revision": "abc123"}
        assert fakes.tf.model_kwargs()["revision"] == "abc123"

    @pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
    def test_dtype_maps_to_torch_dtype(self, make, fakes, dtype: str) -> None:
        make().load({"dtype": dtype, "hf": {"quantization": None}})
        assert fakes.tf.model_kwargs()["dtype"] is getattr(fakes.torch, dtype)

    def test_path_model_path_becomes_string(self, make, fakes, tmp_path: Path) -> None:
        model = make()
        model.load({"model_path": tmp_path})
        assert fakes.tf.calls[0][1] == str(tmp_path)
        assert model.describe()["model_path"] == str(tmp_path)

    @pytest.mark.parametrize(
        "config, match",
        [
            ({"dtype": "fp8"}, "dtype"),
            ({"dtype": None}, "dtype"),
            ({"hf": {"quantization": "nf8"}}, "quantization"),
            ({"hf": {"quantization": "4bit"}}, "quantization"),
            ({"hf": {"quantization": "NF4"}}, "quantization"),
            ({"quantization": True}, "quantization"),
            ({"model_path": ""}, "model_path"),
            ({"model_path": "   "}, "model_path"),
            ({"model_path": None}, "model_path"),
            ({"model_path": 5}, "model_path"),
            ({"revision": ""}, "revision"),
            ({"revision": 3}, "revision"),
            ({"hf": "nf4"}, "model.hf"),
        ],
    )
    def test_invalid_config_raises_before_any_import(self, make, fakes, monkeypatch, config, match) -> None:
        _block_heavy_imports(monkeypatch)
        model = make()
        with pytest.raises(ValueError, match=match):
            model.load(config)
        assert fakes.tf.calls == [] and fakes.tf.bnb_configs == []
        assert not model.loaded
        model.unload()

    def test_invalid_config_without_fakes_is_value_error_not_import_error(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        with pytest.raises(ValueError):
            JoyCaptionHFModel().load({"dtype": "int4"})

    @pytest.mark.parametrize("config", [None, [], "model"])
    def test_non_dict_config(self, make, config) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            make().load(config)

    def test_all_problems_reported_together(self, make) -> None:
        with pytest.raises(ValueError) as info:
            make().load({"model_path": "", "dtype": "fp8", "hf": {"quantization": "x"}})
        message = str(info.value)
        assert "model_path" in message and "dtype" in message and "quantization" in message


# ---------------------------------------------------------------------------
# 3. Quantization and the removed hack (Q3)
# ---------------------------------------------------------------------------


class TestQuantization:
    def test_nf4_config(self, make, fakes) -> None:
        make().load({"dtype": "bfloat16", "hf": {"quantization": "nf4"}})
        (config,) = fakes.tf.bnb_configs
        kw = config.kwargs
        assert kw["load_in_4bit"] is True
        assert kw["bnb_4bit_quant_type"] == "nf4"
        assert kw["bnb_4bit_use_double_quant"] is True
        assert kw["bnb_4bit_compute_dtype"] is fakes.torch.bfloat16
        assert kw["bnb_4bit_quant_storage"] is fakes.torch.bfloat16
        assert "load_in_8bit" not in kw

    @pytest.mark.parametrize("quantization", ["nf4", "int8"])
    def test_vision_tower_and_projector_are_skipped(self, make, fakes, quantization: str) -> None:
        make().load({"hf": {"quantization": quantization}})
        skip = fakes.tf.bnb_configs[0].kwargs["llm_int8_skip_modules"]
        assert "vision_tower" in skip and "multi_modal_projector" in skip
        assert "lm_head" in skip  # explicit list replaces transformers' default (lm_head)
        assert isinstance(skip, list)

    def test_nf4_compute_dtype_follows_dtype(self, make, fakes) -> None:
        make().load({"dtype": "float16"})
        kw = fakes.tf.bnb_configs[0].kwargs
        assert kw["bnb_4bit_compute_dtype"] is fakes.torch.float16
        assert fakes.tf.model_kwargs()["dtype"] is fakes.torch.float16

    def test_int8_config(self, make, fakes) -> None:
        make().load({"hf": {"quantization": "int8"}})
        kw = fakes.tf.bnb_configs[0].kwargs
        assert kw["load_in_8bit"] is True
        assert not any(key.startswith("bnb_4bit") for key in kw) and "load_in_4bit" not in kw

    def test_no_quantization(self, make, fakes) -> None:
        make().load({"hf": {"quantization": None}})
        assert fakes.tf.bnb_configs == []
        assert "quantization_config" not in fakes.tf.model_kwargs()

    def test_no_out_proj_hack(self, make, fakes) -> None:
        make().load({})
        attention = fakes.model.model.vision_tower.head.attention
        assert attention.out_proj is OUT_PROJ
        assert fakes.torch.linear_calls == []

    def test_processor_default_resize_untouched(self, make, fakes) -> None:
        make().load({})
        assert fakes.processor.image_processor.size == {"height": 384, "width": 384}
        assert fakes.processor.image_processor.do_resize is True

    def test_model_put_in_eval_mode(self, make, fakes) -> None:
        make().load({})
        assert fakes.model.eval_calls == 1


# ---------------------------------------------------------------------------
# 4. load()/unload() lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_processor_load_failure(self, make, fakes) -> None:
        fakes.tf.processor_error = OSError("no such repo")
        model = make()
        with pytest.raises(OSError, match="no such repo"):
            model.load({})
        assert not model.loaded
        model.unload()
        model.unload()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.caption_batch([_request()])

    def test_partial_load_releases_processor(self, make, fakes) -> None:
        fakes.tf.model_error = MemoryError("weights too big")
        model = make()
        with pytest.raises(MemoryError):
            model.load({})
        assert model._processor is None and model._model is None
        assert ("empty_cache",) in fakes.log
        model.unload()

    def test_load_without_torch_installed(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        model = JoyCaptionHFModel()
        with pytest.raises(ImportError):
            model.load({})
        assert not model.loaded
        model.unload()
        model.unload()

    def test_load_logs_failure(self, make, fakes, caplog) -> None:
        fakes.tf.model_error = RuntimeError("bad weights")
        with caplog.at_level(logging.ERROR, logger=joycaption_hf.logger.name):
            with pytest.raises(RuntimeError):
                make().load({"model_path": "./Models"})
        assert "./Models" in caplog.text

    def test_load_twice_unloads_first(self, make, fakes) -> None:
        model = make()
        model.load({})
        fakes.log.clear()
        model.load({"hf": {"quantization": None}})
        assert ("empty_cache",) in fakes.log
        assert [c[0] for c in fakes.tf.calls] == ["processor", "model", "processor", "model"]
        assert model.loaded and model.describe()["quantization"] is None

    def test_unload_before_load_never_imports_torch(self, monkeypatch) -> None:
        _block_heavy_imports(monkeypatch)
        model = JoyCaptionHFModel()
        model.unload()
        model.unload()

    def test_unload_after_load_twice(self, loaded, fakes) -> None:
        loaded.unload()
        assert not loaded.loaded
        assert loaded._model is None and loaded._processor is None
        assert fakes.log.count(("empty_cache",)) == 1
        loaded.unload()
        assert fakes.log.count(("empty_cache",)) == 2
        with pytest.raises(RuntimeError, match="not loaded"):
            loaded.caption_batch([_request()])

    def test_unload_skips_empty_cache_without_cuda(self, loaded, fakes) -> None:
        fakes.torch.cuda.available = False
        loaded.unload()
        assert ("empty_cache",) not in fakes.log

    def test_unload_never_raises_when_empty_cache_fails(self, loaded, fakes, caplog) -> None:
        fakes.torch.cuda.empty_cache_error = RuntimeError("driver gone")
        with caplog.at_level(logging.WARNING, logger=joycaption_hf.logger.name):
            loaded.unload()
        assert "empty_cache" in caplog.text
        assert not loaded.loaded

    def test_caption_batch_before_load(self, make) -> None:
        with pytest.raises(RuntimeError, match="not loaded"):
            make().caption_batch([_request()])

    def test_no_eos_id_warns_and_reports_length(self, make, fakes, caplog) -> None:
        fakes.model.generation_config = SimpleNamespace(eos_token_id=None)
        fakes.processor.tokenizer.eos_token_id = None
        model = make()
        with caplog.at_level(logging.WARNING, logger=joycaption_hf.logger.name):
            model.load({})
        assert "EOS" in caplog.text
        fakes.model.script = [[1, 2, EOS]]
        (result,) = model.caption_batch([_request(max_new_tokens=3)])
        assert result.finish_reason == "length"


# ---------------------------------------------------------------------------
# 5. Messages and the HF call pattern (AGENTS.md §6.6, D9)
# ---------------------------------------------------------------------------


class TestCallPattern:
    def test_chat_template_call(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(prompt="Write a long description.")])
        ((args, kwargs),) = fakes.processor.template_calls
        assert args == (
            [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": "Write a long description."},
            ],
        )
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}

    def test_processor_call_one_conversation_one_image(self, loaded, fakes) -> None:
        request = _request(prompt="Short please.")
        loaded.caption_batch([request])
        ((args, kwargs),) = fakes.processor.calls
        assert args == ()
        assert kwargs["text"] == ["<convo>Short please.</convo>"]
        assert len(kwargs["images"]) == 1 and kwargs["images"][0] is request.image
        assert kwargs["return_tensors"] == "pt"
        assert set(kwargs) == {"text", "images", "return_tensors"}

    def test_one_hf_call_per_request_in_order(self, loaded, fakes) -> None:
        requests = [_request(i) for i in range(3)]
        results = loaded.caption_batch(requests)
        assert len(results) == 3
        assert [c[0][0][1]["content"] for c in fakes.processor.template_calls] == [r.prompt for r in requests]
        assert [c[1]["images"][0] for c in fakes.processor.calls] == [r.image for r in requests]
        assert len(fakes.model.calls) == 3
        steps = [entry[0] for entry in fakes.log if entry[0] in ("template", "process", "generate")]
        assert steps == ["template", "process", "generate"] * 3

    def test_empty_batch(self, loaded, fakes) -> None:
        assert loaded.caption_batch([]) == []
        assert fakes.model.calls == [] and fakes.processor.calls == []

    def test_caption_wrapper(self, loaded) -> None:
        result = loaded.caption(_request())
        assert isinstance(result, CaptionResult) and result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 6. Tensors, dtype cast and generate() kwargs
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_inputs_moved_to_device(self, loaded, fakes) -> None:
        loaded.caption_batch([_request()])
        kwargs = fakes.model.calls[0]["kwargs"]
        for key in ("input_ids", "attention_mask", "pixel_values"):
            assert kwargs[key].device == "cuda:0", key

    @pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
    def test_pixel_values_cast_to_model_dtype(self, make, fakes, dtype: str) -> None:
        model = make()
        model.load({"dtype": dtype})
        model.caption_batch([_request()])
        kwargs = fakes.model.calls[0]["kwargs"]
        assert kwargs["pixel_values"].dtype is getattr(fakes.torch, dtype)
        assert kwargs["input_ids"].dtype == "int64"  # only pixel_values is cast

    def test_generate_kwargs(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(max_new_tokens=77, temperature=0.4, top_p=0.8, top_k=20)])
        kwargs = dict(fakes.model.calls[0]["kwargs"])
        for key in ("input_ids", "attention_mask", "pixel_values"):
            kwargs.pop(key)
        assert kwargs == {
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 20,
            "max_new_tokens": 77,
            "use_cache": True,
            "suppress_tokens": None,
        }

    def test_generate_runs_under_inference_mode(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(), _request(1)])
        assert [c["inference"] for c in fakes.model.calls] == [True, True]
        assert fakes.torch.in_inference is False

    @pytest.mark.parametrize("key", ["max_new_tokens", "temperature", "top_p", "top_k"])
    def test_none_or_missing_gen_params_use_backend_defaults(self, loaded, fakes, key: str) -> None:
        none_request = _request(**{key: None})
        missing_request = _request()
        del missing_request.gen_params[key]
        loaded.caption_batch([none_request, missing_request])
        for call in fakes.model.calls:
            assert call["kwargs"][key] == DEFAULT_GEN_PARAMS[key]

    def test_empty_gen_params(self, loaded, fakes) -> None:
        request = _request()
        request.gen_params = {}
        (result,) = loaded.caption_batch([request])
        assert result.finish_reason == "stop"
        kwargs = fakes.model.calls[0]["kwargs"]
        assert (kwargs["max_new_tokens"], kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (512, 0.6, 0.9, None)

    def test_unknown_gen_params_are_not_forwarded(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(repetition_penalty=1.3)])
        assert "repetition_penalty" not in fakes.model.calls[0]["kwargs"]

    def test_temperature_zero_is_greedy(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(temperature=0)])
        kwargs = fakes.model.calls[0]["kwargs"]
        assert kwargs["do_sample"] is False
        assert not {"temperature", "top_p", "top_k"} & set(kwargs)
        assert kwargs["max_new_tokens"] == 16

    @pytest.mark.parametrize(
        "gen",
        [
            {"max_new_tokens": 0},
            {"max_new_tokens": -3},
            {"max_new_tokens": "16"},
            {"max_new_tokens": True},
            {"max_new_tokens": 2.5},
            {"temperature": -0.1},
            {"temperature": "hot"},
            {"top_p": 0},
            {"top_p": 1.5},
            {"top_k": -1},
            {"top_k": 2.5},
        ],
    )
    def test_invalid_gen_params_give_error_result_without_generate(self, loaded, fakes, gen) -> None:
        results = loaded.caption_batch([_request(0, **gen), _request(1)])
        assert results[0].finish_reason == "error"
        assert results[0].error.startswith("ValueError: ")
        assert results[1].finish_reason == "stop"
        assert len(fakes.model.calls) == 1


# ---------------------------------------------------------------------------
# 7. Prompt trimming and decoding
# ---------------------------------------------------------------------------


class TestDecoding:
    @pytest.mark.parametrize("prompt", ["a", "ab", "abc", "abcd", "abcde"])
    def test_only_new_tokens_are_decoded(self, loaded, fakes, prompt: str) -> None:
        fakes.model.script = [[7, 8, EOS]]
        (result,) = loaded.caption_batch([_request(prompt=prompt)])
        ((tokens, kwargs),) = fakes.processor.tokenizer.calls
        assert tokens == [7, 8, EOS]
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        assert result.text == "w7 w8"
        assert result.num_output_tokens == 3

    def test_text_is_only_stripped(self, loaded, fakes) -> None:
        fakes.processor.tokenizer.decode = lambda ids, **kw: "\n  A  red   square.\t\n"
        (result,) = loaded.caption_batch([_request()])
        assert result.text == "A  red   square."
        assert result.error is None

    def test_results_follow_request_order(self, loaded, fakes) -> None:
        fakes.model.script = [[1, EOS], [2, 2, EOS], [3, 3, 3, EOS]]
        results = loaded.caption_batch([_request(i) for i in range(3)])
        assert [r.text for r in results] == ["w1", "w2 w2", "w3 w3 w3"]
        assert [r.num_output_tokens for r in results] == [2, 3, 4]


# ---------------------------------------------------------------------------
# 8. finish_reason
# ---------------------------------------------------------------------------


class TestFinishReason:
    @pytest.mark.parametrize(
        "new_tokens, max_new_tokens, expected",
        [
            ([5, 6, EOS], 16, "stop"),  # EOS before the limit
            ([5, 6, EOS], 3, "stop"),  # limit reached exactly on EOS
            ([5, 6, EOT], 3, "stop"),  # second EOS id from generation_config
            ([5, 6, 7], 3, "length"),  # limit reached, no EOS
            ([5, 6], 16, "stop"),  # ended early without EOS (other stopping criterion)
            ([], 16, "stop"),  # nothing generated
            ([1] * 8, 8, "length"),
        ],
    )
    def test_finish_reason(self, loaded, fakes, new_tokens, max_new_tokens, expected) -> None:
        fakes.model.script = [new_tokens]
        (result,) = loaded.caption_batch([_request(max_new_tokens=max_new_tokens)])
        assert result.finish_reason == expected
        assert result.num_output_tokens == len(new_tokens)
        assert result.error is None

    def test_eos_from_tokenizer_when_generation_config_has_none(self, make, fakes) -> None:
        fakes.model.generation_config = SimpleNamespace(eos_token_id=None)
        model = make()
        model.load({})
        fakes.model.script = [[5, EOS], [5, EOT]]
        results = model.caption_batch([_request(max_new_tokens=2), _request(1, max_new_tokens=2)])
        assert [r.finish_reason for r in results] == ["stop", "length"]

    def test_int_eos_in_generation_config(self, make, fakes) -> None:
        fakes.model.generation_config = SimpleNamespace(eos_token_id=EOT)
        fakes.processor.tokenizer.eos_token_id = None
        model = make()
        model.load({})
        fakes.model.script = [[5, EOT]]
        (result,) = model.caption_batch([_request(max_new_tokens=2)])
        assert result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 9. Per-request seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_each_request_seeded_before_its_generate(self, loaded, fakes) -> None:
        requests = [_request(0, seed=11), _request(1, seed=22), _request(2, seed=33)]
        loaded.caption_batch(requests)
        assert [c["seed"] for c in fakes.model.calls] == [11, 22, 33]
        events = [e for e in fakes.log if e[0] in ("seed", "cuda_seed", "generate")]
        assert events == [
            ("seed", 11), ("cuda_seed", 11), ("generate",),
            ("seed", 22), ("cuda_seed", 22), ("generate",),
            ("seed", 33), ("cuda_seed", 33), ("generate",),
        ]

    def test_same_seed_is_set_again_for_every_request(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(0, seed=5), _request(1, seed=5)])
        assert [e for e in fakes.log if e[0] == "seed"] == [("seed", 5), ("seed", 5)]

    def test_seed_set_after_a_failed_request(self, loaded, fakes) -> None:
        fakes.model.script = [RuntimeError("boom"), [1, EOS]]
        results = loaded.caption_batch([_request(0, seed=1), _request(1, seed=2)])
        assert [r.finish_reason for r in results] == ["error", "stop"]
        assert fakes.model.calls[1]["seed"] == 2

    def test_seed_is_first_step_of_a_request(self, loaded, fakes) -> None:
        loaded.caption_batch([_request(0, seed=9)])
        names = [e[0] for e in fakes.log]
        assert names.index("seed") < names.index("template") < names.index("process") < names.index("generate")


# ---------------------------------------------------------------------------
# 10. Error mapping (SPEC §6.1 contract)
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_template_failure_is_isolated(self, loaded, fakes) -> None:
        fakes.processor.fail_on_prompt["bad"] = ValueError("template broke")
        requests = [_request(0), _request(1, prompt="bad"), _request(2)]
        results = loaded.caption_batch(requests)
        assert len(results) == 3
        assert results[1] == CaptionResult(
            text="", finish_reason="error", num_output_tokens=0, error="ValueError: template broke"
        )
        assert results[0].finish_reason == results[2].finish_reason == "stop"
        assert len(fakes.model.calls) == 2

    def test_processor_failure(self, loaded, fakes) -> None:
        fakes.processor.call_error = OSError("image unreadable")
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error" and result.error == "OSError: image unreadable"

    def test_non_string_template_is_an_error(self, loaded, fakes) -> None:
        fakes.processor.template_result = [1, 2, 3]
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error" and result.error.startswith("TypeError: ")
        assert fakes.processor.calls == []

    def test_generate_failure_other_than_oom(self, loaded, fakes) -> None:
        fakes.model.script = [RuntimeError("CUDA error: device-side assert triggered")]
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error"
        assert result.error == "RuntimeError: CUDA error: device-side assert triggered"
        assert result.text == "" and result.num_output_tokens == 0

    def test_decode_failure(self, loaded, fakes) -> None:
        fakes.processor.tokenizer.error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")
        (result,) = loaded.caption_batch([_request()])
        assert result.finish_reason == "error" and result.error.startswith("UnicodeDecodeError: ")

    def test_error_is_logged_with_request_key(self, loaded, fakes, caplog) -> None:
        fakes.model.script = [RuntimeError("boom")]
        with caplog.at_level(logging.WARNING, logger=joycaption_hf.logger.name):
            loaded.caption_batch([_request(7)])
        assert "celeste/0001/000007.jpg" in caplog.text and "long" in caplog.text and "boom" in caplog.text

    def test_cuda_oom_in_generate_is_raised(self, loaded, fakes) -> None:
        fakes.model.script = [[1, EOS], FakeOOM("CUDA out of memory")]
        with pytest.raises(FakeOOM, match="out of memory"):
            loaded.caption_batch([_request(0), _request(1), _request(2)])
        assert len(fakes.model.calls) == 2  # stops at the OOM

    def test_cuda_oom_before_generate_is_raised(self, loaded, fakes) -> None:
        fakes.processor.call_error = FakeOOM("CUDA out of memory while moving tensors")
        with pytest.raises(FakeOOM):
            loaded.caption_batch([_request()])

    def test_cuda_oom_through_caption_wrapper(self, loaded, fakes) -> None:
        fakes.model.script = [FakeOOM("oom")]
        with pytest.raises(FakeOOM):
            loaded.caption(_request())

    def test_keyboard_interrupt_propagates(self, loaded, fakes) -> None:
        fakes.model.script = [KeyboardInterrupt()]
        with pytest.raises(KeyboardInterrupt):
            loaded.caption_batch([_request()])

    def test_model_still_usable_after_errors(self, loaded, fakes) -> None:
        fakes.model.script = [RuntimeError("x")]
        loaded.caption_batch([_request()])
        (result,) = loaded.caption_batch([_request(1)])
        assert result.finish_reason == "stop" and loaded.loaded


# ---------------------------------------------------------------------------
# 11. describe() (SPEC §6.1, D11; proposal P4 not implemented)
# ---------------------------------------------------------------------------


DESCRIBE_KEYS = {
    "model_name",
    "backend",
    "model_path",
    "revision",
    "dtype",
    "quantization",
    "device_map",
    "quant_skip_modules",
    "torch_version",
    "transformers_version",
    "bitsandbytes_version",
}


class TestDescribe:
    def test_before_load(self, monkeypatch) -> None:
        monkeypatch.setattr(joycaption_hf, "_dist_version", lambda name: None)
        info = JoyCaptionHFModel().describe()
        assert set(info) == DESCRIBE_KEYS
        assert info["model_name"] == "joycaption-beta-one" and info["backend"] == "hf"
        assert all(info[k] is None for k in DESCRIBE_KEYS - {"model_name", "backend"})

    def test_after_load(self, make, monkeypatch) -> None:
        monkeypatch.setattr(joycaption_hf, "_dist_version", lambda name: {"bitsandbytes": "0.49.2"}.get(name))
        model = make()
        model.load({"model_path": "./Models", "revision": "deadbeef", "dtype": "float16",
                    "hf": {"quantization": "nf4", "device_map": 0}})
        info = model.describe()
        assert set(info) == DESCRIBE_KEYS
        assert info == {
            "model_name": "joycaption-beta-one",
            "backend": "hf",
            "model_path": "./Models",
            "revision": "deadbeef",
            "dtype": "float16",
            "quantization": "nf4",
            "device_map": 0,
            "quant_skip_modules": list(QUANT_SKIP_MODULES),
            "torch_version": "2.12.0+fake",
            "transformers_version": "5.12.0+fake",
            "bitsandbytes_version": "0.49.2",
        }
        json.dumps(info)

    def test_revision_is_recorded_as_configured(self, make) -> None:
        """D11: a null revision stays null (no resolved commit hash; proposal P4 is pending)."""
        model = make()
        model.load({})
        assert model.describe()["revision"] is None

    def test_no_quantization(self, make) -> None:
        model = make()
        model.load({"hf": {"quantization": None}})
        info = model.describe()
        assert info["quantization"] is None and info["quant_skip_modules"] is None

    def test_settings_kept_after_unload_and_new_dict_each_call(self, make) -> None:
        model = make()
        model.load({"model_path": "./Models"})
        model.unload()
        first, second = model.describe(), model.describe()
        assert first["model_path"] == "./Models"
        assert first == second and first is not second
        first["model_path"] = "changed"
        assert model.describe()["model_path"] == "./Models"

    def test_versions_from_installed_distributions_when_not_loaded(self, monkeypatch) -> None:
        monkeypatch.setattr(joycaption_hf, "_dist_version", lambda name: f"{name}-dist")
        info = JoyCaptionHFModel().describe()
        assert info["torch_version"] == "torch-dist"
        assert info["transformers_version"] == "transformers-dist"
        assert info["bitsandbytes_version"] == "bitsandbytes-dist"

    def test_dist_version_of_missing_package_is_none(self) -> None:
        assert joycaption_hf._dist_version("surely-not-installed-pkg-xyz") is None
