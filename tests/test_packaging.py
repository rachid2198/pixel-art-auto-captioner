"""Packaging checks (SPEC v1.1 §11.2, §14, §15; bug B9).

Static checks of ``pyproject.toml``, the requirements files and
``docker/Dockerfile``. No build, no network, no install is performed here.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
from pathlib import Path

import pytest

import pixel_art_auto_captioner

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
REQ_FILES = ("requirements.txt", "requirements-local.txt", "requirements-dev.txt")
DIST_NAME = "pixel-art-auto-captioner"
VLLM_TAG = "v0.33.0"  # TODO step 9: vllm==0.33.0


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _requirements(name: str) -> list[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _normalise(reqs: list[str]) -> list[str]:
    return sorted(r.replace(" ", "").lower() for r in reqs)


# ---------------------------------------------------------------------------
# pyproject.toml (§15)
# ---------------------------------------------------------------------------


class TestPyproject:
    def test_project_metadata(self) -> None:
        project = _pyproject()["project"]
        assert project["name"] == DIST_NAME
        assert project["requires-python"] == ">=3.10"
        assert project["dependencies"] == ["pillow>=10.0"]
        assert project["dynamic"] == ["version"]

    def test_extras_are_local_and_dev(self) -> None:
        extras = _pyproject()["project"]["optional-dependencies"]
        assert set(extras) == {"local", "dev"}

    def test_no_console_scripts(self) -> None:
        project = _pyproject()["project"]
        assert "scripts" not in project and "gui-scripts" not in project

    def test_src_layout_and_version_source(self) -> None:
        data = _pyproject()
        assert data["build-system"]["build-backend"] == "setuptools.build_meta"
        setuptools_cfg = data["tool"]["setuptools"]
        assert setuptools_cfg["packages"]["find"]["where"] == ["src"]
        assert setuptools_cfg["dynamic"]["version"] == {
            "attr": "pixel_art_auto_captioner.__version__"
        }

    def test_extras_match_requirements_files(self) -> None:
        project = _pyproject()["project"]
        extras = project["optional-dependencies"]
        assert _normalise(project["dependencies"]) == _normalise(_requirements("requirements.txt"))
        assert _normalise(extras["local"]) == _normalise(_requirements("requirements-local.txt"))
        assert _normalise(extras["dev"]) == _normalise(_requirements("requirements-dev.txt"))


# ---------------------------------------------------------------------------
# Requirements files (§15, B9)
# ---------------------------------------------------------------------------


class TestRequirements:
    @pytest.mark.parametrize("name", REQ_FILES)
    def test_no_index_options(self, name: str) -> None:
        """B9: no line may replace or extend the package index."""
        for line in _requirements(name):
            assert not line.startswith("-"), f"{name}: option line {line!r}"
            assert "index-url" not in line and "find-links" not in line

    @pytest.mark.parametrize("name", REQ_FILES)
    def test_torch_family_not_listed(self, name: str) -> None:
        names = {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].lower() for r in _requirements(name)}
        assert not names & {"torch", "torchvision", "torchaudio", "vllm"}

    def test_exact_contents(self) -> None:
        assert _requirements("requirements.txt") == ["pillow>=10.0"]
        assert _requirements("requirements-local.txt") == [
            "transformers==5.12.*",
            "accelerate==1.13.*",
            "bitsandbytes==0.49.*",
        ]
        assert _requirements("requirements-dev.txt") == ["pytest>=8"]


# ---------------------------------------------------------------------------
# docker/Dockerfile (§11.2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def instructions() -> list[str]:
    """Dockerfile instructions with line continuations joined, comments dropped."""
    text = DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


class TestDockerfile:
    def test_pinned_vllm_base(self, instructions: list[str]) -> None:
        froms = [ln for ln in instructions if ln.upper().startswith("FROM ")]
        assert froms == [f"FROM vllm/vllm-openai:{VLLM_TAG}"]

    def test_workdir_env_entrypoint(self, instructions: list[str]) -> None:
        assert "WORKDIR /app" in instructions
        env = " ".join(ln for ln in instructions if ln.startswith("ENV "))
        assert "HF_HOME=/workspace/hf-cache" in env
        assert "PYTHONPATH" not in "\n".join(instructions)  # Q17
        entry = [ln for ln in instructions if ln.startswith("ENTRYPOINT ")]
        assert len(entry) == 1 and "/app/scripts/run_caption.py" in entry[0]

    def test_install_keeps_base_packages(self, instructions: list[str]) -> None:
        runs = [ln for ln in instructions if ln.startswith("RUN ")]
        installs = [r for r in runs if "pip install" in r]
        assert installs and all("--no-deps" in r for r in installs)
        assert not any(re.search(r"\b(torch|vllm|transformers)\b", r.split("pip install", 1)[1].split("&&")[0])
                       for r in installs)

    def test_copies_only_code_and_configs(self, instructions: list[str]) -> None:
        copies = " ".join(ln for ln in instructions if ln.startswith(("COPY ", "ADD ")))
        assert "ADD " not in copies
        for forbidden in ("Models", "input", "output", ".env", "data/", ".git", " . "):
            assert forbidden not in copies, forbidden
        for needed in ("pyproject.toml", "README.md", "src/", "scripts/", "configs/"):
            assert needed in copies, needed

    def test_no_credentials(self, instructions: list[str]) -> None:
        text = "\n".join(instructions)
        assert "HF_TOKEN=" not in text and "hf_" not in text.lower().replace("hf_home", "")


# ---------------------------------------------------------------------------
# Installed package (AGENTS.md §5)
# ---------------------------------------------------------------------------


def test_conftest_does_not_edit_sys_path() -> None:
    tree = ast.parse((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    names = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not any(n.startswith("sys.path") for n in names)


def test_installed_metadata_matches_package() -> None:
    try:
        version = importlib.metadata.version(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("package not installed (run `pip install -e .` from the repo root)")
    assert version == pixel_art_auto_captioner.__version__
