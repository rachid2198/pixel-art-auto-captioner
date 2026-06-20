"""Shared test fixtures for the pixel-art-auto-captioner test suite."""

import sys
from pathlib import Path

# Ensure the package is importable when running tests from the project root.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
