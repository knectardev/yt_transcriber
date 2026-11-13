"""Optional Node harness for reader system-nav / position helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "reader_system_nav_harness.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_reader_system_nav_harness():
    completed = subprocess.run(
        ["node", str(HARNESS)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "reader harness passed" in completed.stdout
