from __future__ import annotations

import re
import subprocess
from pathlib import Path


def verify_math_answer(answer: str, expected: str | float | int) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if not numbers:
        return 0.0
    try:
        expected_value = float(expected)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if abs(float(numbers[-1]) - expected_value) < 1e-6 else 0.0


def verify_code_compiles(source: str, language: str = "python", timeout_seconds: int = 30) -> float:
    if language != "python":
        return 0.0
    path = Path("tmp_singularity_check.py")
    path.write_text(source, encoding="utf-8")
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        path.unlink(missing_ok=True)
