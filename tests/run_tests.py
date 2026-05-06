"""Minimal stdlib runner for the repository's pytest-style smoke tests."""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


TEST_MODULES = [
    "tests.test_gmm_generator",
    "tests.test_generator_training_smoke",
    "tests.test_wasserstein",
    "tests.test_reproducibility_scripts",
]


def run_test(function) -> tuple[bool, str]:
    signature = inspect.signature(function)
    kwargs = {}
    temp_dirs = []
    try:
        if "tmp_path" in signature.parameters:
            temp_dir = tempfile.TemporaryDirectory()
            temp_dirs.append(temp_dir)
            kwargs["tmp_path"] = Path(temp_dir.name)
        function(**kwargs)
        return True, ""
    except Exception:
        return False, traceback.format_exc()
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


def main() -> int:
    failures = []
    total = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            total += 1
            ok, error = run_test(function)
            status = "ok" if ok else "FAIL"
            print(f"{status} {module_name}.{name}")
            if not ok:
                failures.append((module_name, name, error))

    if failures:
        for module_name, name, error in failures:
            print(f"\n{module_name}.{name}\n{error}", file=sys.stderr)
        print(f"\n{len(failures)} failed, {total - len(failures)} passed", file=sys.stderr)
        return 1
    print(f"\n{total} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

