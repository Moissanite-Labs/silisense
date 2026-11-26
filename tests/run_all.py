#!/usr/bin/env python3
"""Deterministic 3-mode isolated (-I) pure-stdlib test runner for SiliSense."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _RecordedTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.recorded_outcomes: list[tuple[str, str]] = []

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self.recorded_outcomes.append((test.id(), "ok"))

    def addFailure(self, test: unittest.case.TestCase, err) -> None:  # type: ignore[no-untyped-def]
        super().addFailure(test, err)
        self.recorded_outcomes.append((test.id(), "fail"))

    def addError(self, test: unittest.case.TestCase, err) -> None:  # type: ignore[no-untyped-def]
        super().addError(test, err)
        self.recorded_outcomes.append((test.id(), "error"))

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.recorded_outcomes.append((test.id(), "skip"))


def main() -> int:
    orig_open = builtins.open

    def _guarded_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        f_str = str(file)
        if "/dev/snd" in f_str or "/dev/video" in f_str or "/dev/v4l" in f_str or "/dev/media" in f_str:
            raise PermissionError(f"FORBIDDEN_LIVE_SENSOR_OPEN during test suite: {f_str}")
        return orig_open(file, *args, **kwargs)

    builtins.open = _guarded_open
    try:
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py", top_level_dir=str(REPO_ROOT))
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=1, resultclass=_RecordedTestResult)
        result: _RecordedTestResult = runner.run(suite)  # type: ignore[assignment]
    finally:
        builtins.open = orig_open

    ok = result.wasSuccessful() and len(result.skipped) == 0
    results_canon = json.dumps(result.recorded_outcomes, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    results_sha256 = hashlib.sha256(results_canon).hexdigest()
    summary = {
        "schema": "silisense.tests.run_all.v1",
        "passed": ok,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "results_sha256": results_sha256,
    }
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    summary["summary_sha256"] = hashlib.sha256(canon).hexdigest()
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    if not ok:
        sys.stderr.write(stream.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

import builtins
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    orig_open = builtins.open

    def _guarded_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        f_str = str(file)
        if "/dev/snd" in f_str or "/dev/video" in f_str or "/dev/v4l" in f_str or "/dev/media" in f_str:
            raise PermissionError(f"FORBIDDEN_LIVE_SENSOR_OPEN during test suite: {f_str}")
        return orig_open(file, *args, **kwargs)

    builtins.open = _guarded_open
    try:
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py", top_level_dir=str(REPO_ROOT))
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=1)
        result = runner.run(suite)
    finally:
        builtins.open = orig_open

    ok = result.wasSuccessful() and len(result.skipped) == 0
    summary = {
        "schema": "silisense.tests.run_all.v1",
        "passed": ok,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    summary["summary_sha256"] = hashlib.sha256(canon).hexdigest()
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    if not ok:
        sys.stderr.write(stream.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
