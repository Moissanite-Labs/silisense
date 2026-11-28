"""Unit tests for fail-closed host CPU-RAM + GPU single-loader resource guard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silisense.errors import HardwareExitCode
from silisense.resource_guard import (
    GUARD_ENV_ACTIVE,
    GUARD_ENV_LOCK,
    HostResourceGuardError,
    _build_exec_child_env,
    _release_lock,
    _require_pos_number,
    evaluate_host_resources,
    run_self_test,
)


class TestResourceGuardR5(unittest.TestCase):
    def test_01_selftest_covers_lock_psi_gpu_timeout_and_r4(self) -> None:
        res = run_self_test()
        self.assertTrue(res["passed"])
        self.assertTrue(res["r4_parity"])
        self.assertGreaterEqual(res["negative_controls_passed"], 10)

    def test_02_exec_exports_env_and_holds_flock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="silisense_rg_") as tmp_str:
            tmp = Path(tmp_str)
            lock_path = tmp / "locks" / "exec.lock"
            mem_ok = tmp / "meminfo"
            mem_ok.write_text("MemTotal: 32535272 kB\nMemAvailable: 22320192 kB\n", encoding="utf-8")
            psi_ok = tmp / "psi"
            psi_ok.write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n", encoding="utf-8")

            script_path = REPO_ROOT / "silisense" / "resource_guard.py"
            child_code = (
                "import os, subprocess, sys; "
                f"ok = (os.environ.get({GUARD_ENV_ACTIVE!r}) == '1' and os.environ.get({GUARD_ENV_LOCK!r}) == sys.argv[1]); "
                "sub = subprocess.run([sys.executable, '-I', '-B', sys.argv[2], '--check', "
                "'--lock-file', sys.argv[1], '--meminfo-file', sys.argv[3], '--psi-file', sys.argv[4], "
                "'--skip-proc-scan', '--json'], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False); "
                "sys.exit(0 if (ok and sub.returncode == 2 and '\"LOADER_LOCK_HELD\"' in sub.stdout) else 9)"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(script_path),
                    "--lock-file",
                    str(lock_path),
                    "--meminfo-file",
                    str(mem_ok),
                    "--psi-file",
                    str(psi_ok),
                    "--skip-proc-scan",
                    "--exec",
                    "--",
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    child_code,
                    str(lock_path),
                    str(script_path),
                    str(mem_ok),
                    str(psi_ok),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", errors="replace"))

            # Lock must be released after --exec finishes.
            rep, held = evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=lock_path,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
                hold_lock=False,
            )
            self.assertTrue(rep["passed"])
            self.assertIsNone(held)

    def test_03_negative_controls_traversal_and_exit_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="silisense_rg_neg_") as tmp_str:
            tmp = Path(tmp_str)
            mem_ok = tmp / "meminfo"
            mem_ok.write_text("MemTotal: 32535272 kB\nMemAvailable: 22320192 kB\n", encoding="utf-8")
            mem_low = tmp / "meminfo_low"
            mem_low.write_text("MemTotal: 32535272 kB\nMemAvailable: 1024 kB\n", encoding="utf-8")
            psi_high = tmp / "psi_high"
            psi_high.write_text("some avg10=40.00 avg60=10.00 avg300=1.00 total=10\n", encoding="utf-8")

            with self.assertRaises(HostResourceGuardError) as trav:
                evaluate_host_resources(
                    lock_path=tmp / ".." / "escape.lock",
                    meminfo_path=mem_ok,
                    skip_proc_scan=True,
                )
            self.assertEqual(trav.exception.code, "PATH_TRAVERSAL_REFUSED")

            with self.assertRaises(HostResourceGuardError) as oom:
                evaluate_host_resources(
                    min_mem_gib=12.0,
                    lock_path=tmp / "a.lock",
                    meminfo_path=mem_low,
                    skip_proc_scan=True,
                )
            self.assertEqual(oom.exception.code, "INSUFFICIENT_CPU_RAM")
            self.assertEqual(oom.exception.exit_code, HardwareExitCode.OOM)

            with self.assertRaises(HostResourceGuardError) as psi:
                evaluate_host_resources(
                    min_mem_gib=12.0,
                    lock_path=tmp / "b.lock",
                    meminfo_path=mem_ok,
                    psi_path=psi_high,
                    skip_proc_scan=True,
                )
            self.assertEqual(psi.exception.code, "HIGH_MEMORY_PRESSURE")
            self.assertEqual(psi.exception.exit_code, HardwareExitCode.MEM_PRESSURE)

            with self.assertRaises(HostResourceGuardError):
                _require_pos_number(-1.0, "timeout_seconds", allow_zero=True)
            with self.assertRaises(HostResourceGuardError):
                _require_pos_number(True, "timeout_seconds", allow_zero=True)

            gpu_env = _build_exec_child_env({"PATH": "/usr/bin"}, lock_path=str(tmp / "l.lock"), require_gpu_idle=True)
            self.assertEqual(gpu_env.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")
            cpu_env = _build_exec_child_env({}, lock_path=str(tmp / "l.lock"), require_gpu_idle=False)
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", cpu_env)

            # Verify held lock manual release helper cleanly unlocks.
            _, fd = evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=tmp / "c.lock",
                meminfo_path=mem_ok,
                skip_proc_scan=True,
                hold_lock=True,
            )
            self.assertIsNotNone(fd)
            if fd is not None:
                _release_lock(fd)


if __name__ == "__main__":
    unittest.main()
