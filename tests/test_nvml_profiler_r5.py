"""Unit tests for fail-closed NVML 4-field energy & thermal profiler (CPU-only)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silisense.errors import HardwareExitCode
from silisense.nvml_profiler import (
    ENERGY_CONTROL_TAG,
    ENERGY_DISCLAIMER,
    REQUIRED_NVML_ENERGY_FIELDS,
    CtypesNvmlBackend,
    NvmlEnergyMonitor,
    NvmlEnergyProfiler,
    NvmlProfilerError,
    NvmlSample,
    run_nvml_profiler_selftest,
    summarize_nvml_samples,
    validate_nvml_energy_report,
)


class _StubClock:
    def __init__(self, times: list[float]) -> None:
        self._times = times
        self._idx = 0

    def __call__(self) -> float:
        v = self._times[min(self._idx, len(self._times) - 1)]
        self._idx += 1
        return v


class _StubBackend:
    def __init__(
        self,
        samples: list[tuple[float, int, int]],
        *,
        init_exc: Exception | None = None,
        energy_readings: tuple[int | None, int | None] = (500_000, 560_000),
    ) -> None:
        self._samples = samples
        self._idx = 0
        self._init_exc = init_exc
        self._energy = energy_readings
        self.closed = False

    def open_device(self, gpu_index: int) -> None:
        if self._init_exc is not None:
            raise self._init_exc

    def read_sample(self) -> tuple[float, int, int]:
        row = self._samples[min(self._idx, len(self._samples) - 1)]
        self._idx += 1
        return row

    def read_total_energy_mj(self) -> int | None:
        return self._energy[0] if self._idx == 0 else self._energy[1]

    def close_device(self) -> None:
        self.closed = True


class TestNvmlProfilerR5(unittest.TestCase):
    def test_01_trapezoidal_integration_and_four_fields(self) -> None:
        # Power rises linearly 60 W -> 80 W over [0.0, 2.0] s => exact 140.0 J, avg 70.0 W.
        samples = [
            NvmlSample(0.0, 60.0, 66, 1650),
            NvmlSample(1.0, 70.0, 74, 1700),
            NvmlSample(2.0, 80.0, 71, 1710),
        ]
        rep = summarize_nvml_samples(
            samples,
            total_tokens=280,
            raw_e0_mj=100_000,
            raw_e1_mj=240_000,
        )
        validate_nvml_energy_report(rep)
        for key in REQUIRED_NVML_ENERGY_FIELDS:
            self.assertIn(key, rep)
        self.assertTrue(math.isclose(rep["nvml_avg_power_w"], 70.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(rep["energy_j_per_tok"], 0.5, rel_tol=1e-6))
        self.assertTrue(math.isclose(rep["energy_tok_per_j"], 2.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(rep["energy_wh_run"], 140.0 / 3600.0, abs_tol=1e-6))
        self.assertEqual(rep["peak_temp_c"], 74)
        self.assertEqual(rep["start_temp_c"], 66)
        self.assertEqual(rep["end_temp_c"], 71)
        self.assertTrue(math.isclose(rep["raw_counter_total_j"], 140.0, rel_tol=1e-6))
        self.assertEqual(rep["energy_power_status"], ENERGY_CONTROL_TAG)
        self.assertEqual(rep["energy_disclaimer"], ENERGY_DISCLAIMER)

    def test_02_context_manager_offline_stub_and_alias(self) -> None:
        self.assertIs(NvmlEnergyMonitor, NvmlEnergyProfiler)
        clk = _StubClock([0.0, 0.0, 0.5, 1.0, 1.0])
        backend = _StubBackend([(60.0, 65, 1600), (60.0, 70, 1680), (60.0, 69, 1670)])
        with NvmlEnergyMonitor(gpu_index=0, interval_s=0.01, backend=backend, clock_fn=clk, background_poll=False) as mon:
            mon.sample_once()
        self.assertTrue(backend.closed)
        rep = mon.summary(total_tokens=120)
        validate_nvml_energy_report(rep)
        self.assertTrue(math.isclose(rep["nvml_avg_power_w"], 60.0, rel_tol=1e-6))
        self.assertEqual(rep["peak_temp_c"], 70)

    def test_03_fail_closed_init_and_non_fail_closed_payload(self) -> None:
        bad_backend = _StubBackend([], init_exc=NvmlProfilerError("ERR_NVML_INIT_FAILED", "no driver"))
        with self.assertRaises(NvmlProfilerError) as ctx:
            with NvmlEnergyProfiler(backend=bad_backend, fail_closed=True, background_poll=False):
                pass
        self.assertEqual(ctx.exception.code, "ERR_NVML_INIT_FAILED")
        self.assertTrue(bad_backend.closed)

        soft_backend = _StubBackend([], init_exc=NvmlProfilerError("ERR_NVML_INIT_FAILED", "no driver"))
        with NvmlEnergyProfiler(backend=soft_backend, fail_closed=False, background_poll=False) as mon:
            pass
        soft_rep = mon.summary(total_tokens=16, fail_closed=False)
        self.assertFalse(soft_rep["nvml_ok"])
        self.assertEqual(soft_rep["energy_power_status"], ENERGY_CONTROL_TAG)
        self.assertEqual(soft_rep["energy_disclaimer"], ENERGY_DISCLAIMER)

    def test_04_negative_controls_reject_invalid_inputs_and_tampering(self) -> None:
        with self.assertRaises(NvmlProfilerError):
            NvmlSample(0.0, float("nan"), 60)
        with self.assertRaises(NvmlProfilerError):
            NvmlSample(0.0, -1.0, 60)
        with self.assertRaises(NvmlProfilerError):
            NvmlEnergyProfiler(gpu_index=True)  # type: ignore[arg-type]
        with self.assertRaises(NvmlProfilerError):
            NvmlEnergyProfiler(interval_s=True)  # type: ignore[arg-type]
        with self.assertRaises(NvmlProfilerError) as zero_tok:
            summarize_nvml_samples([NvmlSample(0.0, 50.0, 60), NvmlSample(1.0, 50.0, 61)], total_tokens=0)
        self.assertEqual(zero_tok.exception.code, "ERR_NVML_INVALID_ARG")

        with self.assertRaises(NvmlProfilerError) as mono_err:
            summarize_nvml_samples([NvmlSample(2.0, 50.0, 60), NvmlSample(1.0, 50.0, 61)], total_tokens=10)
        self.assertEqual(mono_err.exception.code, "ERR_NVML_NON_MONOTONIC_CLOCK")

        with self.assertRaises(NvmlProfilerError) as zero_e:
            summarize_nvml_samples([NvmlSample(0.0, 0.0, 60), NvmlSample(1.0, 0.0, 61)], total_tokens=10)
        self.assertEqual(zero_e.exception.code, "ERR_NVML_ZERO_ENERGY")

        with self.assertRaises(NvmlProfilerError) as therm_err:
            summarize_nvml_samples(
                [NvmlSample(0.0, 75.0, 80), NvmlSample(1.0, 75.0, 92)],
                total_tokens=10,
                max_temp_c=85.0,
            )
        self.assertEqual(therm_err.exception.code, "ERR_NVML_THERMAL_LIMIT")
        self.assertEqual(therm_err.exception.exit_code, HardwareExitCode.THERMAL_STOP)

        valid = summarize_nvml_samples([NvmlSample(0.0, 50.0, 62), NvmlSample(1.0, 50.0, 64)], total_tokens=25)
        for bad_patch in (
            {"energy_power_status": "CONTROLLED"},
            {"energy_disclaimer": "altered"},
            {"nvml_avg_power_w": 999.0},
            {"energy_j_per_tok": 999.0},
            {"summary_sha256": "0" * 64},
        ):
            with self.assertRaises(NvmlProfilerError):
                validate_nvml_energy_report({**valid, **bad_patch})

        with self.assertRaises(NvmlProfilerError) as missing_lib:
            CtypesNvmlBackend("libnonexistent_nvml_test.so").open_device(0)
        self.assertEqual(missing_lib.exception.code, "ERR_NVML_LIBRARY_LOAD")

    def test_05_selftest_deterministic(self) -> None:
        r1 = run_nvml_profiler_selftest()
        r2 = run_nvml_profiler_selftest()
        self.assertTrue(r1["passed"])
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
