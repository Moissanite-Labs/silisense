#!/usr/bin/env python3
"""Fail-closed 100 Hz NVML 4-field energy & thermal profiler for SiliSense.

Computes trapezoidal energy integration across `(power_w, timestamp_s)` samples and
reports the canonical 4 energy fields (`nvml_avg_power_w`, `energy_j_per_tok`,
`energy_tok_per_j`, `energy_wh_run`) plus `peak_temp_c`, unconditionally tagged
with `UNCONTROLLED_LAPTOP_NOT_CLAIMED` and the mandatory uncontrolled-variables
energy disclaimer.

Pure Python stdlib (`ctypes` only when querying `libnvidia-ml.so.1` live),
zero bare `assert`s, LF-only, R4-parity safe across `-I -B`, `-I -O -B`, and
`-I -B -W error`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense.errors import HardwareExitCode, SiliSenseValueError


NVML_PROFILER_SCHEMA = "silisense.nvml.energy_profile.v1"
NVML_SELFTEST_SCHEMA = "silisense.nvml.self_test.v1"

ENERGY_CONTROL_TAG = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
ENERGY_DISCLAIMER = (
    "Energy readings are not guaranteed to be consistent due to uncontrollable variables "
    "(thermal state, ambient temperature, fan curve, and dynamic power limits)."
)
REQUIRED_NVML_ENERGY_FIELDS: tuple[str, ...] = (
    "nvml_avg_power_w",
    "energy_j_per_tok",
    "energy_tok_per_j",
    "energy_wh_run",
    "peak_temp_c",
)

_NVML_TEMPERATURE_GPU = 0
_NVML_CLOCK_SM = 1
_MAX_SAMPLES_DEFAULT = 2_000_000


class NvmlProfilerError(SiliSenseValueError):
    """Fail-closed NVML profiler error with stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        exit_code: HardwareExitCode = HardwareExitCode.INFRA_ERROR,
    ) -> None:
        super().__init__(code, message, exit_code=exit_code)


def _require_nvml(
    condition: bool,
    code: str,
    message: str,
    *,
    exit_code: HardwareExitCode = HardwareExitCode.INFRA_ERROR,
) -> None:
    if not condition:
        raise NvmlProfilerError(code, message, exit_code=exit_code)


def _require_nonbool_int(val: Any, name: str, *, minimum: int = 0) -> int:
    _require_nvml(
        type(val) is int,
        "ERR_NVML_INVALID_ARG",
        f"{name} must be strict int (not {type(val).__name__})",
        exit_code=HardwareExitCode.PREFLIGHT_FAIL,
    )
    _require_nvml(
        val >= minimum,
        "ERR_NVML_INVALID_ARG",
        f"{name} must be >= {minimum}, got {val}",
        exit_code=HardwareExitCode.PREFLIGHT_FAIL,
    )
    return val


def _require_finite_float(
    val: Any,
    name: str,
    *,
    allow_zero: bool = True,
    max_val: float | None = None,
) -> float:
    _require_nvml(
        type(val) in (int, float),
        "ERR_NVML_INVALID_ARG",
        f"{name} must be int or float (not {type(val).__name__})",
        exit_code=HardwareExitCode.PREFLIGHT_FAIL,
    )
    fval = float(val)
    _require_nvml(
        math.isfinite(fval),
        "ERR_NVML_NON_FINITE",
        f"{name} must be finite, got {val!r}",
        exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
    )
    if allow_zero:
        _require_nvml(
            fval >= 0.0,
            "ERR_NVML_NEGATIVE_VALUE",
            f"{name} must be >= 0, got {fval}",
            exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
        )
    else:
        _require_nvml(
            fval > 0.0,
            "ERR_NVML_NON_POSITIVE",
            f"{name} must be > 0, got {fval}",
            exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
        )
    if max_val is not None:
        _require_nvml(
            fval <= max_val,
            "ERR_NVML_OUT_OF_RANGE",
            f"{name} must be <= {max_val}, got {fval}",
            exit_code=HardwareExitCode.PREFLIGHT_FAIL,
        )
    return fval


def canonical_nvml_sha256(payload: Mapping[str, Any], *, exclude_key: str = "summary_sha256") -> str:
    """Compute canonical SHA-256 hex digest of an NVML energy payload."""
    filtered = {k: v for k, v in payload.items() if k != exclude_key}
    raw = json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NvmlSample:
    """Single validated GPU power/thermal/clock sample."""

    timestamp_s: float
    power_w: float
    temp_c: int
    sm_clock_mhz: int = 0

    def __post_init__(self) -> None:
        _require_finite_float(self.timestamp_s, "timestamp_s", allow_zero=True)
        _require_finite_float(self.power_w, "power_w", allow_zero=True)
        _require_nonbool_int(self.temp_c, "temp_c", minimum=0)
        _require_nonbool_int(self.sm_clock_mhz, "sm_clock_mhz", minimum=0)


class NvmlBackend(Protocol):
    """Abstract backend interface enabling offline deterministic testing."""

    def open_device(self, gpu_index: int) -> None: ...
    def read_sample(self) -> tuple[float, int, int]: ...
    def read_total_energy_mj(self) -> int | None: ...
    def close_device(self) -> None: ...


class CtypesNvmlBackend:
    """Fail-closed ctypes binding to `libnvidia-ml.so.1`."""

    def __init__(self, library_name: str = "libnvidia-ml.so.1") -> None:
        self._library_name = library_name
        self._lib: Any = None
        self._dev: ctypes.c_void_p | None = None
        self._initialized = False

    def open_device(self, gpu_index: int) -> None:
        idx = _require_nonbool_int(gpu_index, "gpu_index", minimum=0)
        try:
            lib = ctypes.CDLL(self._library_name)
        except OSError as exc:
            raise NvmlProfilerError("ERR_NVML_LIBRARY_LOAD", f"Cannot load {self._library_name}: {exc}") from exc

        try:
            lib.nvmlInit_v2.argtypes = []
            lib.nvmlInit_v2.restype = ctypes.c_int
            lib.nvmlShutdown.argtypes = []
            lib.nvmlShutdown.restype = ctypes.c_int
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            lib.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
            lib.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
            lib.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int
            lib.nvmlDeviceGetTotalEnergyConsumption.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
            lib.nvmlDeviceGetTotalEnergyConsumption.restype = ctypes.c_int
        except AttributeError as exc:
            raise NvmlProfilerError("ERR_NVML_SYMBOL_MISSING", f"Missing NVML symbol: {exc}") from exc

        rc_init = int(lib.nvmlInit_v2())
        _require_nvml(rc_init == 0, "ERR_NVML_INIT_FAILED", f"nvmlInit_v2 returned rc={rc_init}")

        dev = ctypes.c_void_p()
        rc_handle = int(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(idx), ctypes.byref(dev)))
        if rc_handle != 0:
            try:
                lib.nvmlShutdown()
            except Exception:
                pass
            raise NvmlProfilerError(
                "ERR_NVML_HANDLE_FAILED",
                f"nvmlDeviceGetHandleByIndex_v2({idx}) returned rc={rc_handle}",
            )

        self._lib = lib
        self._dev = dev
        self._initialized = True

    def read_sample(self) -> tuple[float, int, int]:
        _require_nvml(
            self._initialized and self._lib is not None and self._dev is not None,
            "ERR_NVML_NOT_OPEN",
            "NVML device is not open",
        )
        mw = ctypes.c_uint(0)
        temp = ctypes.c_uint(0)
        clk = ctypes.c_uint(0)
        rc_p = int(self._lib.nvmlDeviceGetPowerUsage(self._dev, ctypes.byref(mw)))
        _require_nvml(rc_p == 0, "ERR_NVML_POWER_READ_FAILED", f"nvmlDeviceGetPowerUsage returned rc={rc_p}")
        rc_t = int(self._lib.nvmlDeviceGetTemperature(self._dev, _NVML_TEMPERATURE_GPU, ctypes.byref(temp)))
        _require_nvml(rc_t == 0, "ERR_NVML_TEMP_READ_FAILED", f"nvmlDeviceGetTemperature returned rc={rc_t}")
        rc_c = int(self._lib.nvmlDeviceGetClockInfo(self._dev, _NVML_CLOCK_SM, ctypes.byref(clk)))
        sm_mhz = int(clk.value) if rc_c == 0 else 0
        return float(mw.value) / 1000.0, int(temp.value), sm_mhz

    def read_total_energy_mj(self) -> int | None:
        if not self._initialized or self._lib is None or self._dev is None:
            return None
        mj = ctypes.c_ulonglong(0)
        rc = int(self._lib.nvmlDeviceGetTotalEnergyConsumption(self._dev, ctypes.byref(mj)))
        if rc != 0:
            return None
        return int(mj.value)

    def close_device(self) -> None:
        if self._initialized and self._lib is not None:
            try:
                self._lib.nvmlShutdown()
            finally:
                self._initialized = False
                self._dev = None
                self._lib = None


def summarize_nvml_samples(
    samples: Sequence[NvmlSample],
    *,
    total_tokens: int,
    wall_s: float | None = None,
    raw_e0_mj: int | None = None,
    raw_e1_mj: int | None = None,
    max_temp_c: float | None = None,
    require_positive_temp: bool = True,
) -> dict[str, Any]:
    """Integrate `(timestamp_s, power_w)` via trapezoid rule and validate all 4 energy + thermal fields."""
    tok_n = _require_nonbool_int(total_tokens, "total_tokens", minimum=1)
    _require_nvml(
        len(samples) >= 2,
        "ERR_NVML_INSUFFICIENT_SAMPLES",
        f"Need >= 2 NVML samples for trapezoidal integration, got {len(samples)}",
        exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
    )
    if max_temp_c is not None:
        max_temp_limit = _require_finite_float(max_temp_c, "max_temp_c", allow_zero=False)
    else:
        max_temp_limit = None

    validated: list[NvmlSample] = []
    for idx, s in enumerate(samples):
        _require_nvml(
            isinstance(s, NvmlSample),
            "ERR_NVML_SAMPLE_TYPE",
            f"Sample {idx} must be NvmlSample, got {type(s).__name__}",
        )
        if idx > 0:
            _require_nvml(
                s.timestamp_s >= validated[-1].timestamp_s,
                "ERR_NVML_NON_MONOTONIC_CLOCK",
                f"Sample {idx} timestamp {s.timestamp_s} < previous {validated[-1].timestamp_s}",
                exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
            )
        validated.append(s)

    span_s = validated[-1].timestamp_s - validated[0].timestamp_s
    eff_wall_s = span_s if wall_s is None else _require_finite_float(wall_s, "wall_s", allow_zero=False)
    _require_nvml(
        eff_wall_s > 0.0,
        "ERR_NVML_ZERO_DURATION",
        f"Integration window duration must be > 0, got {eff_wall_s}",
        exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
    )

    tot_j = 0.0
    for idx in range(1, len(validated)):
        prev = validated[idx - 1]
        cur = validated[idx]
        dt = cur.timestamp_s - prev.timestamp_s
        tot_j += 0.5 * (prev.power_w + cur.power_w) * dt

    _require_nvml(
        math.isfinite(tot_j) and tot_j > 0.0,
        "ERR_NVML_ZERO_ENERGY",
        f"Integrated NVML energy must be finite and > 0 J, got {tot_j!r}",
        exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
    )

    powers = [s.power_w for s in validated]
    temps = [s.temp_c for s in validated if s.temp_c > 0]
    if require_positive_temp:
        _require_nvml(
            bool(temps),
            "ERR_NVML_MISSING_THERMAL",
            "Thermal telemetry missing (no positive GPU temperature samples recorded)",
            exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
        )
    peak_c = max(temps) if temps else 0
    if max_temp_limit is not None and float(peak_c) > max_temp_limit:
        raise NvmlProfilerError(
            "ERR_NVML_THERMAL_LIMIT",
            f"Peak GPU temperature {peak_c} C exceeded limit {max_temp_limit:.1f} C",
            exit_code=HardwareExitCode.THERMAL_STOP,
        )

    clks = [s.sm_clock_mhz for s in validated if s.sm_clock_mhz > 0]
    tok_f = float(tok_n)
    avg_w = tot_j / eff_wall_s
    raw_j: float | None = None
    if raw_e0_mj is not None and raw_e1_mj is not None:
        e0 = _require_nonbool_int(raw_e0_mj, "raw_e0_mj", minimum=0)
        e1 = _require_nonbool_int(raw_e1_mj, "raw_e1_mj", minimum=0)
        _require_nvml(
            e1 >= e0,
            "ERR_NVML_COUNTER_REGRESSION",
            f"Total energy counter regressed: {e1} mJ < {e0} mJ",
            exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
        )
        raw_j = float(e1 - e0) / 1000.0

    payload: dict[str, Any] = {
        "schema": NVML_PROFILER_SCHEMA,
        "nvml_ok": True,
        "nvml_available": True,
        "nvml_samples": len(validated),
        "wall_s": round(eff_wall_s, 6),
        "total_tokens": tok_n,
        "tok_per_s": round(tok_f / eff_wall_s, 4),
        "ms_per_tok": round((eff_wall_s * 1000.0) / tok_f, 4),
        "nvml_avg_power_w": round(avg_w, 4),
        "nvml_peak_power_w": round(max(powers), 4),
        "energy_j": round(tot_j, 6),
        "total_j": round(tot_j, 6),
        "energy_wh_run": round(tot_j / 3600.0, 6),
        "energy_j_per_tok": round(tot_j / tok_f, 6),
        "energy_tok_per_j": round(tok_f / tot_j, 4),
        "start_temp_c": temps[0] if temps else 0,
        "peak_temp_c": peak_c,
        "end_temp_c": temps[-1] if temps else 0,
        "mean_sm_mhz": round(sum(clks) / float(len(clks)), 1) if clks else 0.0,
        "raw_counter_total_j": round(raw_j, 6) if raw_j is not None else None,
        "energy_power_status": ENERGY_CONTROL_TAG,
        "energy_disclaimer": ENERGY_DISCLAIMER,
    }
    payload["summary_sha256"] = canonical_nvml_sha256(payload)
    return payload


def validate_nvml_energy_report(
    report: Mapping[str, Any],
    *,
    require_positive_temp: bool = True,
    verify_sha256: bool = True,
) -> Mapping[str, Any]:
    """Fail-closed verifier for NVML 4-field energy, thermal, and disclaimer contracts."""
    _require_nvml(isinstance(report, Mapping), "ERR_NVML_SCHEMA", "Report must be a mapping")
    _require_nvml(report.get("nvml_ok") is True, "ERR_NVML_NOT_OK", "nvml_ok must be strictly True")
    _require_nvml(report.get("nvml_available") is True, "ERR_NVML_UNAVAILABLE", "nvml_available must be strictly True")
    _require_nvml(
        report.get("energy_power_status") == ENERGY_CONTROL_TAG,
        "ERR_NVML_STATUS_TAG",
        f"energy_power_status must equal {ENERGY_CONTROL_TAG!r}",
    )
    _require_nvml(
        report.get("energy_disclaimer") == ENERGY_DISCLAIMER,
        "ERR_NVML_DISCLAIMER",
        "Missing or altered mandatory uncontrolled-laptop energy disclaimer",
    )

    samples_n = _require_nonbool_int(report.get("nvml_samples"), "nvml_samples", minimum=2)
    _ = samples_n
    total_tokens = _require_nonbool_int(report.get("total_tokens"), "total_tokens", minimum=1)
    wall_s = _require_finite_float(report.get("wall_s"), "wall_s", allow_zero=False)
    total_j = _require_finite_float(report.get("total_j"), "total_j", allow_zero=False)

    parsed_fields: dict[str, float] = {}
    for key in REQUIRED_NVML_ENERGY_FIELDS:
        _require_nvml(key in report, "ERR_NVML_MISSING_FIELD", f"Missing required NVML field {key!r}")
        allow_zero_temp = (key == "peak_temp_c") and (not require_positive_temp)
        parsed_fields[key] = _require_finite_float(report[key], key, allow_zero=allow_zero_temp)

    avg_w = parsed_fields["nvml_avg_power_w"]
    j_per_tok = parsed_fields["energy_j_per_tok"]
    tok_per_j = parsed_fields["energy_tok_per_j"]
    wh_run = parsed_fields["energy_wh_run"]

    # Cross-field consistency invariants against rounded fields (tolerant of display rounding).
    _require_nvml(
        abs((avg_w * wall_s) - total_j) <= max(0.01 * total_j, 1e-3),
        "ERR_NVML_POWER_DURATION_DRIFT",
        f"nvml_avg_power_w * wall_s ({avg_w * wall_s:.6f}) inconsistent with total_j ({total_j:.6f})",
    )
    _require_nvml(
        abs((j_per_tok * float(total_tokens)) - total_j) <= max(0.01 * total_j, 1e-3),
        "ERR_NVML_TOKEN_ENERGY_DRIFT",
        "energy_j_per_tok * total_tokens inconsistent with total_j",
    )
    _require_nvml(
        abs((j_per_tok * tok_per_j) - 1.0) <= 0.02,
        "ERR_NVML_RECIPROCAL_DRIFT",
        f"energy_j_per_tok * energy_tok_per_j = {j_per_tok * tok_per_j:.6f} != 1.0",
    )
    expected_wh = total_j / 3600.0
    _require_nvml(
        abs(wh_run - expected_wh) <= max(0.01 * expected_wh, 2e-6),
        "ERR_NVML_WH_CONVERSION_DRIFT",
        f"energy_wh_run ({wh_run:.6f}) inconsistent with total_j / 3600 ({expected_wh:.6f})",
    )

    if verify_sha256 and "summary_sha256" in report:
        expected_sha = canonical_nvml_sha256(report)
        _require_nvml(
            report.get("summary_sha256") == expected_sha,
            "ERR_NVML_SHA_MISMATCH",
            "NVML summary_sha256 mismatch",
        )
    return report


class NvmlEnergyProfiler:
    """100 Hz NVML trapezoidal energy & thermal sampler with fail-closed invariants."""

    def __init__(
        self,
        gpu_index: int = 0,
        interval_s: float = 0.01,
        *,
        backend: NvmlBackend | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        fail_closed: bool = True,
        max_temp_c: float | None = None,
        background_poll: bool = True,
        max_samples: int = _MAX_SAMPLES_DEFAULT,
    ) -> None:
        self.gpu_index = _require_nonbool_int(gpu_index, "gpu_index", minimum=0)
        self.interval_s = _require_finite_float(interval_s, "interval_s", allow_zero=False, max_val=10.0)
        _require_nvml(type(fail_closed) is bool, "ERR_NVML_INVALID_ARG", "fail_closed must be bool")
        _require_nvml(type(background_poll) is bool, "ERR_NVML_INVALID_ARG", "background_poll must be bool")
        self._max_samples = _require_nonbool_int(max_samples, "max_samples", minimum=2)
        self.max_temp_c = (
            _require_finite_float(max_temp_c, "max_temp_c", allow_zero=False)
            if max_temp_c is not None
            else None
        )
        self._backend: NvmlBackend = CtypesNvmlBackend() if backend is None else backend
        self._clock_fn = clock_fn
        self._fail_closed = fail_closed
        self._background_poll = background_poll

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[NvmlSample] = []
        self._t0 = 0.0
        self._t1 = 0.0
        self._e0_mj: int | None = None
        self._e1_mj: int | None = None
        self.nvml_ok = False
        self.err_code = ""
        self.err_msg = ""

    def __enter__(self) -> "NvmlEnergyProfiler":
        try:
            self._backend.open_device(self.gpu_index)
            self.nvml_ok = True
            self._e0_mj = self._backend.read_total_energy_mj()
            self._t0 = float(self._clock_fn())
            self.sample_once()
            if self._background_poll:
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
        except NvmlProfilerError as exc:
            self.nvml_ok = False
            self.err_code = exc.code
            self.err_msg = exc.message
            try:
                self._backend.close_device()
            except Exception:
                pass
            if self._fail_closed:
                raise
        except Exception as exc:
            self.nvml_ok = False
            self.err_code = "ERR_NVML_RUNTIME"
            self.err_msg = str(exc)
            try:
                self._backend.close_device()
            except Exception:
                pass
            if self._fail_closed:
                raise NvmlProfilerError(self.err_code, self.err_msg) from exc
        return self

    def record_sample(
        self,
        *,
        timestamp_s: float,
        power_w: float,
        temp_c: int,
        sm_clock_mhz: int = 0,
    ) -> NvmlSample:
        """Append a validated manual/synthetic sample (thread-safe)."""
        sample = NvmlSample(
            timestamp_s=timestamp_s,
            power_w=power_w,
            temp_c=temp_c,
            sm_clock_mhz=sm_clock_mhz,
        )
        with self._lock:
            _require_nvml(
                len(self._samples) < self._max_samples,
                "ERR_NVML_SAMPLE_CAP",
                f"Sample cap exceeded ({self._max_samples})",
            )
            if self._samples:
                _require_nvml(
                    sample.timestamp_s >= self._samples[-1].timestamp_s,
                    "ERR_NVML_NON_MONOTONIC_CLOCK",
                    f"Sample timestamp {sample.timestamp_s} < previous {self._samples[-1].timestamp_s}",
                    exit_code=HardwareExitCode.SIGNAL_GATE_FAIL,
                )
            self._samples.append(sample)
        return sample

    def sample_once(self) -> NvmlSample | None:
        if not self.nvml_ok:
            return None
        try:
            power_w, temp_c, sm_mhz = self._backend.read_sample()
            ts = float(self._clock_fn())
            return self.record_sample(
                timestamp_s=ts,
                power_w=power_w,
                temp_c=temp_c,
                sm_clock_mhz=sm_mhz,
            )
        except NvmlProfilerError as exc:
            self.err_code = exc.code
            self.err_msg = exc.message
            if self._fail_closed:
                raise
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample_once()
            except Exception as exc:
                self.err_code = getattr(exc, "code", "ERR_NVML_POLL_FAILED")
                self.err_msg = str(exc)
                break

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.nvml_ok:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            try:
                self.sample_once()
                self._t1 = float(self._clock_fn())
                if self._e0_mj is not None:
                    self._e1_mj = self._backend.read_total_energy_mj()
            except NvmlProfilerError as sample_exc:
                self.err_code = sample_exc.code
                self.err_msg = sample_exc.message
                if exc_type is None and self._fail_closed:
                    self._backend.close_device()
                    raise
            finally:
                self._backend.close_device()

    def summary(
        self,
        total_tokens: int,
        *,
        fail_closed: bool | None = None,
        require_positive_temp: bool = True,
    ) -> dict[str, Any]:
        """Finalize energy, thermal, and disclaimer report for `total_tokens`."""
        strict = self._fail_closed if fail_closed is None else bool(fail_closed)
        if not self.nvml_ok or self.err_code:
            code = self.err_code or "ERR_NVML_UNAVAILABLE"
            msg = self.err_msg or "NVML profiler did not initialize cleanly"
            if strict:
                raise NvmlProfilerError(code, msg)
            tok_n = _require_nonbool_int(total_tokens, "total_tokens", minimum=1)
            wall_s = max(self._t1 - self._t0, 0.0)
            return {
                "schema": NVML_PROFILER_SCHEMA,
                "nvml_ok": False,
                "nvml_available": False,
                "error_code": code,
                "error": msg,
                "wall_s": round(wall_s, 6),
                "total_tokens": tok_n,
                "energy_power_status": ENERGY_CONTROL_TAG,
                "energy_disclaimer": ENERGY_DISCLAIMER,
            }

        with self._lock:
            samples = list(self._samples)
        wall_s = (self._t1 - self._t0) if self._t1 > self._t0 else None
        try:
            report = summarize_nvml_samples(
                samples,
                total_tokens=total_tokens,
                wall_s=wall_s,
                raw_e0_mj=self._e0_mj,
                raw_e1_mj=self._e1_mj,
                max_temp_c=self.max_temp_c,
                require_positive_temp=require_positive_temp,
            )
            validate_nvml_energy_report(report, require_positive_temp=require_positive_temp)
            return report
        except NvmlProfilerError as exc:
            if strict:
                raise
            return {
                "schema": NVML_PROFILER_SCHEMA,
                "nvml_ok": False,
                "nvml_available": False,
                "error_code": exc.code,
                "error": exc.message,
                "wall_s": round(max(self._t1 - self._t0, 0.0), 6),
                "total_tokens": total_tokens,
                "energy_power_status": ENERGY_CONTROL_TAG,
                "energy_disclaimer": ENERGY_DISCLAIMER,
            }


NvmlEnergyMonitor = NvmlEnergyProfiler


def run_nvml_profiler_selftest() -> dict[str, Any]:
    """Deterministic pure-CPU selftest of trapezoidal NVML math and fail-closed guards."""

    class _FakeClock:
        def __init__(self, times: Sequence[float]) -> None:
            self._times = list(times)
            self._idx = 0

        def __call__(self) -> float:
            val = self._times[min(self._idx, len(self._times) - 1)]
            self._idx += 1
            return val

    class _FakeBackend:
        def __init__(self, rows: Sequence[tuple[float, int, int]]) -> None:
            self._rows = list(rows)
            self._idx = 0
            self._closed = False

        def open_device(self, gpu_index: int) -> None:
            _ = gpu_index

        def read_sample(self) -> tuple[float, int, int]:
            row = self._rows[min(self._idx, len(self._rows) - 1)]
            self._idx += 1
            return row

        def read_total_energy_mj(self) -> int | None:
            return 1_000_000 if self._idx == 0 else 1_075_000

        def close_device(self) -> None:
            self._closed = True

    clk = _FakeClock([10.0, 10.0, 10.5, 11.0, 11.0])
    backend = _FakeBackend([(70.0, 68, 1695), (80.0, 74, 1710), (75.0, 72, 1700)])
    with NvmlEnergyProfiler(gpu_index=0, interval_s=0.01, backend=backend, clock_fn=clk, background_poll=False) as prof:
        prof.sample_once()
    rep = prof.summary(total_tokens=100)
    validate_nvml_energy_report(rep)
    for field_name in REQUIRED_NVML_ENERGY_FIELDS:
        _require_nvml(field_name in rep, "SELFTEST_FAIL", f"Missing {field_name}")

    # Negative controls: bool rejection, zero tokens, zero energy, missing disclaimer
    neg_codes: list[str] = []
    for fn in (
        lambda: summarize_nvml_samples([NvmlSample(0.0, 70.0, 60), NvmlSample(1.0, 70.0, 61)], total_tokens=True),  # type: ignore[arg-type]
        lambda: summarize_nvml_samples([NvmlSample(0.0, 70.0, 60)], total_tokens=10),
        lambda: summarize_nvml_samples([NvmlSample(0.0, 0.0, 60), NvmlSample(1.0, 0.0, 61)], total_tokens=10),
        lambda: validate_nvml_energy_report({**rep, "energy_power_status": "CLAIMED_CONTROLLED"}),
    ):
        try:
            fn()
            raise NvmlProfilerError("SELFTEST_FAIL", "Negative control succeeded unexpectedly")
        except NvmlProfilerError as exc:
            neg_codes.append(exc.code)

    out = {
        "schema": NVML_SELFTEST_SCHEMA,
        "passed": True,
        "required_fields": list(REQUIRED_NVML_ENERGY_FIELDS),
        "sample_summary_sha256": rep["summary_sha256"],
        "negative_controls": neg_codes,
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SiliSense NVML 4-field energy & thermal profiler.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic CPU selftest.")
    parser.add_argument("--validate-json", type=Path, help="Validate an existing NVML JSON report.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.validate_json is not None:
            raw = json.loads(args.validate_json.read_text(encoding="utf-8"))
            validate_nvml_energy_report(raw)
            sys.stdout.write(json.dumps({"passed": True, "summary_sha256": raw.get("summary_sha256")}, sort_keys=True) + "\n")
            return 0
        rep = run_nvml_profiler_selftest()
        sys.stdout.write(json.dumps(rep, sort_keys=True) + "\n")
        return 0
    except (NvmlProfilerError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, NvmlProfilerError) else "ERR_NVML_IO"
        sys.stderr.write(json.dumps({"passed": False, "error": code, "message": str(exc)}, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
