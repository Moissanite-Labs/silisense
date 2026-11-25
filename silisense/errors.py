"""Typed fail-closed error hierarchy and hardware exit taxonomy (zero bare asserts)."""

from __future__ import annotations

from enum import IntEnum


class HardwareExitCode(IntEnum):
    """Structured overnight/autotuner hardware exit taxonomy (75..81)."""

    OK = 0
    THERMAL_STOP = 75
    OOM = 76
    PREFLIGHT_FAIL = 77
    MEM_PRESSURE = 78
    SIGNAL_GATE_FAIL = 79
    CALIBRATE_FAIL = 80
    INFRA_ERROR = 81


class SiliSenseError(Exception):
    """Base typed exception for all SiliSense modules."""

    def __init__(self, code: str, message: str, exit_code: HardwareExitCode = HardwareExitCode.PREFLIGHT_FAIL) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.exit_code = exit_code


class SiliSensePermissionError(SiliSenseError):
    """Raised when the Two-Mode Sensor & Privacy Airlock blocks an endpoint or action."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, exit_code=HardwareExitCode.PREFLIGHT_FAIL)


class SiliSenseValueError(SiliSenseError):
    """Raised on invalid telemetry schemas, non-finite values, or violated invariants."""

    def __init__(self, code: str, message: str, exit_code: HardwareExitCode = HardwareExitCode.SIGNAL_GATE_FAIL) -> None:
        super().__init__(code, message, exit_code=exit_code)


def require_ok(
    condition: bool,
    code: str,
    message: str,
    exit_code: HardwareExitCode = HardwareExitCode.SIGNAL_GATE_FAIL,
) -> None:
    """Fail-closed invariant check replacing bare `assert` statements."""
    if not condition:
        raise SiliSenseValueError(code, message, exit_code=exit_code)


def require_perm(condition: bool, code: str, message: str) -> None:
    """Fail-closed permission check for sensor and privacy boundaries."""
    if not condition:
        raise SiliSensePermissionError(code, message)
