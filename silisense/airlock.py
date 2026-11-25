"""Two-Mode Privacy & Sensor Airlock and Mode-1 Read-Only Sysfs/NVML Readers.

Hard Invariants:
  1. Camera, V4L2, PipeWire-video, and IR endpoints (/dev/video*, /dev/media*,
     ir_led, infrared) are UNCONDITIONALLY FORBIDDEN (ERR_CAMERA_FORBIDDEN)
     across all modes (justified by Track S113 H3/H4 zero-leakage proof).
  2. Mode 1 ('telemetry-only', default) forbids all microphone/ALSA/PipeWire
     audio endpoints (ERR_DMIC_DISABLED_DEFAULT) and reads strictly non-acoustic
     intel-rapl, nvidia-smi/NVML, /proc/stat IRQ/ctxt, and cpufreq counters.
  3. Mode 2 ('dmic-authorized') requires explicit enable_dmic_authorized=True,
     bounded authorize_audio_window_sec in (0, 300], and ASCII authorization
     token of length >= 8, enforced on a monotonic deadline clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Callable

from silisense.errors import SiliSensePermissionError, require_ok, require_perm

MAX_AUDIO_WINDOW_SEC = 300.0
MIN_AUTH_TOKEN_LEN = 8

_CAMERA_OR_IR_RE = re.compile(
    r"(^|/)(dev/video\d*|dev/media\d*|v4l(/|$)|v4l2|video4linux|pipewire.*camera|libcamera|ir_led|infrared|ir_camera|tof_sensor)",
    re.IGNORECASE,
)
_AUDIO_DEV_RE = re.compile(
    r"(^|/)(dev/snd(/|$)|snd/|alsa|pcmC\d+D\d+|arecord|pw-record|parec|pipewire.*audio)",
    re.IGNORECASE,
)


def is_camera_or_ir_identifier(text: str) -> bool:
    return bool(_CAMERA_OR_IR_RE.search(text))


def validate_endpoint_path(endpoint: str | Path, auth_gate: "AudioAuthorizationGate | None" = None) -> None:
    """Validate an OS path or device URI against the Two-Mode Sensor Airlock."""
    text = str(endpoint).strip()
    require_ok(len(text) > 0, "ERR_EMPTY_ENDPOINT", "Endpoint path must be non-empty")
    if _CAMERA_OR_IR_RE.search(text):
        raise SiliSensePermissionError(
            "ERR_CAMERA_FORBIDDEN",
            f"Camera/IR endpoint {text!r} is permanently forbidden in SiliSense (S113 H3/H4 zero-leakage invariant)",
        )
    if _AUDIO_DEV_RE.search(text):
        require_perm(
            auth_gate is not None,
            "ERR_DMIC_DISABLED_DEFAULT",
            f"Audio endpoint {text!r} refused: Mode 1 ('telemetry-only') forbids microphone access",
        )
        if auth_gate is not None:
            auth_gate.verify_active()


@dataclass(frozen=True)
class AudioAuthorizationGate:
    """Monotonic-deadline capability token gating Mode-2 48 kHz DMIC lenses."""

    enable_dmic_authorized: bool
    authorize_audio_window_sec: float
    pi_authorization_token: str
    created_mono: float
    clock_fn: Callable[[], float] = time.monotonic

    @classmethod
    def create(
        cls,
        *,
        enable_dmic_authorized: bool,
        authorize_audio_window_sec: float | None,
        pi_authorization_token: str | None,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> "AudioAuthorizationGate":
        require_perm(
            enable_dmic_authorized is True,
            "ERR_DMIC_UNAUTHORIZED",
            "Mode 2 DMIC access requires explicit enable_dmic_authorized=True",
        )
        require_perm(
            authorize_audio_window_sec is not None
            and isinstance(authorize_audio_window_sec, (int, float))
            and not isinstance(authorize_audio_window_sec, bool)
            and math.isfinite(float(authorize_audio_window_sec)),
            "ERR_AUDIO_WINDOW_MISSING",
            "Mode 2 DMIC access requires explicit finite --authorize-audio-window-sec",
        )
        win_sec = float(authorize_audio_window_sec)  # type: ignore[arg-type]
        require_perm(
            0.0 < win_sec <= MAX_AUDIO_WINDOW_SEC,
            "ERR_AUDIO_WINDOW_BOUNDS",
            f"authorize_audio_window_sec must lie in (0, {MAX_AUDIO_WINDOW_SEC}], got {win_sec}",
        )
        tok = (pi_authorization_token or "").strip()
        require_perm(
            len(tok) >= MIN_AUTH_TOKEN_LEN and tok.isascii(),
            "ERR_AUTH_TOKEN_INVALID",
            f"Mode 2 DMIC access requires ASCII pi_authorization_token of length >= {MIN_AUTH_TOKEN_LEN}",
        )
        return cls(
            enable_dmic_authorized=True,
            authorize_audio_window_sec=win_sec,
            pi_authorization_token=tok,
            created_mono=float(clock_fn()),
            clock_fn=clock_fn,
        )

    def remaining_sec(self) -> float:
        return self.authorize_audio_window_sec - (float(self.clock_fn()) - self.created_mono)

    def verify_active(self) -> None:
        require_perm(self.enable_dmic_authorized is True, "ERR_DMIC_UNAUTHORIZED", "DMIC gate not enabled")
        elapsed = float(self.clock_fn()) - self.created_mono
        require_perm(elapsed >= -1e-9, "ERR_CLOCK_SKEW", "Monotonic clock moved backwards unexpectedly")
        require_perm(
            elapsed <= self.authorize_audio_window_sec,
            "ERR_AUDIO_WINDOW_EXPIRED",
            f"Authorized audio window ({self.authorize_audio_window_sec:.2f}s) expired ({elapsed:.2f}s elapsed)",
        )


def check_dmic_permission(auth_gate: AudioAuthorizationGate | None, allow_offline_array: bool) -> None:
    if auth_gate is not None:
        auth_gate.verify_active()
        return
    require_perm(
        allow_offline_array is True,
        "ERR_DMIC_DISABLED_DEFAULT",
        "DMIC observables (A1/A2/A3-62us/A5-80ms) are disabled in default Mode 1 ('telemetry-only'); "
        "pass an active AudioAuthorizationGate or explicit allow_offline_array=True for synthetic/offline arrays.",
    )


class RaplSysfsReader:
    """Wraparound-safe read-only intel-rapl sysfs energy-counter reader (Mode 1)."""

    def __init__(self, powercap_root: Path = Path("/sys/class/powercap")) -> None:
        validate_endpoint_path(powercap_root, auth_gate=None)
        self.powercap_root = powercap_root
        self._domains: dict[str, tuple[Path, int]] = {}
        self._last_energy_uj: dict[str, int] = {}
        self._last_mono: float | None = None

    def discover(self) -> list[str]:
        if not self.powercap_root.is_dir():
            return []
        found: dict[str, tuple[Path, int]] = {}
        for d in sorted(self.powercap_root.glob("intel-rapl:*")):
            if d.is_symlink():
                continue
            name_file = d / "name"
            energy_file = d / "energy_uj"
            max_file = d / "max_energy_range_uj"
            if name_file.is_file() and energy_file.is_file():
                try:
                    raw_name = name_file.read_text(encoding="utf-8").strip().lower()
                    max_uj = int(max_file.read_text(encoding="utf-8").strip()) if max_file.is_file() else (1 << 32)
                    key = (
                        "psys_w"
                        if "psys" in raw_name
                        else "pkg_w"
                        if raw_name.startswith("package")
                        else "core_w"
                        if raw_name == "core"
                        else "uncore_w"
                        if raw_name == "uncore"
                        else f"rapl_{raw_name}_w"
                    )
                    found[key] = (energy_file, max_uj)
                except OSError:
                    continue
        self._domains = found
        return sorted(found.keys())

    def sample_watts(self, clock_fn: Callable[[], float] = time.monotonic) -> dict[str, float]:
        if not self._domains:
            self.discover()
        now_m = float(clock_fn())
        cur_uj = {key: int(efile.read_text(encoding="utf-8").strip()) for key, (efile, _) in self._domains.items()}
        if self._last_mono is None:
            self._last_mono = now_m
            self._last_energy_uj = cur_uj
            return {k: 0.0 for k in cur_uj}
        dt = max(now_m - self._last_mono, 1e-6)
        watts: dict[str, float] = {}
        for key, val in cur_uj.items():
            prev = self._last_energy_uj.get(key, val)
            _, max_uj = self._domains[key]
            delta_uj = val - prev
            if delta_uj < 0:
                delta_uj += max_uj
            watts[key] = float(delta_uj) * 1e-6 / dt
        self._last_mono = now_m
        self._last_energy_uj = cur_uj
        return watts


def query_proc_irq_and_cpu_khz(
    proc_stat_path: Path = Path("/proc/stat"),
    cpufreq_glob_root: Path = Path("/sys/devices/system/cpu"),
) -> dict[str, float | int]:
    """Read Mode-1 non-acoustic OS IRQ / context-switch counters and mean CPU clock kHz."""
    validate_endpoint_path(proc_stat_path, auth_gate=None)
    validate_endpoint_path(cpufreq_glob_root, auth_gate=None)
    intr_total = 0
    ctxt_total = 0
    if proc_stat_path.is_file():
        for line in proc_stat_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("intr "):
                parts = line.split()
                if len(parts) >= 2:
                    intr_total = int(parts[1])
            elif line.startswith("ctxt "):
                parts = line.split()
                if len(parts) >= 2:
                    ctxt_total = int(parts[1])
    khz_vals: list[float] = []
    if cpufreq_glob_root.is_dir():
        for fpath in sorted(cpufreq_glob_root.glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
            try:
                khz_vals.append(float(fpath.read_text(encoding="utf-8").strip()))
            except OSError:
                continue
    mean_khz = (sum(khz_vals) / float(len(khz_vals))) if khz_vals else 0.0
    return {
        "intr_total": intr_total,
        "ctxt_total": ctxt_total,
        "mean_cpu_clock_khz": round(mean_khz, 2),
    }


def query_nvidia_smi_once(timeout_sec: float = 2.0) -> dict[str, float | int]:
    """Query single GPU power/clock/throttle sample via list-argv nvidia-smi (0 shell=True)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw,clocks.current.sm,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    require_ok(proc.returncode == 0, "ERR_NVIDIA_SMI_FAILED", proc.stderr.strip() or "nvidia-smi returned non-zero")
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    require_ok(len(parts) >= 5, "ERR_NVIDIA_SMI_PARSE", f"Unexpected nvidia-smi CSV: {line!r}")
    return {
        "gpu_power_w": float(parts[0]),
        "sm_clock_mhz": float(parts[1]),
        "gpu_temp_c": float(parts[2]),
        "sw_power_cap_active": 1 if parts[3].lower() == "active" else 0,
        "hw_thermal_slowdown_active": 1 if parts[4].lower() == "active" else 0,
    }
