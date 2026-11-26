"""Two-Mode Privacy & Sensor Airlock and Mode-1 Read-Only Sysfs/NVML Readers.

Hard Invariants:
  1. Camera, V4L2 (including char major 81 aliases), PipeWire-video, and IR
     endpoints (/dev/video*, /dev/media*, /dev/char/81:*, ir_led, infrared) are
     UNCONDITIONALLY FORBIDDEN (ERR_CAMERA_FORBIDDEN) across all modes, checked
     both before and after path normalization and symlink resolution.
  2. Mode 1 ('telemetry-only', default) forbids all microphone/ALSA/PipeWire
     audio endpoints (ERR_DMIC_DISABLED_DEFAULT) and reads strictly non-acoustic
     intel-rapl, nvidia-smi/NVML, /proc/stat IRQ/ctxt, and cpufreq counters.
  3. Mode 2 ('dmic-authorized') requires explicit enable_dmic_authorized=True,
     bounded authorize_audio_window_sec in (0, 300], and ASCII authorization
     token of length >= 8 (redacted from repr), enforced on a monotonic clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import posixpath
import re
import stat
import subprocess
import time
from typing import Callable

from silisense.errors import (
    HardwareExitCode,
    SiliSensePermissionError,
    SiliSenseValueError,
    require_ok,
    require_perm,
)

MAX_AUDIO_WINDOW_SEC = 300.0
MIN_AUTH_TOKEN_LEN = 8
V4L2_CHAR_MAJOR = 81

_CAMERA_OR_IR_RE = re.compile(
    r"(^|/)(dev/video\d*|dev/media\d*|dev/char/81:\d*|v4l(/|$)|v4l2|video4linux|pipewire.*camera|libcamera|ir_led|infrared|ir_camera|tof_sensor)",
    re.IGNORECASE,
)
_AUDIO_DEV_RE = re.compile(
    r"(^|/)(dev/snd(/|$)|snd/|alsa|pcmC\d+D\d+|arecord|pw-record|parec|pipewire.*audio)",
    re.IGNORECASE,
)


def _candidate_normalized_strings(raw_text: str) -> list[str]:
    cleaned = raw_text.replace("\\", "/")
    norm_posix = posixpath.normpath(cleaned)
    norm_os = os.path.normpath(raw_text)
    candidates = [raw_text, cleaned, norm_posix, norm_os]
    # Only resolve filesystem symlinks if the path looks like a local filesystem path (not a scheme URI)
    if "://" not in raw_text and not raw_text.startswith("alsa:"):
        try:
            resolved = str(Path(raw_text).resolve(strict=False))
            candidates.append(resolved)
        except OSError:
            pass
    return candidates


def is_camera_or_ir_identifier(text: str) -> bool:
    s = str(text).strip()
    if not s:
        return False
    for cand in _candidate_normalized_strings(s):
        if _CAMERA_OR_IR_RE.search(cand):
            return True
    return False


def _is_v4l2_char_device(path_text: str) -> bool:
    if "://" in path_text or path_text.startswith("alsa:"):
        return False
    for check_fn in (os.lstat, os.stat):
        try:
            st = check_fn(path_text)
            if stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == V4L2_CHAR_MAJOR:
                return True
        except OSError:
            continue
    return False


def validate_endpoint_path(endpoint: str | Path, auth_gate: "AudioAuthorizationGate | None" = None) -> None:
    """Validate an OS path or device URI against the Two-Mode Sensor Airlock.

    Normalizes '.', '..', '//', resolves symlinks, and checks V4L2 char-major 81
    strictly BEFORE checking audio endpoints so traversal paths such as
    '/dev/snd/../video0', '/dev/./video0', '/dev//video0', or '/dev/char/81:0'
    always fail closed with ERR_CAMERA_FORBIDDEN even when Mode 2 is active.
    """
    text = str(endpoint).strip()
    require_ok(len(text) > 0, "ERR_EMPTY_ENDPOINT", "Endpoint path must be non-empty", exit_code=HardwareExitCode.PREFLIGHT_FAIL)

    candidates = _candidate_normalized_strings(text)
    for cand in candidates:
        if _CAMERA_OR_IR_RE.search(cand):
            raise SiliSensePermissionError(
                "ERR_CAMERA_FORBIDDEN",
                f"Camera/IR endpoint {text!r} (normalized {cand!r}) is permanently forbidden in SiliSense",
            )
    if _is_v4l2_char_device(text):
        raise SiliSensePermissionError(
            "ERR_CAMERA_FORBIDDEN",
            f"V4L2 character device (major {V4L2_CHAR_MAJOR}) {text!r} is permanently forbidden in SiliSense",
        )

    # Check audio on normalized path(s) so '/dev/snd/../null' is not misclassified, while real audio paths are gated
    norm_primary = candidates[2]
    is_audio = bool(_AUDIO_DEV_RE.search(norm_primary)) or any(
        bool(_AUDIO_DEV_RE.search(c)) for c in candidates[4:]
    )
    if is_audio:
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
    pi_authorization_token: str = field(repr=False)
    created_mono: float = 0.0
    clock_fn: Callable[[], float] = field(default=time.monotonic, repr=False)

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
        tok = (pi_authorization_token or "").strip() if isinstance(pi_authorization_token, str) else ""
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
    """Wraparound-safe read-only intel-rapl sysfs energy-counter reader (Mode 1).

    Safely follows Linux `/sys/class/powercap/intel-rapl:*` symlinks ONLY when
    their resolved realpath remains strictly inside an allowed powercap sysfs
    subtree (`/sys/devices/virtual/powercap`, `/sys/devices/platform`, or the
    caller's explicit `allowed_real_root`), failing closed on escape symlinks.
    """

    def __init__(
        self,
        powercap_root: Path = Path("/sys/class/powercap"),
        *,
        allowed_real_roots: tuple[Path, ...] | None = None,
    ) -> None:
        validate_endpoint_path(powercap_root, auth_gate=None)
        self.powercap_root = powercap_root
        if allowed_real_roots is not None:
            self._allowed_roots = tuple(r.resolve(strict=False) for r in allowed_real_roots)
        else:
            self._allowed_roots = (
                powercap_root.resolve(strict=False),
                Path("/sys/devices/virtual/powercap").resolve(strict=False),
                Path("/sys/devices/platform").resolve(strict=False),
            )
        self._domains: dict[str, tuple[Path, int]] = {}
        self._last_energy_uj: dict[str, int] = {}
        self._last_mono: float | None = None

    def _verify_confined_domain_dir(self, candidate: Path) -> Path | None:
        validate_endpoint_path(candidate, auth_gate=None)
        try:
            real_dir = candidate.resolve(strict=True)
        except OSError:
            return None
        validate_endpoint_path(real_dir, auth_gate=None)
        confined = any(
            real_dir == root or root in real_dir.parents
            for root in self._allowed_roots
        )
        require_perm(
            confined,
            "ERR_SYMLINK_ESCAPE",
            f"RAPL entry {str(candidate)!r} resolves outside allowed powercap roots: {str(real_dir)!r}",
        )
        return real_dir if real_dir.is_dir() else None

    def discover(self) -> list[str]:
        if not self.powercap_root.is_dir():
            return []
        found: dict[str, tuple[Path, int]] = {}
        visited_real: set[Path] = set()
        queue = sorted(self.powercap_root.glob("intel-rapl:*"))
        while queue:
            entry = queue.pop(0)
            real_d = self._verify_confined_domain_dir(entry)
            if real_d is None or real_d in visited_real:
                continue
            visited_real.add(real_d)
            for child in sorted(real_d.glob("intel-rapl:*")):
                queue.append(child)

            name_file = real_d / "name"
            energy_file = real_d / "energy_uj"
            max_file = real_d / "max_energy_range_uj"
            if name_file.is_file() and energy_file.is_file():
                try:
                    raw_name = name_file.read_text(encoding="utf-8").strip().lower()
                    max_uj = int(max_file.read_text(encoding="utf-8").strip()) if max_file.is_file() else (1 << 32)
                    require_ok(max_uj > 0, "ERR_RAPL_MAX_RANGE", "max_energy_range_uj must be > 0", exit_code=HardwareExitCode.INFRA_ERROR)
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
                except (OSError, ValueError):
                    continue
        self._domains = found
        return sorted(found.keys())

    def sample_watts(self, clock_fn: Callable[[], float] = time.monotonic) -> dict[str, float]:
        if not self._domains:
            self.discover()
        now_m = float(clock_fn())
        cur_uj: dict[str, int] = {}
        for key, (efile, _) in self._domains.items():
            try:
                cur_uj[key] = int(efile.read_text(encoding="utf-8").strip())
            except PermissionError as exc:
                raise SiliSensePermissionError("ERR_RAPL_UNREADABLE", f"Permission denied reading {efile}: {exc}") from exc
            except (OSError, ValueError) as exc:
                raise SiliSenseValueError("ERR_RAPL_READ_FAIL", f"Failed reading {efile}: {exc}", exit_code=HardwareExitCode.INFRA_ERROR) from exc

        if self._last_mono is None:
            self._last_mono = now_m
            self._last_energy_uj = cur_uj
            return {k: 0.0 for k in cur_uj}
        dt = max(now_m - self._last_mono, 1e-6)
        watts: dict[str, float] = {}
        for key, val in cur_uj.items():
            prev = self._last_energy_uj.get(key, val)
            _, max_uj = self._domains[key]
            delta_uj = (val - prev) % max_uj if val < prev else (val - prev)
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
            except (OSError, ValueError):
                continue
    mean_khz = (sum(khz_vals) / float(len(khz_vals))) if khz_vals else 0.0
    return {
        "intr_total": intr_total,
        "ctxt_total": ctxt_total,
        "mean_cpu_clock_khz": round(mean_khz, 2),
    }


def query_nvidia_smi_once(timeout_sec: float = 2.0, gpu_index: int = 0) -> dict[str, float | int]:
    """Query single GPU power/clock/throttle sample via list-argv nvidia-smi (0 shell=True)."""
    require_ok(isinstance(gpu_index, int) and not isinstance(gpu_index, bool) and gpu_index >= 0, "ERR_GPU_INDEX", "Invalid gpu_index")
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=power.draw,clocks.current.sm,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise SiliSenseValueError(
            "ERR_NVIDIA_SMI_FAILED",
            f"nvidia-smi execution failed: {exc}",
            exit_code=HardwareExitCode.INFRA_ERROR,
        ) from exc
    require_ok(
        proc.returncode == 0,
        "ERR_NVIDIA_SMI_FAILED",
        proc.stderr.strip() or "nvidia-smi returned non-zero",
        exit_code=HardwareExitCode.INFRA_ERROR,
    )
    lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    require_ok(len(lines) >= 1, "ERR_NVIDIA_SMI_PARSE", "Empty stdout from nvidia-smi", exit_code=HardwareExitCode.INFRA_ERROR)
    parts = [p.strip() for p in lines[0].split(",")]
    require_ok(len(parts) >= 5, "ERR_NVIDIA_SMI_PARSE", f"Unexpected nvidia-smi CSV: {lines[0]!r}", exit_code=HardwareExitCode.INFRA_ERROR)
    try:
        gpu_w = float(parts[0])
        sm_mhz = float(parts[1])
        temp_c = float(parts[2])
    except ValueError as exc:
        raise SiliSenseValueError(
            "ERR_NVIDIA_SMI_PARSE",
            f"Non-numeric field in nvidia-smi output: {lines[0]!r}",
            exit_code=HardwareExitCode.INFRA_ERROR,
        ) from exc
    require_ok(
        math.isfinite(gpu_w) and math.isfinite(sm_mhz) and math.isfinite(temp_c),
        "ERR_NVIDIA_SMI_PARSE",
        "Non-finite value in nvidia-smi output",
        exit_code=HardwareExitCode.INFRA_ERROR,
    )
    return {
        "gpu_power_w": gpu_w,
        "sm_clock_mhz": sm_mhz,
        "gpu_temp_c": temp_c,
        "sw_power_cap_active": 1 if parts[3].lower() == "active" else 0,
        "hw_thermal_slowdown_active": 1 if parts[4].lower() == "active" else 0,
    }
