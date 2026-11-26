"""HostContentionSentinel: Process, GPU-app, and power control-variable metrology.

Pure-stdlib Linux process, GPU-app, and power-source control-variable collector,
canonical receipt binder, fail-closed R1-R5 verifier, and pure-Q contention-band
stratification gate (addressing Moissanite-Labs/silikernels#7).

Privacy & Safety Invariants:
- Reads only /proc/<pid>/stat, /proc/<pid>/comm, /proc/loadavg, /proc/stat,
  /sys/class/power_supply/AC0/online, read-only nvidia-smi queries, and
  optional desktop foreground class name.
- NEVER reads /proc/<pid>/cmdline, full file paths, window titles, user secrets,
  or audio/video devices.
- Enforces the uncontrolled-laptop energy policy: rejects uncontrolled laptop
  energy/power keys, units, and string claims across all receipt structures
  (energy_power_status == "UNCONTROLLED_LAPTOP_NOT_CLAIMED").
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence as AbcSequence, Set as AbcSet
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import unicodedata

from silisense.errors import SiliSenseValueError, require_ok


SNAPSHOT_SCHEMA = "silisense.contention.host_snapshot.v1"
ENVIRONMENT_SCHEMA = "silisense.contention.environment.v1"
RECEIPT_SCHEMA = "silisense.contention.bench_receipt.v1"
COMPARISON_SCHEMA = "silisense.contention.band_comparison.v1"
PROBE_SCHEMA = "silisense.contention.probe.v1"
ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"

# NVML active throttle bitmask fault bits (excludes 0x1 GPU_IDLE and 0x2 APP_CLOCKS).
NVML_FAULT_THROTTLE_MASK = 0xFC

FORBIDDEN_ENERGY_KEYS: frozenset[str] = frozenset(
    {
        "w",
        "pw",
        "nw",
        "uw",
        "mw",
        "kw",
        "gw",
        "tw",
        "wh",
        "whr",
        "whrs",
        "mwh",
        "kwh",
        "j",
        "pj",
        "nj",
        "uj",
        "mj",
        "kj",
        "gj",
        "tj",
        "tdp",
        "pwr",
        "watt",
        "watts",
        "joule",
        "joules",
        "millijoule",
        "millijoules",
        "energy",
        "power",
        "edp",
    }
)
_FORBIDDEN_METADATA_VALUE_KEYS: frozenset[str] = FORBIDDEN_ENERGY_KEYS - frozenset({"w", "j"})

_ALLOWED_SCHEMA_POLICY_PATHS: frozenset[Tuple[str, ...]] = frozenset(
    {
        ("energy_power_status",),
        ("power_band",),
        ("environment", "energy_power_status"),
        ("environment", "power_band"),
        ("quiet_receipt", "energy_power_status"),
        ("quiet_receipt", "environment", "energy_power_status"),
        ("quiet_receipt", "environment", "power_band"),
        ("contended_receipt", "energy_power_status"),
        ("contended_receipt", "environment", "energy_power_status"),
        ("contended_receipt", "environment", "power_band"),
    }
)
_VALID_POWER_BANDS: frozenset[str] = frozenset({"ac", "battery", "unknown"})

_ENV_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "energy_power_status",
        "pre_snapshot_sha256",
        "post_snapshot_sha256",
        "in_run_summary",
        "ac_online_all",
        "power_band",
        "external_gpu_apps_detected",
        "cpu_load_max_ratio",
        "cpu_load_max_ratio_q",
        "bg_cpu_util_max_ratio",
        "bg_cpu_util_max_ratio_q",
        "foreground_apps",
        "thresholds",
        "contended",
        "contention_reasons",
        "environment_sha256",
    }
)

_SUMMARY_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "sample_count",
        "cpu_load_ratio_median_q",
        "cpu_load_ratio_max_q",
        "bg_cpu_util_median_q",
        "bg_cpu_util_max_q",
        "ac_online_observations",
        "ac_offline_observations",
        "ac_unknown_observations",
        "gpu_status",
        "gpu_util_max_pct",
        "gpu_sm_clock_min_mhz",
        "gpu_sm_clock_max_mhz",
        "gpu_temp_max_c",
        "gpu_throttle_fault_any",
        "external_gpu_apps_max",
    }
)

_RECEIPT_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "energy_power_status",
        "arm_name",
        "metric_name",
        "metric_unit",
        "repeat_values_q",
        "spread",
        "environment",
        "metadata",
        "result_sha256",
    }
)

_PROBE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "energy_power_status",
        "cpu_count",
        "n_repeats_per_band",
        "synthetic_burner_workers",
        "quiet_receipt",
        "contended_receipt",
        "cross_band_comparison",
        "stratified_bands",
        "band_separation_verified",
        "probe_sha256",
    }
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_Q_STR_RE = re.compile(r"^-?[0-9]+/[1-9][0-9]*$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_LOADAVG_NUM_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_FORBIDDEN_COMPACT_KEY_RE = re.compile(
    r"(?:(?:tera|giga|mega|kilo|milli|micro|nano|pico)?(?:watt|joule)|power|energy|mjpertok|mjtok|jpertok|ujpertok|kwh|mwh|whpertok|tdp)"
)
_ENERGY_PFX = r"(?:p|n|u|\u03bc|m|k|g|t|pico|nano|micro|milli|kilo|mega|giga|tera)"
_ENERGY_UNIT = r"(?:watt[^a-z0-9]*hours?|watthours?|watts?|joules?|whrs?|wh|w|j|tdp)"
_ENERGY_SEP = r"[^a-z0-9]*"
_ENERGY_NUM = r"\d[\d.,]*(?:e[+-]?\d+)?"
_FORBIDDEN_ENERGY_VALUE_RE = re.compile(
    "(?:"
    rf"{_ENERGY_NUM}{_ENERGY_SEP}(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}(?![a-z0-9])(?!/\s)"
    rf"|(?<![a-z0-9])(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}\s*[:=]\s*{_ENERGY_NUM}"
    rf"|(?<![a-z0-9])tdp[^a-z0-9]*{_ENERGY_NUM}"
    rf"|^[^a-z0-9]*(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}[^a-z0-9]*$"
    r"|\b(?:joules?|watts?|mj|uj|kwh|wh)\s*(?:per|/)\s*tok(?:en)?s?\b"
    r"|\b(?:less|lower|reduced|fewer)\s+(?:in\s+)?(?:energy|power)\b"
    r"|\b(?:energy|power)\s+(?:use|usage|reduction|savings?|dropped|consumption)\b"
    ")"
)


DEFAULT_CONTENTION_THRESHOLDS: Dict[str, Any] = {
    "max_cpu_load_ratio_q": "1/2",
    "max_bg_cpu_util_ratio_q": "2/5",
    "max_external_gpu_util_pct": 5,
    "max_gpu_temp_c": 80,
    "min_repeats": 3,
    "require_ac_online": True,
    "allow_external_gpu_apps": False,
    "allow_gpu_throttle": False,
}


def _is_strict_int(value: Any) -> bool:
    return type(value) is int


def _is_strict_bool(value: Any) -> bool:
    return type(value) is bool


def _is_strict_finite_num(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON serialization (rejects NaN/Infinity as SiliSenseValueError)."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SiliSenseValueError("BAD_JSON_PAYLOAD", f"non-canonical JSON payload: {exc}") from exc


def canonical_sha256(obj: Any, *, exclude_key: Optional[str] = None) -> str:
    """SHA-256 hex digest of canonical JSON, optionally excluding one top-level key."""
    if exclude_key is None:
        payload = obj
    else:
        payload = {k: v for k, v in obj.items() if k != exclude_key}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def format_q(value: Fraction) -> str:
    """Canonical reduced rational string 'num/den' with den >= 1."""
    return f"{value.numerator}/{value.denominator}"


def parse_q(raw: Any, *, code: str = "BAD_RATIONAL") -> Fraction:
    """Parse a strict canonical 'num/den' string into a Fraction."""
    require_ok(type(raw) is str and bool(_Q_STR_RE.match(raw)), code, f"invalid rational: {raw!r}")
    num_s, den_s = raw.split("/", 1)
    q = Fraction(int(num_s), int(den_s))
    require_ok(format_q(q) == raw, code, f"non-canonical rational: {raw!r}")
    return q


def sanitize_proc_name(raw: Any) -> str:
    """Scrub process or desktop class string to a short privacy-safe executable basename."""
    if type(raw) is not str:
        return "unknown"
    text = raw.strip()
    if not text:
        return "unknown"
    first_tok = text.split()[0]
    base = first_tok.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("_")
    if not cleaned:
        return "unknown"
    return cleaned[:32]


def _strip_nfkd_marks(raw: str) -> str:
    nfkd = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", raw))
    return "".join(ch for ch in nfkd if unicodedata.category(ch) not in ("Cf", "Mn"))


def _has_non_ascii_letter(norm: str) -> bool:
    return any(not ch.isascii() and ch.isalpha() and ch != "\u03bc" for ch in norm)


def _is_forbidden_energy_value_str(raw: str) -> bool:
    """Check a string value for uncontrolled laptop energy/power quantity or unit claims."""
    norm = _strip_nfkd_marks(raw).casefold()
    return _has_non_ascii_letter(norm) or bool(_FORBIDDEN_ENERGY_VALUE_RE.search(norm))


def _is_forbidden_energy_label(raw: str, *, allow_bare_wj: bool = False) -> bool:
    """Check a normalized key, arm_name, metric_name, or metric_unit against forbidden terms."""
    cleaned = _strip_nfkd_marks(raw)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", cleaned).replace("_", " ")
    if _is_forbidden_energy_value_str(spaced):
        return True
    tokens = _ALNUM_TOKEN_RE.findall(spaced.casefold())
    key_set = _FORBIDDEN_METADATA_VALUE_KEYS if allow_bare_wj else FORBIDDEN_ENERGY_KEYS
    for tok in tokens:
        if tok in key_set:
            return True
    compact = "".join(tokens)
    return bool(_FORBIDDEN_COMPACT_KEY_RE.search(compact))


def _is_allowed_schema_policy_entry(path: Tuple[str, ...], value: Any) -> bool:
    if path not in _ALLOWED_SCHEMA_POLICY_PATHS:
        return False
    leaf = path[-1]
    if leaf == "energy_power_status":
        return value == ENERGY_POLICY_STATUS
    return type(value) is str and value in _VALID_POWER_BANDS


def has_forbidden_energy_key(obj: Any, *, _path: Tuple[str, ...] = ()) -> bool:
    """Recursively reject uncontrolled laptop energy/power keys, units, and string claims."""
    if isinstance(obj, str):
        if _is_forbidden_energy_value_str(obj):
            return True
        return "metadata" in _path and _is_forbidden_energy_label(obj, allow_bare_wj=True)
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str):
                child_path = _path + (k,)
                if _is_allowed_schema_policy_entry(child_path, v):
                    continue
                if _is_forbidden_energy_label(k):
                    return True
                if has_forbidden_energy_key(v, _path=child_path):
                    return True
            else:
                if has_forbidden_energy_key(v, _path=_path):
                    return True
        return False
    if isinstance(obj, (AbcSequence, AbcSet)) and not isinstance(obj, (bytes, bytearray)):
        for item in obj:
            if has_forbidden_energy_key(item, _path=_path):
                return True
    return False


def validate_contention_thresholds(thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate and normalize contention threshold configuration."""
    if thresholds is None:
        cfg = dict(DEFAULT_CONTENTION_THRESHOLDS)
    else:
        require_ok(type(thresholds) is dict, "THRESH_TYPE", "thresholds must be a JSON object")
        cfg = dict(thresholds)
    req_keys = set(DEFAULT_CONTENTION_THRESHOLDS.keys())
    require_ok(set(cfg.keys()) == req_keys, "THRESH_KEYS", "threshold keys mismatch")
    max_load = parse_q(cfg["max_cpu_load_ratio_q"], code="THRESH_RATIONAL")
    max_bg = parse_q(cfg["max_bg_cpu_util_ratio_q"], code="THRESH_RATIONAL")
    require_ok(Fraction(0) < max_load <= Fraction(4), "THRESH_RANGE", "max_cpu_load_ratio_q out of (0,4]")
    require_ok(Fraction(0) < max_bg <= Fraction(1), "THRESH_RANGE", "max_bg_cpu_util_ratio_q out of (0,1]")
    require_ok(
        _is_strict_int(cfg["max_external_gpu_util_pct"]) and 0 <= cfg["max_external_gpu_util_pct"] <= 100,
        "THRESH_RANGE",
        "max_external_gpu_util_pct must be strict int in [0,100]",
    )
    require_ok(
        _is_strict_int(cfg["max_gpu_temp_c"]) and 30 <= cfg["max_gpu_temp_c"] <= 105,
        "THRESH_RANGE",
        "max_gpu_temp_c must be strict int in [30,105]",
    )
    require_ok(
        _is_strict_int(cfg["min_repeats"]) and cfg["min_repeats"] >= 3,
        "THRESH_MIN_REPEATS",
        "min_repeats must be strict int >= 3",
    )
    require_ok(
        _is_strict_bool(cfg["require_ac_online"])
        and _is_strict_bool(cfg["allow_external_gpu_apps"])
        and _is_strict_bool(cfg["allow_gpu_throttle"]),
        "THRESH_BOOL",
        "boolean threshold flags must be strict bool",
    )
    return cfg


def _parse_proc_stat_cpu(text: str) -> Tuple[int, int]:
    """Return (total_jiffies, idle_jiffies) from /proc/stat aggregate cpu line."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()[1:]
            require_ok(len(parts) >= 4, "PROC_STAT_PARSE", "malformed /proc/stat cpu line")
            vals = [int(p) for p in parts]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            return sum(vals), idle
    raise SiliSenseValueError("PROC_STAT_PARSE", "missing aggregate cpu line in /proc/stat")


def _parse_loadavg(text: str) -> Tuple[Fraction, Fraction, Fraction]:
    """Parse 1m, 5m, 15m load averages from /proc/loadavg as exact Fractions."""
    parts = text.strip().split()
    require_ok(len(parts) >= 3, "LOADAVG_PARSE", "malformed /proc/loadavg")
    out: List[Fraction] = []
    for tok in parts[:3]:
        require_ok(bool(_LOADAVG_NUM_RE.match(tok)), "LOADAVG_PARSE", f"bad loadavg token: {tok!r}")
        out.append(Fraction(tok))
    return out[0], out[1], out[2]


def _parse_pid_stat(text: str) -> Optional[Tuple[int, int]]:
    """Return (cpu_ticks, rss_pages) from /proc/<pid>/stat, or None if malformed."""
    rparen = text.rfind(")")
    if rparen <= 0:
        return None
    tail = text[rparen + 2 :].split()
    if len(tail) < 22:
        return None
    try:
        utime = int(tail[11])
        stime = int(tail[12])
        rss = int(tail[21])
    except ValueError:
        return None
    return max(0, utime + stime), max(0, rss)


def _parse_gpu_compute_apps(stdout: str, exclude_pids: Set[int]) -> List[Dict[str, Any]]:
    """Parse nvidia-smi compute-apps CSV into privacy-scrubbed sorted records."""
    apps: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            continue
        try:
            pid = int(cols[0])
            mem_mib = int(float(cols[2]))
        except ValueError:
            continue
        if pid in exclude_pids:
            continue
        apps.append(
            {
                "pid": pid,
                "proc": sanitize_proc_name(cols[1]),
                "used_memory_mib": max(0, mem_mib),
            }
        )
    apps.sort(key=lambda item: (-item["used_memory_mib"], item["pid"], item["proc"]))
    return apps


def _parse_gpu_telemetry(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse single-GPU nvidia-smi utilization/clock/temp/throttle CSV."""
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            return None
        try:
            util_pct = int(cols[0])
            sm_clock_mhz = int(cols[1])
            temp_c = int(cols[2])
            throttle_mask = int(cols[3], 16)
        except ValueError:
            return None
        return {
            "gpu_util_pct": util_pct,
            "gpu_sm_clock_mhz": sm_clock_mhz,
            "gpu_temp_c": temp_c,
            "gpu_throttle_mask": throttle_mask,
            "gpu_fault_throttled": (throttle_mask & NVML_FAULT_THROTTLE_MASK) != 0,
        }
    return None


def _default_read_text(path: str) -> Optional[str]:
    content: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        content = None
    return content


def _default_list_proc_pids(proc_root: str = "/proc") -> List[int]:
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return []
    pids: List[int] = []
    for entry in entries:
        if entry.isdigit():
            pids.append(int(entry))
    pids.sort()
    return pids


def _default_run_cmd(argv: Sequence[str], timeout_s: float = 2.0) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 127, ""


class HostContentionCollector:
    """Injectable Linux host telemetry collector (zero shell=True, zero cmdline/title reads)."""

    def __init__(
        self,
        *,
        read_text: Callable[[str], Optional[str]] = _default_read_text,
        list_proc_pids: Callable[[], List[int]] = _default_list_proc_pids,
        run_cmd: Callable[[Sequence[str]], Tuple[int, str]] = _default_run_cmd,
        cpu_count: Optional[int] = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        enable_gpu_probe: bool = True,
        enable_foreground_probe: bool = True,
    ) -> None:
        self.read_text = read_text
        self.list_proc_pids = list_proc_pids
        self.run_cmd = run_cmd
        detected_cpus = os.cpu_count()
        default_cpus = detected_cpus if detected_cpus is not None else 1
        self.cpu_count = cpu_count if cpu_count is not None else default_cpus
        require_ok(_is_strict_int(self.cpu_count) and self.cpu_count >= 1, "BAD_CPU_COUNT", "cpu_count must be >= 1")
        self.monotonic_ns = monotonic_ns
        self.sleep = sleep
        self.enable_gpu_probe = enable_gpu_probe
        self.enable_foreground_probe = enable_foreground_probe

    def read_ac_online(self) -> Optional[bool]:
        """Read AC mains online state from /sys/class/power_supply/AC0/online."""
        raw = self.read_text("/sys/class/power_supply/AC0/online")
        if raw is None:
            return None
        val = raw.strip()
        if val == "1":
            return True
        if val == "0":
            return False
        return None

    def read_foreground_app(self) -> str:
        """Read privacy-scrubbed active window class only (never window title)."""
        if not self.enable_foreground_probe:
            return "unavailable"
        rc, out = self.run_cmd(["hyprctl", "activewindow", "-j"])
        if rc != 0:
            return "unavailable"
        try:
            doc = json.loads(out)
        except ValueError:
            return "unavailable"
        if type(doc) is not dict:
            return "unavailable"
        raw_cls = doc.get("class")
        if not raw_cls:
            raw_cls = doc.get("initialClass")
        if not raw_cls:
            return "unavailable"
        return sanitize_proc_name(raw_cls)

    def read_gpu_compute_apps(self, exclude_pids: Set[int]) -> Tuple[str, List[Dict[str, Any]]]:
        """Query external GPU compute apps via nvidia-smi (gracefully unavailable)."""
        if not self.enable_gpu_probe:
            return "unavailable", []
        rc, out = self.run_cmd(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        if rc != 0:
            return "unavailable", []
        return "available", _parse_gpu_compute_apps(out, exclude_pids)

    def read_gpu_telemetry(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Query read-only GPU utilization, SM clock, temperature, and throttle mask."""
        if not self.enable_gpu_probe:
            return "unavailable", None
        rc, out = self.run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ]
        )
        if rc != 0:
            return "unavailable", None
        parsed = _parse_gpu_telemetry(out)
        if parsed is None:
            return "unavailable", None
        return "available", parsed

    def read_cpu_and_load(self, exclude_pids: Set[int]) -> Dict[str, Any]:
        """Read /proc/stat jiffies, /proc/loadavg, and excluded PIDs' CPU ticks."""
        stat_txt = self.read_text("/proc/stat")
        require_ok(stat_txt is not None, "PROC_STAT_MISSING", "unable to read /proc/stat")
        total_jiffies, idle_jiffies = _parse_proc_stat_cpu(stat_txt)
        load_txt = self.read_text("/proc/loadavg")
        require_ok(load_txt is not None, "LOADAVG_MISSING", "unable to read /proc/loadavg")
        load1, load5, load15 = _parse_loadavg(load_txt)
        self_ticks = 0
        for pid in sorted(exclude_pids):
            pstat = self.read_text(f"/proc/{pid}/stat")
            if pstat is not None:
                parsed = _parse_pid_stat(pstat)
                if parsed is not None:
                    self_ticks += parsed[0]
        return {
            "cpu_jiffies_total": total_jiffies,
            "cpu_jiffies_idle": idle_jiffies,
            "self_cpu_ticks": self_ticks,
            "loadavg_1m_q": load1,
            "loadavg_5m_q": load5,
            "loadavg_15m_q": load15,
        }

    def capture_snapshot(
        self,
        *,
        phase: str,
        exclude_pids: Optional[Iterable[int]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Capture privacy-scrubbed pre/post host process, GPU, and power snapshot."""
        require_ok(phase in ("pre", "post"), "SNAPSHOT_PHASE", f"invalid snapshot phase: {phase!r}")
        require_ok(_is_strict_int(top_k) and 1 <= top_k <= 20, "SNAPSHOT_TOPK", "top_k must be in [1,20]")
        excl = set(exclude_pids) if exclude_pids is not None else {os.getpid()}
        cpu_info = self.read_cpu_and_load(excl)
        procs: List[Dict[str, Any]] = []
        for pid in self.list_proc_pids():
            if pid in excl:
                continue
            pstat = self.read_text(f"/proc/{pid}/stat")
            if pstat is None:
                continue
            parsed = _parse_pid_stat(pstat)
            if parsed is None:
                continue
            comm_raw = self.read_text(f"/proc/{pid}/comm")
            procs.append(
                {
                    "pid": pid,
                    "comm": sanitize_proc_name(comm_raw),
                    "cpu_ticks": parsed[0],
                    "rss_pages": parsed[1],
                }
            )
        top_cpu = sorted(procs, key=lambda r: (-r["cpu_ticks"], r["pid"]))[:top_k]
        top_rss = sorted(procs, key=lambda r: (-r["rss_pages"], r["pid"]))[:top_k]
        gpu_apps_status, gpu_apps = self.read_gpu_compute_apps(excl)
        load1_ratio = cpu_info["loadavg_1m_q"] / Fraction(self.cpu_count)
        snap: Dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA,
            "phase": phase,
            "cpu_count": self.cpu_count,
            "loadavg_1m_q": format_q(cpu_info["loadavg_1m_q"]),
            "loadavg_5m_q": format_q(cpu_info["loadavg_5m_q"]),
            "loadavg_15m_q": format_q(cpu_info["loadavg_15m_q"]),
            "cpu_load_ratio_q": format_q(load1_ratio),
            "cpu_jiffies_total": cpu_info["cpu_jiffies_total"],
            "cpu_jiffies_idle": cpu_info["cpu_jiffies_idle"],
            "self_cpu_ticks": cpu_info["self_cpu_ticks"],
            "ac_online": self.read_ac_online(),
            "foreground_app": self.read_foreground_app(),
            "process_count_observed": len(procs),
            "top_cpu_processes": top_cpu,
            "top_rss_processes": top_rss,
            "gpu_compute_apps_status": gpu_apps_status,
            "gpu_compute_apps": gpu_apps,
        }
        snap["snapshot_sha256"] = canonical_sha256(snap)
        return snap

    def collect_in_run_sample(
        self,
        *,
        prev_cpu_total: int,
        prev_cpu_idle: int,
        prev_self_ticks: int,
        exclude_pids: Set[int],
    ) -> Tuple[Dict[str, Any], int, int, int]:
        """Collect one in-run contention sample using /proc/stat delta."""
        cpu_info = self.read_cpu_and_load(exclude_pids)
        dt_total = cpu_info["cpu_jiffies_total"] - prev_cpu_total
        dt_idle = cpu_info["cpu_jiffies_idle"] - prev_cpu_idle
        dt_self = max(0, cpu_info["self_cpu_ticks"] - prev_self_ticks)
        if dt_total > 0:
            busy_ticks = max(0, dt_total - dt_idle - dt_self)
            bg_util_q = min(Fraction(1), Fraction(busy_ticks, dt_total))
        else:
            bg_util_q = Fraction(0)
        load1_ratio = cpu_info["loadavg_1m_q"] / Fraction(self.cpu_count)
        gpu_status, gpu_tel = self.read_gpu_telemetry()
        apps_status, gpu_apps = self.read_gpu_compute_apps(exclude_pids)
        sample: Dict[str, Any] = {
            "cpu_load_ratio_q": format_q(load1_ratio),
            "bg_cpu_util_ratio_q": format_q(bg_util_q),
            "ac_online": self.read_ac_online(),
            "gpu_status": gpu_status,
            "gpu_util_pct": gpu_tel["gpu_util_pct"] if gpu_tel is not None else None,
            "gpu_sm_clock_mhz": gpu_tel["gpu_sm_clock_mhz"] if gpu_tel is not None else None,
            "gpu_temp_c": gpu_tel["gpu_temp_c"] if gpu_tel is not None else None,
            "gpu_fault_throttled": gpu_tel["gpu_fault_throttled"] if gpu_tel is not None else False,
            "external_gpu_apps_count": len(gpu_apps) if apps_status == "available" else 0,
        }
        return (
            sample,
            cpu_info["cpu_jiffies_total"],
            cpu_info["cpu_jiffies_idle"],
            cpu_info["self_cpu_ticks"],
        )


def exact_quantile_q(sorted_vals: Sequence[Fraction], num: int, den: int) -> Fraction:
    """Exact pure-Q inclusive quantile via linear interpolation."""
    n = len(sorted_vals)
    require_ok(n >= 1, "EMPTY_SAMPLES", "cannot compute quantile of empty sequence")
    pos = Fraction(num * (n - 1), den)
    lo = pos.numerator // pos.denominator
    frac = pos - lo
    if frac == 0:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[lo + 1] - sorted_vals[lo]) * frac


def compute_spread_stats_q(values: Sequence[Fraction]) -> Dict[str, Any]:
    """Compute exact pure-Q spread statistics (min, Q1, median, Q3, max, IQR, MAD, rel_IQR)."""
    svals = sorted(values)
    require_ok(len(svals) >= 3, "SPREAD_MIN_REPEATS", "at least 3 repeats required for spread reporting")
    q1 = exact_quantile_q(svals, 1, 4)
    med = exact_quantile_q(svals, 1, 2)
    q3 = exact_quantile_q(svals, 3, 4)
    iqr = q3 - q1
    abs_devs = sorted(abs(v - med) for v in svals)
    mad = exact_quantile_q(abs_devs, 1, 2)
    require_ok(med > 0, "NONPOSITIVE_MEDIAN", "median must be > 0 for relative IQR")
    rel_iqr = iqr / med
    return {
        "n_repeats": len(svals),
        "min_q": format_q(svals[0]),
        "q1_q": format_q(q1),
        "median_q": format_q(med),
        "q3_q": format_q(q3),
        "max_q": format_q(svals[-1]),
        "iqr_q": format_q(iqr),
        "mad_q": format_q(mad),
        "rel_iqr_q": format_q(rel_iqr),
        "median": round(float(med), 6),
        "iqr": round(float(iqr), 6),
        "mad": round(float(mad), 6),
    }


def _compute_contention_reasons(
    *,
    ac_online_all: bool,
    external_gpu_apps_detected: bool,
    cpu_load_max_q: Fraction,
    bg_cpu_util_max_q: Fraction,
    in_run_summary: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[str]:
    """Deterministically evaluate pure-Q contention rules against thresholds."""
    reasons: List[str] = []
    if thresholds["require_ac_online"] and not ac_online_all:
        reasons.append("ac_power_lost_or_unknown")
    if cpu_load_max_q > parse_q(thresholds["max_cpu_load_ratio_q"], code="THRESH_RATIONAL"):
        reasons.append("cpu_load_ratio_exceeded")
    if bg_cpu_util_max_q > parse_q(thresholds["max_bg_cpu_util_ratio_q"], code="THRESH_RATIONAL"):
        reasons.append("bg_cpu_util_exceeded")
    if not thresholds["allow_external_gpu_apps"] and external_gpu_apps_detected:
        reasons.append("external_gpu_apps_present")
    if in_run_summary["gpu_status"] == "available":
        if in_run_summary["gpu_util_max_pct"] > thresholds["max_external_gpu_util_pct"]:
            reasons.append("external_gpu_util_exceeded")
        if in_run_summary["gpu_temp_max_c"] > thresholds["max_gpu_temp_c"]:
            reasons.append("gpu_temp_exceeded")
        if not thresholds["allow_gpu_throttle"] and in_run_summary["gpu_throttle_fault_any"]:
            reasons.append("gpu_throttle_active")
    return reasons


def verify_host_snapshot(snapshot: Dict[str, Any], *, expected_phase: str) -> Dict[str, Any]:
    """Fail-closed verifier for a single pre/post host snapshot."""
    require_ok(type(snapshot) is dict, "SNAPSHOT_TYPE", "snapshot must be a JSON object")
    require_ok(snapshot.get("schema_version") == SNAPSHOT_SCHEMA, "SNAPSHOT_SCHEMA", "bad snapshot schema")
    require_ok(snapshot.get("phase") == expected_phase, "SNAPSHOT_PHASE", "unexpected snapshot phase")
    require_ok(
        _is_strict_int(snapshot.get("cpu_count")) and snapshot["cpu_count"] >= 1,
        "SNAPSHOT_CPU_COUNT",
        "snapshot cpu_count must be strict int >= 1",
    )
    snap_sha = snapshot.get("snapshot_sha256")
    require_ok(
        type(snap_sha) is str and bool(_HEX64_RE.match(snap_sha)),
        "SNAPSHOT_SHA_FORMAT",
        "invalid snapshot_sha256 format",
    )
    require_ok(
        canonical_sha256(snapshot, exclude_key="snapshot_sha256") == snap_sha,
        "SNAPSHOT_SHA_MISMATCH",
        "snapshot_sha256 mismatch",
    )
    parse_q(snapshot.get("cpu_load_ratio_q"), code="SNAPSHOT_RATIONAL")
    return snapshot


def build_environment_block(
    *,
    pre_snapshot: Dict[str, Any],
    post_snapshot: Dict[str, Any],
    in_run_samples: Sequence[Dict[str, Any]],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bind pre/post snapshot hashes, in-run sample summary, and contention flags."""
    verify_host_snapshot(pre_snapshot, expected_phase="pre")
    verify_host_snapshot(post_snapshot, expected_phase="post")
    require_ok(
        pre_snapshot["cpu_count"] == post_snapshot["cpu_count"],
        "ENV_CPU_COUNT_MISMATCH",
        "pre/post cpu_count mismatch",
    )
    require_ok(len(in_run_samples) >= 1, "ENV_NO_SAMPLES", "at least one in-run sample required")
    cfg = validate_contention_thresholds(thresholds)

    load_vals: List[Fraction] = [
        parse_q(pre_snapshot["cpu_load_ratio_q"], code="ENV_RATIONAL"),
        parse_q(post_snapshot["cpu_load_ratio_q"], code="ENV_RATIONAL"),
    ]
    bg_vals: List[Fraction] = []
    ac_states: List[Optional[bool]] = [pre_snapshot["ac_online"], post_snapshot["ac_online"]]
    gpu_utils: List[int] = []
    gpu_clocks: List[int] = []
    gpu_temps: List[int] = []
    gpu_throttle_any = False
    ext_gpu_apps_max = max(len(pre_snapshot["gpu_compute_apps"]), len(post_snapshot["gpu_compute_apps"]))

    for sample in in_run_samples:
        load_vals.append(parse_q(sample["cpu_load_ratio_q"], code="ENV_RATIONAL"))
        bg_vals.append(parse_q(sample["bg_cpu_util_ratio_q"], code="ENV_RATIONAL"))
        ac_states.append(sample["ac_online"])
        ext_gpu_apps_max = max(ext_gpu_apps_max, sample["external_gpu_apps_count"])
        if sample["gpu_status"] == "available":
            gpu_utils.append(sample["gpu_util_pct"])
            gpu_clocks.append(sample["gpu_sm_clock_mhz"])
            gpu_temps.append(sample["gpu_temp_c"])
            if sample["gpu_fault_throttled"]:
                gpu_throttle_any = True

    load_sorted = sorted(load_vals)
    bg_sorted = sorted(bg_vals)
    cpu_load_max_q = load_sorted[-1]
    bg_cpu_util_max_q = bg_sorted[-1]

    ac_online_count = sum(1 for s in ac_states if s is True)
    ac_offline_count = sum(1 for s in ac_states if s is False)
    ac_unknown_count = sum(1 for s in ac_states if s is None)
    ac_online_all = ac_online_count == len(ac_states)
    if ac_online_all:
        power_band = "ac"
    elif ac_offline_count > 0:
        power_band = "battery"
    else:
        power_band = "unknown"

    gpu_available = len(gpu_utils) > 0
    in_run_summary: Dict[str, Any] = {
        "sample_count": len(in_run_samples),
        "cpu_load_ratio_median_q": format_q(exact_quantile_q(load_sorted, 1, 2)),
        "cpu_load_ratio_max_q": format_q(cpu_load_max_q),
        "bg_cpu_util_median_q": format_q(exact_quantile_q(bg_sorted, 1, 2)),
        "bg_cpu_util_max_q": format_q(bg_cpu_util_max_q),
        "ac_online_observations": ac_online_count,
        "ac_offline_observations": ac_offline_count,
        "ac_unknown_observations": ac_unknown_count,
        "gpu_status": "available" if gpu_available else "unavailable",
        "gpu_util_max_pct": max(gpu_utils) if gpu_available else None,
        "gpu_sm_clock_min_mhz": min(gpu_clocks) if gpu_available else None,
        "gpu_sm_clock_max_mhz": max(gpu_clocks) if gpu_available else None,
        "gpu_temp_max_c": max(gpu_temps) if gpu_available else None,
        "gpu_throttle_fault_any": gpu_throttle_any,
        "external_gpu_apps_max": ext_gpu_apps_max,
    }

    external_gpu_apps_detected = ext_gpu_apps_max > 0
    reasons = _compute_contention_reasons(
        ac_online_all=ac_online_all,
        external_gpu_apps_detected=external_gpu_apps_detected,
        cpu_load_max_q=cpu_load_max_q,
        bg_cpu_util_max_q=bg_cpu_util_max_q,
        in_run_summary=in_run_summary,
        thresholds=cfg,
    )
    foreground_apps = sorted({pre_snapshot["foreground_app"], post_snapshot["foreground_app"]})

    env: Dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "pre_snapshot_sha256": pre_snapshot["snapshot_sha256"],
        "post_snapshot_sha256": post_snapshot["snapshot_sha256"],
        "in_run_summary": in_run_summary,
        "ac_online_all": ac_online_all,
        "power_band": power_band,
        "external_gpu_apps_detected": external_gpu_apps_detected,
        "cpu_load_max_ratio": round(float(cpu_load_max_q), 6),
        "cpu_load_max_ratio_q": format_q(cpu_load_max_q),
        "bg_cpu_util_max_ratio": round(float(bg_cpu_util_max_q), 6),
        "bg_cpu_util_max_ratio_q": format_q(bg_cpu_util_max_q),
        "foreground_apps": foreground_apps,
        "thresholds": cfg,
        "contended": len(reasons) > 0,
        "contention_reasons": reasons,
    }
    env["environment_sha256"] = canonical_sha256(env)
    return env


def verify_environment_block(env: Any) -> Dict[str, Any]:
    """Strict fail-closed R1-R5 verifier for a receipt environment block."""
    require_ok(type(env) is dict, "ENV_TYPE", "environment block must be a JSON object")
    require_ok(env.get("schema_version") == ENVIRONMENT_SCHEMA, "ENV_SCHEMA", "invalid environment schema")
    require_ok(
        env.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_key(env),
        "ENV_ENERGY_POLICY",
        "environment violates uncontrolled-laptop energy policy",
    )
    require_ok(set(env.keys()) == _ENV_REQUIRED_KEYS, "ENV_KEYS", "environment keys mismatch")
    for sha_key in ("pre_snapshot_sha256", "post_snapshot_sha256", "environment_sha256"):
        val = env.get(sha_key)
        require_ok(type(val) is str and bool(_HEX64_RE.match(val)), "ENV_HASH_FORMAT", f"bad {sha_key}")
    require_ok(
        canonical_sha256(env, exclude_key="environment_sha256") == env["environment_sha256"],
        "ENV_HASH_MISMATCH",
        "environment_sha256 mismatch",
    )
    require_ok(
        _is_strict_bool(env.get("ac_online_all"))
        and _is_strict_bool(env.get("external_gpu_apps_detected"))
        and _is_strict_bool(env.get("contended")),
        "ENV_STRICT_TYPES",
        "environment booleans must be strict bool",
    )
    require_ok(
        _is_strict_finite_num(env.get("cpu_load_max_ratio"))
        and _is_strict_finite_num(env.get("bg_cpu_util_max_ratio")),
        "ENV_STRICT_TYPES",
        "ratio floats must be strict finite numbers",
    )
    summary = env.get("in_run_summary")
    require_ok(type(summary) is dict, "ENV_SUMMARY_TYPE", "in_run_summary must be a dict")
    require_ok(set(summary.keys()) == _SUMMARY_REQUIRED_KEYS, "ENV_SUMMARY_KEYS", "in_run_summary keys mismatch")
    require_ok(
        _is_strict_int(summary.get("sample_count")) and summary["sample_count"] >= 1,
        "ENV_STRICT_TYPES",
        "sample_count must be strict int >= 1",
    )
    for obs_key in ("ac_online_observations", "ac_offline_observations", "ac_unknown_observations", "external_gpu_apps_max"):
        require_ok(
            _is_strict_int(summary.get(obs_key)) and summary[obs_key] >= 0,
            "ENV_STRICT_TYPES",
            f"{obs_key} must be strict non-negative int",
        )
    require_ok(
        _is_strict_bool(summary.get("gpu_throttle_fault_any")),
        "ENV_STRICT_TYPES",
        "gpu_throttle_fault_any must be strict bool",
    )
    raw_thresholds = env.get("thresholds")
    require_ok(type(raw_thresholds) is dict, "THRESH_TYPE", "environment thresholds must be a JSON object")
    cfg = validate_contention_thresholds(raw_thresholds)

    cpu_load_max_q = parse_q(env.get("cpu_load_max_ratio_q"), code="ENV_RATIONAL")
    bg_cpu_max_q = parse_q(env.get("bg_cpu_util_max_ratio_q"), code="ENV_RATIONAL")
    cpu_load_med_q = parse_q(summary.get("cpu_load_ratio_median_q"), code="ENV_RATIONAL")
    bg_cpu_med_q = parse_q(summary.get("bg_cpu_util_median_q"), code="ENV_RATIONAL")
    require_ok(
        parse_q(summary.get("cpu_load_ratio_max_q"), code="ENV_RATIONAL") == cpu_load_max_q,
        "ENV_SUMMARY_MISMATCH",
        "summary cpu_load_ratio_max_q mismatch",
    )
    require_ok(
        parse_q(summary.get("bg_cpu_util_max_q"), code="ENV_RATIONAL") == bg_cpu_max_q,
        "ENV_SUMMARY_MISMATCH",
        "summary bg_cpu_util_max_q mismatch",
    )
    require_ok(
        Fraction(0) <= cpu_load_med_q <= cpu_load_max_q
        and Fraction(0) <= bg_cpu_med_q <= bg_cpu_max_q <= Fraction(1),
        "ENV_RANGE",
        "CPU load or background utilization ratio out of physical bounds",
    )
    require_ok(
        abs(env["cpu_load_max_ratio"] - float(cpu_load_max_q)) <= 1e-5
        and abs(env["bg_cpu_util_max_ratio"] - float(bg_cpu_max_q)) <= 1e-5,
        "ENV_FLOAT_MISMATCH",
        "float ratio projection disagrees with exact rational",
    )

    total_ac_obs = (
        summary["ac_online_observations"]
        + summary["ac_offline_observations"]
        + summary["ac_unknown_observations"]
    )
    require_ok(
        total_ac_obs == summary["sample_count"] + 2,
        "ENV_POWER_BAND",
        "AC observation count must equal sample_count + 2",
    )
    expected_ac_all = summary["ac_online_observations"] == total_ac_obs
    require_ok(env["ac_online_all"] == expected_ac_all, "ENV_AC_ALL_MISMATCH", "ac_online_all inconsistent")
    if expected_ac_all:
        expected_power_band = "ac"
    elif summary["ac_offline_observations"] > 0:
        expected_power_band = "battery"
    else:
        expected_power_band = "unknown"
    require_ok(env.get("power_band") == expected_power_band, "ENV_POWER_BAND", "power_band inconsistent")

    gpu_status = summary.get("gpu_status")
    require_ok(gpu_status in ("available", "unavailable"), "ENV_GPU_STATUS", "invalid gpu_status")
    if gpu_status == "available":
        for gk in ("gpu_util_max_pct", "gpu_sm_clock_min_mhz", "gpu_sm_clock_max_mhz", "gpu_temp_max_c"):
            require_ok(_is_strict_int(summary.get(gk)), "ENV_STRICT_TYPES", f"{gk} must be strict int")
        require_ok(
            0 <= summary["gpu_util_max_pct"] <= 100
            and 0 <= summary["gpu_sm_clock_min_mhz"] <= summary["gpu_sm_clock_max_mhz"]
            and 0 <= summary["gpu_temp_max_c"] <= 130,
            "ENV_GPU_RANGE",
            "GPU telemetry fields out of physical bounds",
        )
    else:
        require_ok(
            summary["gpu_util_max_pct"] is None
            and summary["gpu_sm_clock_min_mhz"] is None
            and summary["gpu_sm_clock_max_mhz"] is None
            and summary["gpu_temp_max_c"] is None
            and summary["gpu_throttle_fault_any"] is False,
            "ENV_GPU_UNAVAILABLE",
            "unavailable GPU must have null metrics and gpu_throttle_fault_any == False",
        )
    require_ok(
        env["external_gpu_apps_detected"] == (summary["external_gpu_apps_max"] > 0),
        "ENV_GPU_APPS_MISMATCH",
        "external_gpu_apps_detected inconsistent",
    )

    expected_reasons = _compute_contention_reasons(
        ac_online_all=env["ac_online_all"],
        external_gpu_apps_detected=env["external_gpu_apps_detected"],
        cpu_load_max_q=cpu_load_max_q,
        bg_cpu_util_max_q=bg_cpu_max_q,
        in_run_summary=summary,
        thresholds=cfg,
    )
    require_ok(
        env.get("contention_reasons") == expected_reasons and env["contended"] == (len(expected_reasons) > 0),
        "ENV_CONTENTION_MISMATCH",
        "contention classification mismatch",
    )
    return env


def _verify_metric_labels_allowed(arm_name: Any, metric_name: Any, metric_unit: Any) -> None:
    require_ok(
        type(arm_name) is str
        and bool(arm_name.strip())
        and type(metric_name) is str
        and bool(metric_name.strip())
        and type(metric_unit) is str
        and bool(metric_unit.strip()),
        "RECEIPT_METRIC_TYPE",
        "arm_name, metric_name, and metric_unit must be non-empty strings",
    )
    require_ok(
        not _is_forbidden_energy_label(arm_name)
        and not _is_forbidden_energy_label(metric_name)
        and not _is_forbidden_energy_label(metric_unit),
        "RECEIPT_ENERGY_POLICY",
        "forbidden energy/power arm_name, metric_name, or metric_unit",
    )


def bind_receipt_with_environment(
    *,
    arm_name: str,
    metric_name: str,
    metric_unit: str,
    repeat_values_q: Sequence[Fraction],
    environment: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bind repeat measurements and verified environment block under canonical result_sha256."""
    verify_environment_block(environment)
    min_rep = environment["thresholds"]["min_repeats"]
    require_ok(
        isinstance(repeat_values_q, AbcSequence)
        and not isinstance(repeat_values_q, (str, bytes, bytearray)),
        "RECEIPT_VALUES_TYPE",
        "repeat_values_q must be a sequence of Fractions",
    )
    require_ok(
        len(repeat_values_q) >= min_rep,
        "RECEIPT_MIN_REPEATS",
        f"at least {min_rep} repeats required for spread reporting",
    )
    for val in repeat_values_q:
        require_ok(
            isinstance(val, Fraction) and val > 0,
            "RECEIPT_BAD_VALUE",
            "repeat values must be positive Fractions",
        )
    require_ok(
        metadata is None or type(metadata) is dict,
        "RECEIPT_METADATA_TYPE",
        "metadata must be a dict or None",
    )
    meta = dict(metadata) if metadata is not None else {}
    require_ok(
        not has_forbidden_energy_key(meta, _path=("metadata",)),
        "RECEIPT_ENERGY_POLICY",
        "forbidden energy key or claim in metadata",
    )
    _verify_metric_labels_allowed(arm_name, metric_name, metric_unit)
    spread = compute_spread_stats_q(repeat_values_q)
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "arm_name": arm_name,
        "metric_name": metric_name,
        "metric_unit": metric_unit,
        "repeat_values_q": [format_q(v) for v in repeat_values_q],
        "spread": spread,
        "environment": environment,
        "metadata": meta,
    }
    receipt["result_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_bound_receipt(receipt: Any) -> Dict[str, Any]:
    """Fail-closed verifier covering result_sha256, environment, spread, and energy policy."""
    require_ok(type(receipt) is dict, "RECEIPT_TYPE", "receipt must be a JSON object")
    require_ok(receipt.get("schema_version") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA", "invalid receipt schema")
    require_ok(
        type(receipt.get("metadata")) is dict,
        "RECEIPT_METADATA_TYPE",
        "receipt metadata must be a JSON object",
    )
    require_ok(
        receipt.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_key(receipt),
        "RECEIPT_ENERGY_POLICY",
        "receipt violates uncontrolled-laptop energy policy",
    )
    require_ok(
        set(receipt.keys()) == _RECEIPT_REQUIRED_KEYS,
        "RECEIPT_KEYS",
        "receipt keys mismatch required schema",
    )
    _verify_metric_labels_allowed(receipt.get("arm_name"), receipt.get("metric_name"), receipt.get("metric_unit"))
    rsha = receipt.get("result_sha256")
    require_ok(type(rsha) is str and bool(_HEX64_RE.match(rsha)), "RECEIPT_SHA_FORMAT", "bad result_sha256 format")
    require_ok(
        canonical_sha256(receipt, exclude_key="result_sha256") == rsha,
        "RECEIPT_SHA_MISMATCH",
        "result_sha256 mismatch",
    )
    env = verify_environment_block(receipt.get("environment"))
    raw_vals = receipt.get("repeat_values_q")
    require_ok(type(raw_vals) is list, "RECEIPT_VALUES_TYPE", "repeat_values_q must be a list")
    min_rep = env["thresholds"]["min_repeats"]
    require_ok(len(raw_vals) >= min_rep, "RECEIPT_MIN_REPEATS", "fewer repeats than min_repeats")
    parsed_vals = [parse_q(v, code="RECEIPT_RATIONAL") for v in raw_vals]
    for v in parsed_vals:
        require_ok(v > 0, "RECEIPT_BAD_VALUE", "repeat values must be > 0")
    expected_spread = compute_spread_stats_q(parsed_vals)
    require_ok(receipt.get("spread") == expected_spread, "RECEIPT_SPREAD_MISMATCH", "spread statistics mismatch")
    return receipt


def contention_band_key(env: Dict[str, Any]) -> str:
    """Return canonical contention-band stratification key bound to threshold profile hash."""
    prefix = "contended" if env["contended"] else "quiet"
    thresh_tag = canonical_sha256(env["thresholds"])[:12]
    return f"{prefix}_{env['power_band']}_{thresh_tag}"


def compare_bound_receipts(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    allow_stratified: bool = False,
) -> Dict[str, Any]:
    """Compare two verified receipts in pure-Q, failing closed across contention bands or thresholds."""
    verify_bound_receipt(baseline)
    verify_bound_receipt(candidate)
    require_ok(
        baseline["metric_name"] == candidate["metric_name"]
        and baseline["metric_unit"] == candidate["metric_unit"],
        "COMPARE_METRIC_MISMATCH",
        "cannot compare different metrics or units",
    )
    base_band = contention_band_key(baseline["environment"])
    cand_band = contention_band_key(candidate["environment"])
    base_med = parse_q(baseline["spread"]["median_q"], code="COMPARE_RATIONAL")
    cand_med = parse_q(candidate["spread"]["median_q"], code="COMPARE_RATIONAL")

    if baseline["environment"]["thresholds"] != candidate["environment"]["thresholds"]:
        require_ok(
            allow_stratified,
            "COMPARE_THRESHOLD_MISMATCH",
            f"cross-threshold comparison forbidden ({base_band} vs {cand_band})",
        )
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": "STRATIFIED_UNCOMPARABLE",
            "baseline_band": base_band,
            "candidate_band": cand_band,
            "baseline_median_q": format_q(base_med),
            "candidate_median_q": format_q(cand_med),
            "ratio_q": None,
            "delta_q": None,
            "non_overlapping_iqr": None,
        }

    if base_band != cand_band:
        require_ok(
            allow_stratified,
            "COMPARE_BAND_MISMATCH",
            f"cross-band comparison forbidden ({base_band} vs {cand_band})",
        )
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": "STRATIFIED_UNCOMPARABLE",
            "baseline_band": base_band,
            "candidate_band": cand_band,
            "baseline_median_q": format_q(base_med),
            "candidate_median_q": format_q(cand_med),
            "ratio_q": None,
            "delta_q": None,
            "non_overlapping_iqr": None,
        }

    base_q1 = parse_q(baseline["spread"]["q1_q"], code="COMPARE_RATIONAL")
    base_q3 = parse_q(baseline["spread"]["q3_q"], code="COMPARE_RATIONAL")
    cand_q1 = parse_q(candidate["spread"]["q1_q"], code="COMPARE_RATIONAL")
    cand_q3 = parse_q(candidate["spread"]["q3_q"], code="COMPARE_RATIONAL")
    ratio_q = cand_med / base_med
    delta_q = cand_med - base_med
    non_overlap = cand_q1 > base_q3 or base_q1 > cand_q3
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "COMPARABLE_SAME_BAND",
        "band": base_band,
        "baseline_median_q": format_q(base_med),
        "candidate_median_q": format_q(cand_med),
        "ratio_q": format_q(ratio_q),
        "ratio": round(float(ratio_q), 6),
        "delta_q": format_q(delta_q),
        "non_overlapping_iqr": non_overlap,
    }


def stratify_receipts(receipts: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Verify and partition receipts into separate contention+threshold bands."""
    bands: Dict[str, List[Dict[str, Any]]] = {}
    for rec in receipts:
        verify_bound_receipt(rec)
        key = contention_band_key(rec["environment"])
        bands.setdefault(key, []).append(
            {
                "arm_name": rec["arm_name"],
                "result_sha256": rec["result_sha256"],
                "environment_sha256": rec["environment"]["environment_sha256"],
                "median_q": rec["spread"]["median_q"],
                "iqr_q": rec["spread"]["iqr_q"],
                "mad_q": rec["spread"]["mad_q"],
                "contention_reasons": rec["environment"]["contention_reasons"],
            }
        )
    return {k: bands[k] for k in sorted(bands.keys())}


def run_monitored_arm(
    *,
    arm_name: str,
    metric_name: str,
    metric_unit: str,
    n_repeats: int,
    workload_fn: Callable[[], Fraction],
    collector: HostContentionCollector,
    thresholds: Optional[Dict[str, Any]] = None,
    sample_interval_s: float = 0.05,
    exclude_pids: Optional[Set[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run an arm with pre/post snapshots and background in-run contention sampling."""
    cfg = validate_contention_thresholds(thresholds)
    require_ok(
        _is_strict_int(n_repeats) and n_repeats >= cfg["min_repeats"],
        "ARM_MIN_REPEATS",
        "n_repeats below min_repeats",
    )
    excl = set(exclude_pids) if exclude_pids is not None else {os.getpid()}
    pre_snap = collector.capture_snapshot(phase="pre", exclude_pids=excl)
    prev_total = pre_snap["cpu_jiffies_total"]
    prev_idle = pre_snap["cpu_jiffies_idle"]
    prev_self = pre_snap["self_cpu_ticks"]

    samples: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    def _sampler_loop() -> None:
        nonlocal prev_total, prev_idle, prev_self
        while not stop_event.is_set():
            collector.sleep(sample_interval_s)
            samp, prev_total, prev_idle, prev_self = collector.collect_in_run_sample(
                prev_cpu_total=prev_total,
                prev_cpu_idle=prev_idle,
                prev_self_ticks=prev_self,
                exclude_pids=excl,
            )
            samples.append(samp)

    sampler_thread = threading.Thread(target=_sampler_loop, daemon=True)
    sampler_thread.start()
    repeat_vals: List[Fraction] = []
    try:
        for _ in range(n_repeats):
            repeat_vals.append(workload_fn())
    finally:
        stop_event.set()
        sampler_thread.join(timeout=5.0)

    if not samples:
        samp, _, _, _ = collector.collect_in_run_sample(
            prev_cpu_total=prev_total,
            prev_cpu_idle=prev_idle,
            prev_self_ticks=prev_self,
            exclude_pids=excl,
        )
        samples.append(samp)

    post_snap = collector.capture_snapshot(phase="post", exclude_pids=excl)
    env = build_environment_block(
        pre_snapshot=pre_snap,
        post_snapshot=post_snap,
        in_run_samples=samples,
        thresholds=cfg,
    )
    return bind_receipt_with_environment(
        arm_name=arm_name,
        metric_name=metric_name,
        metric_unit=metric_unit,
        repeat_values_q=repeat_vals,
        environment=env,
        metadata=metadata,
    )


def _cpu_hash_workload(rounds: int = 20000) -> Fraction:
    """Deterministic pure-stdlib CPU micro-workload returning exact ops/s Fraction."""
    t0 = time.monotonic_ns()
    acc = b"silisense-host-contention-control-seed"
    for _ in range(rounds):
        acc = hashlib.sha256(acc).digest()
    elapsed_ns = max(1, time.monotonic_ns() - t0)
    return Fraction(rounds * 1_000_000_000, elapsed_ns)


def run_live_contention_probe(
    *,
    n_repeats: int = 5,
    rounds_per_repeat: int = 80000,
    burner_workers: int = 10,
    collector: Optional[HostContentionCollector] = None,
    workload_fn: Optional[Callable[[], Fraction]] = None,
) -> Dict[str, Any]:
    """Run live CPU-only quiet vs controlled synthetic CPU-load probe."""
    col = collector if collector is not None else HostContentionCollector()
    work = workload_fn if workload_fn is not None else (lambda: _cpu_hash_workload(rounds_per_repeat))

    quiet_receipt = run_monitored_arm(
        arm_name="quiet_cpu_sha256_band",
        metric_name="sha256_rounds_per_sec",
        metric_unit="ops/s",
        n_repeats=n_repeats,
        workload_fn=work,
        collector=col,
        metadata={"synthetic_burner_workers": 0, "workload": "stdlib_sha256_chain"},
    )

    burners: List[subprocess.Popen[str]] = []
    burner_code = "x=1\nwhile True:\n x=(x*1664525+1013904223)&0xFFFFFFFF\n"
    try:
        for _ in range(burner_workers):
            burners.append(
                subprocess.Popen(
                    [sys.executable, "-I", "-B", "-c", burner_code],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            )
        col.sleep(0.25)
        contended_receipt = run_monitored_arm(
            arm_name="synthetic_contended_cpu_sha256_band",
            metric_name="sha256_rounds_per_sec",
            metric_unit="ops/s",
            n_repeats=n_repeats,
            workload_fn=work,
            collector=col,
            metadata={"synthetic_burner_workers": burner_workers, "workload": "stdlib_sha256_chain"},
        )
    finally:
        for proc in burners:
            proc.terminate()
        for proc in burners:
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)

    cross_band = compare_bound_receipts(quiet_receipt, contended_receipt, allow_stratified=True)
    stratified = stratify_receipts([quiet_receipt, contended_receipt])
    probe: Dict[str, Any] = {
        "schema_version": PROBE_SCHEMA,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "cpu_count": col.cpu_count,
        "n_repeats_per_band": n_repeats,
        "synthetic_burner_workers": burner_workers,
        "quiet_receipt": quiet_receipt,
        "contended_receipt": contended_receipt,
        "cross_band_comparison": cross_band,
        "stratified_bands": stratified,
        "band_separation_verified": (
            not quiet_receipt["environment"]["contended"]
            and contended_receipt["environment"]["contended"]
            and cross_band["status"] == "STRATIFIED_UNCOMPARABLE"
        ),
    }
    probe["probe_sha256"] = canonical_sha256(probe)
    return probe


def verify_live_probe(probe: Any) -> Dict[str, Any]:
    """Fail-closed verifier for a saved live quiet-vs-contended probe JSON."""
    require_ok(type(probe) is dict, "PROBE_TYPE", "probe must be a JSON object")
    require_ok(probe.get("schema_version") == PROBE_SCHEMA, "PROBE_SCHEMA", "invalid probe schema")
    require_ok(
        probe.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_key(probe),
        "PROBE_ENERGY_POLICY",
        "probe violates uncontrolled-laptop energy policy",
    )
    require_ok(
        set(probe.keys()) == _PROBE_REQUIRED_KEYS,
        "PROBE_KEYS",
        "probe keys mismatch required schema",
    )
    psha = probe.get("probe_sha256")
    require_ok(type(psha) is str and bool(_HEX64_RE.match(psha)), "PROBE_SHA_FORMAT", "bad probe_sha256 format")
    require_ok(
        canonical_sha256(probe, exclude_key="probe_sha256") == psha,
        "PROBE_SHA_MISMATCH",
        "probe_sha256 mismatch",
    )
    quiet = verify_bound_receipt(probe.get("quiet_receipt"))
    contended = verify_bound_receipt(probe.get("contended_receipt"))
    require_ok(
        _is_strict_int(probe.get("cpu_count")) and probe["cpu_count"] >= 1,
        "PROBE_META",
        "cpu_count must be a positive integer",
    )
    require_ok(
        _is_strict_int(probe.get("n_repeats_per_band"))
        and probe["n_repeats_per_band"] == len(quiet["repeat_values_q"])
        and probe["n_repeats_per_band"] == len(contended["repeat_values_q"]),
        "PROBE_META",
        "n_repeats_per_band must match embedded receipt repeat counts",
    )
    require_ok(
        _is_strict_int(probe.get("synthetic_burner_workers")) and probe["synthetic_burner_workers"] >= 0,
        "PROBE_META",
        "synthetic_burner_workers must be a non-negative integer",
    )
    expected_cmp = compare_bound_receipts(quiet, contended, allow_stratified=True)
    require_ok(probe.get("cross_band_comparison") == expected_cmp, "PROBE_CMP_MISMATCH", "cross_band_comparison mismatch")
    expected_strat = stratify_receipts([quiet, contended])
    require_ok(probe.get("stratified_bands") == expected_strat, "PROBE_STRAT_MISMATCH", "stratified_bands mismatch")
    expected_sep = (
        not quiet["environment"]["contended"]
        and contended["environment"]["contended"]
        and expected_cmp["status"] == "STRATIFIED_UNCOMPARABLE"
    )
    require_ok(
        probe.get("band_separation_verified") == expected_sep,
        "PROBE_SEP_MISMATCH",
        "band_separation_verified mismatch",
    )
    return probe


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for live probe capture or offline receipt verification."""
    parser = argparse.ArgumentParser(description="SiliSense HostContentionSentinel")
    parser.add_argument("--verify-probe", type=str, default=None, help="Verify saved live probe JSON")
    parser.add_argument("--run-live-probe", type=str, default=None, help="Run live CPU-only probe and write JSON")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per band (>=3)")
    parser.add_argument("--rounds", type=int, default=80000, help="SHA-256 rounds per repeat")
    parser.add_argument("--burners", type=int, default=10, help="External synthetic CPU burner processes")
    args = parser.parse_args(argv)

    try:
        if args.verify_probe is not None:
            with open(args.verify_probe, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            verify_live_probe(doc)
            status = "PASS" if doc["band_separation_verified"] else "VERIFIED_NO_BAND_SEPARATION"
            summary = {
                "status": status,
                "probe_sha256": doc["probe_sha256"],
                "quiet_contended": doc["quiet_receipt"]["environment"]["contended"],
                "contended_contended": doc["contended_receipt"]["environment"]["contended"],
                "cross_band_status": doc["cross_band_comparison"]["status"],
                "band_separation_verified": doc["band_separation_verified"],
            }
            sys.stdout.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
            return 0

        if args.run_live_probe is not None:
            probe = run_live_contention_probe(
                n_repeats=args.repeats,
                rounds_per_repeat=args.rounds,
                burner_workers=args.burners,
            )
            with open(args.run_live_probe, "w", encoding="utf-8") as handle:
                json.dump(probe, handle, sort_keys=True, indent=2)
                handle.write("\n")
            os.chmod(args.run_live_probe, 0o644)
            sys.stdout.write(json.dumps({"status": "WROTE", "probe_sha256": probe["probe_sha256"]}, sort_keys=True) + "\n")
            return 0
    except SiliSenseValueError as exc:
        sys.stderr.write(
            json.dumps({"status": "ERROR", "code": exc.code, "message": str(exc)}, sort_keys=True) + "\n"
        )
        return 2
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            json.dumps({"status": "ERROR", "code": "CLI_IO_ERROR", "message": str(exc)}, sort_keys=True) + "\n"
        )
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
