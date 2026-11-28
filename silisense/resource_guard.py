#!/usr/bin/env python3
"""Host CPU-RAM + GPU single-loader serialization guard (`flock`, `O_NOFOLLOW`, PSI, GPU-idle).

Prevents concurrent multi-GB model or embedding loads from triggering host
OOM kills on memory-bounded workstations even when GPU VRAM appears idle.

Enforces:
  1. Exclusive `fcntl.flock` lockfile (`$XDG_CACHE_HOME/silisense-locks/heavy_model_loader.lock`,
     ancestor-verified with `O_DIRECTORY | O_NOFOLLOW`, `0o600`, `st_nlink == 1`).
  2. Live `/proc/meminfo` `MemAvailable >= min_mem_gib` (default `12.0` GiB) and
     optional `/proc/pressure/memory` `some avg10 < max_psi_avg10`.
  3. Competing heavy-loader `/proc/<pid>/cmdline` process check.
  4. Optional GPU idle check (`nvidia-smi memory.used < max_gpu_mib` and `0`
     competing compute apps).
  5. Under `--exec`, exports `SILISENSE_RESOURCE_GUARD_ACTIVE=1` and
     `SILISENSE_RESOURCE_GUARD_LOCK=<lock_path>` into the guarded child env,
     enforces `--timeout-seconds` process-group termination on hang, and releases
     the exclusive lock cleanly on exit.

Pure Python stdlib, zero bare `assert`s, LF-only, R4-parity safe across `-I -B`,
`-I -O -B`, and `-I -B -W error`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import errno as _errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense.errors import HardwareExitCode, SiliSenseValueError


SCHEMA_CHECK = "silisense.resource_guard.check.v1"
SCHEMA_EXEC = "silisense.resource_guard.exec.v1"
SCHEMA_SELF_TEST = "silisense.resource_guard.self_test.v1"

GUARD_ENV_ACTIVE = "SILISENSE_RESOURCE_GUARD_ACTIVE"
GUARD_ENV_LOCK = "SILISENSE_RESOURCE_GUARD_LOCK"

DEFAULT_MIN_MEM_GIB = 12.0
DEFAULT_MAX_GPU_MIB = 300.0
DEFAULT_MAX_PSI_AVG10 = 25.0
DEFAULT_WAIT_SECONDS = 0
_KIB_PER_GIB = 1024 * 1024

_HEAVY_CMD_PATTERNS: tuple[str, ...] = (
    "from_pretrained",
    "AutoModel",
    "SentenceTransformer",
    "transformers.Trainer",
    "vllm.entrypoints",
    "--run-live-gpu",
    "--measure-cpu",
    "--torch-selftest",
)
_LIGHTWEIGHT_EXCLUSIONS: tuple[str, ...] = (
    "--cpu-selftest",
    "--self-test",
    "--check",
    "resource_guard.py",
)
_MEMINFO_LINE_RE = re.compile(r"^([A-Za-z0-9_()]+):\s+(\d+)\s*(?:kB)?\s*$")
_PSI_AVG10_RE = re.compile(r"^some\s+avg10=([0-9]+(?:\.[0-9]+)?)\b")


class HostResourceGuardError(SiliSenseValueError):
    """Fail-closed domain error with stable machine-readable error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: HardwareExitCode = HardwareExitCode.PREFLIGHT_FAIL,
    ) -> None:
        super().__init__(code, message, exit_code=exit_code)


def _require(
    cond: bool,
    code: str,
    message: str,
    *,
    exit_code: HardwareExitCode = HardwareExitCode.PREFLIGHT_FAIL,
) -> None:
    if not cond:
        raise HostResourceGuardError(code, message, exit_code=exit_code)


def _require_pos_number(val: Any, name: str, *, allow_zero: bool = False) -> float:
    _require(
        type(val) in (int, float),
        "INVALID_ARGUMENT",
        f"{name} must be int or float, got {type(val).__name__}",
    )
    fval = float(val)
    _require(math.isfinite(fval), "INVALID_ARGUMENT", f"{name} must be finite")
    if allow_zero:
        _require(fval >= 0.0, "INVALID_ARGUMENT", f"{name} must be >= 0")
    else:
        _require(fval > 0.0, "INVALID_ARGUMENT", f"{name} must be > 0")
    return fval


def _require_nonneg_int(val: Any, name: str) -> int:
    _require(type(val) is int and val >= 0, "INVALID_ARGUMENT", f"{name} must be non-negative int")
    return val


def default_lock_path() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    return base / "silisense-locks" / "heavy_model_loader.lock"


def _open_parent_dir_nofollow(parent: Path, label: str, *, create_dirs: bool = False) -> int:
    parts = parent.parts
    _require(bool(parts), "INVALID_PATH", f"{label} empty path")
    for comp in parts:
        _require(comp != "..", "PATH_TRAVERSAL_REFUSED", f"{label} '..' traversal refused: {parent}")

    if create_dirs:
        existing_prefix = Path(os.sep) if parent.is_absolute() else Path(".")
        start_idx = 1 if parent.is_absolute() else 0
        for comp in parts[start_idx:]:
            if comp in ("", "."):
                continue
            cand = existing_prefix / comp
            try:
                st = os.lstat(cand)
                if stat.S_ISLNK(st.st_mode):
                    raise HostResourceGuardError("SYMLINK_REFUSED", f"{label} symlink directory refused: {cand}")
                if not stat.S_ISDIR(st.st_mode):
                    raise HostResourceGuardError("NOT_DIRECTORY", f"{label} non-directory parent: {cand}")
            except FileNotFoundError:
                os.mkdir(cand, 0o700)
            existing_prefix = cand

    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    anchor = os.sep if parent.is_absolute() else "."
    start_idx = 1 if parent.is_absolute() else 0
    open_fds: list[int] = []
    try:
        cur_fd = os.open(anchor, dir_flags)
        open_fds.append(cur_fd)
        for comp in parts[start_idx:]:
            if comp in ("", "."):
                continue
            next_fd = os.open(comp, dir_flags, dir_fd=cur_fd)
            open_fds.append(next_fd)
            os.close(cur_fd)
            open_fds.pop(0)
            cur_fd = next_fd
        return cur_fd
    except FileNotFoundError:
        for fd in open_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise HostResourceGuardError("DIR_NOT_FOUND", f"{label} directory missing: {parent}") from None
    except OSError as exc:
        for fd in open_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        if exc.errno == _errno.ELOOP:
            raise HostResourceGuardError("SYMLINK_REFUSED", f"{label} symlink directory refused: {parent}") from None
        raise HostResourceGuardError("IO_ERROR", f"{label} directory open failed: {parent}") from None


def _open_regular_nofollow_text(path: Path, *, max_bytes: int = 1048576, label: str) -> str:
    parts = path.parts
    _require(bool(parts), "INVALID_PATH", f"{label} path is empty")
    leaf = parts[-1]
    _require(leaf not in ("", ".", ".."), "INVALID_PATH", f"{label} invalid leaf: {path}")

    parent_fd = _open_parent_dir_nofollow(path.parent, label, create_dirs=False)
    leaf_fd = -1
    try:
        try:
            leaf_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except FileNotFoundError:
            raise HostResourceGuardError("FILE_NOT_FOUND", f"{label} missing: {path}") from None
        except OSError as exc:
            if exc.errno == _errno.ELOOP:
                raise HostResourceGuardError("SYMLINK_REFUSED", f"{label} symlink refused: {path}") from None
            raise HostResourceGuardError("IO_ERROR", f"{label} open failed: {path}") from None
        st = os.fstat(leaf_fd)
        _require(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE", f"{label} must be regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(leaf_fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            _require(total <= max_bytes, "BYTE_CAP_EXCEEDED", f"{label} exceeds {max_bytes} B: {path}")
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="strict")
    finally:
        if leaf_fd >= 0:
            try:
                os.close(leaf_fd)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass


def read_meminfo(meminfo_path: Path | None = None) -> dict[str, int]:
    """Read Linux `/proc/meminfo` fields in KiB fail-closed."""
    target = Path("/proc/meminfo") if meminfo_path is None else meminfo_path
    text = _open_regular_nofollow_text(target, label="meminfo")
    fields: dict[str, int] = {}
    for raw_line in text.splitlines():
        m = _MEMINFO_LINE_RE.match(raw_line.strip())
        if m is not None:
            fields[m.group(1)] = int(m.group(2))
    mem_total = fields.get("MemTotal")
    mem_avail = fields.get("MemAvailable")
    _require(type(mem_total) is int and mem_total > 0, "MEMINFO_INVALID", f"{target} missing positive MemTotal")
    _require(type(mem_avail) is int and mem_avail >= 0, "MEMINFO_INVALID", f"{target} missing non-negative MemAvailable")
    return {
        "mem_total_kib": mem_total,
        "mem_available_kib": mem_avail,
        "swap_total_kib": fields.get("SwapTotal", 0),
        "swap_free_kib": fields.get("SwapFree", 0),
    }


def read_memory_psi_avg10(psi_path: Path | None = None) -> float | None:
    """Return `/proc/pressure/memory` `some avg10` percentage if available."""
    target = Path("/proc/pressure/memory") if psi_path is None else psi_path
    try:
        text = _open_regular_nofollow_text(target, label="psi_memory")
    except HostResourceGuardError as exc:
        if exc.code in ("FILE_NOT_FOUND", "DIR_NOT_FOUND"):
            return None
        raise
    for line in text.splitlines():
        m = _PSI_AVG10_RE.match(line.strip())
        if m is not None:
            return float(m.group(1))
    return None


def _read_ancestor_pids() -> set[int]:
    ancestors: set[int] = set()
    cur = os.getpid()
    for _ in range(32):
        if cur <= 1 or cur in ancestors:
            break
        ancestors.add(cur)
        stat_path = Path(f"/proc/{cur}/stat")
        try:
            raw = stat_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            break
        rparen = raw.rfind(")")
        if rparen < 0:
            break
        rest = raw[rparen + 2 :].split()
        if len(rest) < 2:
            break
        try:
            cur = int(rest[1])
        except ValueError:
            break
    return ancestors


def _default_proc_reader() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return out
    try:
        with os.scandir(proc_root) as it:
            for entry in it:
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                try:
                    raw = (Path(entry.path) / "cmdline").read_bytes()
                except OSError:
                    continue
                if not raw:
                    continue
                cmd = " ".join(part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part)
                if cmd:
                    out.append((pid, cmd))
    except OSError:
        return out
    return out


def find_competing_heavy_loaders(
    *,
    proc_reader: Callable[[], Sequence[tuple[int, str]]] | None = None,
    ignore_pids: set[int] | None = None,
    heavy_patterns: Sequence[str] = _HEAVY_CMD_PATTERNS,
) -> list[dict[str, Any]]:
    """Detect concurrent Python processes loading multi-GB checkpoints or embeddings."""
    reader = _default_proc_reader if proc_reader is None else proc_reader
    skip_pids = _read_ancestor_pids() if ignore_pids is None else set(ignore_pids)
    competing: list[dict[str, Any]] = []
    for pid, cmd in reader():
        if pid in skip_pids:
            continue
        tokens = cmd.split()
        if not tokens:
            continue
        exe_base = os.path.basename(tokens[0]).lower()
        if not exe_base.startswith("python"):
            continue
        if any(ex in cmd for ex in _LIGHTWEIGHT_EXCLUSIONS):
            continue
        matched = [pat for pat in heavy_patterns if pat in cmd]
        if matched:
            competing.append({"pid": pid, "patterns": matched, "cmd_head": cmd[:180]})
    return competing


def _default_gpu_probe() -> tuple[float, list[str]]:
    mem_proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    _require(mem_proc.returncode == 0, "NVIDIA_SMI_FAILED", f"nvidia-smi memory query failed: {mem_proc.stderr.strip()}")
    vals = [float(ln.strip()) for ln in mem_proc.stdout.splitlines() if ln.strip()]
    _require(bool(vals), "NVIDIA_SMI_FAILED", "nvidia-smi returned no GPU memory rows")
    used_mib = max(vals)

    app_proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    _require(app_proc.returncode == 0, "NVIDIA_SMI_FAILED", f"nvidia-smi apps query failed: {app_proc.stderr.strip()}")
    apps = [ln.strip() for ln in app_proc.stdout.splitlines() if ln.strip()]
    return used_mib, apps


def _acquire_exclusive_lock(lock_path: Path, *, wait_seconds: int = 0) -> tuple[int, str]:
    """Open `lock_path` fail-closed (`O_NOFOLLOW`, `0o600`, `st_nlink==1`) and acquire `LOCK_EX`."""
    parts = lock_path.parts
    _require(bool(parts), "INVALID_PATH", "lock_path is empty")
    leaf = parts[-1]
    _require(leaf not in ("", ".", ".."), "INVALID_PATH", f"invalid lock_path leaf: {lock_path}")

    parent_fd = _open_parent_dir_nofollow(lock_path.parent, "lock_dir", create_dirs=True)
    lock_fd = -1
    try:
        try:
            lock_fd = os.open(leaf, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == _errno.ELOOP:
                raise HostResourceGuardError("SYMLINK_REFUSED", f"lockfile symlink refused: {lock_path}") from None
            raise HostResourceGuardError("IO_ERROR", f"lockfile open failed ({exc.strerror or exc}): {lock_path}") from None

        st = os.fstat(lock_fd)
        _require(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE", f"lockfile must be regular file: {lock_path}")
        _require(st.st_nlink == 1, "HARDLINK_REFUSED", f"hardlink lockfile refused (st_nlink={st.st_nlink}): {lock_path}")
        os.fchmod(lock_fd, 0o600)

        deadline = time.monotonic() + float(wait_seconds)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    holder = ""
                    try:
                        os.lseek(lock_fd, 0, os.SEEK_SET)
                        holder = os.read(lock_fd, 512).decode("utf-8", errors="replace").strip()
                    except OSError:
                        holder = ""
                    msg = f"heavy_model_loader lock held: {lock_path}"
                    if holder:
                        msg += f" ({holder})"
                    raise HostResourceGuardError("LOADER_LOCK_HELD", msg) from None
                time.sleep(0.5)

        meta = json.dumps({"pid": os.getpid(), "lock_path": str(lock_path)}, sort_keys=True) + "\n"
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, meta.encode("utf-8"))
        os.fsync(lock_fd)
        ret_fd = lock_fd
        lock_fd = -1
        return ret_fd, meta.strip()
    finally:
        if lock_fd >= 0:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass


def _release_lock(lock_fd: int) -> None:
    try:
        try:
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.ftruncate(lock_fd, 0)
        except OSError:
            pass
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def evaluate_host_resources(
    *,
    min_mem_gib: float = DEFAULT_MIN_MEM_GIB,
    require_gpu_idle: bool = False,
    max_gpu_mib: float = DEFAULT_MAX_GPU_MIB,
    max_psi_avg10: float = DEFAULT_MAX_PSI_AVG10,
    lock_path: Path | None = None,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    meminfo_path: Path | None = None,
    psi_path: Path | None = None,
    skip_proc_scan: bool = False,
    proc_reader: Callable[[], Sequence[tuple[int, str]]] | None = None,
    ignore_pids: set[int] | None = None,
    gpu_probe: Callable[[], tuple[float, list[str]]] | None = None,
    hold_lock: bool = False,
) -> tuple[dict[str, Any], int | None]:
    """Validate lock, host CPU RAM, memory PSI, competing processes, and optional GPU idle."""
    min_mem = _require_pos_number(min_mem_gib, "min_mem_gib")
    max_gpu = _require_pos_number(max_gpu_mib, "max_gpu_mib")
    max_psi = _require_pos_number(max_psi_avg10, "max_psi_avg10")
    wait_s = _require_nonneg_int(wait_seconds, "wait_seconds")

    eff_lock = default_lock_path() if lock_path is None else lock_path
    lock_fd, holder_meta = _acquire_exclusive_lock(eff_lock, wait_seconds=wait_s)
    try:
        mem = read_meminfo(meminfo_path)
        req_kib = int(min_mem * float(_KIB_PER_GIB))
        avail_kib = mem["mem_available_kib"]
        _require(
            avail_kib >= req_kib,
            "INSUFFICIENT_CPU_RAM",
            f"MemAvailable {avail_kib / _KIB_PER_GIB:.2f} GiB < required {min_mem:.2f} GiB",
            exit_code=HardwareExitCode.OOM,
        )

        psi_avg10 = read_memory_psi_avg10(psi_path)
        if psi_avg10 is not None:
            _require(
                psi_avg10 < max_psi,
                "HIGH_MEMORY_PRESSURE",
                f"/proc/pressure/memory some avg10={psi_avg10:.2f}% >= {max_psi:.2f}%",
                exit_code=HardwareExitCode.MEM_PRESSURE,
            )

        competing: list[dict[str, Any]] = []
        if not skip_proc_scan:
            competing = find_competing_heavy_loaders(proc_reader=proc_reader, ignore_pids=ignore_pids)
            if competing:
                raise HostResourceGuardError(
                    "COMPETING_HEAVY_LOADER",
                    f"competing heavy model/embedding process active: {competing[0]}",
                )

        gpu_used_mib: float | None = None
        gpu_apps: list[str] = []
        if require_gpu_idle:
            probe_fn = _default_gpu_probe if gpu_probe is None else gpu_probe
            gpu_used_mib, gpu_apps = probe_fn()
            _require(
                gpu_used_mib < max_gpu,
                "GPU_MEMORY_BUSY",
                f"GPU memory.used={gpu_used_mib:.1f} MiB >= {max_gpu:.1f} MiB",
            )
            if gpu_apps:
                raise HostResourceGuardError(
                    "GPU_COMPUTE_BUSY",
                    f"active GPU compute apps present: {gpu_apps}",
                )

        report = {
            "schema": SCHEMA_CHECK,
            "passed": True,
            "lock_path": str(eff_lock),
            "lock_acquired": True,
            "holder": holder_meta,
            "min_mem_gib": min_mem,
            "mem_total_kib": mem["mem_total_kib"],
            "mem_available_kib": avail_kib,
            "swap_free_kib": mem["swap_free_kib"],
            "psi_memory_some_avg10": psi_avg10,
            "competing_heavy_loaders": len(competing),
            "require_gpu_idle": bool(require_gpu_idle),
            "gpu_used_mib": gpu_used_mib,
            "gpu_apps_count": len(gpu_apps),
        }
        if hold_lock:
            ret_fd = lock_fd
            lock_fd = -1
            return report, ret_fd
        return report, None
    finally:
        if lock_fd >= 0:
            _release_lock(lock_fd)


def _build_exec_child_env(
    base_env: Mapping[str, str],
    *,
    lock_path: str,
    require_gpu_idle: bool,
) -> dict[str, str]:
    """Build guarded subprocess env, defaulting CUDA expandable segments when GPU-guarded."""
    child_env = dict(base_env)
    child_env[GUARD_ENV_ACTIVE] = "1"
    child_env[GUARD_ENV_LOCK] = lock_path
    if require_gpu_idle:
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return child_env


def _expect_error(code: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except HostResourceGuardError as exc:
        if exc.code != code:
            raise HostResourceGuardError(
                "SELF_TEST_FAILED",
                f"expected {code!r}, got {exc.code!r}: {exc.message}",
            ) from None
        return
    raise HostResourceGuardError("SELF_TEST_FAILED", f"expected failure {code!r} succeeded unexpectedly")


def run_self_test() -> dict[str, Any]:
    """Verify RAM/GPU/PSI/process/lock gates, negative controls, `--timeout-seconds`, and 3x `-I` parity."""
    with tempfile.TemporaryDirectory(prefix="silisense_guard_") as tmp_str:
        tmp = Path(tmp_str)
        lock_ok = tmp / "locks" / "heavy.lock"
        mem_ok = tmp / "meminfo_ok"
        mem_ok.write_text(
            "MemTotal:       32535272 kB\n"
            "MemAvailable:   22320192 kB\n"
            "SwapTotal:      65069792 kB\n"
            "SwapFree:       65069792 kB\n",
            encoding="utf-8",
        )
        psi_ok = tmp / "psi_ok"
        psi_ok.write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n", encoding="utf-8")

        rep, held_fd = evaluate_host_resources(
            min_mem_gib=12.0,
            require_gpu_idle=True,
            lock_path=lock_ok,
            meminfo_path=mem_ok,
            psi_path=psi_ok,
            proc_reader=lambda: [(999991, "python3 -I -B silisense/resource_guard.py --self-test")],
            ignore_pids=set(),
            gpu_probe=lambda: (2.0, []),
            hold_lock=False,
        )
        _require(rep["passed"] is True and held_fd is None, "SELF_TEST_FAILED", "clean check failed")

        mem_low = tmp / "meminfo_low"
        mem_low.write_text("MemTotal: 32535272 kB\nMemAvailable: 8388608 kB\n", encoding="utf-8")
        _expect_error(
            "INSUFFICIENT_CPU_RAM",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=lock_ok,
                meminfo_path=mem_low,
                psi_path=psi_ok,
                skip_proc_scan=True,
            ),
        )

        psi_high = tmp / "psi_high"
        psi_high.write_text("some avg10=50.56 avg60=12.00 avg300=2.00 total=100\n", encoding="utf-8")
        _expect_error(
            "HIGH_MEMORY_PRESSURE",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=lock_ok,
                meminfo_path=mem_ok,
                psi_path=psi_high,
                skip_proc_scan=True,
            ),
        )

        for heavy_cmd in (
            "python3 model_loader.py --from_pretrained",
            "python3 worker.py --run-live-gpu",
        ):
            _expect_error(
                "COMPETING_HEAVY_LOADER",
                lambda cmd=heavy_cmd: evaluate_host_resources(
                    min_mem_gib=12.0,
                    lock_path=lock_ok,
                    meminfo_path=mem_ok,
                    psi_path=psi_ok,
                    proc_reader=lambda c=cmd: [(999992, c)],
                    ignore_pids=set(),
                ),
            )

        _expect_error(
            "GPU_MEMORY_BUSY",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                require_gpu_idle=True,
                lock_path=lock_ok,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
                gpu_probe=lambda: (7633.0, []),
            ),
        )
        _expect_error(
            "GPU_COMPUTE_BUSY",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                require_gpu_idle=True,
                lock_path=lock_ok,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
                gpu_probe=lambda: (10.0, ["12345, python, 4280 MiB"]),
            ),
        )

        sym_lock = tmp / "sym.lock"
        sym_lock.symlink_to(lock_ok)
        _expect_error(
            "SYMLINK_REFUSED",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=sym_lock,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
            ),
        )

        hard_src = tmp / "locks" / "hard_src.lock"
        hard_src.write_text("x\n", encoding="utf-8")
        hard_lock = tmp / "locks" / "hard.lock"
        os.link(hard_src, hard_lock)
        _expect_error(
            "HARDLINK_REFUSED",
            lambda: evaluate_host_resources(
                min_mem_gib=12.0,
                lock_path=hard_lock,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
            ),
        )
        hard_lock.unlink(missing_ok=True)
        hard_src.unlink(missing_ok=True)

        _expect_error(
            "INVALID_ARGUMENT",
            lambda: evaluate_host_resources(
                min_mem_gib=True,  # type: ignore[arg-type]
                lock_path=lock_ok,
                meminfo_path=mem_ok,
                psi_path=psi_ok,
                skip_proc_scan=True,
            ),
        )

        _, lock_fd = evaluate_host_resources(
            min_mem_gib=12.0,
            lock_path=lock_ok,
            meminfo_path=mem_ok,
            psi_path=psi_ok,
            skip_proc_scan=True,
            hold_lock=True,
        )
        _require(lock_fd is not None, "SELF_TEST_FAILED", "expected held lock fd")
        script_path = Path(__file__).resolve()
        try:
            c_proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(script_path),
                    "--check",
                    "--lock-file",
                    str(lock_ok),
                    "--meminfo-file",
                    str(mem_ok),
                    "--psi-file",
                    str(psi_ok),
                    "--skip-proc-scan",
                    "--json",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=15,
                check=False,
            )
            _require(c_proc.returncode == 2, "SELF_TEST_FAILED", f"contended lock expected rc=2, got {c_proc.returncode}")
            c_obj = json.loads(c_proc.stdout.decode("utf-8"))
            _require(c_obj.get("error") == "LOADER_LOCK_HELD", "SELF_TEST_FAILED", f"expected LOADER_LOCK_HELD, got {c_obj}")
        finally:
            if lock_fd is not None:
                _release_lock(lock_fd)

        r4_lock = tmp / "locks" / "r4_fixed.lock"
        r4_modes = (
            ("B", [sys.executable, "-I", "-B", str(script_path), "--check", "--lock-file", str(r4_lock), "--meminfo-file", str(mem_ok), "--psi-file", str(psi_ok), "--skip-proc-scan", "--json"]),
            ("O_B", [sys.executable, "-I", "-O", "-B", str(script_path), "--check", "--lock-file", str(r4_lock), "--meminfo-file", str(mem_ok), "--psi-file", str(psi_ok), "--skip-proc-scan", "--json"]),
            ("B_Werror", [sys.executable, "-I", "-B", "-W", "error", str(script_path), "--check", "--lock-file", str(r4_lock), "--meminfo-file", str(mem_ok), "--psi-file", str(psi_ok), "--skip-proc-scan", "--json"]),
        )
        r4_hashes: set[str] = set()
        for label, cmd in r4_modes:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=False)
            _require(
                proc.returncode == 0,
                "R4_MODE_FAILED",
                f"R4 mode {label} exited {proc.returncode}: {proc.stderr.decode('utf-8', errors='replace')}",
            )
            parsed = json.loads(proc.stdout.decode("utf-8"))
            parsed.pop("holder", None)
            parsed.pop("lock_path", None)
            canon = json.dumps(parsed, sort_keys=True).encode("utf-8")
            r4_hashes.add(hashlib.sha256(canon).hexdigest())
        _require(len(r4_hashes) == 1, "R4_PARITY_MISMATCH", f"R4 canonical check differed: {sorted(r4_hashes)}")

        exec_lock = tmp / "locks" / "exec_guard.lock"
        to_fail = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(script_path),
                "--lock-file",
                str(exec_lock),
                "--meminfo-file",
                str(mem_ok),
                "--psi-file",
                str(psi_ok),
                "--skip-proc-scan",
                "--timeout-seconds",
                "0.2",
                "--exec",
                "--",
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import time; time.sleep(10)",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            check=False,
        )
        _require(
            to_fail.returncode == 2 and b"EXEC_TIMEOUT" in to_fail.stderr,
            "SELF_TEST_FAILED",
            f"--timeout-seconds expected EXEC_TIMEOUT rc=2, got rc={to_fail.returncode}",
        )
        rep_after_to, held_after_to = evaluate_host_resources(
            min_mem_gib=12.0,
            lock_path=exec_lock,
            meminfo_path=mem_ok,
            psi_path=psi_ok,
            skip_proc_scan=True,
            hold_lock=False,
        )
        _require(rep_after_to["passed"] is True and held_after_to is None, "SELF_TEST_FAILED", "lock not released after timeout")

    return {
        "schema": SCHEMA_SELF_TEST,
        "passed": True,
        "negative_controls_passed": 10,
        "exec_env_exported": True,
        "r4_parity": True,
        "r4_canonical_sha256": next(iter(r4_hashes)),
    }


def _run_guarded_subprocess(cmd: list[str], *, env: dict[str, str], timeout_seconds: float) -> int:
    """Run guarded child command, killing its process group on `timeout_seconds` expiry."""
    if timeout_seconds <= 0.0:
        return int(subprocess.run(cmd, env=env, check=False).returncode)
    proc = subprocess.Popen(cmd, env=env, start_new_session=(os.name == "posix"))
    try:
        return int(proc.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
        else:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise HostResourceGuardError(
            "EXEC_TIMEOUT",
            f"guarded command exceeded timeout_seconds={timeout_seconds:g}",
        ) from None


class _GuardArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise HostResourceGuardError("INVALID_CLI_ARGS", message)


def _build_parser() -> _GuardArgumentParser:
    parser = _GuardArgumentParser(
        description="Host CPU-RAM + GPU single-loader serialization guard.",
        allow_abbrev=False,
    )
    parser.add_argument("--check", action="store_true", help="Acquire/release lock and verify CPU-RAM (+ optional GPU) headroom.")
    parser.add_argument(
        "--exec",
        action="store_true",
        help="Run command after `--` under exclusive loader flock (exports SILISENSE_RESOURCE_GUARD_ACTIVE=1).",
    )
    parser.add_argument("--min-mem-gib", type=float, default=DEFAULT_MIN_MEM_GIB, help="Minimum /proc/meminfo MemAvailable in GiB (default: 12.0).")
    parser.add_argument("--require-gpu-idle", action="store_true", help="Verify nvidia-smi memory.used < --max-gpu-mib and 0 compute apps.")
    parser.add_argument("--max-gpu-mib", type=float, default=DEFAULT_MAX_GPU_MIB, help="Maximum allowed GPU memory.used MiB when --require-gpu-idle.")
    parser.add_argument("--max-psi-avg10", type=float, default=DEFAULT_MAX_PSI_AVG10, help="Maximum /proc/pressure/memory some avg10 %% (default: 25.0).")
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS, help="Seconds to wait for lockfile (default: 0 fail-closed).")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock timeout in seconds for --exec (kills process group on expiry; 0 = unbounded).",
    )
    parser.add_argument("--lock-file", type=Path, help="Override lockfile path.")
    parser.add_argument("--meminfo-file", type=Path, help="Override /proc/meminfo path.")
    parser.add_argument("--psi-file", type=Path, help="Override /proc/pressure/memory path.")
    parser.add_argument("--skip-proc-scan", action="store_true", help="Skip /proc competing heavy-loader scan.")
    parser.add_argument("--json", action="store_true", help="Emit canonical sorted JSON.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic self-test suite.")
    return parser


def _split_exec_argv(raw_argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split guard flags and child command around `--exec ... -- <cmd>` (or trailing `--exec <cmd>`)."""
    if "--exec" not in raw_argv:
        return raw_argv, None
    exec_idx = raw_argv.index("--exec")
    after = raw_argv[exec_idx + 1 :]
    if after and after[0].startswith("-") and "--" in after:
        sep_idx = raw_argv.index("--", exec_idx + 1)
        guard_argv = raw_argv[:exec_idx] + ["--exec"] + raw_argv[exec_idx + 1 : sep_idx]
        exec_cmd = raw_argv[sep_idx + 1 :]
        return guard_argv, exec_cmd
    guard_argv = raw_argv[:exec_idx] + ["--exec"]
    exec_cmd = after
    return guard_argv, exec_cmd


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        try:
            rep = run_self_test()
        except HostResourceGuardError as exc:
            sys.stderr.write(f"ERROR[{exc.code}]: {exc.message}\n")
            return 2
        sys.stdout.write(json.dumps(rep, sort_keys=True) + "\n")
        return 0

    exec_cmd: list[str] | None = None
    args: Any = None
    try:
        guard_argv, exec_cmd = _split_exec_argv(raw_argv)
        parser = _build_parser()
        args = parser.parse_args(guard_argv)
        if exec_cmd is not None and (args.check or args.self_test):
            raise HostResourceGuardError("INVALID_CLI_ARGS", "--exec cannot be combined with --check or --self-test")
    except HostResourceGuardError as exc:
        if "--json" in raw_argv and exec_cmd is None:
            sys.stdout.write(json.dumps({"passed": False, "error": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"ERROR[{exc.code}]: {exc.message}\n")
        return 2

    try:
        if args.self_test:
            rep = run_self_test()
            sys.stdout.write(json.dumps(rep, sort_keys=True) + "\n")
            return 0

        if exec_cmd is not None:
            cmd = list(exec_cmd)
            _require(bool(cmd), "MISSING_EXEC_COMMAND", "--exec requires a command after `--`")
            timeout_s = _require_pos_number(args.timeout_seconds, "timeout_seconds", allow_zero=True)
            rep, held_fd = evaluate_host_resources(
                min_mem_gib=args.min_mem_gib,
                require_gpu_idle=args.require_gpu_idle,
                max_gpu_mib=args.max_gpu_mib,
                max_psi_avg10=args.max_psi_avg10,
                lock_path=args.lock_file,
                wait_seconds=args.wait_seconds,
                meminfo_path=args.meminfo_file,
                psi_path=args.psi_file,
                skip_proc_scan=args.skip_proc_scan,
                hold_lock=True,
            )
            child_env = _build_exec_child_env(
                os.environ,
                lock_path=str(rep["lock_path"]),
                require_gpu_idle=bool(args.require_gpu_idle),
            )
            try:
                rc = _run_guarded_subprocess(cmd, env=child_env, timeout_seconds=timeout_s)
                if args.json:
                    payload = {
                        "schema": SCHEMA_EXEC,
                        "guard": rep,
                        "returncode": rc,
                    }
                    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
                return rc
            finally:
                if held_fd is not None:
                    _release_lock(held_fd)

        rep, _ = evaluate_host_resources(
            min_mem_gib=args.min_mem_gib,
            require_gpu_idle=args.require_gpu_idle,
            max_gpu_mib=args.max_gpu_mib,
            max_psi_avg10=args.max_psi_avg10,
            lock_path=args.lock_file,
            wait_seconds=args.wait_seconds,
            meminfo_path=args.meminfo_file,
            psi_path=args.psi_file,
            skip_proc_scan=args.skip_proc_scan,
            hold_lock=False,
        )
        if args.json:
            sys.stdout.write(json.dumps(rep, sort_keys=True) + "\n")
        else:
            avail_gib = rep["mem_available_kib"] / float(_KIB_PER_GIB)
            sys.stdout.write(
                f"HOST_GUARD_PASS mem_available_gib={avail_gib:.2f} "
                f"competing={rep['competing_heavy_loaders']} gpu_used_mib={rep['gpu_used_mib']}\n"
            )
        return 0
    except HostResourceGuardError as exc:
        if getattr(args, "json", False):
            sys.stdout.write(json.dumps({"passed": False, "error": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"ERROR[{exc.code}]: {exc.message}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
