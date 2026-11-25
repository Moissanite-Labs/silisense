"""Cache-line-aligned 64 B `<8Q>` telemetry frame & `zone-canon-v0` hashing.

Eliminates `float32` timestamp precision loss (`128 s` ULP jumps at Unix epoch
and `1 ms` drift after `2.27 h`) by encoding monotonic nanoseconds and all
cumulative energy counters as exact unsigned 64-bit integers (`<8Q>`, 64 bytes
= 1 x86_64 L1 cache line).

Also implements:
  - `canonicalize_zone_canon_v0`: Deterministic float-free canonical JSON
    encoder rejecting floats, control byte `0x7F` (DEL), and non-ASCII keys
    so Python, Rust, C11, and Lean 4 compute identical SHA-256 digests.
  - `domain_tree_leaf_sha256`: Injective length-prefixed domain-separated leaf
    digest `SHA-256(b"silisense-tree-v1\\x00" || u64_le(len(path)) || path || raw_sha256)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Any, Mapping

from silisense.errors import require_ok

_FRAME64_STRUCT = struct.Struct("<8Q")
FRAME64_BYTES = 64
U64_MAX = (1 << 64) - 1

ZONE_CANON_VERSION = "zone-canon-v0"
TREE_DOMAIN_PREFIX = b"silisense-tree-v1\x00"

require_ok(_FRAME64_STRUCT.size == FRAME64_BYTES, "ERR_FRAME64_LAYOUT", "Struct <8Q> size must be 64 bytes")


def _require_u64(val: Any, name: str) -> int:
    require_ok(
        isinstance(val, int) and not isinstance(val, bool) and 0 <= val <= U64_MAX,
        "ERR_U64_BOUNDS",
        f"{name} must be an exact uint64 in [0, 2^64 - 1], got {val!r}",
    )
    return int(val)


def pack_flags_temp_mhz(
    *,
    sw_power_cap: bool,
    hw_thermal_slowdown: bool,
    gpu_temp_deci_c: int,
    sm_clock_mhz: int,
) -> int:
    """Pack throttle flags, 0.1 C GPU temperature, and SM clock MHz into one uint64 slot."""
    require_ok(0 <= gpu_temp_deci_c <= 0xFFFF, "ERR_TEMP_BOUNDS", f"gpu_temp_deci_c out of uint16 range: {gpu_temp_deci_c}")
    require_ok(0 <= sm_clock_mhz <= 0xFFFF, "ERR_CLOCK_BOUNDS", f"sm_clock_mhz out of uint16 range: {sm_clock_mhz}")
    flags = (1 if sw_power_cap else 0) | ((1 if hw_thermal_slowdown else 0) << 1)
    return flags | (int(gpu_temp_deci_c) << 16) | (int(sm_clock_mhz) << 32)


def unpack_flags_temp_mhz(word: int) -> dict[str, int | bool]:
    w = _require_u64(word, "flags_temp_mhz")
    return {
        "sw_power_cap": bool(w & 0x1),
        "hw_thermal_slowdown": bool((w >> 1) & 0x1),
        "gpu_temp_deci_c": int((w >> 16) & 0xFFFF),
        "sm_clock_mhz": int((w >> 32) & 0xFFFF),
    }


@dataclass(frozen=True)
class TelemetryFrame64:
    """Single 64-byte L1-cache-line pure-integer telemetry frame (`<8Q>`)."""

    ts_mono_ns: int
    pkg_uj: int
    core_uj: int
    uncore_uj: int
    psys_uj: int
    gpu_mj: int
    step_us: int
    flags_temp_mhz: int

    def validate(self) -> None:
        _require_u64(self.ts_mono_ns, "ts_mono_ns")
        _require_u64(self.pkg_uj, "pkg_uj")
        _require_u64(self.core_uj, "core_uj")
        _require_u64(self.uncore_uj, "uncore_uj")
        _require_u64(self.psys_uj, "psys_uj")
        _require_u64(self.gpu_mj, "gpu_mj")
        _require_u64(self.step_us, "step_us")
        _require_u64(self.flags_temp_mhz, "flags_temp_mhz")

    def to_bytes(self) -> bytes:
        self.validate()
        return _FRAME64_STRUCT.pack(
            self.ts_mono_ns,
            self.pkg_uj,
            self.core_uj,
            self.uncore_uj,
            self.psys_uj,
            self.gpu_mj,
            self.step_us,
            self.flags_temp_mhz,
        )

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> "TelemetryFrame64":
        raw = bytes(buf)
        require_ok(
            len(raw) == FRAME64_BYTES,
            "ERR_FRAME64_LEN",
            f"Expected exactly {FRAME64_BYTES} bytes, got {len(raw)}",
        )
        fields = _FRAME64_STRUCT.unpack(raw)
        obj = cls(*fields)
        obj.validate()
        return obj

    def to_canon_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "canon": ZONE_CANON_VERSION,
            "core_uj": self.core_uj,
            "flags_temp_mhz": self.flags_temp_mhz,
            "gpu_mj": self.gpu_mj,
            "pkg_uj": self.pkg_uj,
            "psys_uj": self.psys_uj,
            "step_us": self.step_us,
            "ts_mono_ns": self.ts_mono_ns,
            "uncore_uj": self.uncore_uj,
        }

    def frame_sha256(self) -> str:
        return hashlib.sha256(canonicalize_zone_canon_v0(self.to_canon_dict())).hexdigest()


def _validate_zone_canon_node(node: Any) -> None:
    if node is None or isinstance(node, bool):
        return
    if isinstance(node, int):
        return
    if isinstance(node, float):
        require_ok(False, "ERR_ZONE_CANON_FLOAT_FORBIDDEN", f"Floats are forbidden in {ZONE_CANON_VERSION}: {node!r}")
    if isinstance(node, str):
        for ch in node:
            code = ord(ch)
            require_ok(
                0x20 <= code <= 0x7E,
                "ERR_ZONE_CANON_CHAR",
                f"Control/DEL/non-ASCII byte 0x{code:02x} forbidden in {ZONE_CANON_VERSION}",
            )
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _validate_zone_canon_node(item)
        return
    if isinstance(node, Mapping):
        for k, v in node.items():
            require_ok(isinstance(k, str), "ERR_ZONE_CANON_KEY", "Mapping keys must be ASCII strings")
            _validate_zone_canon_node(k)
            _validate_zone_canon_node(v)
        return
    require_ok(False, "ERR_ZONE_CANON_TYPE", f"Unsupported type {type(node).__name__} in {ZONE_CANON_VERSION}")


def canonicalize_zone_canon_v0(payload: Mapping[str, Any]) -> bytes:
    """Serialize a float-free mapping to deterministic ASCII `zone-canon-v0` bytes."""
    require_ok(isinstance(payload, Mapping), "ERR_ZONE_CANON_ROOT", "Root payload must be a mapping")
    _validate_zone_canon_node(payload)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return text.encode("ascii")


def domain_tree_leaf_sha256(relpath: str, raw_content_sha256_hex: str) -> str:
    """Compute collision-free u64_le length-prefixed domain-separated Merkle leaf digest."""
    require_ok(len(relpath) > 0 and "\x00" not in relpath and "\x7f" not in relpath, "ERR_BAD_RELPATH", f"Invalid relpath {relpath!r}")
    require_ok(
        len(raw_content_sha256_hex) == 64 and all(c in "0123456789abcdef" for c in raw_content_sha256_hex),
        "ERR_BAD_SHA256_HEX",
        f"Expected lowercase 64-hex sha256, got {raw_content_sha256_hex!r}",
    )
    path_bytes = relpath.encode("utf-8")
    raw_digest = bytes.fromhex(raw_content_sha256_hex)
    buf = TREE_DOMAIN_PREFIX + struct.pack("<Q", len(path_bytes)) + path_bytes + raw_digest
    return hashlib.sha256(buf).hexdigest()
