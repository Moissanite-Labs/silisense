"""One-way SiliSense -> HoloSphere step-boundary telemetry & tuning bridge.

Wire- and `profile_sha256()`-identical to `holosphere/telemetry.py` with ZERO
imports of `holosphere` or `torch`, preserving the strict one-way boundary:
`Moissanite-Labs/holosphere` never imports `silisense`, and `silisense` never
mutates model weights or `WeftAuditShell` security thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from silisense.errors import require_ok
from silisense.governor import PrecursorThrottleGovernor

TELEMETRY_SCHEMA = "holosphere.step_telemetry.v1"
TUNING_PROFILE_SCHEMA = "holosphere.tuning_profile.v1"

VALID_EXECUTION_BACKENDS = frozenset({"cuda_sm86", "triton", "torch_cpu"})
VALID_TUNING_BACKENDS = frozenset({"auto", "cuda_sm86", "triton", "torch_cpu"})
VALID_BLOCK_TS = frozenset({32, 64, 128, 256})
MAX_COOLDOWN_MS = 60000.0


def _req_int_no_bool(val: Any, name: str, code: str) -> int:
    require_ok(isinstance(val, int) and not isinstance(val, bool), code, f"{name} must be int (not bool)")
    return int(val)


def _req_finite(val: Any, name: str, code: str) -> float:
    require_ok(isinstance(val, (int, float)) and not isinstance(val, bool), code, f"{name} must be numeric")
    fval = float(val)
    require_ok(math.isfinite(fval), code, f"{name} must be finite, got {val!r}")
    return fval


@dataclass(frozen=True)
class HoloSphereStepTelemetry:
    """Wire-identical mirror of `holosphere.telemetry.HoloSphereStepTelemetry`."""

    step_idx: int
    seq_len: int
    batch_size: int
    r_loops: int
    backend: str
    block_t: int
    step_ms: float
    tok_per_sec: float
    norm_drift_max: float
    schema: str = TELEMETRY_SCHEMA

    def validate(self) -> None:
        require_ok(self.schema == TELEMETRY_SCHEMA, "TEL_SCHEMA", f"Unexpected schema {self.schema!r}")
        step_idx = _req_int_no_bool(self.step_idx, "step_idx", "TEL_STEP")
        seq_len = _req_int_no_bool(self.seq_len, "seq_len", "TEL_SEQ")
        batch_size = _req_int_no_bool(self.batch_size, "batch_size", "TEL_BATCH")
        r_loops = _req_int_no_bool(self.r_loops, "r_loops", "TEL_RLOOPS")
        block_t = _req_int_no_bool(self.block_t, "block_t", "TEL_BLOCKT")
        step_ms = _req_finite(self.step_ms, "step_ms", "TEL_STEP_MS")
        tps = _req_finite(self.tok_per_sec, "tok_per_sec", "TEL_TPS")
        drift = _req_finite(self.norm_drift_max, "norm_drift_max", "TEL_DRIFT")

        require_ok(step_idx >= 0, "TEL_STEP", "step_idx must be >= 0")
        require_ok(seq_len >= 1, "TEL_SEQ", "seq_len must be >= 1")
        require_ok(batch_size >= 1, "TEL_BATCH", "batch_size must be >= 1")
        require_ok(r_loops >= 1, "TEL_RLOOPS", "r_loops must be >= 1")
        require_ok(self.backend in VALID_EXECUTION_BACKENDS, "TEL_BACKEND", f"Unsupported backend {self.backend!r}")
        require_ok(block_t in VALID_BLOCK_TS, "TEL_BLOCKT", f"Unsupported block_t {block_t}")
        require_ok(step_ms >= 0.0, "TEL_STEP_MS", "step_ms must be >= 0")
        require_ok(tps >= 0.0, "TEL_TPS", "tok_per_sec must be >= 0")
        require_ok(drift >= 0.0, "TEL_DRIFT", "norm_drift_max must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HoloSphereStepTelemetry":
        require_ok(isinstance(data, Mapping), "TEL_PAYLOAD", "Telemetry payload must be a mapping")
        allowed = {
            "schema",
            "step_idx",
            "seq_len",
            "batch_size",
            "r_loops",
            "backend",
            "block_t",
            "step_ms",
            "tok_per_sec",
            "norm_drift_max",
        }
        extra = set(data.keys()) - allowed
        require_ok(not extra, "TEL_FIELDS", f"Unexpected telemetry fields: {sorted(extra)}")
        missing = (allowed - {"schema"}) - set(data.keys())
        require_ok(not missing, "TEL_FIELDS", f"Missing telemetry fields: {sorted(missing)}")
        obj = cls(
            step_idx=data["step_idx"],
            seq_len=data["seq_len"],
            batch_size=data["batch_size"],
            r_loops=data["r_loops"],
            backend=str(data["backend"]),
            block_t=data["block_t"],
            step_ms=data["step_ms"],
            tok_per_sec=data["tok_per_sec"],
            norm_drift_max=data["norm_drift_max"],
            schema=str(data.get("schema", TELEMETRY_SCHEMA)),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class HoloSphereTuningProfile:
    """Wire- and SHA256-identical mirror of `holosphere.telemetry.HoloSphereTuningProfile`."""

    backend: str = "auto"
    block_t: int = 128
    microbatch_size: int | None = None
    inter_step_cooldown_ms: float = 0.0
    r_loops_override: int | None = None
    schema: str = TUNING_PROFILE_SCHEMA

    def validate(self) -> None:
        require_ok(self.schema == TUNING_PROFILE_SCHEMA, "TUNE_SCHEMA", f"Unexpected schema {self.schema!r}")
        require_ok(self.backend in VALID_TUNING_BACKENDS, "TUNE_BACKEND", f"Unsupported backend {self.backend!r}")
        block_t = _req_int_no_bool(self.block_t, "block_t", "TUNE_BLOCKT")
        require_ok(block_t in VALID_BLOCK_TS, "TUNE_BLOCKT", f"Unsupported block_t {block_t}")
        if self.microbatch_size is not None:
            mbs = _req_int_no_bool(self.microbatch_size, "microbatch_size", "TUNE_MBS")
            require_ok(mbs >= 1, "TUNE_MBS", "microbatch_size must be >= 1")
        cooldown = _req_finite(self.inter_step_cooldown_ms, "inter_step_cooldown_ms", "TUNE_COOLDOWN")
        require_ok(0.0 <= cooldown <= MAX_COOLDOWN_MS, "TUNE_COOLDOWN", f"Cooldown out of range: {cooldown}")
        if self.r_loops_override is not None:
            rl = _req_int_no_bool(self.r_loops_override, "r_loops_override", "TUNE_RLOOPS")
            require_ok(rl >= 1, "TUNE_RLOOPS", "r_loops_override must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def profile_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HoloSphereTuningProfile":
        require_ok(isinstance(data, Mapping), "TUNE_PAYLOAD", "Tuning profile payload must be a mapping")
        allowed = {
            "schema",
            "backend",
            "block_t",
            "microbatch_size",
            "inter_step_cooldown_ms",
            "r_loops_override",
        }
        extra = set(data.keys()) - allowed
        require_ok(not extra, "TUNE_FIELDS", f"Unexpected tuning fields: {sorted(extra)}")
        obj = cls(
            backend=str(data.get("backend", "auto")),
            block_t=data.get("block_t", 128),
            microbatch_size=data.get("microbatch_size", None),
            inter_step_cooldown_ms=data.get("inter_step_cooldown_ms", 0.0),
            r_loops_override=data.get("r_loops_override", None),
            schema=str(data.get("schema", TUNING_PROFILE_SCHEMA)),
        )
        obj.validate()
        return obj


def build_step_telemetry(
    *,
    step_idx: int,
    seq_len: int,
    batch_size: int,
    r_loops: int,
    backend: str,
    block_t: int,
    step_ms: float,
    norm_drift_max: float = 0.0,
) -> HoloSphereStepTelemetry:
    ms = _req_finite(step_ms, "step_ms", "TEL_STEP_MS")
    require_ok(ms >= 0.0, "TEL_STEP_MS", "step_ms must be >= 0")
    bsz = _req_int_no_bool(batch_size, "batch_size", "TEL_BATCH")
    slen = _req_int_no_bool(seq_len, "seq_len", "TEL_SEQ")
    tps = (float(bsz * slen) * 1000.0 / ms) if ms > 0.0 else 0.0
    rec = HoloSphereStepTelemetry(
        step_idx=step_idx,
        seq_len=slen,
        batch_size=bsz,
        r_loops=r_loops,
        backend=backend,
        block_t=block_t,
        step_ms=round(ms, 6),
        tok_per_sec=round(tps, 3),
        norm_drift_max=round(_req_finite(norm_drift_max, "norm_drift_max", "TEL_DRIFT"), 9),
    )
    rec.validate()
    return rec


class SiliSenseHoloSphereBridge:
    """Step-boundary controller driving `HoloSphereTuningProfile` from Mode-1 telemetry."""

    def __init__(
        self,
        governor: PrecursorThrottleGovernor | None = None,
        base_profile: HoloSphereTuningProfile | None = None,
    ) -> None:
        self.governor = governor if governor is not None else PrecursorThrottleGovernor()
        self.base_profile = base_profile if base_profile is not None else HoloSphereTuningProfile()
        self.base_profile.validate()

    def on_step_boundary(self, step_tel: HoloSphereStepTelemetry, fast_power_w: float) -> HoloSphereTuningProfile:
        step_tel.validate()
        require_ok(step_tel.step_ms > 0.0 and step_tel.tok_per_sec > 0.0, "ERR_BAD_STEP_TEL", "Step ms and tok/s must be > 0")
        require_ok(step_tel.norm_drift_max <= 1e-3, "ERR_NORM_DRIFT_EXCEEDED", f"Unit-norm drift {step_tel.norm_drift_max} > 1e-3")
        scale, _ = self.governor.step(fast_power_w)
        cooldown = 0.0 if scale >= 0.999 else min(250.0, round(float(step_tel.step_ms) * (1.0 / scale - 1.0), 4))
        out = HoloSphereTuningProfile(
            backend=self.base_profile.backend,
            block_t=self.base_profile.block_t,
            microbatch_size=self.base_profile.microbatch_size,
            inter_step_cooldown_ms=cooldown,
            r_loops_override=self.base_profile.r_loops_override,
        )
        out.validate()
        return out
