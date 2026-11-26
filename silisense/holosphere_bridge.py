"""One-way SiliSense -> HoloSphere step-boundary telemetry & tuning contract.

Wire-compatible and `profile_sha256()`-identical across all canonicalized float
and integer inputs with `holosphere.telemetry`, while importing ZERO `holosphere`
or `torch` modules so the strict one-way dependency boundary holds:
`Moissanite-Labs/holosphere` never imports `silisense`, and `silisense` never
mutates model weights or downstream audit thresholds.
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
MAX_TOKEN_DIM = 1_000_000_000


def _require_int_not_bool(val: Any, name: str, code: str) -> int:
    require_ok(
        isinstance(val, int) and not isinstance(val, bool),
        code,
        f"{name} must be an int (not bool), got {type(val).__name__}",
    )
    return int(val)


def _require_finite_float(val: Any, name: str, code: str) -> float:
    require_ok(
        isinstance(val, (int, float)) and not isinstance(val, bool),
        code,
        f"{name} must be numeric, got {type(val).__name__}",
    )
    try:
        fval = float(val)
    except OverflowError as exc:
        raise ValueError(code) from exc
    require_ok(math.isfinite(fval), code, f"{name} must be finite, got {val!r}")
    return 0.0 if fval == 0.0 else fval


def _safe_finite_float(val: Any, name: str, code: str) -> float:
    require_ok(
        isinstance(val, (int, float)) and not isinstance(val, bool),
        code,
        f"{name} must be numeric, got {type(val).__name__}",
    )
    try:
        fval = float(val)
    except OverflowError:
        require_ok(False, code, f"{name} overflowed float64")
        return 0.0
    require_ok(math.isfinite(fval), code, f"{name} must be finite, got {val!r}")
    return 0.0 if fval == 0.0 else fval


def _require_str(val: Any, name: str, code: str) -> str:
    require_ok(isinstance(val, str), code, f"{name} must be a str, got {type(val).__name__}")
    return val


@dataclass(frozen=True)
class HoloSphereStepTelemetry:
    """Read-only per-step summary consumed at step boundaries from `holosphere`."""

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _require_str(self.schema, "schema", "TEL_SCHEMA"))
        object.__setattr__(self, "step_idx", _require_int_not_bool(self.step_idx, "step_idx", "TEL_STEP"))
        object.__setattr__(self, "seq_len", _require_int_not_bool(self.seq_len, "seq_len", "TEL_SEQ"))
        object.__setattr__(self, "batch_size", _require_int_not_bool(self.batch_size, "batch_size", "TEL_BATCH"))
        object.__setattr__(self, "r_loops", _require_int_not_bool(self.r_loops, "r_loops", "TEL_RLOOPS"))
        object.__setattr__(self, "backend", _require_str(self.backend, "backend", "TEL_BACKEND"))
        object.__setattr__(self, "block_t", _require_int_not_bool(self.block_t, "block_t", "TEL_BLOCKT"))
        object.__setattr__(self, "step_ms", _safe_finite_float(self.step_ms, "step_ms", "TEL_STEP_MS"))
        object.__setattr__(self, "tok_per_sec", _safe_finite_float(self.tok_per_sec, "tok_per_sec", "TEL_TPS"))
        object.__setattr__(self, "norm_drift_max", _safe_finite_float(self.norm_drift_max, "norm_drift_max", "TEL_DRIFT"))
        self.validate()

    def validate(self) -> None:
        schema = _require_str(self.schema, "schema", "TEL_SCHEMA")
        require_ok(schema == TELEMETRY_SCHEMA, "TEL_SCHEMA", f"Unexpected schema {schema!r}")
        step_idx = _require_int_not_bool(self.step_idx, "step_idx", "TEL_STEP")
        seq_len = _require_int_not_bool(self.seq_len, "seq_len", "TEL_SEQ")
        batch_size = _require_int_not_bool(self.batch_size, "batch_size", "TEL_BATCH")
        r_loops = _require_int_not_bool(self.r_loops, "r_loops", "TEL_RLOOPS")
        backend = _require_str(self.backend, "backend", "TEL_BACKEND")
        block_t = _require_int_not_bool(self.block_t, "block_t", "TEL_BLOCKT")
        step_ms = _safe_finite_float(self.step_ms, "step_ms", "TEL_STEP_MS")
        tps = _safe_finite_float(self.tok_per_sec, "tok_per_sec", "TEL_TPS")
        drift = _safe_finite_float(self.norm_drift_max, "norm_drift_max", "TEL_DRIFT")

        require_ok(step_idx >= 0, "TEL_STEP", "step_idx must be >= 0")
        require_ok(1 <= seq_len <= MAX_TOKEN_DIM, "TEL_SEQ", f"seq_len must lie in [1, {MAX_TOKEN_DIM}]")
        require_ok(1 <= batch_size <= MAX_TOKEN_DIM, "TEL_BATCH", f"batch_size must lie in [1, {MAX_TOKEN_DIM}]")
        require_ok(1 <= r_loops <= 65536, "TEL_RLOOPS", "r_loops must lie in [1, 65536]")
        require_ok(backend in VALID_EXECUTION_BACKENDS, "TEL_BACKEND", f"Unsupported backend {backend!r}")
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
        return cls(
            step_idx=data["step_idx"],
            seq_len=data["seq_len"],
            batch_size=data["batch_size"],
            r_loops=data["r_loops"],
            backend=data["backend"],
            block_t=data["block_t"],
            step_ms=data["step_ms"],
            tok_per_sec=data["tok_per_sec"],
            norm_drift_max=data["norm_drift_max"],
            schema=data.get("schema", TELEMETRY_SCHEMA),
        )


@dataclass(frozen=True)
class HoloSphereTuningProfile:
    """Semantics-preserving execution knobs with canonicalized `profile_sha256()`.

    Numeric fields are normalized in `__post_init__` so `0`, `0.0`, `-0.0`, and
    `from_dict({"inter_step_cooldown_ms": 0})` produce identical `profile_sha256()`
    digests matching `holosphere.telemetry.HoloSphereTuningProfile`.
    """

    backend: str = "auto"
    block_t: int = 128
    microbatch_size: int | None = None
    inter_step_cooldown_ms: float = 0.0
    r_loops_override: int | None = None
    schema: str = TUNING_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _require_str(self.schema, "schema", "TUNE_SCHEMA"))
        object.__setattr__(self, "backend", _require_str(self.backend, "backend", "TUNE_BACKEND"))
        object.__setattr__(self, "block_t", _require_int_not_bool(self.block_t, "block_t", "TUNE_BLOCKT"))
        if self.microbatch_size is not None:
            object.__setattr__(
                self, "microbatch_size", _require_int_not_bool(self.microbatch_size, "microbatch_size", "TUNE_MBS")
            )
        object.__setattr__(
            self,
            "inter_step_cooldown_ms",
            _safe_finite_float(self.inter_step_cooldown_ms, "inter_step_cooldown_ms", "TUNE_COOLDOWN"),
        )
        if self.r_loops_override is not None:
            object.__setattr__(
                self, "r_loops_override", _require_int_not_bool(self.r_loops_override, "r_loops_override", "TUNE_RLOOPS")
            )
        self.validate()

    def validate(self) -> None:
        schema = _require_str(self.schema, "schema", "TUNE_SCHEMA")
        require_ok(schema == TUNING_PROFILE_SCHEMA, "TUNE_SCHEMA", f"Unexpected schema {schema!r}")
        backend = _require_str(self.backend, "backend", "TUNE_BACKEND")
        require_ok(backend in VALID_TUNING_BACKENDS, "TUNE_BACKEND", f"Unsupported backend {backend!r}")
        block_t = _require_int_not_bool(self.block_t, "block_t", "TUNE_BLOCKT")
        require_ok(block_t in VALID_BLOCK_TS, "TUNE_BLOCKT", f"Unsupported block_t {block_t}")
        if self.microbatch_size is not None:
            mbs = _require_int_not_bool(self.microbatch_size, "microbatch_size", "TUNE_MBS")
            require_ok(1 <= mbs <= MAX_TOKEN_DIM, "TUNE_MBS", "microbatch_size out of range")
        cooldown = _safe_finite_float(self.inter_step_cooldown_ms, "inter_step_cooldown_ms", "TUNE_COOLDOWN")
        require_ok(0.0 <= cooldown <= MAX_COOLDOWN_MS, "TUNE_COOLDOWN", f"Cooldown out of range: {cooldown}")
        if self.r_loops_override is not None:
            rl = _require_int_not_bool(self.r_loops_override, "r_loops_override", "TUNE_RLOOPS")
            require_ok(1 <= rl <= 65536, "TUNE_RLOOPS", "r_loops_override out of range")

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
        return cls(
            backend=data.get("backend", "auto"),
            block_t=data.get("block_t", 128),
            microbatch_size=data.get("microbatch_size", None),
            inter_step_cooldown_ms=data.get("inter_step_cooldown_ms", 0.0),
            r_loops_override=data.get("r_loops_override", None),
            schema=data.get("schema", TUNING_PROFILE_SCHEMA),
        )


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
    ms = _safe_finite_float(step_ms, "step_ms", "TEL_STEP_MS")
    require_ok(ms >= 0.0, "TEL_STEP_MS", "step_ms must be >= 0")
    bsz = _require_int_not_bool(batch_size, "batch_size", "TEL_BATCH")
    slen = _require_int_not_bool(seq_len, "seq_len", "TEL_SEQ")
    require_ok(1 <= bsz <= MAX_TOKEN_DIM and 1 <= slen <= MAX_TOKEN_DIM, "TEL_DIM", "Invalid batch_size or seq_len")
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
        norm_drift_max=round(_safe_finite_float(norm_drift_max, "norm_drift_max", "TEL_DRIFT"), 9),
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
        require_ok(isinstance(step_tel, HoloSphereStepTelemetry), "ERR_BAD_STEP_TEL", "Expected HoloSphereStepTelemetry")
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
