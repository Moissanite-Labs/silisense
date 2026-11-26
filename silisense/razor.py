"""Closed-loop 30 kHz BER-canary Razor controller, Weft rollback, and Krein-Holonomy detector.

Pure-stdlib, exact-rational (`int` / `Fraction`) control plane matching the 64-byte
cacheline-aligned `.sys` canary ABI in `include/razor_abi_4byte_sys.h`:

1. Inline `[16, 11, 4]` Block-H4 LSB SECDED (`j = (s >> 1) & 0xF`) sampling physical
   bit-flips at `30 kHz` (`33.33 us/tok`) vs `10 Hz` NVML ZOH (`3,000x` faster) with
   zero PCIe atomic traffic on clean cycles.
2. Vmin carry-path upper-bit closure (nibble bits `1..3`) via three exact mechanisms:
   - Folded `bit1 ^ bit2 ^ bit3` upper-bit parity (`1` spare codeword bit, `0` extra DRAM B,
     catches `48/48` single upper-bit flips),
   - Epoch-rotated nibble bit-plane selection (`b = epoch & 3`, closes all `4` planes over `4` epochs),
   - Dual-channel `block_crc16_ccitt` (catches `48/48` single and multi-bit upper-bit faults).
3. Three-tier exact-Q self-healing Razor controller with `N = 256`-token epoch hysteresis
   (`alpha = 1/32` EWMA, step-up on `sec_delta >= 2` or `ewma >= 2/256`, `4`-epoch dwell
   hold, `16` consecutive clean epochs for margin descent, `64`-epoch DED rollback latch)
   plus `Weft` `16 B` `Pi_{t_0}` chunk-32 `SU(2)` rollback curing right-coset drift.
4. Exact-Q Non-Scalar Krein-Holonomy phase detector `K_t`:
   `K_t = diag(G - A^T G A, 3 - |tr(Delta U_t)|^2)`, proving scalar `G = I_3` blindness
   (`n_-(UL) == n_-(LU) == 1`) vs non-scalar `G_3 = diag(3, 1, 1)` order separation
   (`n_-(UL) == 1 != 2 == n_-(LU)`).
5. Native `silisense.holosphere_bridge.HoloSphereTuningProfile` emission guaranteeing
   bit-identical `profile_sha256()` digests with `Moissanite-Labs/holosphere`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import struct
from typing import Any, Mapping, Sequence

from silisense.errors import require_ok
from silisense.holosphere_bridge import HoloSphereTuningProfile, MAX_COOLDOWN_MS

CANARY64_STRUCT = struct.Struct("<16I")
CANARY64_BYTES = 64
U32_MAX = (1 << 32) - 1
CHUNK_TOKENS = 32
PI_ROLLBACK_BYTES = 16
F32_MAX_Q = Fraction(((1 << 24) - 1) << 104, 1)
F32_MIN_SUBNORMAL_Q = Fraction(1, 1 << 149)

# Exact N=256-token epoch hysteresis parameters (anti-chatter law)
EPOCH_TOKENS = 256
EWMA_Q16_SCALE = 1 << 16                        # Q16 fixed-point grid matching ewma_ber_q16_u32
HYSTERESIS_EWMA_ALPHA_Q = Fraction(1, 32)
HYSTERESIS_SEC_STEP_DOWN_COUNT = 2
HYSTERESIS_EWMA_STEP_DOWN_Q = Fraction(2, 256)  # 1/128 = 0.0078125
HYSTERESIS_HOLD_EPOCHS = 4                      # 1,024 tokens
HYSTERESIS_STEP_UP_CLEAN_EPOCHS = 16            # 4,096 tokens
HYSTERESIS_DED_LATCH_CLEAN_EPOCHS = 64          # 16,384 tokens

MAX_COOLDOWN_MS_Q = Fraction(int(MAX_COOLDOWN_MS), 1)

# [16, 11, 4] Extended Hamming layout matching CUDA `razor_a16_secded_syndrome`
PARITY_POSITIONS = (0, 1, 2, 4, 8)
DATA_POSITIONS = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)

# Exact rational sampling periods (10 Hz NVML ZOH vs 30 kHz inline BER canary)
NVML_ZOH_PERIOD_US_Q = Fraction(100_000, 1)  # 100 ms = 10 Hz
RAZOR_BER_PERIOD_US_Q = Fraction(100, 3)     # 33.333... us/tok = 30 kHz
DELAY_BW_LIMIT_Q = Fraction(1, 6)            # f * tau < 1/6


def _req_int(val: Any, name: str, code: str) -> int:
    require_ok(
        isinstance(val, int) and not isinstance(val, bool),
        code,
        f"{name} must be int (not bool), got {type(val).__name__}",
    )
    return int(val)


def _req_u32(val: Any, name: str, code: str) -> int:
    iv = _req_int(val, name, code)
    require_ok(0 <= iv <= U32_MAX, code, f"{name} out of uint32 bounds: {iv}")
    return iv


def _req_fraction(val: Any, name: str, code: str) -> Fraction:
    require_ok(
        isinstance(val, Fraction),
        code,
        f"{name} must be exact Fraction (zero floats on trust path), got {type(val).__name__}",
    )
    return val


def _frac_str(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------------------
# 1. Exact [16, 11, 4] Extended Hamming SECDED + Vmin Upper-Bit Closure
# ---------------------------------------------------------------------------

def a16_secded_syndrome(w16: int) -> int:
    """Exact 5-bit `[16, 11, 4]` syndrome matching `razor_a16_secded_syndrome`."""
    word = _req_int(w16, "w16", "ERR_A16_WORD")
    require_ok(0 <= word <= 0xFFFF, "ERR_A16_WORD", f"w16 must lie in [0, 0xFFFF], got {word}")
    p0 = word.bit_count() & 1
    p1 = (word & 0xAAAA).bit_count() & 1
    p2 = (word & 0xCCCC).bit_count() & 1
    p3 = (word & 0xF0F0).bit_count() & 1
    p4 = (word & 0xFF00).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def a16_encode_lsb16(data11: int) -> int:
    """Encode 11 data bits (0..2047) into a 16-bit `[16, 11, 4]` codeword (`syndrome == 0`)."""
    payload = _req_int(data11, "data11", "ERR_A16_DATA11")
    require_ok(0 <= payload <= 0x7FF, "ERR_A16_DATA11", f"data11 must lie in [0, 2047], got {payload}")
    w16 = 0
    for bit_idx, pos in enumerate(DATA_POSITIONS):
        if (payload >> bit_idx) & 1:
            w16 |= 1 << pos
    if (w16 & 0xAAAA).bit_count() & 1:
        w16 |= 1 << 1
    if (w16 & 0xCCCC).bit_count() & 1:
        w16 |= 1 << 2
    if (w16 & 0xF0F0).bit_count() & 1:
        w16 |= 1 << 4
    if (w16 & 0xFF00).bit_count() & 1:
        w16 |= 1 << 8
    if w16.bit_count() & 1:
        w16 |= 1 << 0
    return w16


def a16_secded_classify(syndrome: int) -> tuple[str, int | None]:
    """Classify 5-bit syndrome -> `('CLEAN', None) | ('SEC', lane_j) | ('DED', None)`."""
    s = _req_int(syndrome, "syndrome", "ERR_A16_SYND")
    require_ok(0 <= s <= 0x1F, "ERR_A16_SYND", f"syndrome out of 5-bit range: {s}")
    if s == 0:
        return ("CLEAN", None)
    if (s & 1) == 1:
        return ("SEC", (s >> 1) & 0xF)
    return ("DED", None)


def _validate_nibbles16(nibbles_16: Sequence[int]) -> list[int]:
    require_ok(
        isinstance(nibbles_16, (list, tuple)),
        "ERR_NIBBLE_TYPE",
        f"nibbles_16 must be a list or tuple, got {type(nibbles_16).__name__}",
    )
    require_ok(len(nibbles_16) == 16, "ERR_NIBBLE_LEN", f"Expected 16 nibbles, got {len(nibbles_16)}")
    out: list[int] = []
    for idx, val in enumerate(nibbles_16):
        iv = _req_int(val, f"nibbles_16[{idx}]", "ERR_NIBBLE_VAL")
        require_ok(-8 <= iv <= 7, "ERR_NIBBLE_VAL", f"Nibble[{idx}]={iv} outside signed 4-bit [-8, 7]")
        out.append(iv)
    return out


def a16_rotate_bitplane_idx(epoch_u32: int) -> int:
    """Epoch-rotated nibble bit-plane selector `b = epoch & 3` (closes bits `0..3` over 4 epochs)."""
    return _req_u32(epoch_u32, "epoch_u32", "ERR_EPOCH_U32") & 3


def a16_extract_bitplane16(nibbles_16: Sequence[int], bit_plane: int) -> int:
    """Extract 16-bit plane `bit_plane` (`0..3`) across 16 signed 4-bit nibbles."""
    nibs = _validate_nibbles16(nibbles_16)
    bp = _req_int(bit_plane, "bit_plane", "ERR_BIT_PLANE")
    require_ok(0 <= bp <= 3, "ERR_BIT_PLANE", f"bit_plane must be in 0..3, got {bp}")
    w16 = 0
    for lane, val in enumerate(nibs):
        if ((val & 0xF) >> bp) & 1:
            w16 |= 1 << lane
    return w16


def a16_plane_syndromes_4(nibbles_16: Sequence[int]) -> tuple[int, int, int, int]:
    """Compute reference 5-bit `[16, 11, 4]` syndromes for all 4 nibble bit planes (`0..3`)."""
    nibs = _validate_nibbles16(nibbles_16)
    return (
        a16_secded_syndrome(a16_extract_bitplane16(nibs, 0)),
        a16_secded_syndrome(a16_extract_bitplane16(nibs, 1)),
        a16_secded_syndrome(a16_extract_bitplane16(nibs, 2)),
        a16_secded_syndrome(a16_extract_bitplane16(nibs, 3)),
    )


def a16_folded_upper_parity_bit(nibbles_16: Sequence[int]) -> int:
    """Exact folded upper-bit parity `XOR_{j=0..15} (bit1 ^ bit2 ^ bit3)` (`0` or `1`).

    Costs `1` spare codeword bit (`0` extra DRAM bytes) and detects `48/48` (`100%`) of all
    single Vmin carry-path flips in nibble bits `1..3`.
    """
    nibs = _validate_nibbles16(nibbles_16)
    acc = 0
    for val in nibs:
        u = val & 0xF
        acc ^= ((u >> 1) ^ (u >> 2) ^ (u >> 3)) & 1
    return acc


def block_crc16_ccitt(nibbles_16: Sequence[int]) -> int:
    """Exact 16-bit CRC-CCITT (poly `0x1021`) over all `4` bits x `16` nibbles."""
    nibs = _validate_nibbles16(nibbles_16)
    packed_bytes = bytes((((nibs[2 * i] & 0xF) << 4) | (nibs[2 * i + 1] & 0xF)) for i in range(8))
    crc = 0xFFFF
    for byte in packed_bytes:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def a16_inline_secded_step(
    nibbles_16: Sequence[int],
    *,
    expected_crc16: int | None = None,
    expected_upper_fold_bit: int | None = None,
    epoch_u32: int = 0,
    expected_plane_syndromes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run inline `[16, 11, 4]` LSB SECDED plus optional Vmin upper-bit closure guards."""
    nibs = _validate_nibbles16(nibbles_16)
    rot_plane = a16_rotate_bitplane_idx(epoch_u32)
    w16 = a16_extract_bitplane16(nibs, 0)
    synd = a16_secded_syndrome(w16)
    kind, fixed_lane = a16_secded_classify(synd)

    healed = list(nibs)
    sec_delta = 0
    ded_delta = 0
    if kind == "SEC" and fixed_lane is not None:
        healed[fixed_lane] = healed[fixed_lane] ^ 1
        sec_delta = 1
    elif kind == "DED":
        ded_delta = 1

    non_lsb_delta = 0
    if expected_upper_fold_bit is not None:
        exp_f = _req_int(expected_upper_fold_bit, "expected_upper_fold_bit", "ERR_BAD_FOLD_BIT")
        require_ok(exp_f in (0, 1), "ERR_BAD_FOLD_BIT", f"expected_upper_fold_bit must be 0 or 1, got {exp_f}")
        if ded_delta == 0 and a16_folded_upper_parity_bit(healed) != exp_f:
            non_lsb_delta = 1

    if expected_plane_syndromes is not None:
        require_ok(
            isinstance(expected_plane_syndromes, (list, tuple)) and len(expected_plane_syndromes) == 4,
            "ERR_BAD_PLANE_SYNDS",
            "expected_plane_syndromes must be a 4-element list or tuple",
        )
        exp_rot = _req_int(expected_plane_syndromes[rot_plane], "expected_plane_syndromes[b]", "ERR_BAD_PLANE_SYNDS")
        require_ok(0 <= exp_rot <= 0x1F, "ERR_BAD_PLANE_SYNDS", f"plane syndrome out of 5-bit range: {exp_rot}")
        if ded_delta == 0 and rot_plane > 0:
            cur_rot_synd = a16_secded_syndrome(a16_extract_bitplane16(healed, rot_plane))
            if cur_rot_synd != exp_rot:
                non_lsb_delta = 1

    if expected_crc16 is not None:
        exp_c = _req_int(expected_crc16, "expected_crc16", "ERR_BAD_CRC16")
        require_ok(0 <= exp_c <= 0xFFFF, "ERR_BAD_CRC16", f"expected_crc16 out of range: {exp_c}")
        if ded_delta == 0 and block_crc16_ccitt(healed) != exp_c:
            non_lsb_delta = 1

    return {
        "status": "NON_LSB_FAULT" if non_lsb_delta else kind,
        "syndrome": synd,
        "rotated_plane": rot_plane,
        "fixed_lane": fixed_lane,
        "healed_nibbles": tuple(healed),
        "sec_delta": sec_delta,
        "ded_delta": ded_delta,
        "non_lsb_delta": non_lsb_delta,
        "pcie_atomic_writes": 0 if (sec_delta == 0 and ded_delta == 0 and non_lsb_delta == 0) else 1,
    }


def inject_nibble_bit_flip(nibbles_16: Sequence[int], lane: int, bit_pos: int) -> tuple[int, ...]:
    """Flip exact bit `bit_pos` (`0..3`) of signed 4-bit nibble at `lane` (`0..15`)."""
    nibs = _validate_nibbles16(nibbles_16)
    ln = _req_int(lane, "lane", "ERR_FLIP_LANE")
    bp = _req_int(bit_pos, "bit_pos", "ERR_FLIP_BIT")
    require_ok(0 <= ln <= 15, "ERR_FLIP_LANE", f"lane must be in 0..15, got {ln}")
    require_ok(0 <= bp <= 3, "ERR_FLIP_BIT", f"bit_pos must be in 0..3, got {bp}")
    raw4 = (nibs[ln] & 0xF) ^ (1 << bp)
    signed4 = raw4 - 16 if raw4 >= 8 else raw4
    nibs[ln] = signed4
    return tuple(nibs)


# ---------------------------------------------------------------------------
# 2. Cacheline-Aligned 64 B D2H .sys Canary Wire Struct (`razor_abi_4byte_sys.h`)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RazorCanary64:
    """Exact 64-byte `<16I>` mirror of `HoloSphereRazorCanary64` in `include/razor_abi_4byte_sys.h`."""

    sec_count_u32: int = 0
    ded_count_u32: int = 0
    epoch_u32: int = 0
    rollback_token_u32: int = 0
    non_lsb_canary_u32: int = 0
    last_syndrome_u32: int = 0
    ewma_ber_q16_u32: int = 0
    clean_streak_u32: int = 0
    hold_remaining_u32: int = 0
    ded_latch_remaining_u32: int = 0
    reserved_u32: tuple[int, ...] = (0,) * 6

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _req_u32(self.sec_count_u32, "sec_count_u32", "ERR_CANARY_U32")
        _req_u32(self.ded_count_u32, "ded_count_u32", "ERR_CANARY_U32")
        _req_u32(self.epoch_u32, "epoch_u32", "ERR_CANARY_U32")
        _req_u32(self.rollback_token_u32, "rollback_token_u32", "ERR_CANARY_U32")
        _req_u32(self.non_lsb_canary_u32, "non_lsb_canary_u32", "ERR_CANARY_U32")
        _req_u32(self.last_syndrome_u32, "last_syndrome_u32", "ERR_CANARY_U32")
        _req_u32(self.ewma_ber_q16_u32, "ewma_ber_q16_u32", "ERR_CANARY_U32")
        _req_u32(self.clean_streak_u32, "clean_streak_u32", "ERR_CANARY_U32")
        _req_u32(self.hold_remaining_u32, "hold_remaining_u32", "ERR_CANARY_U32")
        _req_u32(self.ded_latch_remaining_u32, "ded_latch_remaining_u32", "ERR_CANARY_U32")
        require_ok(
            isinstance(self.reserved_u32, tuple) and len(self.reserved_u32) == 6,
            "ERR_CANARY_PAD",
            "reserved_u32 must be a 6-element zero tuple",
        )
        for idx, v in enumerate(self.reserved_u32):
            iv = _req_u32(v, f"reserved_u32[{idx}]", "ERR_CANARY_PAD")
            require_ok(iv == 0, "ERR_CANARY_PAD", f"reserved_u32[{idx}] must be 0, got {iv}")

    def to_bytes(self) -> bytes:
        self.validate()
        return CANARY64_STRUCT.pack(
            self.sec_count_u32,
            self.ded_count_u32,
            self.epoch_u32,
            self.rollback_token_u32,
            self.non_lsb_canary_u32,
            self.last_syndrome_u32,
            self.ewma_ber_q16_u32,
            self.clean_streak_u32,
            self.hold_remaining_u32,
            self.ded_latch_remaining_u32,
            *self.reserved_u32,
        )

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> "RazorCanary64":
        raw = bytes(buf)
        require_ok(len(raw) == CANARY64_BYTES, "ERR_CANARY_LEN", f"Expected 64 B, got {len(raw)} B")
        words = CANARY64_STRUCT.unpack(raw)
        return cls(
            sec_count_u32=words[0],
            ded_count_u32=words[1],
            epoch_u32=words[2],
            rollback_token_u32=words[3],
            non_lsb_canary_u32=words[4],
            last_syndrome_u32=words[5],
            ewma_ber_q16_u32=words[6],
            clean_streak_u32=words[7],
            hold_remaining_u32=words[8],
            ded_latch_remaining_u32=words[9],
            reserved_u32=tuple(words[10:]),
        )

    def canary_sha256(self) -> str:
        self.validate()
        payload = {
            "canon": "zone-canon-v0",
            "clean_streak_u32": self.clean_streak_u32,
            "ded_count_u32": self.ded_count_u32,
            "ded_latch_remaining_u32": self.ded_latch_remaining_u32,
            "epoch_u32": self.epoch_u32,
            "ewma_ber_q16_u32": self.ewma_ber_q16_u32,
            "hold_remaining_u32": self.hold_remaining_u32,
            "last_syndrome_u32": self.last_syndrome_u32,
            "non_lsb_canary_u32": self.non_lsb_canary_u32,
            "rollback_token_u32": self.rollback_token_u32,
            "sec_count_u32": self.sec_count_u32,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# 3. Exact-Q SU(2) Holonomy, Adjoint SO(3), & Weft 16 B Pi_{t_0} Rollback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuatQ:
    """Exact rational quaternion in `Q^4` (zero IEEE-754 floats)."""

    w: Fraction
    x: Fraction
    y: Fraction
    z: Fraction

    def __post_init__(self) -> None:
        _req_fraction(self.w, "w", "ERR_QUAT_Q")
        _req_fraction(self.x, "x", "ERR_QUAT_Q")
        _req_fraction(self.y, "y", "ERR_QUAT_Q")
        _req_fraction(self.z, "z", "ERR_QUAT_Q")

    def norm_sq(self) -> Fraction:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def conj(self) -> "QuatQ":
        return QuatQ(self.w, -self.x, -self.y, -self.z)

    def mul(self, other: "QuatQ") -> "QuatQ":
        require_ok(isinstance(other, QuatQ), "ERR_QUAT_TYPE", "Expected QuatQ")
        return QuatQ(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def diff_norm_sq(self, other: "QuatQ") -> Fraction:
        require_ok(isinstance(other, QuatQ), "ERR_QUAT_TYPE", "Expected QuatQ")
        return (
            (self.w - other.w) ** 2
            + (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def to_pi16_bytes(self) -> bytes:
        """Pack exact QuatQ into the 16-byte little-endian IEEE-754 `HoloSpherePiRollback16` ABI."""
        vals: list[float] = []
        for name, q in (("w", self.w), ("x", self.x), ("y", self.y), ("z", self.z)):
            aq = abs(q)
            require_ok(
                aq <= F32_MAX_Q and (aq == 0 or aq >= F32_MIN_SUBNORMAL_Q),
                "ERR_PI16_RANGE",
                f"QuatQ.{name} outside finite non-underflowing IEEE-754 float32 range",
            )
            vals.append(float(q))
        return struct.pack("<4f", *vals)


QUAT_IDENTITY = QuatQ(Fraction(1), Fraction(0), Fraction(0), Fraction(0))


def cayley_su2_q(wx: Fraction, wy: Fraction, wz: Fraction) -> QuatQ:
    """Exact rational trig-free Cayley map `Q^3 -> S^3` (`norm_sq == 1` identically)."""
    _req_fraction(wx, "wx", "ERR_CAYLEY_Q")
    _req_fraction(wy, "wy", "ERR_CAYLEY_Q")
    _req_fraction(wz, "wz", "ERR_CAYLEY_Q")
    n2 = wx * wx + wy * wy + wz * wz
    den = Fraction(1) + n2
    return QuatQ((Fraction(1) - n2) / den, (Fraction(2) * wx) / den, (Fraction(2) * wy) / den, (Fraction(2) * wz) / den)


def su2_to_so3_q(q: QuatQ) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    """Exact rational adjoint map `rho: SU(2) cap Q^4 -> SO(3) cap GL_3(Q)`."""
    require_ok(isinstance(q, QuatQ), "ERR_SO3_QUAT", "Expected QuatQ")
    require_ok(q.norm_sq() == Fraction(1), "ERR_SO3_UNIT", f"Unit quaternion required, got norm_sq={q.norm_sq()}")
    w, x, y, z = q.w, q.x, q.y, q.z
    two = Fraction(2)
    one = Fraction(1)
    return (
        (one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)),
        (two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x)),
        (two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y)),
    )


class WeftHolonomyRollbackBuffer:
    """Chunk-32 checkpointing with `16 B` `Pi_{t_0}` rollback curing right-coset drift."""

    def __init__(self, chunk_size: int = CHUNK_TOKENS) -> None:
        cs = _req_int(chunk_size, "chunk_size", "ERR_CHUNK_SIZE")
        require_ok(cs >= 1, "ERR_CHUNK_SIZE", "chunk_size must be >= 1")
        self.chunk_size = cs
        self.token_idx: int = 0
        self.u_state: QuatQ = QUAT_IDENTITY
        self.s_fast_acc: Fraction = Fraction(0)
        self.ckpt_token_idx: int = 0
        self.ckpt_pi_t0: QuatQ = QUAT_IDENTITY
        self.ckpt_s_fast_acc: Fraction = Fraction(0)
        self.rollback_count: int = 0

    def step_token(
        self,
        wx: Fraction,
        wy: Fraction,
        wz: Fraction,
        s_delta: Fraction,
        *,
        corrupt_delta: QuatQ | None = None,
        commit_boundary: bool = True,
    ) -> QuatQ:
        """Advance holonomy `U_{t+1} = U_t * cayley_su2_q(w_t)`, committing verified clean boundaries."""
        _req_fraction(s_delta, "s_delta", "ERR_SFAST_Q")
        require_ok(isinstance(commit_boundary, bool), "ERR_COMMIT_BOOL", "commit_boundary must be bool")
        if commit_boundary and (self.token_idx % self.chunk_size == 0):
            self.ckpt_token_idx = self.token_idx
            self.ckpt_pi_t0 = self.u_state
            self.ckpt_s_fast_acc = self.s_fast_acc
        dq = cayley_su2_q(wx, wy, wz)
        if corrupt_delta is not None:
            require_ok(isinstance(corrupt_delta, QuatQ), "ERR_CORRUPT_Q", "corrupt_delta must be QuatQ")
            dq = dq.mul(corrupt_delta)
        self.u_state = self.u_state.mul(dq)
        self.s_fast_acc = self.s_fast_acc + s_delta
        self.token_idx += 1
        return self.u_state

    def pack_checkpoint_pi16(self) -> bytes:
        """Pack the saved `Pi_{t_0}` holonomy checkpoint into the 16-byte `HoloSpherePiRollback16` ABI."""
        return self.ckpt_pi_t0.to_pi16_bytes()

    def rollback_to_last_clean_chunk(self) -> dict[str, Any]:
        """Execute `16 B` `Pi_{t_0}` `O(1)` rollback snapping `U_t` and `S_fast` to last `C=32` boundary."""
        rewound_tokens = self.token_idx - self.ckpt_token_idx
        self.token_idx = self.ckpt_token_idx
        self.u_state = self.ckpt_pi_t0
        self.s_fast_acc = self.ckpt_s_fast_acc
        self.rollback_count += 1
        return {
            "rollback_token_u32": self.ckpt_token_idx,
            "rewound_tokens": rewound_tokens,
            "rollback_state_bytes": PI_ROLLBACK_BYTES,
            "rollback_count": self.rollback_count,
        }


# ---------------------------------------------------------------------------
# 4. Exact-Q Non-Scalar Krein-Holonomy Phase Detector K_t
# ---------------------------------------------------------------------------

Mat3Q = list[list[Fraction]]


def _req_mat3_q(mat: Sequence[Sequence[Fraction]], name: str, code: str) -> Mat3Q:
    require_ok(
        isinstance(mat, (list, tuple))
        and len(mat) == 3
        and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in mat),
        code,
        f"{name} must be a 3x3 list/tuple exact Fraction matrix",
    )
    return [[_req_fraction(val, f"{name}[{i}][{j}]", code) for j, val in enumerate(row)] for i, row in enumerate(mat)]


def _mat3_mul(a: Mat3Q, b: Mat3Q) -> Mat3Q:
    return [[sum((a[i][k] * b[k][j] for k in range(3)), Fraction(0)) for j in range(3)] for i in range(3)]


def _mat3_transpose(a: Mat3Q) -> Mat3Q:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _mat3_det(a: Mat3Q) -> Fraction:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _req_so3_mat3_q(mat: Sequence[Sequence[Fraction]], name: str) -> Mat3Q:
    m = _req_mat3_q(mat, name, "ERR_KREIN_SO3")
    mt_m = _mat3_mul(_mat3_transpose(m), m)
    eye = [[Fraction(1) if i == j else Fraction(0) for j in range(3)] for i in range(3)]
    require_ok(mt_m == eye and _mat3_det(m) == Fraction(1), "ERR_KREIN_SO3", f"{name} must lie in SO(3) (U^T U = I, det = 1)")
    return m


def _req_spd_mat3_q(mat: Sequence[Sequence[Fraction]], name: str) -> Mat3Q:
    g = _req_mat3_q(mat, name, "ERR_KREIN_SPD")
    require_ok(g == _mat3_transpose(g), "ERR_KREIN_SPD", f"{name} must be symmetric")
    d1 = g[0][0]
    d2 = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    d3 = _mat3_det(g)
    require_ok(d1 > 0 and d2 > 0 and d3 > 0, "ERR_KREIN_SPD", f"{name} must be strictly SPD (leading minors > 0)")
    return g


def _symmetric_mat3_inertia_q(m: Mat3Q) -> tuple[int, int, int]:
    """Exact Sylvester inertia `(n_pos, n_zero, n_neg)` of symmetric `3x3` `M` via Descartes' Rule."""
    c1 = m[0][0] + m[1][1] + m[2][2]
    c2 = (
        (m[0][0] * m[1][1] - m[0][1] * m[1][0])
        + (m[0][0] * m[2][2] - m[0][2] * m[2][0])
        + (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    )
    c3 = _mat3_det(m)
    pos_seq = [v for v in (Fraction(1), -c1, c2, -c3) if v != 0]
    neg_seq = [v for v in (Fraction(1), c1, c2, c3) if v != 0]
    n_pos = sum(1 for i in range(len(pos_seq) - 1) if (pos_seq[i] > 0) != (pos_seq[i + 1] > 0))
    n_neg = sum(1 for i in range(len(neg_seq) - 1) if (neg_seq[i] > 0) != (neg_seq[i + 1] > 0))
    n_zero = 3 - n_pos - n_neg
    return (n_pos, n_zero, n_neg)


def nonscalar_krein_holonomy_detector(
    u_rot: Sequence[Sequence[Fraction]] | QuatQ,
    l_mat: Sequence[Sequence[Fraction]],
    g_metric: Sequence[Sequence[Fraction]],
    delta_u: Sequence[Sequence[Fraction]] | QuatQ,
    *,
    order: str = "UL",
) -> dict[str, Any]:
    """Exact-Q Non-Scalar Krein-Holonomy Phase Detector `K_t`.

    Computes exact rational Sylvester inertia of:
      `K_t = diag(G - A^T G A,  3 - |tr(Delta U_t)|^2)`
    where `A = rho(U_t) L_t` (`order="UL"`) or `L_t rho(U_t)` (`order="LU"`),
    `G` is SPD in `GL_3(Q)`, and `Delta U_t` lies in `SO(3)`.
    """
    require_ok(order in ("UL", "LU"), "ERR_KREIN_ORDER", f"order must be 'UL' or 'LU', got {order!r}")
    u_m = _req_so3_mat3_q(su2_to_so3_q(u_rot), "u_rot") if isinstance(u_rot, QuatQ) else _req_so3_mat3_q(u_rot, "u_rot")
    l_m = _req_mat3_q(l_mat, "l_mat", "ERR_KREIN_LMAT")
    g_m = _req_spd_mat3_q(g_metric, "g_metric")
    du_m = _req_so3_mat3_q(su2_to_so3_q(delta_u), "delta_u") if isinstance(delta_u, QuatQ) else _req_so3_mat3_q(delta_u, "delta_u")

    a_m = _mat3_mul(u_m, l_m) if order == "UL" else _mat3_mul(l_m, u_m)
    at_g_a = _mat3_mul(_mat3_mul(_mat3_transpose(a_m), g_m), a_m)
    block3 = [[g_m[i][j] - at_g_a[i][j] for j in range(3)] for i in range(3)]
    b_pos, b_zero, b_neg = _symmetric_mat3_inertia_q(block3)

    tr_du = du_m[0][0] + du_m[1][1] + du_m[2][2]
    phase_slot = Fraction(3) - tr_du * tr_du
    p_pos = 1 if phase_slot > 0 else 0
    p_zero = 1 if phase_slot == 0 else 0
    p_neg = 1 if phase_slot < 0 else 0

    return {
        "order": order,
        "block3_inertia": {"n_pos": b_pos, "n_zero": b_zero, "n_neg": b_neg},
        "tr_delta_u_q": _frac_str(tr_du),
        "phase_slot_q": _frac_str(phase_slot),
        "k4_total_inertia": {
            "n_pos": b_pos + p_pos,
            "n_zero": b_zero + p_zero,
            "n_neg": b_neg + p_neg,
        },
        "overshoot_or_phase_active": bool(b_neg > 0),
    }


def krein_holonomy_order_witness_q() -> dict[str, Any]:
    """Exact order-separation witness: scalar `I_3` blind `(1, 1)` vs `G_3=diag(3,1,1)` split `(1, 2)`."""
    u_cycle3 = (
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(0)),
    )
    l_diag = (
        (Fraction(3, 5), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(6, 5), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(3, 5)),
    )
    g_scalar = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    g_nonscalar = (
        (Fraction(3), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    s_ul = nonscalar_krein_holonomy_detector(u_cycle3, l_diag, g_scalar, u_cycle3, order="UL")
    s_lu = nonscalar_krein_holonomy_detector(u_cycle3, l_diag, g_scalar, u_cycle3, order="LU")
    ns_ul = nonscalar_krein_holonomy_detector(u_cycle3, l_diag, g_nonscalar, u_cycle3, order="UL")
    ns_lu = nonscalar_krein_holonomy_detector(u_cycle3, l_diag, g_nonscalar, u_cycle3, order="LU")

    n_neg_s_ul = s_ul["block3_inertia"]["n_neg"]
    n_neg_s_lu = s_lu["block3_inertia"]["n_neg"]
    n_neg_ns_ul = ns_ul["block3_inertia"]["n_neg"]
    n_neg_ns_lu = ns_lu["block3_inertia"]["n_neg"]
    return {
        "scalar_I3_n_neg_UL": n_neg_s_ul,
        "scalar_I3_n_neg_LU": n_neg_s_lu,
        "scalar_I3_blind": bool(n_neg_s_ul == 1 and n_neg_s_lu == 1),
        "nonscalar_G3_n_neg_UL": n_neg_ns_ul,
        "nonscalar_G3_n_neg_LU": n_neg_ns_lu,
        "nonscalar_G3_splits_order": bool(n_neg_ns_ul == 1 and n_neg_ns_lu == 2),
        "phase_slot_order3_q": ns_ul["phase_slot_q"],
    }


# ---------------------------------------------------------------------------
# 5. SiliSense Exact-Q Health Verdict & 3-Tier Epoch-Hysteresis Controller
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiliSenseHealthSnapshotQ:
    """Exact-Q SiliSense `OP1` / `OP2` / `OP3` + 5-gate anti-Goodhart health verdict."""

    op1_leverage_q: Fraction
    op2_phi_asym_q: Fraction
    op3_neg_inertia: int
    ratchet_accepted: bool

    def __post_init__(self) -> None:
        lv = _req_fraction(self.op1_leverage_q, "op1_leverage_q", "ERR_HEALTH_Q")
        ph = _req_fraction(self.op2_phi_asym_q, "op2_phi_asym_q", "ERR_HEALTH_Q")
        require_ok(Fraction(0) <= lv <= Fraction(1), "ERR_HEALTH_RANGE", "op1_leverage_q must lie in [0, 1]")
        require_ok(Fraction(0) <= ph <= Fraction(1), "ERR_HEALTH_RANGE", "op2_phi_asym_q must lie in [0, 1]")
        ni = _req_int(self.op3_neg_inertia, "op3_neg_inertia", "ERR_HEALTH_INT")
        require_ok(0 <= ni <= 3, "ERR_HEALTH_RANGE", "op3_neg_inertia must lie in 0..3")
        require_ok(isinstance(self.ratchet_accepted, bool), "ERR_HEALTH_BOOL", "ratchet_accepted must be bool")

    @classmethod
    def from_ratchet_result(
        cls,
        *,
        op1_leverage_q: Fraction,
        op2_phi_asym_q: Fraction,
        op3_neg_inertia: int,
        ratchet_result: Mapping[str, Any],
    ) -> "SiliSenseHealthSnapshotQ":
        """Build a health snapshot directly from `FiveGateAntiGoodhartRatchet.evaluate(...)`."""
        require_ok(isinstance(ratchet_result, Mapping), "ERR_HEALTH_RATCHET", "ratchet_result must be a mapping")
        require_ok("accepted" in ratchet_result, "ERR_HEALTH_RATCHET", "ratchet_result missing 'accepted' key")
        return cls(
            op1_leverage_q=op1_leverage_q,
            op2_phi_asym_q=op2_phi_asym_q,
            op3_neg_inertia=op3_neg_inertia,
            ratchet_accepted=ratchet_result["accepted"],
        )


def compare_nvml_vs_razor_latency_q(
    nvml_period_us_q: Fraction = NVML_ZOH_PERIOD_US_Q,
    razor_period_us_q: Fraction = RAZOR_BER_PERIOD_US_Q,
) -> dict[str, Any]:
    """Exact rational comparison of `10 Hz` NVML ZOH vs `30 kHz` `.sys` BER canary."""
    tn = _req_fraction(nvml_period_us_q, "nvml_period_us_q", "ERR_LATENCY_Q")
    tr = _req_fraction(razor_period_us_q, "razor_period_us_q", "ERR_LATENCY_Q")
    require_ok(tn > 0 and tr > 0, "ERR_LATENCY_POS", "Periods must be > 0")
    speedup_q = tn / tr
    nvml_hz_q = Fraction(1_000_000, 1) / tn
    razor_hz_q = Fraction(1_000_000, 1) / tr
    nvml_max_ctrl_hz_q = nvml_hz_q * DELAY_BW_LIMIT_Q
    razor_max_ctrl_hz_q = razor_hz_q * DELAY_BW_LIMIT_Q
    blind_tokens_nvml_q = tn / tr
    return {
        "nvml_period_us_q": _frac_str(tn),
        "razor_period_us_q": _frac_str(tr),
        "detection_speedup_exact_q": _frac_str(speedup_q),
        "nvml_sample_hz_q": _frac_str(nvml_hz_q),
        "razor_sample_hz_q": _frac_str(razor_hz_q),
        "nvml_max_stable_ctrl_hz_q": _frac_str(nvml_max_ctrl_hz_q),
        "razor_max_stable_ctrl_hz_q": _frac_str(razor_max_ctrl_hz_q),
        "worst_case_blind_tokens_nvml_q": _frac_str(blind_tokens_nvml_q),
        "worst_case_blind_tokens_razor_q": "1/1",
    }


class HoloSphereRazorController:
    """Three-tier exact-Q self-healing Razor controller with `N=256`-token epoch hysteresis."""

    def __init__(
        self,
        *,
        base_profile: HoloSphereTuningProfile | None = None,
        initial_margin_mv_q: Fraction = Fraction(40, 1),
        min_margin_mv_q: Fraction = Fraction(0, 1),
        max_margin_mv_q: Fraction = Fraction(100, 1),
        initial_power_cap_w_q: Fraction = Fraction(100, 1),
        min_power_cap_w_q: Fraction = Fraction(70, 1),
        max_power_cap_w_q: Fraction = Fraction(108, 1),  # strictly < 110 W SwPowerCap cliff
        tier0_descent_mv_q: Fraction = Fraction(2, 1),
        tier0_power_step_w_q: Fraction = Fraction(1, 1),
        tier1_step_up_mv_q: Fraction = Fraction(12, 1),
        tier2_step_up_mv_q: Fraction = Fraction(25, 1),
        tier1_cooldown_ms_q: Fraction = Fraction(1, 2),
        tier2_cooldown_ms_q: Fraction = Fraction(2, 1),
        max_op1_leverage_q: Fraction = Fraction(1, 2),
        max_op2_phi_asym_q: Fraction = Fraction(1, 4),
        epoch_tokens: int = EPOCH_TOKENS,
        ewma_alpha_q: Fraction = HYSTERESIS_EWMA_ALPHA_Q,
        sec_step_down_count: int = HYSTERESIS_SEC_STEP_DOWN_COUNT,
        ewma_step_down_q: Fraction = HYSTERESIS_EWMA_STEP_DOWN_Q,
        hold_epochs: int = HYSTERESIS_HOLD_EPOCHS,
        step_up_clean_epochs: int = HYSTERESIS_STEP_UP_CLEAN_EPOCHS,
        ded_latch_clean_epochs: int = HYSTERESIS_DED_LATCH_CLEAN_EPOCHS,
    ) -> None:
        if base_profile is not None:
            require_ok(
                isinstance(base_profile, HoloSphereTuningProfile),
                "ERR_CTRL_PROFILE",
                "base_profile must be a HoloSphereTuningProfile",
            )
            self.base_profile = base_profile
        else:
            self.base_profile = HoloSphereTuningProfile(backend="cuda_sm86", block_t=128)
        self.base_profile.validate()

        for nm, qv in (
            ("initial_margin_mv_q", initial_margin_mv_q),
            ("min_margin_mv_q", min_margin_mv_q),
            ("max_margin_mv_q", max_margin_mv_q),
            ("initial_power_cap_w_q", initial_power_cap_w_q),
            ("min_power_cap_w_q", min_power_cap_w_q),
            ("max_power_cap_w_q", max_power_cap_w_q),
            ("tier0_descent_mv_q", tier0_descent_mv_q),
            ("tier0_power_step_w_q", tier0_power_step_w_q),
            ("tier1_step_up_mv_q", tier1_step_up_mv_q),
            ("tier2_step_up_mv_q", tier2_step_up_mv_q),
            ("tier1_cooldown_ms_q", tier1_cooldown_ms_q),
            ("tier2_cooldown_ms_q", tier2_cooldown_ms_q),
            ("max_op1_leverage_q", max_op1_leverage_q),
            ("max_op2_phi_asym_q", max_op2_phi_asym_q),
            ("ewma_alpha_q", ewma_alpha_q),
            ("ewma_step_down_q", ewma_step_down_q),
        ):
            _req_fraction(qv, nm, "ERR_CTRL_Q")

        et = _req_int(epoch_tokens, "epoch_tokens", "ERR_HYST_INT")
        sc = _req_int(sec_step_down_count, "sec_step_down_count", "ERR_HYST_INT")
        he = _req_int(hold_epochs, "hold_epochs", "ERR_HYST_INT")
        se = _req_int(step_up_clean_epochs, "step_up_clean_epochs", "ERR_HYST_INT")
        de = _req_int(ded_latch_clean_epochs, "ded_latch_clean_epochs", "ERR_HYST_INT")

        require_ok(
            Fraction(0) <= min_margin_mv_q <= initial_margin_mv_q <= max_margin_mv_q,
            "ERR_CTRL_MARGIN_BOUNDS",
            "Invalid margin bounds",
        )
        require_ok(
            Fraction(0) < min_power_cap_w_q <= initial_power_cap_w_q <= max_power_cap_w_q < Fraction(110, 1),
            "ERR_CTRL_POWER_BOUNDS",
            "Power cap bounds must stay strictly below 110 W SwPowerCap cliff",
        )
        require_ok(
            tier0_descent_mv_q > 0 and tier1_step_up_mv_q > tier0_descent_mv_q and tier2_step_up_mv_q > tier1_step_up_mv_q,
            "ERR_CTRL_STEPS",
            "Require 0 < tier0_descent < tier1_step_up < tier2_step_up (asymmetric safety recovery)",
        )
        require_ok(
            tier0_power_step_w_q > 0 and Fraction(0) <= tier1_cooldown_ms_q <= tier2_cooldown_ms_q <= MAX_COOLDOWN_MS_Q,
            "ERR_CTRL_COOLDOWN",
            "Invalid power step or cooldown bounds",
        )
        require_ok(
            et >= 1 and sc >= 1 and he >= 1 and se >= he and de >= se,
            "ERR_HYST_BOUNDS",
            "Require epoch_tokens>=1, sec_step_down_count>=1, 1<=hold_epochs<=step_up_clean_epochs<=ded_latch_clean_epochs",
        )
        require_ok(
            Fraction(0) < ewma_alpha_q <= Fraction(1) and Fraction(0) < ewma_step_down_q <= Fraction(1),
            "ERR_HYST_ALPHA",
            "ewma_alpha_q and ewma_step_down_q must lie in (0, 1]",
        )

        self.margin_mv_q = initial_margin_mv_q
        self.min_margin_mv_q = min_margin_mv_q
        self.max_margin_mv_q = max_margin_mv_q
        self.power_cap_w_q = initial_power_cap_w_q
        self.min_power_cap_w_q = min_power_cap_w_q
        self.max_power_cap_w_q = max_power_cap_w_q
        self.tier0_descent_mv_q = tier0_descent_mv_q
        self.tier0_power_step_w_q = tier0_power_step_w_q
        self.tier1_step_up_mv_q = tier1_step_up_mv_q
        self.tier2_step_up_mv_q = tier2_step_up_mv_q
        self.tier1_cooldown_ms_q = tier1_cooldown_ms_q
        self.tier2_cooldown_ms_q = tier2_cooldown_ms_q
        self.max_op1_leverage_q = max_op1_leverage_q
        self.max_op2_phi_asym_q = max_op2_phi_asym_q

        self.epoch_tokens = et
        self.ewma_alpha_q = ewma_alpha_q
        self.sec_step_down_count = sc
        self.ewma_step_down_q = ewma_step_down_q
        self.hold_epochs = he
        self.step_up_clean_epochs = se
        self.ded_latch_clean_epochs = de

        self.prev_sec_u32: int = 0
        self.prev_ded_u32: int = 0
        self.prev_non_lsb_u32: int = 0
        self.epoch_u32: int = 0
        self.ewma_ber_q16_u32: int = 0
        self.ewma_ber_q: Fraction = Fraction(0, 1)
        self.clean_streak: int = 0
        self.hold_remaining: int = 0
        self.ded_latch_remaining: int = 0

    def _consume_canary_deltas(
        self,
        canary: RazorCanary64,
        health: SiliSenseHealthSnapshotQ,
        rollback_buf: WeftHolonomyRollbackBuffer | None = None,
    ) -> tuple[int, int, int]:
        require_ok(isinstance(canary, RazorCanary64), "ERR_BAD_CANARY", "Expected RazorCanary64")
        require_ok(isinstance(health, SiliSenseHealthSnapshotQ), "ERR_BAD_HEALTH", "Expected SiliSenseHealthSnapshotQ")
        if rollback_buf is not None:
            require_ok(
                isinstance(rollback_buf, WeftHolonomyRollbackBuffer),
                "ERR_BAD_ROLLBACK_BUF",
                f"rollback_buf must be WeftHolonomyRollbackBuffer, got {type(rollback_buf).__name__}",
            )
        canary.validate()
        sec_delta = canary.sec_count_u32 - self.prev_sec_u32
        ded_delta = canary.ded_count_u32 - self.prev_ded_u32
        non_lsb_delta = canary.non_lsb_canary_u32 - self.prev_non_lsb_u32
        require_ok(sec_delta >= 0, "ERR_CANARY_NONMONOTONIC", f"sec_count regressed: {canary.sec_count_u32} < {self.prev_sec_u32}")
        require_ok(ded_delta >= 0, "ERR_CANARY_NONMONOTONIC", f"ded_count regressed: {canary.ded_count_u32} < {self.prev_ded_u32}")
        require_ok(
            non_lsb_delta >= 0,
            "ERR_CANARY_NONMONOTONIC",
            f"non_lsb_canary regressed: {canary.non_lsb_canary_u32} < {self.prev_non_lsb_u32}",
        )
        self.prev_sec_u32 = canary.sec_count_u32
        self.prev_ded_u32 = canary.ded_count_u32
        self.prev_non_lsb_u32 = canary.non_lsb_canary_u32
        return sec_delta, ded_delta, non_lsb_delta

    def _silisense_healthy(self, health: SiliSenseHealthSnapshotQ) -> bool:
        return (
            health.op1_leverage_q <= self.max_op1_leverage_q
            and health.op2_phi_asym_q <= self.max_op2_phi_asym_q
            and health.op3_neg_inertia == 0
            and health.ratchet_accepted
        )

    def _emit_step_result(
        self,
        *,
        tier: int,
        action: str,
        sec_delta: int,
        ded_delta: int,
        non_lsb_delta: int,
        cooldown_q: Fraction,
        canary: RazorCanary64,
        rollback_receipt: dict[str, Any] | None,
    ) -> dict[str, Any]:
        profile = HoloSphereTuningProfile(
            backend=self.base_profile.backend,
            block_t=self.base_profile.block_t,
            microbatch_size=self.base_profile.microbatch_size,
            inter_step_cooldown_ms=float(cooldown_q),
            r_loops_override=self.base_profile.r_loops_override,
        )
        return {
            "tier": tier,
            "action": action,
            "sec_delta": sec_delta,
            "ded_delta": ded_delta,
            "non_lsb_delta": non_lsb_delta,
            "epoch_u32": self.epoch_u32,
            "rotated_bitplane": a16_rotate_bitplane_idx(self.epoch_u32),
            "ewma_ber_q": _frac_str(self.ewma_ber_q),
            "ewma_ber_q16_u32": self.ewma_ber_q16_u32,
            "clean_streak": self.clean_streak,
            "hold_remaining": self.hold_remaining,
            "ded_latch_remaining": self.ded_latch_remaining,
            "margin_mv_q": _frac_str(self.margin_mv_q),
            "power_cap_w_q": _frac_str(self.power_cap_w_q),
            "cooldown_ms_q": _frac_str(cooldown_q),
            "profile": profile.to_dict(),
            "profile_sha256": profile.profile_sha256(),
            "canary_sha256": canary.canary_sha256(),
            "rollback_receipt": rollback_receipt,
        }

    def step(
        self,
        canary: RazorCanary64,
        health: SiliSenseHealthSnapshotQ,
        *,
        rollback_buf: WeftHolonomyRollbackBuffer | None = None,
    ) -> dict[str, Any]:
        """Immediate single-poll 3-tier Razor action that strictly respects active safety latches."""
        sec_delta, ded_delta, non_lsb_delta = self._consume_canary_deltas(canary, health, rollback_buf)
        rollback_receipt: dict[str, Any] | None = None
        if ded_delta > 0 or non_lsb_delta > 0:
            tier = 2
            action = "TIER2_DED_ROLLBACK_AND_MARGIN_UP"
            self.margin_mv_q = min(self.max_margin_mv_q, self.margin_mv_q + self.tier2_step_up_mv_q)
            self.power_cap_w_q = max(self.min_power_cap_w_q, self.power_cap_w_q - self.tier0_power_step_w_q)
            self.clean_streak = 0
            self.hold_remaining = max(self.hold_remaining, self.hold_epochs)
            self.ded_latch_remaining = self.ded_latch_clean_epochs
            cooldown_q = self.tier2_cooldown_ms_q
            if rollback_buf is not None:
                rollback_receipt = rollback_buf.rollback_to_last_clean_chunk()
        elif sec_delta > 0:
            tier = 1
            action = "TIER1_SEC_INLINE_HEALED_MARGIN_UP"
            self.margin_mv_q = min(self.max_margin_mv_q, self.margin_mv_q + self.tier1_step_up_mv_q)
            self.clean_streak = 0
            self.hold_remaining = max(self.hold_remaining, self.hold_epochs)
            if self.ded_latch_remaining > 0:
                self.ded_latch_remaining = self.ded_latch_clean_epochs
            cooldown_q = self.tier1_cooldown_ms_q
        else:
            tier = 0
            if self.ded_latch_remaining > 0:
                action = "TIER0_HOLD_DED_LATCH"
                cooldown_q = self.tier1_cooldown_ms_q
            elif self.hold_remaining > 0:
                action = "TIER0_HOLD_MIN_DWELL"
                cooldown_q = self.tier1_cooldown_ms_q
            elif not self._silisense_healthy(health):
                action = "TIER0_NOMINAL_HOLD_HEALTH_GATE"
                cooldown_q = self.tier1_cooldown_ms_q
            elif self.clean_streak < self.step_up_clean_epochs:
                action = "TIER0_HOLD_CLEAN_WARMUP"
                cooldown_q = Fraction(0, 1)
            else:
                action = "TIER0_NOMINAL_RAZOR_DESCENT"
                self.margin_mv_q = max(self.min_margin_mv_q, self.margin_mv_q - self.tier0_descent_mv_q)
                self.power_cap_w_q = max(self.min_power_cap_w_q, self.power_cap_w_q - self.tier0_power_step_w_q)
                self.clean_streak = 0
                self.hold_remaining = self.hold_epochs
                cooldown_q = Fraction(0, 1)
        return self._emit_step_result(
            tier=tier,
            action=action,
            sec_delta=sec_delta,
            ded_delta=ded_delta,
            non_lsb_delta=non_lsb_delta,
            cooldown_q=cooldown_q,
            canary=canary,
            rollback_receipt=rollback_receipt,
        )

    def step_epoch(
        self,
        canary: RazorCanary64,
        health: SiliSenseHealthSnapshotQ,
        *,
        rollback_buf: WeftHolonomyRollbackBuffer | None = None,
    ) -> dict[str, Any]:
        """Exact-Q `N=256`-token epoch hysteresis governor (Q16-quantized EWMA eliminating denominator blowup)."""
        sec_delta, ded_delta, non_lsb_delta = self._consume_canary_deltas(canary, health, rollback_buf)
        rate_q = Fraction(sec_delta, self.epoch_tokens)
        raw_ewma_q = (Fraction(1) - self.ewma_alpha_q) * self.ewma_ber_q + self.ewma_alpha_q * rate_q
        q16 = (raw_ewma_q.numerator * EWMA_Q16_SCALE) // raw_ewma_q.denominator
        self.ewma_ber_q16_u32 = min(U32_MAX, q16)
        self.ewma_ber_q = Fraction(self.ewma_ber_q16_u32, EWMA_Q16_SCALE)
        self.epoch_u32 += 1
        rollback_receipt: dict[str, Any] | None = None

        if ded_delta > 0 or non_lsb_delta > 0:
            tier = 2
            action = "TIER2_DED_ROLLBACK_LATCH64"
            self.margin_mv_q = min(self.max_margin_mv_q, self.margin_mv_q + self.tier2_step_up_mv_q)
            self.power_cap_w_q = max(self.min_power_cap_w_q, self.power_cap_w_q - self.tier0_power_step_w_q)
            self.clean_streak = 0
            self.hold_remaining = max(self.hold_remaining, self.hold_epochs)
            self.ded_latch_remaining = self.ded_latch_clean_epochs
            cooldown_q = self.tier2_cooldown_ms_q
            if rollback_buf is not None:
                rollback_receipt = rollback_buf.rollback_to_last_clean_chunk()
        elif sec_delta >= self.sec_step_down_count or self.ewma_ber_q >= self.ewma_step_down_q:
            tier = 1
            action = "TIER1_SEC_STEP_UP_HOLD4"
            self.margin_mv_q = min(self.max_margin_mv_q, self.margin_mv_q + self.tier1_step_up_mv_q)
            self.clean_streak = 0
            self.hold_remaining = self.hold_epochs
            if self.ded_latch_remaining > 0:
                self.ded_latch_remaining = self.ded_latch_clean_epochs
            cooldown_q = self.tier1_cooldown_ms_q
        elif sec_delta > 0:
            tier = 1
            action = "TIER1_SEC_DEADBAND_HOLD"
            self.clean_streak = 0
            if self.ded_latch_remaining > 0:
                self.ded_latch_remaining = self.ded_latch_clean_epochs
            cooldown_q = self.tier1_cooldown_ms_q
        else:
            tier = 0
            self.clean_streak += 1
            if self.ded_latch_remaining > 0:
                self.ded_latch_remaining -= 1
            if self.hold_remaining > 0:
                self.hold_remaining -= 1
            if self.ded_latch_remaining > 0:
                action = "TIER0_HOLD_DED_LATCH"
                cooldown_q = self.tier1_cooldown_ms_q
            elif self.hold_remaining > 0:
                action = "TIER0_HOLD_MIN_DWELL"
                cooldown_q = self.tier1_cooldown_ms_q
            elif not self._silisense_healthy(health):
                action = "TIER0_NOMINAL_HOLD_HEALTH_GATE"
                cooldown_q = self.tier1_cooldown_ms_q
            elif self.clean_streak < self.step_up_clean_epochs:
                action = "TIER0_HOLD_CLEAN_WARMUP"
                cooldown_q = Fraction(0, 1)
            else:
                action = "TIER0_NOMINAL_RAZOR_DESCENT"
                self.margin_mv_q = max(self.min_margin_mv_q, self.margin_mv_q - self.tier0_descent_mv_q)
                self.power_cap_w_q = max(self.min_power_cap_w_q, self.power_cap_w_q - self.tier0_power_step_w_q)
                self.clean_streak = 0
                self.hold_remaining = self.hold_epochs
                cooldown_q = Fraction(0, 1)

        return self._emit_step_result(
            tier=tier,
            action=action,
            sec_delta=sec_delta,
            ded_delta=ded_delta,
            non_lsb_delta=non_lsb_delta,
            cooldown_q=cooldown_q,
            canary=canary,
            rollback_receipt=rollback_receipt,
        )


def run_razor_selftest() -> dict[str, Any]:
    """Deterministic self-check certificate exercising SECDED, upper-bit closure, hysteresis, and Krein."""
    base_lsb = a16_encode_lsb16(0x5A3)
    clean_nibs = tuple((((i % 7) - 3) & ~1) | ((base_lsb >> i) & 1) for i in range(16))
    clean_nibs = tuple(v - 16 if v >= 8 else v for v in (x & 0xF for x in clean_nibs))
    crc16 = block_crc16_ccitt(clean_nibs)
    fold_bit = a16_folded_upper_parity_bit(clean_nibs)
    plane_synds = a16_plane_syndromes_4(clean_nibs)

    sec_all_healed = all(
        (res := a16_inline_secded_step(inject_nibble_bit_flip(clean_nibs, lane, 0), expected_crc16=crc16))["status"] == "SEC"
        and res["fixed_lane"] == lane
        and res["healed_nibbles"] == clean_nibs
        for lane in range(16)
    )

    ded_pairs = sum(
        1
        for i in range(16)
        for j in range(i + 1, 16)
        if (res := a16_inline_secded_step(inject_nibble_bit_flip(inject_nibble_bit_flip(clean_nibs, i, 0), j, 0)))["status"] == "DED"
        and res["ded_delta"] == 1
        and res["sec_delta"] == 0
    )

    lsb_only_missed_upper = 0
    folded_caught_upper = 0
    rotated_caught_upper = 0
    crc16_caught_upper = 0
    for lane in range(16):
        for bp in (1, 2, 3):
            cu = inject_nibble_bit_flip(clean_nibs, lane, bp)
            if a16_inline_secded_step(cu)["status"] == "CLEAN":
                lsb_only_missed_upper += 1
            if a16_inline_secded_step(cu, expected_upper_fold_bit=fold_bit)["non_lsb_delta"] == 1:
                folded_caught_upper += 1
            if a16_inline_secded_step(cu, epoch_u32=bp, expected_plane_syndromes=plane_synds)["non_lsb_delta"] == 1:
                rotated_caught_upper += 1
            if a16_inline_secded_step(cu, expected_crc16=crc16)["non_lsb_delta"] == 1:
                crc16_caught_upper += 1

    return {
        "schema": "silisense.razor_selftest.v1",
        "sec_16_of_16_inline_healed": sec_all_healed,
        "ded_120_of_120_pairs_detected": ded_pairs == 120,
        "upper_bits_1_3_lsb_only_missed_48_of_48": lsb_only_missed_upper == 48,
        "upper_bits_1_3_folded_parity_caught_48_of_48": folded_caught_upper == 48,
        "upper_bits_1_3_rotated_plane_caught_48_of_48": rotated_caught_upper == 48,
        "upper_bits_1_3_crc16_caught_48_of_48": crc16_caught_upper == 48,
        "hysteresis_spec": {
            "epoch_tokens": EPOCH_TOKENS,
            "ewma_alpha_q": _frac_str(HYSTERESIS_EWMA_ALPHA_Q),
            "ewma_step_down_q": _frac_str(HYSTERESIS_EWMA_STEP_DOWN_Q),
            "hold_epochs": HYSTERESIS_HOLD_EPOCHS,
            "step_up_clean_epochs": HYSTERESIS_STEP_UP_CLEAN_EPOCHS,
            "ded_latch_clean_epochs": HYSTERESIS_DED_LATCH_CLEAN_EPOCHS,
        },
        "krein_holonomy_order_witness": krein_holonomy_order_witness_q(),
        "latency_comparison": compare_nvml_vs_razor_latency_q(),
        "canary_struct_bytes": CANARY64_BYTES,
        "rollback_state_bytes": PI_ROLLBACK_BYTES,
    }


if __name__ == "__main__":
    print(json.dumps(run_razor_selftest(), indent=2, sort_keys=True))
