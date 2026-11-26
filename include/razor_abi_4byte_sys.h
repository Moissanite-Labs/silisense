/*
 * razor_abi_4byte_sys.h — Cacheline-Aligned 64 B D2H .sys Telemetry & 16 B Rollback ABI
 *
 * Bridges inline [16, 11, 4] A_16 Block-H4 SECDED BER detection (~30 kHz / 33.3 us/tok) +
 * rotating nibble bit-plane / folded bit1^bit2^bit3 upper-bit parity + N=256-token epoch
 * hysteresis from HoloSphere sm_86 CUDA kernels to Moissanite-Labs/silisense with ZERO
 * PCIe atomic traffic on clean cycles and drop-in offset-0 compatibility with a 4-byte
 * `uint32_t* host_ber_canary` pointer.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SILISENSE_RAZOR_ABI_4BYTE_SYS_H
#define SILISENSE_RAZOR_ABI_4BYTE_SYS_H

#include <stddef.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef __cplusplus
#define RAZOR_ALIGNAS(n) alignas(n)
#define RAZOR_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
extern "C" {
#else
#define RAZOR_ALIGNAS(n) _Alignas(n)
#define RAZOR_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
RAZOR_STATIC_ASSERT(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Razor wire ABI requires little-endian byte order");
#endif

#define HOLOSPHERE_RAZOR_CANARY_BYTES        64u
#define HOLOSPHERE_RAZOR_ROLLBACK_BYTES      16u
#define HOLOSPHERE_RAZOR_CHUNK_TOKENS        32u
#define HOLOSPHERE_RAZOR_EPOCH_TOKENS        256u
#define HOLOSPHERE_RAZOR_HOLD_EPOCHS         4u
#define HOLOSPHERE_RAZOR_STEP_UP_EPOCHS      16u
#define HOLOSPHERE_RAZOR_DED_LATCH_EPOCHS    64u
#define HOLOSPHERE_RAZOR_ABI_VERSION         2u

/*
 * Cacheline-aligned 64-byte D2H telemetry record placed in cacheable
 * `cudaHostAllocMapped` host memory (`cudaHostAllocDefault` / mapped, NOT WC).
 *
 * Offset 0 (`sec_count_u32`) is layout-compatible with a 4-byte `uint32_t*`
 * host BER canary pointer. GPU writes ONLY via `atom.release.sys.global.add.u32`
 * (strictly `.sys` scope) guarded behind `__ballot_sync(0xFFFFFFFFu, fault != 0u)`
 * -> 0 PCIe writes on clean cycles.
 */
typedef struct HoloSphereRazorCanary64 {
    RAZOR_ALIGNAS(64) uint32_t sec_count_u32;       /* [0..3]   Tier-1 1-bit LSB SEC count (synd != 0 && (synd & 1) == 1) */
    uint32_t ded_count_u32;                         /* [4..7]   Tier-2 2-bit LSB DED count (synd != 0 && (synd & 1) == 0) */
    uint32_t epoch_u32;                             /* [8..11]  Monotonic N=256-token epoch (plane b = epoch_u32 & 3u) */
    uint32_t rollback_token_u32;                    /* [12..15] Last clean C=32 chunk boundary token index t_0 */
    uint32_t non_lsb_canary_u32;                    /* [16..19] Upper-bit (bits 1..3) rotated/folded/CRC fault count -> Tier 2 */
    uint32_t last_syndrome_u32;                     /* [20..23] Diagnostic: (plane << 10) | (lane_grp << 8) | (is_ded << 5) | (synd & 0x1F) */
    uint32_t ewma_ber_q16_u32;                      /* [24..27] Q16 fixed-point EWMA BER (alpha = 1/32 over N=256-token epochs) */
    uint32_t clean_streak_u32;                      /* [28..31] Consecutive clean epochs (step-up gated on >= 16) */
    uint32_t hold_remaining_u32;                    /* [32..35] Remaining min-dwell hold epochs (4 epochs = 1,024 tokens) */
    uint32_t ded_latch_remaining_u32;               /* [36..39] Remaining DED rollback recovery latch epochs (64 epochs = 16,384 tokens) */
    uint32_t reserved_u32[6];                       /* [40..63] Explicit zero padding to 64 B (1 L1 cacheline) */
} HoloSphereRazorCanary64;

RAZOR_STATIC_ASSERT(sizeof(HoloSphereRazorCanary64) == 64u, "HoloSphereRazorCanary64 must be exactly 64 bytes");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, sec_count_u32) == 0u, "sec_count_u32 must be at offset 0");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, ded_count_u32) == 4u, "ded_count_u32 must be at offset 4");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, epoch_u32) == 8u, "epoch_u32 must be at offset 8");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, rollback_token_u32) == 12u, "rollback_token_u32 must be at offset 12");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, non_lsb_canary_u32) == 16u, "non_lsb_canary_u32 must be at offset 16");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, last_syndrome_u32) == 20u, "last_syndrome_u32 must be at offset 20");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, ewma_ber_q16_u32) == 24u, "ewma_ber_q16_u32 must be at offset 24");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, clean_streak_u32) == 28u, "clean_streak_u32 must be at offset 28");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, hold_remaining_u32) == 32u, "hold_remaining_u32 must be at offset 32");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, ded_latch_remaining_u32) == 36u, "ded_latch_remaining_u32 must be at offset 36");
RAZOR_STATIC_ASSERT(offsetof(HoloSphereRazorCanary64, reserved_u32) == 40u, "reserved_u32 must be at offset 40");

#ifdef __cplusplus
RAZOR_STATIC_ASSERT(alignof(HoloSphereRazorCanary64) == 64u, "HoloSphereRazorCanary64 must be 64-byte aligned");
#else
RAZOR_STATIC_ASSERT(_Alignof(HoloSphereRazorCanary64) == 64u, "HoloSphereRazorCanary64 must be 64-byte aligned");
#endif

/*
 * Exact 16-byte Weft SU(2) holonomy rollback snapshot Pi_{t_0} = (w, x, y, z) in S^3.
 * Snaps U_t back to the last clean C=32 chunk boundary in O(1) without KV-cache copy.
 */
typedef struct HoloSpherePiRollback16 {
    RAZOR_ALIGNAS(16) float w;
    float x;
    float y;
    float z;
} HoloSpherePiRollback16;

RAZOR_STATIC_ASSERT(sizeof(HoloSpherePiRollback16) == 16u, "HoloSpherePiRollback16 must be exactly 16 bytes");
RAZOR_STATIC_ASSERT(offsetof(HoloSpherePiRollback16, w) == 0u, "w offset must be 0");
RAZOR_STATIC_ASSERT(offsetof(HoloSpherePiRollback16, x) == 4u, "x offset must be 4");
RAZOR_STATIC_ASSERT(offsetof(HoloSpherePiRollback16, y) == 8u, "y offset must be 8");
RAZOR_STATIC_ASSERT(offsetof(HoloSpherePiRollback16, z) == 12u, "z offset must be 12");

#ifdef __cplusplus
} /* extern "C" */
#endif

/* --------------------------------------------------------------------------
 * Host-side acquire snapshot helper (C11 / C++17) for polling mapped D2H line.
 * Uses __atomic_load_n(..., __ATOMIC_ACQUIRE) directly on plain uint32_t words
 * (zero C11/C++17 atomic-pointer type-punning UB). Note: counters are loaded
 * per-field with acquire ordering rather than as one cross-field transaction;
 * monotonic counters remain exact lower bounds per poll.
 * -------------------------------------------------------------------------- */
typedef struct HoloSphereRazorSnapshot {
    uint32_t sec_count_u32;
    uint32_t ded_count_u32;
    uint32_t epoch_u32;
    uint32_t rollback_token_u32;
    uint32_t non_lsb_canary_u32;
    uint32_t last_syndrome_u32;
    uint32_t ewma_ber_q16_u32;
    uint32_t clean_streak_u32;
    uint32_t hold_remaining_u32;
    uint32_t ded_latch_remaining_u32;
} HoloSphereRazorSnapshot;

static inline uint32_t holosphere_razor_load_acquire_u32(const uint32_t* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
    {
        uint32_t v = *(const volatile uint32_t*)ptr;
        _ReadWriteBarrier();
        return v;
    }
#else
    return *(const volatile uint32_t*)ptr;
#endif
}

static inline HoloSphereRazorSnapshot holosphere_razor_poll_acquire(const HoloSphereRazorCanary64* line) {
    HoloSphereRazorSnapshot out;
    out.sec_count_u32           = holosphere_razor_load_acquire_u32(&line->sec_count_u32);
    out.ded_count_u32           = holosphere_razor_load_acquire_u32(&line->ded_count_u32);
    out.epoch_u32               = holosphere_razor_load_acquire_u32(&line->epoch_u32);
    out.rollback_token_u32      = holosphere_razor_load_acquire_u32(&line->rollback_token_u32);
    out.non_lsb_canary_u32      = holosphere_razor_load_acquire_u32(&line->non_lsb_canary_u32);
    out.last_syndrome_u32       = holosphere_razor_load_acquire_u32(&line->last_syndrome_u32);
    out.ewma_ber_q16_u32        = holosphere_razor_load_acquire_u32(&line->ewma_ber_q16_u32);
    out.clean_streak_u32        = holosphere_razor_load_acquire_u32(&line->clean_streak_u32);
    out.hold_remaining_u32      = holosphere_razor_load_acquire_u32(&line->hold_remaining_u32);
    out.ded_latch_remaining_u32 = holosphere_razor_load_acquire_u32(&line->ded_latch_remaining_u32);
    return out;
}

/* --------------------------------------------------------------------------
 * CUDA Device Helpers (sm_86+): [16, 11, 4] Inline SECDED + Vmin Upper-Bit
 * Rotation (b = epoch & 3) + Folded (bit1^bit2^bit3) Parity + .sys Canary
 * -------------------------------------------------------------------------- */
#ifdef __CUDACC__

__device__ __forceinline__ uint32_t razor_a16_rotated_bitplane(uint32_t epoch_u32) {
    return epoch_u32 & 3u;
}

__device__ __forceinline__ uint32_t razor_a16_extract_plane16(int8_t w_raw, uint32_t bit_plane, int lane) {
    uint32_t bit = (uint32_t(uint8_t(w_raw)) >> (bit_plane & 3u)) & 1u;
    uint32_t mask32 = __ballot_sync(0xFFFFFFFFu, bit != 0u);
    return (lane < 16) ? (mask32 & 0xFFFFu) : (mask32 >> 16);
}

/*
 * Side-channel folded upper-bit parity (bit1 ^ bit2 ^ bit3) across 16 lanes.
 * Closes the Vmin carry-path hole where droop flips nibble bits 1..3 without flipping LSB 0.
 */
__device__ __forceinline__ uint32_t razor_a16_folded_upper_parity16(int8_t w_raw, int lane) {
    uint32_t u = uint32_t(uint8_t(w_raw));
    uint32_t fold_bit = ((u >> 1) ^ (u >> 2) ^ (u >> 3)) & 1u;
    uint32_t mask32 = __ballot_sync(0xFFFFFFFFu, fold_bit != 0u);
    uint32_t w16 = (lane < 16) ? (mask32 & 0xFFFFu) : (mask32 >> 16);
    return uint32_t(__popc(w16)) & 1u;
}

__device__ __forceinline__ uint32_t razor_a16_secded_syndrome(uint32_t w16) {
    uint32_t p0 = uint32_t(__popc(w16 & 0xFFFFu)) & 1u;
    uint32_t p1 = uint32_t(__popc(w16 & 0xAAAAu)) & 1u;
    uint32_t p2 = uint32_t(__popc(w16 & 0xCCCCu)) & 1u;
    uint32_t p3 = uint32_t(__popc(w16 & 0xF0F0u)) & 1u;
    uint32_t p4 = uint32_t(__popc(w16 & 0xFF00u)) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

/*
 * Inline 1-bit LSB SECDED heal BEFORE IMMA.16832:
 *   - synd == 0:             CLEAN (0 flips)
 *   - (synd & 1u) == 1u:     SEC   (1-bit LSB error at lane j = (synd >> 1) & 0xFu -> flip LSB!)
 *   - synd != 0 && !(s & 1): DED   (2-bit LSB error -> do NOT miscorrect; trip Tier 2 rollback!)
 */
__device__ __forceinline__ int8_t razor_a16_inline_secded_fix_lane(
    int8_t w_raw,
    uint32_t synd,
    uint32_t lane16
) {
    if ((synd & 1u) != 0u) {
        uint32_t j = (synd >> 1) & 0xFu;
        if (lane16 == j) {
            w_raw = int8_t(int(w_raw) ^ 1);
        }
    }
    return w_raw;
}

__device__ __forceinline__ void razor_atom_release_sys_add_u32(uint32_t* host_ptr, uint32_t delta) {
    asm volatile(
        "atom.release.sys.global.add.u32 _, [%0], %1;"
        :
        : "l"(host_ptr), "r"(delta)
        : "memory"
    );
}

__device__ __forceinline__ void razor_st_release_sys_u32(uint32_t* host_ptr, uint32_t val) {
    asm volatile(
        "st.release.sys.global.u32 [%0], %1;"
        :
        : "l"(host_ptr), "r"(val)
        : "memory"
    );
}

/*
 * Warp-aggregated zero-overhead D2H .sys canary emitter (up to 4 .sys release ops:
 * 1..3 conditional atom.release.sys + 1 st.release.sys on fault cycles).
 * Clean cycles execute ONLY __ballot_sync and exit -> ZERO PCIe traffic.
 */
__device__ __forceinline__ void razor_canary_emit_sys(
    HoloSphereRazorCanary64* host_canary,
    uint32_t lane_sec_fault,
    uint32_t lane_ded_fault,
    uint32_t lane_non_lsb_fault,
    uint32_t diag_word,
    int lane
) {
    uint32_t sec_mask     = __ballot_sync(0xFFFFFFFFu, lane_sec_fault != 0u);
    uint32_t ded_mask     = __ballot_sync(0xFFFFFFFFu, lane_ded_fault != 0u);
    uint32_t non_lsb_mask = __ballot_sync(0xFFFFFFFFu, lane_non_lsb_fault != 0u);
    uint32_t any_mask     = sec_mask | ded_mask | non_lsb_mask;
    if (any_mask != 0u && lane == 0 && host_canary != nullptr) {
        if (sec_mask != 0u) {
            razor_atom_release_sys_add_u32(&host_canary->sec_count_u32, uint32_t(__popc(sec_mask)));
        }
        if (ded_mask != 0u) {
            razor_atom_release_sys_add_u32(&host_canary->ded_count_u32, uint32_t(__popc(ded_mask)));
        }
        if (non_lsb_mask != 0u) {
            razor_atom_release_sys_add_u32(&host_canary->non_lsb_canary_u32, uint32_t(__popc(non_lsb_mask)));
        }
        razor_st_release_sys_u32(&host_canary->last_syndrome_u32, diag_word);
    }
}

#endif /* __CUDACC__ */

#endif /* SILISENSE_RAZOR_ABI_4BYTE_SYS_H */
