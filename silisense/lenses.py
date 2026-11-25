"""Pure-stdlib S113 non-textbook silicon lenses, S20c exact-Q operators, and 7-tone Fisher ID.

Runs 100% on stock Python stdlib (`math`, `cmath`, `fractions.Fraction`, seeded
`random.Random`) with zero mandatory third-party packages:
  - S113 Non-Textbook Lenses:
      * A3: Microscopic third-order time-reversal shockwave asymmetry + Westfall-Young max-|z| IAAFT
      * A5: Causal EMA positive-innovation precursor lead-lag + circular-shift null
      * A7: 2-pass Modified Gram-Schmidt multi-lag Schur residualization + Miller-Madow copula MI
      * A1 (Mode-2 gated): Kim-Powers [6, 23] kHz ultrasonic auto-bicoherence
      * A2 (Mode-2 gated): Nolte [18.5, 22.2] kHz imaginary coherency + parabolic GCC-PHAT VRM TDOA
  - S20c Zero-Float Exact-Q Monotone-Invariant Trio + WHT_PHASE:
      * OP1_SEAM1: Sliding W=32 exact Fraction Sherman-Morrison leverage shock h_t = q_t / (1 + q_t)
      * OP2_SEAM3: Exact m=3 7x7 ordinal Hodge-curl Phi_asym in Q cap [0, 1] + O(N) HVG beta_1
      * OP3_F3: Exact 3x3 rational non-normality commutator kappa_F3 + Sylvester In(I_3 - T^T T)
      * wht_phase_surrogate: Exact Walsh-Hadamard sequency-preserving sign randomization (paired w/ IAAFT)
  - ASKA-XFER-H1 Delay-Locked 7-Tone Fisher Impedance Collinearity Detector:
      * Refuses free transport delay (ERR_FREE_DELAY_FORBIDDEN); strips A5-locked tau_d and
        evaluates all 35 (7 choose 3) collinearity determinants det[1, -Im H / u, Re H].
"""

from __future__ import annotations

import cmath
from fractions import Fraction
import math
import random
from typing import Any, Sequence

from silisense.airlock import (
    AudioAuthorizationGate,
    check_dmic_permission,
    is_camera_or_ir_identifier,
)
from silisense.errors import SiliSensePermissionError, require_ok, require_perm

PREREG_S115_ID = "PREREG-S115-SILICON-PHYSICS-FEEDBACK-OPTIMIZER-V1"

_FFT_CACHE: dict[tuple[int, bool], tuple[list[int], list[list[complex]]]] = {}


def _get_fft_plan(n: int, invert: bool) -> tuple[list[int], list[list[complex]]]:
    key = (n, invert)
    cached = _FFT_CACHE.get(key)
    if cached is not None:
        return cached
    require_ok(n >= 2 and (n & (n - 1)) == 0, "ERR_FFT_POW2", f"Radix-2 FFT requires power of 2, got {n}")
    bits = n.bit_length() - 1
    bitrev = [0] * n
    for i in range(n):
        rev = 0
        val = i
        for _ in range(bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        bitrev[i] = rev
    sign = 2.0 * math.pi if invert else -2.0 * math.pi
    stages: list[list[complex]] = []
    length = 2
    while length <= n:
        half = length >> 1
        ang = sign / float(length)
        stages.append([cmath.rect(1.0, ang * float(k)) for k in range(half)])
        length <<= 1
    _FFT_CACHE[key] = (bitrev, stages)
    return bitrev, stages


def fft_radix2(seq: Sequence[complex | float], invert: bool = False) -> list[complex]:
    """Iterative pure-stdlib Radix-2 Cooley-Tukey FFT / IFFT with cached twiddles."""
    n = len(seq)
    bitrev, stages = _get_fft_plan(n, invert)
    a = [complex(seq[bitrev[i]]) for i in range(n)]
    length = 2
    for twiddles in stages:
        half = length >> 1
        for i in range(0, n, length):
            for j in range(half):
                u = a[i + j]
                v = a[i + j + half] * twiddles[j]
                a[i + j] = u + v
                a[i + j + half] = u - v
        length <<= 1
    if invert:
        inv_n = 1.0 / float(n)
        return [z * inv_n for z in a]
    return a


def _clean_1d_floats(seq: Sequence[float], name: str, min_len: int = 16) -> list[float]:
    n = len(seq)
    require_ok(n >= min_len, "ERR_TOO_SHORT", f"{name} length {n} < {min_len}")
    out: list[float] = []
    for v in seq:
        require_ok(isinstance(v, (int, float)) and not isinstance(v, bool), "ERR_TYPE", f"{name} element not numeric")
        fv = float(v)
        require_ok(math.isfinite(fv), "ERR_NONFINITE", f"{name} contains NaN or Inf")
        out.append(fv)
    return out


def _mean_std(vals: Sequence[float], ddof: int = 0) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / float(n)
    if n <= ddof:
        return mu, 0.0
    var = sum((v - mu) * (v - mu) for v in vals) / float(n - ddof)
    return mu, math.sqrt(max(0.0, var))


def holm_bonferroni(pvals: Sequence[float]) -> list[float]:
    """Exact step-down Holm-Bonferroni multiple-testing adjustment."""
    m = len(pvals)
    if m == 0:
        return []
    for p in pvals:
        require_ok(math.isfinite(float(p)) and 0.0 <= float(p) <= 1.0, "ERR_BAD_PVAL", f"Invalid p-value {p}")
    order = sorted(range(m), key=lambda idx: float(pvals[idx]))
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = min(1.0, float(pvals[idx]) * float(m - rank))
        if scaled > running_max:
            running_max = scaled
        adjusted[idx] = running_max
    return adjusted


def iaaft_surrogates_1d(
    series: Sequence[float],
    n_surr: int,
    rng: random.Random,
    max_iter: int = 10,
) -> list[list[float]]:
    """Pure-stdlib Iterative Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz 1996)."""
    x = _clean_1d_floats(series, "iaaft_input", min_len=16)
    require_ok(n_surr >= 1, "ERR_BAD_NSURR", f"n_surr must be >= 1, got {n_surr}")
    n = len(x)
    mu, sd = _mean_std(x, ddof=0)
    if sd <= 1e-15:
        return [list(x) for _ in range(n_surr)]

    sorted_x = sorted(x)
    n_fft = 1 << (n - 1).bit_length()
    padded = [v - mu for v in x] + [0.0] * (n_fft - n)
    orig_spec = fft_radix2(padded, invert=False)
    target_amp = [abs(z) for z in orig_spec]
    half = n_fft >> 1

    surrogates: list[list[float]] = []
    for _ in range(n_surr):
        cur = list(x)
        rng.shuffle(cur)
        prev_ranks: list[int] | None = None
        for _it in range(max_iter):
            c_mu = sum(cur) / float(n)
            c_pad = [v - c_mu for v in cur] + [0.0] * (n_fft - n)
            spec = fft_radix2(c_pad, invert=False)
            new_spec = [0j] * n_fft
            new_spec[0] = complex(orig_spec[0].real, 0.0)
            new_spec[half] = complex(target_amp[half] if spec[half].real >= 0.0 else -target_amp[half], 0.0)
            for k in range(1, half):
                sk = spec[k]
                mag = abs(sk)
                zc = (sk / mag) * target_amp[k] if mag > 1e-30 else complex(target_amp[k], 0.0)
                new_spec[k] = zc
                new_spec[n_fft - k] = zc.conjugate()
            y_full = fft_radix2(new_spec, invert=True)
            y = [y_full[i].real for i in range(n)]
            order = sorted(range(n), key=lambda idx: y[idx])
            ranks = [0] * n
            for r_pos, idx in enumerate(order):
                ranks[idx] = r_pos
            cur = [sorted_x[ranks[i]] for i in range(n)]
            if prev_ranks is not None and ranks == prev_ranks:
                break
            prev_ranks = ranks
        surrogates.append(cur)
    return surrogates


# =============================================================================
# S113 Non-Textbook Silicon Lenses (A3, A5, A7 Default + Mode-2 A1, A2)
# =============================================================================


def _third_order_asymmetry(x: Sequence[float], lag: int) -> float:
    m = len(x) - lag
    if m <= 0:
        return 0.0
    s2 = 0.0
    s3 = 0.0
    for i in range(m):
        d = x[i + lag] - x[i]
        d2 = d * d
        s2 += d2
        s3 += d2 * d
    var = s2 / float(m)
    if var <= 1e-24:
        return 0.0
    return (s3 / float(m)) / (var ** 1.5)


def compute_a3_time_reversal_asymmetry(
    series: Sequence[float],
    *,
    dt_ms: float = 5.0,
    lags: Sequence[int] = (1, 2, 3),
    n_iaaft: int = 249,
    seed: int = 42,
) -> dict[str, Any]:
    """Lens A3: Microscopic third-order time-reversal asymmetry + Westfall-Young max-|z| IAAFT."""
    x = _clean_1d_floats(series, "a3_series", min_len=32)
    require_ok(dt_ms > 0.0 and math.isfinite(dt_ms), "ERR_BAD_DT", "dt_ms must be > 0")
    require_ok(len(lags) >= 1, "ERR_EMPTY_LAGS", "At least one lag required")
    for lag in lags:
        require_ok(isinstance(lag, int) and 1 <= lag < len(x) // 4, "ERR_BAD_LAG", f"Invalid lag {lag}")

    rng = random.Random(int(seed))
    surr = iaaft_surrogates_1d(x, n_iaaft, rng)

    per_lag: list[dict[str, Any]] = []
    raw_pvals: list[float] = []
    surr_abs_z: list[list[float]] = [[0.0] * len(lags) for _ in range(n_iaaft)]

    for l_idx, lag in enumerate(lags):
        obs = _third_order_asymmetry(x, lag)
        null_vals = [_third_order_asymmetry(s_row, lag) for s_row in surr]
        mu, sd = _mean_std(null_vals, ddof=1 if n_iaaft > 1 else 0)
        z_val = 0.0 if sd <= 1e-15 else (obs - mu) / sd
        if sd > 1e-15:
            for s_i, nv in enumerate(null_vals):
                surr_abs_z[s_i][l_idx] = abs(nv - mu) / sd
        ge_cnt = sum(1 for nv in null_vals if abs(nv - mu) >= abs(obs - mu) - 1e-15)
        p_emp = float(1 + ge_cnt) / float(1 + n_iaaft)
        raw_pvals.append(p_emp)
        per_lag.append(
            {
                "lag_samples": int(lag),
                "lag_ms": round(float(lag) * float(dt_ms), 6),
                "asymmetry": round(obs, 6),
                "abs_asymmetry": round(abs(obs), 6),
                "iaaft_mean": round(mu, 6),
                "iaaft_std": round(sd, 6),
                "z_iaaft": round(z_val, 6),
                "p_emp": round(p_emp, 6),
            }
        )

    for row, ph in zip(per_lag, holm_bonferroni(raw_pvals)):
        row["p_holm"] = round(ph, 6)

    best = max(per_lag, key=lambda r: abs(float(r["z_iaaft"])))
    obs_max_z = abs(float(best["z_iaaft"]))
    ge_max_t = sum(1 for row_z in surr_abs_z if max(row_z) >= obs_max_z - 1e-12)
    p_max_t = float(1 + ge_max_t) / float(1 + n_iaaft)

    return {
        "lens": "A3_time_reversal_asymmetry",
        "n_samples": len(x),
        "n_iaaft": int(n_iaaft),
        "dominant_lag_samples": best["lag_samples"],
        "dominant_lag_ms": best["lag_ms"],
        "asymmetry": best["asymmetry"],
        "abs_asymmetry": best["abs_asymmetry"],
        "z_iaaft": best["z_iaaft"],
        "p_emp": best["p_emp"],
        "p_holm": round(p_max_t, 6),
        "significant_0p005": bool(p_max_t < 0.005 and obs_max_z >= 3.5),
        "per_lag": per_lag,
    }


def causal_innovation_zscores(series: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Strictly causal O(N) positive-step innovation z-score clipped to [-15, 15]."""
    x = _clean_1d_floats(series, "innovation_series", min_len=16)
    require_ok(0.0 < alpha < 1.0, "ERR_BAD_ALPHA", f"alpha must lie in (0, 1), got {alpha}")
    n = len(x)
    pos_step = [0.0] + [max(0.0, x[t] - x[t - 1]) for t in range(1, n)]
    warmup = min(12, n // 2)
    mu, sd0 = _mean_std(pos_step[1 : warmup + 1], ddof=0)
    var = max(sd0 * sd0, 1e-4)

    z_out = [0.0] * n
    for t in range(1, n):
        val = pos_step[t]
        if t >= warmup:
            sd = math.sqrt(max(var, 1e-8))
            raw_z = 0.0 if sd <= 1e-6 else (val - mu) / sd
            z_out[t] = max(-15.0, min(15.0, raw_z))
        diff = val - mu
        mu = (1.0 - alpha) * mu + alpha * val
        var = (1.0 - alpha) * var + alpha * (diff * diff)
    return z_out


def compute_a5_precursor_lead_lag(
    fast_series: Sequence[float],
    slow_series: Sequence[float],
    *,
    dt_ms: float = 5.0,
    max_lag_ms: float = 150.0,
    z_thresh: float = 2.5,
    n_shifts: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Lens A5: Causal fast-rail innovation precursor lead-lag vs circular-shift null."""
    f_arr = _clean_1d_floats(fast_series, "a5_fast_series", min_len=64)
    s_arr = _clean_1d_floats(slow_series, "a5_slow_series", min_len=64)
    require_ok(len(f_arr) == len(s_arr), "ERR_SHAPE_MISMATCH", "fast and slow series must match length")
    require_ok(dt_ms > 0.0 and max_lag_ms >= dt_ms, "ERR_BAD_LAG_MS", "Invalid dt_ms or max_lag_ms")
    require_ok(n_shifts >= 50, "ERR_TOO_FEW_SHIFTS", "n_shifts must be >= 50")

    n = len(f_arr)
    max_lag = max(1, min(int(round(max_lag_ms / dt_ms)), n // 6))
    guard = max(2 * max_lag + 2, n // 10)
    require_ok(n > 2 * guard + 8, "ERR_SERIES_TOO_SHORT", f"Need length > {2 * guard + 8}, got {n}")

    zf = causal_innovation_zscores(f_arr)
    zs = causal_innovation_zscores(s_arr)
    mu_f, sd_f = _mean_std(zf, ddof=0)
    mu_s, sd_s = _mean_std(zs, ddof=0)
    if sd_f <= 1e-12 or sd_s <= 1e-12:
        return {
            "lens": "A5_precursor_lead_lag",
            "n_samples": n,
            "peak_precursor_lag_samples": -1,
            "peak_precursor_lag_ms": round(-dt_ms, 4),
            "peak_precursor_xcorr": 0.0,
            "precursor_event_rate": 0.0,
            "z_circ": 0.0,
            "p_circ": 1.0,
            "significant_0p005": False,
        }

    zf_n = [(v - mu_f) / sd_f for v in zf]
    zs_n = [(v - mu_s) / sd_s for v in zs]

    def _max_precursor_corr(f_vec: list[float]) -> tuple[float, int]:
        best_c = -1e9
        best_k = 1
        for k in range(1, max_lag + 1):
            acc = 0.0
            limit = n - k
            for i in range(limit):
                acc += f_vec[i] * zs_n[i + k]
            c = acc / float(limit)
            if c > best_c:
                best_c = c
                best_k = k
        return best_c, best_k

    obs_peak, best_lag_k = _max_precursor_corr(zf_n)

    rng = random.Random(int(seed))
    null_peaks: list[float] = []
    for _ in range(n_shifts):
        sh = rng.randint(guard, n - guard - 1)
        shifted = zf_n[-sh:] + zf_n[:-sh]
        c_sh, _ = _max_precursor_corr(shifted)
        null_peaks.append(c_sh)

    n_mu, n_sd = _mean_std(null_peaks, ddof=1)
    z_circ = 0.0 if n_sd <= 1e-15 else (obs_peak - n_mu) / n_sd
    ge_cnt = sum(1 for v in null_peaks if v >= obs_peak - 1e-15)
    p_circ = float(1 + ge_cnt) / float(1 + n_shifts)

    fast_idx = [i for i, v in enumerate(zf) if v >= z_thresh]
    slow_idx = [i for i, v in enumerate(zs) if v >= z_thresh]
    matched_slow = 0
    for ts in slow_idx:
        if any((ts - max_lag) <= tf <= (ts - 1) for tf in fast_idx):
            matched_slow += 1
    ev_rate = float(matched_slow) / float(len(slow_idx)) if slow_idx else 0.0

    return {
        "lens": "A5_precursor_lead_lag",
        "n_samples": n,
        "peak_precursor_lag_samples": -int(best_lag_k),
        "peak_precursor_lag_ms": round(-float(best_lag_k) * float(dt_ms), 4),
        "peak_precursor_xcorr": round(obs_peak, 6),
        "precursor_event_rate": round(ev_rate, 6),
        "z_circ": round(z_circ, 6),
        "p_circ": round(p_circ, 6),
        "significant_0p005": bool(p_circ < 0.005 and z_circ >= 3.0),
    }


def _mgs_project_orthogonal(y: list[float], cols: list[list[float]]) -> list[float]:
    """2-pass Modified Gram-Schmidt orthogonal projection (DGKS re-orthogonalization)."""
    m = len(y)
    basis: list[list[float]] = []
    for col in cols:
        v = list(col)
        for _pass in range(2):
            for q in basis:
                dot = sum(v[i] * q[i] for i in range(m))
                for i in range(m):
                    v[i] -= dot * q[i]
        norm = math.sqrt(sum(vi * vi for vi in v))
        if norm > 1e-12:
            inv = 1.0 / norm
            basis.append([vi * inv for vi in v])

    res = list(y)
    for _pass in range(2):
        for q in basis:
            dot = sum(res[i] * q[i] for i in range(m))
            for i in range(m):
                res[i] -= dot * q[i]
    return res


def _quantile_copula_mi(bin_u: list[int], ry: Sequence[float], k_bins: int, h_u: float, h_v: float) -> float:
    m = len(bin_u)
    order = sorted(range(m), key=lambda idx: ry[idx])
    bin_v = [0] * m
    for r_pos, idx in enumerate(order):
        bin_v[idx] = (r_pos * k_bins) // m
    counts = [0] * (k_bins * k_bins)
    for i in range(m):
        counts[bin_u[i] * k_bins + bin_v[i]] += 1
    inv_m = 1.0 / float(m)
    h_uv = 0.0
    for c in counts:
        if c > 0:
            p = float(c) * inv_m
            h_uv -= p * math.log(p)
    mm_bias = float((k_bins - 1) * (k_bins - 1)) / float(2 * m)
    return (h_u + h_v - h_uv) - mm_bias


def compute_a7_nonlinear_uncore_copula_mi(
    uncore_w: Sequence[float],
    target_series: Sequence[float],
    *,
    max_linear_lag: int = 5,
    n_bins: int | None = None,
    n_iaaft: int = 249,
    seed: int = 42,
) -> dict[str, Any]:
    """Lens A7: 2-pass MGS multi-lag Schur residualization + rank-copula MI vs IAAFT."""
    u = _clean_1d_floats(uncore_w, "a7_uncore_w", min_len=48)
    y = _clean_1d_floats(target_series, "a7_target_series", min_len=48)
    require_ok(len(u) == len(y), "ERR_SHAPE_MISMATCH", "uncore_w and target_series must match length")
    require_ok(0 <= max_linear_lag <= len(u) // 8, "ERR_BAD_MAX_LAG", f"Invalid max_linear_lag={max_linear_lag}")

    n = len(u)
    lag = int(max_linear_lag)
    m = n - 2 * lag
    u_mid = u[lag : n - lag]
    y_mid = y[lag : n - lag]
    _, sd_u = _mean_std(u_mid)
    _, sd_y = _mean_std(y_mid)
    if sd_u <= 1e-14 or sd_y <= 1e-14:
        return {
            "lens": "A7_nonlinear_uncore_copula_mi",
            "n_interior": m,
            "n_bins": 3,
            "max_abs_residual_linear_corr": 0.0,
            "copula_mi_nats": 0.0,
            "z_iaaft": 0.0,
            "p_emp": 1.0,
            "significant_0p005": False,
        }

    cols = [[1.0] * m] + [u[lag + k : n - lag + k] for k in range(-lag, lag + 1)]
    r_y = _mgs_project_orthogonal(y_mid, cols)

    mu_ry, sd_ry = _mean_std(r_y)
    max_lin_corr = 0.0
    if sd_ry > 1e-14:
        for k in range(-lag, lag + 1):
            uk = u[lag + k : n - lag + k]
            mu_uk, sd_uk = _mean_std(uk)
            if sd_uk > 1e-14:
                c_val = abs(sum((r_y[i] - mu_ry) * (uk[i] - mu_uk) for i in range(m)) / (float(m) * sd_ry * sd_uk))
                if c_val > max_lin_corr:
                    max_lin_corr = c_val

    require_ok(max_lin_corr <= 1e-9, "ERR_SCHUR_RESIDUAL_NOT_ORTHOGONAL", f"Residual corr {max_lin_corr} > 1e-9")

    k_bins = int(n_bins) if n_bins is not None else max(3, min(6, int(math.sqrt(float(m) / 8.0))))
    order_u = sorted(range(m), key=lambda idx: u_mid[idx])
    bin_u = [0] * m
    for r_pos, idx in enumerate(order_u):
        bin_u[idx] = (r_pos * k_bins) // m

    m_counts = [0] * k_bins
    for b in bin_u:
        m_counts[b] += 1
    h_marg = -sum((float(c) / float(m)) * math.log(float(c) / float(m)) for c in m_counts if c > 0)

    obs_raw = _quantile_copula_mi(bin_u, r_y, k_bins, h_marg, h_marg)
    rng = random.Random(int(seed))
    surr_ry = iaaft_surrogates_1d(r_y, n_iaaft, rng)
    null_mi = [_quantile_copula_mi(bin_u, s_row, k_bins, h_marg, h_marg) for s_row in surr_ry]

    mu_n, sd_n = _mean_std(null_mi, ddof=1 if n_iaaft > 1 else 0)
    z_val = 0.0 if sd_n <= 1e-15 else (obs_raw - mu_n) / sd_n
    ge_cnt = sum(1 for v in null_mi if v >= obs_raw - 1e-15)
    p_emp = float(1 + ge_cnt) / float(1 + n_iaaft)
    mi_pos = max(0.0, obs_raw)

    return {
        "lens": "A7_nonlinear_uncore_copula_mi",
        "n_interior": m,
        "n_bins": k_bins,
        "max_abs_residual_linear_corr": round(max_lin_corr, 12),
        "copula_mi_nats": round(mi_pos, 6),
        "copula_mi_bits": round(mi_pos / math.log(2.0), 6),
        "z_iaaft": round(z_val, 6),
        "p_emp": round(p_emp, 6),
        "significant_0p005": bool(p_emp <= 0.005 and z_val >= 3.5),
    }


def compute_a1_ultrasonic_bicoherence(
    dmic_wave: Sequence[float],
    *,
    fs: int = 48000,
    f_min_hz: float = 6000.0,
    f_max_hz: float = 23000.0,
    n_fft: int = 128,
    n_iaaft: int = 69,
    seed: int = 42,
    auth_gate: AudioAuthorizationGate | None = None,
    allow_offline_array: bool = False,
) -> dict[str, Any]:
    """Lens A1 (Mode-2 Opt-In): Pure-stdlib Kim-Powers ultrasonic auto-bicoherence."""
    check_dmic_permission(auth_gate, allow_offline_array)
    x = _clean_1d_floats(dmic_wave, "a1_dmic_wave", min_len=max(4 * n_fft, 512))
    require_ok((n_fft & (n_fft - 1)) == 0 and n_fft >= 64, "ERR_A1_NFFT", "n_fft must be power of 2 >= 64")

    df = float(fs) / float(n_fft)
    valid_bins = [k for k in range(n_fft // 2 + 1) if f_min_hz <= float(k) * df <= f_max_hz]
    require_ok(len(valid_bins) >= 4, "ERR_A1_BAND", "Too few bins in ultrasonic band")
    k_max = valid_bins[-1]
    triads = [(k1, k2, k1 + k2) for k1 in valid_bins for k2 in valid_bins if k2 <= k1 and (k1 + k2) <= k_max]
    require_ok(len(triads) >= 1, "ERR_A1_TRIADS", "Empty bicoherence triangle")

    hann = [0.5 * (1.0 - math.cos(2.0 * math.pi * float(i) / float(n_fft - 1))) for i in range(n_fft)]

    def _max_b2(sig: Sequence[float]) -> tuple[float, tuple[int, int, int]]:
        n_seg = len(sig) // n_fft
        specs: list[list[complex]] = []
        for s in range(n_seg):
            seg = sig[s * n_fft : (s + 1) * n_fft]
            mu_s = sum(seg) / float(n_fft)
            win_seg = [(seg[i] - mu_s) * hann[i] for i in range(n_fft)]
            specs.append(fft_radix2(win_seg, invert=False))
        best_val = 0.0
        best_tri = triads[0]
        inv_s = 1.0 / float(n_seg)
        for k1, k2, ks in triads:
            num = 0j
            d12 = 0.0
            d3 = 0.0
            for sp in specs:
                p12 = sp[k1] * sp[k2]
                s3 = sp[ks]
                num += p12 * s3.conjugate()
                d12 += (p12.real * p12.real + p12.imag * p12.imag)
                d3 += (s3.real * s3.real + s3.imag * s3.imag)
            denom = (d12 * inv_s) * (d3 * inv_s)
            if denom > 1e-30:
                b2 = ((num.real * num.real + num.imag * num.imag) * inv_s * inv_s) / denom
                if b2 > best_val:
                    best_val = b2
                    best_tri = (k1, k2, ks)
        return best_val, best_tri

    obs_max, best_tri = _max_b2(x)
    rng = random.Random(int(seed))
    surr = iaaft_surrogates_1d(x, n_iaaft, rng, max_iter=12)
    null_max = [_max_b2(s_row)[0] for s_row in surr]
    mu_n, sd_n = _mean_std(null_max, ddof=1 if n_iaaft > 1 else 0)
    z_val = 0.0 if sd_n <= 1e-15 else (obs_max - mu_n) / sd_n
    ge_cnt = sum(1 for v in null_max if v >= obs_max - 1e-15)
    p_emp = float(1 + ge_cnt) / float(1 + n_iaaft)

    return {
        "lens": "A1_ultrasonic_bicoherence",
        "max_b2": round(obs_max, 6),
        "peak_triad_hz": [round(best_tri[0] * df, 2), round(best_tri[1] * df, 2), round(best_tri[2] * df, 2)],
        "z_iaaft": round(z_val, 6),
        "p_emp": round(p_emp, 6),
        "significant_0p005": bool(p_emp <= 0.02 and z_val >= 3.5),
    }


def compute_a2_chiral_vrm_balance(
    left_ch: Sequence[float],
    right_ch: Sequence[float],
    *,
    fs: int = 48000,
    chiral_band_hz: tuple[float, float] = (18500.0, 22200.0),
    vrm_band_hz: tuple[float, float] = (12000.0, 18200.0),
    bezel_spacing_m: float = 0.134,
    c_sound_m_s: float = 343.0,
    n_fft: int = 256,
    n_surr: int = 69,
    seed: int = 42,
    auth_gate: AudioAuthorizationGate | None = None,
    allow_offline_array: bool = False,
) -> dict[str, Any]:
    """Lens A2 (Mode-2 Opt-In): Nolte imaginary coherency + parabolic GCC-PHAT lateral VRM TDOA."""
    check_dmic_permission(auth_gate, allow_offline_array)
    xl = _clean_1d_floats(left_ch, "a2_left_ch", min_len=4 * n_fft)
    xr = _clean_1d_floats(right_ch, "a2_right_ch", min_len=4 * n_fft)
    require_ok(len(xl) == len(xr), "ERR_A2_LEN", "left_ch and right_ch must match length")
    require_ok((n_fft & (n_fft - 1)) == 0 and n_fft >= 128, "ERR_A2_NFFT", "n_fft must be power of 2 >= 128")

    hann = [0.5 * (1.0 - math.cos(2.0 * math.pi * float(i) / float(n_fft - 1))) for i in range(n_fft)]
    step = n_fft // 2
    n_seg = 1 + (len(xl) - n_fft) // step
    zl_all: list[list[complex]] = []
    zr_all: list[list[complex]] = []
    for s in range(n_seg):
        s0 = s * step
        sl = xl[s0 : s0 + n_fft]
        sr = xr[s0 : s0 + n_fft]
        mul = sum(sl) / float(n_fft)
        mur = sum(sr) / float(n_fft)
        zl_all.append(fft_radix2([(sl[i] - mul) * hann[i] for i in range(n_fft)], invert=False))
        zr_all.append(fft_radix2([(sr[i] - mur) * hann[i] for i in range(n_fft)], invert=False))

    df = float(fs) / float(n_fft)
    half = n_fft // 2
    c_bins = [k for k in range(half + 1) if chiral_band_hz[0] <= float(k) * df <= chiral_band_hz[1]]
    require_ok(len(c_bins) >= 3, "ERR_A2_CBINS", "Chiral band has < 3 FFT bins")

    inv_s = 1.0 / float(n_seg)
    psd_l_full = [sum( abs(zl_all[s][k]) ** 2 for s in range(n_seg) ) * inv_s for k in range(half + 1)]
    psd_r_full = [sum( abs(zr_all[s][k]) ** 2 for s in range(n_seg) ) * inv_s for k in range(half + 1)]

    obs_max_chi = 0.0
    obs_signed_chi = 0.0
    peak_hz = 0.0
    for k in c_bins:
        slr = sum(zl_all[s][k] * zr_all[s][k].conjugate() for s in range(n_seg)) * inv_s
        denom = math.sqrt(max(psd_l_full[k] * psd_r_full[k], 1e-30))
        im_coh = slr.imag / denom
        if abs(im_coh) > obs_max_chi:
            obs_max_chi = abs(im_coh)
            obs_signed_chi = im_coh
            peak_hz = float(k) * df

    # Locally median-smoothed PSD surrogates (window=5 bins)
    amp_l_c: list[float] = []
    amp_r_c: list[float] = []
    for k in c_bins:
        lo = max(0, k - 2)
        hi = min(half + 1, k + 3)
        med_l = sorted(psd_l_full[lo:hi])[len(psd_l_full[lo:hi]) // 2]
        med_r = sorted(psd_r_full[lo:hi])[len(psd_r_full[lo:hi]) // 2]
        amp_l_c.append(math.sqrt(max(med_l, 1e-30) * 0.5))
        amp_r_c.append(math.sqrt(max(med_r, 1e-30) * 0.5))

    rng = random.Random(int(seed))
    null_chi: list[float] = []
    for _ in range(n_surr):
        max_s = 0.0
        for b_i in range(len(c_bins)):
            al = amp_l_c[b_i]
            ar = amp_r_c[b_i]
            csum = 0j
            pl_s = 0.0
            pr_s = 0.0
            for _seg in range(n_seg):
                zl_g = complex(rng.gauss(0.0, al), rng.gauss(0.0, al))
                zr_g = complex(rng.gauss(0.0, ar), rng.gauss(0.0, ar))
                csum += zl_g * zr_g.conjugate()
                pl_s += zl_g.real * zl_g.real + zl_g.imag * zl_g.imag
                pr_s += zr_g.real * zr_g.real + zr_g.imag * zr_g.imag
            den_s = math.sqrt(max((pl_s * inv_s) * (pr_s * inv_s), 1e-30))
            val_s = abs((csum * inv_s).imag) / den_s
            if val_s > max_s:
                max_s = val_s
        null_chi.append(max_s)

    mu_c, sd_c = _mean_std(null_chi, ddof=1 if n_surr > 1 else 0)
    z_chiral = 0.0 if sd_c <= 1e-15 else (obs_max_chi - mu_c) / sd_c
    ge_cnt = sum(1 for v in null_chi if v >= obs_max_chi - 1e-15)
    p_emp = float(1 + ge_cnt) / float(1 + n_surr)

    # 4x zero-padded bandpass GCC-PHAT lateral TDOA
    interp = 4
    n_pad = interp * n_fft
    phat_spec = [0j] * n_pad
    for k in range(1, half):
        fk = float(k) * df
        if vrm_band_hz[0] <= fk <= vrm_band_hz[1]:
            c_vrm = sum(zl_all[s][k] * zr_all[s][k].conjugate() for s in range(n_seg))
            mag = abs(c_vrm)
            if mag > 1e-30:
                u = c_vrm / mag
                phat_spec[k] = u
                phat_spec[n_pad - k] = u.conjugate()
    cc_raw = [z.real for z in fft_radix2(phat_spec, invert=True)]
    half_pad = n_pad // 2
    cc_shifted = cc_raw[half_pad:] + cc_raw[:half_pad]

    tau_bound = (bezel_spacing_m / c_sound_m_s) * 1.15
    best_g = half_pad
    best_cc = -1e9
    for idx in range(n_pad):
        lag_s = float(idx - half_pad) / float(fs * interp)
        if abs(lag_s) <= tau_bound and cc_shifted[idx] > best_cc:
            best_cc = cc_shifted[idx]
            best_g = idx

    delta = 0.0
    if 0 < best_g < n_pad - 1:
        y0, y1, y2 = cc_shifted[best_g - 1], cc_shifted[best_g], cc_shifted[best_g + 1]
        denom_p = y0 - 2.0 * y1 + y2
        if abs(denom_p) > 1e-18:
            delta = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom_p))

    peak_lag_s = (float(best_g - half_pad) + delta) / float(fs * interp)
    tau_left_leads_s = -peak_lag_s
    tau_us = tau_left_leads_s * 1e6
    lateral_cm = -0.5 * c_sound_m_s * tau_left_leads_s * 100.0
    rail = "GPU_VRM_RIGHT" if lateral_cm > 0.5 else ("CPU_VRM_LEFT" if lateral_cm < -0.5 else "BALANCED_MIDLINE")

    return {
        "lens": "A2_chiral_vrm_balance",
        "max_abs_chirality": round(obs_max_chi, 6),
        "signed_peak_chirality": round(obs_signed_chi, 6),
        "peak_chirality_hz": round(peak_hz, 2),
        "z_chiral": round(z_chiral, 6),
        "p_emp": round(p_emp, 6),
        "tdoa_left_leads_right_us": round(tau_us, 3),
        "lateral_vrm_offset_cm": round(lateral_cm, 4),
        "dominant_vrm_rail": rail,
        "significant_0p005": bool(p_emp <= 0.02 and z_chiral >= 3.5),
    }


# =============================================================================
# S20c Zero-Float Exact-Q Telemetry Trio + WHT_PHASE + 7-Tone Fisher ID
# =============================================================================


def compute_op1_seam1_sherman_morrison_leverage(
    int_series: Sequence[int],
    *,
    window: int = 32,
) -> dict[str, Any]:
    """OP1_SEAM1: Zero-float exact-Q sliding W=32 Sherman-Morrison causal leverage shock."""
    require_ok(window >= 8, "ERR_OP1_WIN", "window must be >= 8")
    require_ok(len(int_series) > window + 2, "ERR_OP1_LEN", f"Need > {window + 2} integer samples")
    xs: list[Fraction] = []
    for v in int_series:
        require_ok(isinstance(v, int) and not isinstance(v, bool), "ERR_OP1_INT", "OP1_SEAM1 requires exact int samples")
        xs.append(Fraction(v))

    max_h = Fraction(0)
    peak_idx = window
    for t in range(window, len(xs)):
        # Exact 2x2 regularized Gram G = I_2 + sum_{s=t-W}^{t-1} [1, x_{s-1}]^T [1, x_{s-1}]
        g00 = Fraction(1 + window)
        g01 = sum((xs[s - 1] for s in range(t - window, t)), Fraction(0))
        g11 = Fraction(1) + sum((xs[s - 1] * xs[s - 1] for s in range(t - window, t)), Fraction(0))
        det_g = g00 * g11 - g01 * g01
        require_ok(det_g > 0, "ERR_OP1_SINGULAR", "Regularized 2x2 Gram must be strictly SPD")

        z0 = Fraction(1)
        z1 = xs[t - 1]
        # G^{-1} z
        u0 = (g11 * z0 - g01 * z1) / det_g
        u1 = (-g01 * z0 + g00 * z1) / det_g
        q_t = z0 * u0 + z1 * u1
        h_t = q_t / (Fraction(1) + q_t)

        # Verify exact Sherman-Morrison identity: (G + z z^T)^{-1} z == G^{-1} z / (1 + q_t)
        gp00, gp01, gp11 = g00 + z0 * z0, g01 + z0 * z1, g11 + z1 * z1
        det_gp = gp00 * gp11 - gp01 * gp01
        up0 = (gp11 * z0 - gp01 * z1) / det_gp
        require_ok(up0 == u0 / (Fraction(1) + q_t), "ERR_OP1_SM_IDENTITY", "Sherman-Morrison identity failed")

        if h_t > max_h:
            max_h = h_t
            peak_idx = t

    return {
        "operator": "OP1_SEAM1_sherman_morrison_leverage",
        "window": window,
        "peak_step": peak_idx,
        "max_leverage_exact_q": f"{max_h.numerator}/{max_h.denominator}",
        "max_leverage_decimal": round(float(max_h), 6),
    }


def _ordinal_state_m3(a: float | int | Fraction, b: float | int | Fraction, c: float | int | Fraction) -> int:
    """Map triple (a, b, c) to 0..5 for strict 3! orderings or 6 for any tie."""
    if a == b or b == c or a == c:
        return 6
    if a < b < c:
        return 0
    if a < c < b:
        return 1
    if b < a < c:
        return 2
    if b < c < a:
        return 3
    if c < a < b:
        return 4
    return 5


def compute_op2_seam3_ordinal_hodge_curl(series: Sequence[float | int | Fraction]) -> dict[str, Any]:
    """OP2_SEAM3: Monotone-invariant 7x7 ordinal Hodge-curl Phi_asym in Q cap [0,1] + O(N) HVG beta_1."""
    n = len(series)
    require_ok(n >= 16, "ERR_OP2_LEN", f"Need >= 16 samples for OP2_SEAM3, got {n}")
    states = [_ordinal_state_m3(series[i], series[i + 1], series[i + 2]) for i in range(n - 2)]

    c_mat = [[0] * 7 for _ in range(7)]
    for t in range(len(states) - 1):
        c_mat[states[t]][states[t + 1]] += 1

    num_q = 0
    den_q = 0
    for i in range(7):
        for j in range(i + 1, 7):
            diff = c_mat[i][j] - c_mat[j][i]
            tot = c_mat[i][j] + c_mat[j][i]
            num_q += diff * diff
            den_q += tot * tot
    phi_q = Fraction(0) if den_q == 0 else Fraction(num_q, den_q)

    # Exact O(N) monotone-stack Horizontal Visibility Graph (HVG) edge & cycle count
    edges = 0
    stack: list[float | int | Fraction] = []
    for val in series:
        while stack and stack[-1] < val:
            edges += 1
            stack.pop()
        if stack:
            edges += 1
        if not stack or stack[-1] > val:
            stack.append(val)
        else:
            # Equal height blocks visibility past tie while remaining top
            stack[-1] = val
    beta1 = max(0, edges - (n - 1))
    beta1_rate = Fraction(beta1, n)

    return {
        "operator": "OP2_SEAM3_ordinal_hodge_curl",
        "phi_asym_exact_q": f"{phi_q.numerator}/{phi_q.denominator}",
        "phi_asym_decimal": round(float(phi_q), 6),
        "hvg_edges": edges,
        "hvg_beta1": beta1,
        "hvg_beta1_per_n_exact_q": f"{beta1_rate.numerator}/{beta1_rate.denominator}",
    }


def compute_op3_f3_nonnormal_commutator(
    t_matrix_q: Sequence[Sequence[int | Fraction]],
) -> dict[str, Any]:
    """OP3_F3: Exact 3x3 rational non-normality commutator kappa_F3 & Sylvester In(I_3 - T^T T)."""
    require_ok(
        len(t_matrix_q) == 3 and all(len(row) == 3 for row in t_matrix_q),
        "ERR_OP3_SHAPE",
        "t_matrix_q must be 3x3",
    )
    t_mat = [[Fraction(v) for v in row] for row in t_matrix_q]

    # T^T T and T T^T
    ttt = [[sum(t_mat[k][i] * t_mat[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tt_t = [[sum(t_mat[i][k] * t_mat[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

    frob_t2 = sum(t_mat[i][j] * t_mat[i][j] for i in range(3) for j in range(3))
    comm_f2 = sum((ttt[i][j] - tt_t[i][j]) ** 2 for i in range(3) for j in range(3))
    kappa_q = Fraction(0) if frob_t2 == 0 else (comm_f2 / (frob_t2 * frob_t2))

    # Exact rational LDL^T / Gaussian pivot signs of M = I_3 - T^T T (transient energy growth iff n_neg > 0)
    m_mat = [[(Fraction(1) if i == j else Fraction(0)) - ttt[i][j] for j in range(3)] for i in range(3)]
    a = [list(row) for row in m_mat]
    n_neg = 0
    for k in range(3):
        piv = a[k][k]
        if piv == 0:
            for r in range(k + 1, 3):
                if a[r][r] != 0:
                    a[k], a[r] = a[r], a[k]
                    for idx in range(3):
                        a[idx][k], a[idx][r] = a[idx][r], a[idx][k]
                    piv = a[k][k]
                    break
        if piv < 0:
            n_neg += 1
        if piv != 0:
            for i in range(k + 1, 3):
                factor = a[i][k] / piv
                for j in range(k + 1, 3):
                    a[i][j] -= factor * a[k][j]

    return {
        "operator": "OP3_F3_nonnormal_commutator",
        "kappa_f3_exact_q": f"{kappa_q.numerator}/{kappa_q.denominator}",
        "kappa_f3_decimal": round(float(kappa_q), 6),
        "gram_contractivity_neg_inertia": n_neg,
        "transient_overshoot_risk": bool(n_neg > 0),
    }


def wht_phase_surrogate(series: Sequence[float], seed: int = 42) -> list[float]:
    """Exact Walsh-Hadamard sequency sign-randomized surrogate (companion to Fourier IAAFT)."""
    x = _clean_1d_floats(series, "wht_input", min_len=8)
    n = len(x)
    require_ok((n & (n - 1)) == 0, "ERR_WHT_POW2", f"wht_phase_surrogate requires power-of-2 length, got {n}")
    buf = list(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = buf[j]
                v = buf[j + h]
                buf[j] = u + v
                buf[j + h] = u - v
        h *= 2
    rng = random.Random(int(seed))
    # Preserve DC slot 0, randomize signs +/-1 on sequency slots 1..n-1
    for k in range(1, n):
        if rng.randint(0, 1) == 0:
            buf[k] = -buf[k]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = buf[j]
                v = buf[j + h]
                buf[j] = u + v
                buf[j + h] = u - v
        h *= 2
    inv_n = 1.0 / float(n)
    return [v * inv_n for v in buf]


def compute_time_locked_7tone_cole_shunt(
    freqs_hz: Sequence[float],
    h_measured: Sequence[complex],
    *,
    locked_tau_d_sec: float | None,
    collinearity_tol: float = 1e-6,
) -> dict[str, Any]:
    """ASKA-XFER-H1: Time-locked 7-tone Fisher Cole-Shunt 35-triple collinearity detector."""
    require_ok(
        locked_tau_d_sec is not None and isinstance(locked_tau_d_sec, (int, float)) and not isinstance(locked_tau_d_sec, bool),
        "ERR_FREE_DELAY_FORBIDDEN",
        "Transport lag locked_tau_d_sec must be locked independently via A5 step edges prior to impedance ID",
    )
    tau_d = float(locked_tau_d_sec)  # type: ignore[arg-type]
    require_ok(math.isfinite(tau_d) and tau_d >= 0.0, "ERR_BAD_TAU_D", f"Invalid locked_tau_d_sec={tau_d}")
    require_ok(len(freqs_hz) == 7 and len(h_measured) == 7, "ERR_7TONE_LEN", "Expected exactly 7 knee-concentrated tones")

    pts: list[tuple[float, float]] = []
    for f_hz, h_val in zip(freqs_hz, h_measured):
        fk = float(f_hz)
        require_ok(fk > 0.0 and math.isfinite(fk), "ERR_BAD_FREQ", f"Invalid tone {fk}")
        u_k = 2.0 * math.pi * fk
        # Strip independently locked pure transport delay exp(-i u_k tau_d)
        h_derot = complex(h_val) * cmath.rect(1.0, u_k * tau_d)
        pts.append((-h_derot.imag / u_k, h_derot.real))

    nonzero_triples = 0
    max_abs_det = 0.0
    for i in range(7):
        x1, y1 = pts[i]
        for j in range(i + 1, 7):
            x2, y2 = pts[j]
            for k in range(j + 1, 7):
                x3, y3 = pts[k]
                det3 = (x2 * y3 - y2 * x3) - (x1 * y3 - y1 * x3) + (x1 * y2 - y1 * x2)
                ad = abs(det3)
                if ad > max_abs_det:
                    max_abs_det = ad
                if ad > collinearity_tol:
                    nonzero_triples += 1

    return {
        "operator": "ASKA_7tone_time_locked_cole_shunt",
        "locked_tau_d_sec": round(tau_d, 6),
        "total_triples": 35,
        "nonzero_collinearity_triples": nonzero_triples,
        "max_abs_collinearity_det": round(max_abs_det, 10),
        "shunt_or_fractional_pole_detected": bool(nonzero_triples > 0),
    }


# =============================================================================
# Unified Mode-1 / Mode-2 Trace Profiler & Synthetic Workload Generator
# =============================================================================


class SiliSenseProfiler:
    """Unified Mode-1 / Mode-2 profiler for HoloSphere & standalone silicon traces."""

    def __init__(
        self,
        *,
        mode: str = "telemetry-only",
        auth_gate: AudioAuthorizationGate | None = None,
        n_iaaft: int = 249,
        n_shifts: int = 300,
        seed: int = 42,
    ) -> None:
        require_ok(mode in ("telemetry-only", "dmic-authorized"), "ERR_UNKNOWN_MODE", f"Invalid mode {mode!r}")
        if mode == "dmic-authorized":
            require_perm(auth_gate is not None, "ERR_DMIC_UNAUTHORIZED", "dmic-authorized requires AudioAuthorizationGate")
            if auth_gate is not None:
                auth_gate.verify_active()
        else:
            require_perm(auth_gate is None, "ERR_MODE_GATE_CONFLICT", "Pass auth_gate=None in telemetry-only mode")
        self.mode = mode
        self.auth_gate = auth_gate
        self.n_iaaft = int(n_iaaft)
        self.n_shifts = int(n_shifts)
        self.seed = int(seed)

    def profile_trace(self, trace: dict[str, Any], *, allow_offline_dmic: bool = False) -> dict[str, Any]:
        for key in trace:
            if is_camera_or_ir_identifier(str(key)):
                raise SiliSensePermissionError("ERR_CAMERA_FORBIDDEN", f"Camera/IR field {key!r} is permanently forbidden")

        dt_ms = float(trace.get("dt_ms", 5.0))
        gpu_w = _clean_1d_floats(trace["gpu_power_w"], "gpu_power_w", min_len=64)
        uncore_w = _clean_1d_floats(trace["uncore_w"], "uncore_w", min_len=64)
        slow_resp = _clean_1d_floats(trace.get("slow_throttle_or_psys", gpu_w), "slow_response", min_len=64)
        fast_rail = _clean_1d_floats(
            trace.get("fast_power_w", [g + u for g, u in zip(gpu_w, uncore_w)]),
            "fast_power_w",
            min_len=64,
        )

        a3_res = compute_a3_time_reversal_asymmetry(gpu_w, dt_ms=dt_ms, n_iaaft=self.n_iaaft, seed=self.seed)
        a5_res = compute_a5_precursor_lead_lag(
            fast_rail, slow_resp, dt_ms=dt_ms, max_lag_ms=max(50.0, 30.0 * dt_ms), n_shifts=self.n_shifts, seed=self.seed + 1
        )
        a7_res = compute_a7_nonlinear_uncore_copula_mi(uncore_w, gpu_w, n_iaaft=self.n_iaaft, seed=self.seed + 2)
        op2_res = compute_op2_seam3_ordinal_hodge_curl(gpu_w)

        mu_g, sd_g = _mean_std(gpu_w, ddof=1)
        mu_u, _ = _mean_std(uncore_w, ddof=1)
        result: dict[str, Any] = {
            "prereg_id": PREREG_S115_ID,
            "mode": self.mode,
            "dmic_authorized": bool(self.mode == "dmic-authorized"),
            "dt_ms": round(dt_ms, 6),
            "mean_gpu_power_w": round(mu_g, 4),
            "std_gpu_power_w": round(sd_g, 4),
            "mean_uncore_w": round(mu_u, 4),
            "observables": {
                "A3_telemetry": a3_res,
                "A5_precursor": a5_res,
                "A7_uncore_copula_mi": a7_res,
                "OP2_SEAM3_ordinal_curl": op2_res,
            },
        }

        has_dmic = ("dmic_left_48k" in trace) or ("dmic_mono_48k" in trace)
        if has_dmic:
            if self.mode != "dmic-authorized" and not allow_offline_dmic:
                raise SiliSensePermissionError(
                    "ERR_DMIC_DISABLED_DEFAULT",
                    "Trace contains DMIC audio channels while profiler is in Mode 1 ('telemetry-only')",
                )
            fs = int(trace.get("dmic_fs_hz", 48000))
            mono = _clean_1d_floats(trace.get("dmic_mono_48k", trace.get("dmic_left_48k", [])), "dmic_mono", min_len=512)
            result["observables"]["A1_ultrasonic_bicoherence"] = compute_a1_ultrasonic_bicoherence(
                mono, fs=fs, auth_gate=self.auth_gate, allow_offline_array=allow_offline_dmic, seed=self.seed + 3
            )
            if "dmic_left_48k" in trace and "dmic_right_48k" in trace:
                result["observables"]["A2_chiral_vrm_balance"] = compute_a2_chiral_vrm_balance(
                    trace["dmic_left_48k"],
                    trace["dmic_right_48k"],
                    fs=fs,
                    auth_gate=self.auth_gate,
                    allow_offline_array=allow_offline_dmic,
                    seed=self.seed + 4,
                )

        p_family = [
            float(a3_res["p_holm"]),
            float(a5_res["p_circ"]),
            float(a7_res["p_emp"]),
        ]
        holm_fam = holm_bonferroni(p_family)
        result["min_family_holm_p"] = round(min(holm_fam), 6)
        result["any_non_textbook_significant_0p005"] = bool(min(holm_fam) < 0.005)
        return result


def make_synthetic_workload_trace(
    kind: str,
    *,
    n_samples: int = 512,
    dt_ms: float = 5.0,
    seed: int = 42,
    target_mean_gpu_w: float = 95.0,
) -> dict[str, Any]:
    """Deterministic pure-stdlib synthetic traces for 'eager_microkernel', 'fused_holosphere', and 'colored_null'."""
    require_ok(kind in ("eager_microkernel", "fused_holosphere", "colored_null"), "ERR_SYNTH_KIND", f"Unknown kind {kind!r}")
    rng = random.Random(int(seed))

    uncore = [0.0] * n_samples
    for t in range(1, n_samples):
        uncore[t] = 0.70 * uncore[t - 1] + rng.gauss(0.0, 1.0)

    fast_spikes = [0.0] * n_samples
    if kind == "eager_microkernel":
        saw = [0.0] * n_samples
        pos = 0
        while pos < n_samples:
            span = min(n_samples - pos, rng.randint(5, 10))
            if span >= 2:
                for i in range(span):
                    saw[pos + i] = -1.0 + 2.0 * float(i) / float(span - 1)
            pos += span
        mean_u2 = sum(u * u for u in uncore) / float(n_samples)
        gpu_raw = [9.5 * saw[t] + 1.65 * (uncore[t] * uncore[t] - mean_u2) + 0.35 * rng.gauss(0.0, 1.0) for t in range(n_samples)]
        mu_raw = sum(gpu_raw) / float(n_samples)
        gpu_w = [target_mean_gpu_w + (v - mu_raw) for v in gpu_raw]

        slow_resp = [105.0 + 0.28 * rng.gauss(0.0, 1.0) for _ in range(n_samples)]
        bt = 22
        while bt < n_samples - 26:
            for k, dv in enumerate((5.5, 11.5, 17.5, 17.5)):
                fast_spikes[bt + k] += dv
            for k, dv in enumerate((5.0, 10.5, 15.0, 15.0, 11.0)):
                slow_resp[bt + 16 + k] += dv
            bt += rng.randint(28, 54)
    elif kind == "fused_holosphere":
        ar_g = [0.0] * n_samples
        for t in range(1, n_samples):
            ar_g[t] = 0.65 * ar_g[t - 1] + 1.7 * rng.gauss(0.0, 1.0)
        raw = [ar_g[t] + 0.35 * uncore[t] for t in range(n_samples)]
        mu_r = sum(raw) / float(n_samples)
        gpu_w = [target_mean_gpu_w + (v - mu_r) for v in raw]
        slow_resp = [105.0 + 0.45 * rng.gauss(0.0, 1.0) for _ in range(n_samples)]
    else:
        ar_g = [0.0] * n_samples
        for t in range(2, n_samples):
            ar_g[t] = 0.78 * ar_g[t - 1] + 1.4 * uncore[t] - 0.85 * uncore[t - 1] + rng.gauss(0.0, 1.0)
        mu_r = sum(ar_g) / float(n_samples)
        gpu_w = [target_mean_gpu_w + (v - mu_r) for v in ar_g]
        slow_resp = [0.0] * n_samples
        for t in range(1, n_samples):
            slow_resp[t] = 0.80 * slow_resp[t - 1] + rng.gauss(0.0, 1.0)
        slow_resp = [105.0 + v for v in slow_resp]

    uncore_pos = [8.0 + u for u in uncore]
    fast_w = [gpu_w[t] + uncore_pos[t] + fast_spikes[t] for t in range(n_samples)]
    return {
        "dt_ms": dt_ms,
        "gpu_power_w": gpu_w,
        "uncore_w": uncore_pos,
        "fast_power_w": fast_w,
        "slow_throttle_or_psys": slow_resp,
    }
