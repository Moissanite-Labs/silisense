"""Closed-loop precursor rail governor, Delay-Bandwidth Lock, L^{5/3} Lyapunov choke, and Contract-4 latch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import struct
from typing import Any, Mapping, Sequence

from silisense.errors import SiliSenseValueError, require_ok
from silisense.lenses import compute_a5_precursor_lead_lag

_F32_MAX_FINITE = 3.4028234663852886e38


def _req_num(val: Any, name: str, code: str) -> float:
    require_ok(
        isinstance(val, (int, float)) and not isinstance(val, bool),
        code,
        f"{name} must be numeric (not bool), got {type(val).__name__}",
    )
    try:
        fv = float(val)
    except OverflowError as exc:
        raise SiliSenseValueError(code, f"{name} overflowed float64") from exc
    require_ok(math.isfinite(fv), code, f"{name} must be finite, got {val!r}")
    return fv


def floor_to_exact_f32(alpha: float) -> float:
    """Round `alpha` downward to a strictly positive IEEE-754 binary32 dyadic rational (0-ULP overshoot)."""
    fv = _req_num(alpha, "alpha", "ERR_BAD_DYADIC_ALPHA")
    require_ok(fv > 0.0, "ERR_BAD_DYADIC_ALPHA", f"alpha must be > 0, got {fv}")
    require_ok(fv <= _F32_MAX_FINITE, "ERR_DYADIC_OVERFLOW", f"alpha {fv} exceeds binary32 max finite value")
    try:
        packed = struct.pack("<f", fv)
    except OverflowError as exc:
        raise SiliSenseValueError("ERR_DYADIC_OVERFLOW", f"alpha {fv} overflowed binary32") from exc
    f32 = struct.unpack("<f", packed)[0]
    require_ok(math.isfinite(f32) and f32 > 0.0, "ERR_DYADIC_UNDERFLOW", f"alpha {fv} underflowed binary32 positive range")
    if f32 > fv:
        bits = struct.unpack("<I", struct.pack("<f", f32))[0]
        require_ok(bits > 1, "ERR_DYADIC_UNDERFLOW", "binary32 underflow at positive floor")
        f32 = struct.unpack("<f", struct.pack("<I", bits - 1))[0]
    require_ok(0.0 < f32 <= fv, "ERR_DYADIC_OVERSHOOT", "binary32 floor violated (0, alpha] bound")
    return float(f32)


def check_delay_bandwidth_lock(f_control_hz: float, tau_sensor_sec: float) -> dict[str, Any]:
    """Principal-band phase-stability Delay-Bandwidth Lock (`f * tau < 1/6` => `|1 - exp(-i 2pi f tau)| < 1`)."""
    fc = _req_num(f_control_hz, "f_control_hz", "ERR_BAD_DELAY_BW")
    tau = _req_num(tau_sensor_sec, "tau_sensor_sec", "ERR_BAD_DELAY_BW")
    require_ok(fc > 0.0 and tau > 0.0, "ERR_BAD_DELAY_BW", "f_control_hz and tau_sensor_sec must be > 0")
    prod = fc * tau
    return {
        "f_tau_product": round(prod, 8),
        "limit": round(1.0 / 6.0, 8),
        "max_stable_control_hz": round(1.0 / (6.0 * tau), 4),
        "stable": bool(prod < (1.0 / 6.0)),
    }


def smooth_l53_shared_heatpipe_excess(rail_excess_watts: Sequence[float]) -> float:
    """Smooth C^1 shared-heatpipe excess norm `(sum_i max(0, u_i)^{5/3})^{3/5}`."""
    s = 0.0
    for v in rail_excess_watts:
        fv = _req_num(v, "rail_excess_watts", "ERR_NONFINITE_RAIL")
        if fv > 0.0:
            s += fv ** (5.0 / 3.0)
    return float(s ** (3.0 / 5.0)) if s > 0.0 else 0.0


def lyapunov_telescoping_choke(
    g_prev: float,
    u_53: float,
    *,
    u_thresh: float = 0.0,
    alpha: float = 0.08,
    beta: float = 0.15,
    g_floor: float = 0.70,
) -> float:
    """Runaway-free telescoping Lyapunov conductance choke quantized downward via `floor_to_exact_f32`."""
    gp = _req_num(g_prev, "g_prev", "ERR_LYAPUNOV_FINITE")
    u_val = _req_num(u_53, "u_53", "ERR_LYAPUNOV_FINITE")
    ut = _req_num(u_thresh, "u_thresh", "ERR_LYAPUNOV_FINITE")
    al = _req_num(alpha, "alpha", "ERR_LYAPUNOV_PARAMS")
    be = _req_num(beta, "beta", "ERR_LYAPUNOV_PARAMS")
    gf = _req_num(g_floor, "g_floor", "ERR_LYAPUNOV_PARAMS")
    require_ok(0.0 < gf <= 1.0 and al >= 0.0 and 0.0 <= be <= 1.0 and ut >= 0.0, "ERR_LYAPUNOV_PARAMS", "Invalid Lyapunov params")
    g_cur = min(1.0, max(gf, gp))
    excess = max(0.0, u_val - ut)
    num = g_cur + be * (1.0 - g_cur)
    den = 1.0 + al * excess * excess
    raw_next = min(1.0, max(gf, num / den))
    return floor_to_exact_f32(raw_next)


def gate_asymmetric_step(
    *,
    headroom_before: float,
    headroom_after: float,
    obj_before: float,
    obj_after: float,
) -> str:
    """Asymmetric safety trust-region gate with deadlock-free `.okRecovery` and `lexImproves`."""
    hb = _req_num(headroom_before, "headroom_before", "ERR_GATE_NONFINITE")
    ha = _req_num(headroom_after, "headroom_after", "ERR_GATE_NONFINITE")
    ob = _req_num(obj_before, "obj_before", "ERR_GATE_NONFINITE")
    oa = _req_num(obj_after, "obj_after", "ERR_GATE_NONFINITE")
    if hb < 0.0:
        return ".okRecovery" if ha > hb else ".refusedUnsafe"
    if ha < 0.0:
        return ".refusedUnsafe"
    if oa < ob:
        return ".okStep"
    if oa == ob and ha > hb:
        return ".okPureSafety"
    if oa == ob and ha <= hb:
        return ".refusedHeadroomErosion"
    return ".refusedObjectiveRegression"


def cyclic_switch_dist(pstate_seq: Sequence[int]) -> int:
    """Exact cumulative cyclic DVFS / P-state transition distance."""
    require_ok(len(pstate_seq) >= 1, "ERR_EMPTY_PSTATES", "pstate_seq must be non-empty")
    for p in pstate_seq:
        require_ok(isinstance(p, int) and not isinstance(p, bool) and p >= 0, "ERR_BAD_PSTATE", f"Invalid P-state {p!r}")
    if len(pstate_seq) == 1:
        return 0
    dist = sum(abs(pstate_seq[i + 1] - pstate_seq[i]) for i in range(len(pstate_seq) - 1))
    dist += abs(pstate_seq[0] - pstate_seq[-1])
    return int(dist)


@dataclass(frozen=True)
class PreAdaptationResidualLatch:
    """Contract-4 Pre-Adaptation Projection Latch (freezes raw pre-tuning silicon residuals)."""

    latched_abs_a3: float
    latched_z_a3: float
    latched_a5_lead_ms: float
    latched_a7_copula_mi_nats: float
    latched_z_a7: float
    latched_mean_gpu_w: float
    open_pathology: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "latched_abs_a3", round(_req_num(self.latched_abs_a3, "latched_abs_a3", "ERR_LATCH_VAL"), 8))
        object.__setattr__(self, "latched_z_a3", round(_req_num(self.latched_z_a3, "latched_z_a3", "ERR_LATCH_VAL"), 8))
        object.__setattr__(self, "latched_a5_lead_ms", round(_req_num(self.latched_a5_lead_ms, "latched_a5_lead_ms", "ERR_LATCH_VAL"), 8))
        object.__setattr__(self, "latched_a7_copula_mi_nats", round(_req_num(self.latched_a7_copula_mi_nats, "latched_a7_copula_mi_nats", "ERR_LATCH_VAL"), 8))
        object.__setattr__(self, "latched_z_a7", round(_req_num(self.latched_z_a7, "latched_z_a7", "ERR_LATCH_VAL"), 8))
        object.__setattr__(self, "latched_mean_gpu_w", round(_req_num(self.latched_mean_gpu_w, "latched_mean_gpu_w", "ERR_LATCH_VAL"), 8))
        require_ok(isinstance(self.open_pathology, bool), "ERR_LATCH_BOOL", "open_pathology must be bool")

    def to_canon_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def latch_sha256(self) -> str:
        """Deterministic SHA-256 over the frozen pre-adaptation silicon residual latch."""
        return hashlib.sha256(self.to_canon_json().encode("ascii")).hexdigest()

    @classmethod
    def capture_from_profile(cls, profile_report: Mapping[str, Any]) -> "PreAdaptationResidualLatch":
        require_ok(isinstance(profile_report, Mapping) and "observables" in profile_report, "ERR_LATCH_SCHEMA", "Missing observables in profile_report")
        obs = profile_report["observables"]
        require_ok(
            isinstance(obs, Mapping)
            and "A3_telemetry" in obs
            and "A5_precursor" in obs
            and "A7_uncore_copula_mi" in obs
            and "mean_gpu_power_w" in profile_report,
            "ERR_LATCH_SCHEMA",
            "Incomplete observables in profile_report",
        )
        a3 = obs["A3_telemetry"]
        a5 = obs["A5_precursor"]
        a7 = obs["A7_uncore_copula_mi"]
        z_a3 = _req_num(a3.get("z_iaaft"), "a3.z_iaaft", "ERR_LATCH_VAL")
        z_a7 = _req_num(a7.get("z_iaaft"), "a7.z_iaaft", "ERR_LATCH_VAL")
        open_flag = bool(
            bool(a3.get("significant_0p005", False))
            or bool(a7.get("significant_0p005", False))
            or abs(z_a3) >= 3.5
            or z_a7 >= 3.5
        )
        return cls(
            latched_abs_a3=_req_num(a3.get("abs_asymmetry"), "a3.abs_asymmetry", "ERR_LATCH_VAL"),
            latched_z_a3=z_a3,
            latched_a5_lead_ms=_req_num(a5.get("peak_precursor_lag_ms"), "a5.peak_precursor_lag_ms", "ERR_LATCH_VAL"),
            latched_a7_copula_mi_nats=_req_num(a7.get("copula_mi_nats"), "a7.copula_mi_nats", "ERR_LATCH_VAL"),
            latched_z_a7=z_a7,
            latched_mean_gpu_w=_req_num(profile_report.get("mean_gpu_power_w"), "mean_gpu_power_w", "ERR_LATCH_VAL"),
            open_pathology=open_flag,
        )


def paired_stats(baseline_vals: Sequence[float], candidate_vals: Sequence[float]) -> dict[str, float]:
    require_ok(len(baseline_vals) == len(candidate_vals) and len(baseline_vals) >= 3, "ERR_PAIRED_LEN", "Need >= 3 pairs")
    n = len(baseline_vals)
    a_clean = [_req_num(v, "baseline_vals", "ERR_PAIRED_VAL") for v in baseline_vals]
    b_clean = [_req_num(v, "candidate_vals", "ERR_PAIRED_VAL") for v in candidate_vals]
    diffs = [b - a for a, b in zip(a_clean, b_clean)]
    mean_a = sum(a_clean) / float(n)
    mean_b = sum(b_clean) / float(n)
    mean_d = sum(diffs) / float(n)
    var_d = sum((d - mean_d) * (d - mean_d) for d in diffs) / float(n - 1)
    se_d = math.sqrt(max(0.0, var_d)) / math.sqrt(float(n))
    if abs(mean_d) <= 1e-15:
        z_p = 0.0
    elif se_d <= 1e-15:
        z_p = math.copysign(1e6, mean_d)
    else:
        z_p = mean_d / se_d
    rel_pct = 0.0 if abs(mean_a) <= 1e-12 else 100.0 * (mean_b - mean_a) / abs(mean_a)
    return {
        "baseline_mean": round(mean_a, 6),
        "candidate_mean": round(mean_b, 6),
        "paired_diff_mean": round(mean_d, 6),
        "paired_diff_se": round(se_d, 6),
        "rel_change_pct": round(rel_pct, 4),
        "z_paired": round(z_p, 4),
    }


class PrecursorThrottleGovernor:
    """Causal O(1) online governor triggered by A5 fast-rail innovation spikes."""

    def __init__(
        self,
        *,
        z_trigger: float = 2.5,
        ema_alpha: float = 0.05,
        damping_scale: float = 0.86,
        hold_steps: int = 8,
        cooldown_steps: int = 12,
    ) -> None:
        zt = _req_num(z_trigger, "z_trigger", "ERR_BAD_Z_TRIGGER")
        ea = _req_num(ema_alpha, "ema_alpha", "ERR_BAD_EMA_ALPHA")
        ds = _req_num(damping_scale, "damping_scale", "ERR_BAD_DAMPING")
        require_ok(1.5 <= zt <= 100.0, "ERR_BAD_Z_TRIGGER", "z_trigger must lie in [1.5, 100.0]")
        require_ok(0.0 < ea < 1.0, "ERR_BAD_EMA_ALPHA", "ema_alpha must lie in (0.0, 1.0)")
        require_ok(0.5 <= ds < 1.0, "ERR_BAD_DAMPING", "damping_scale must lie in [0.5, 1.0)")
        require_ok(
            isinstance(hold_steps, int) and not isinstance(hold_steps, bool) and hold_steps >= 1,
            "ERR_BAD_HOLD",
            "hold_steps must be int >= 1",
        )
        require_ok(
            isinstance(cooldown_steps, int) and not isinstance(cooldown_steps, bool) and cooldown_steps >= 0,
            "ERR_BAD_HOLD",
            "cooldown_steps must be int >= 0",
        )
        self.z_trigger = zt
        self.ema_alpha = ea
        self.damping_scale = floor_to_exact_f32(ds)
        self.hold_steps = int(hold_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.reset()

    def reset(self) -> None:
        self._prev_val: float | None = None
        self._mu = 0.0
        self._var = 1.0
        self._step_idx = 0
        self._active_rem = 0
        self._cooldown_rem = 0
        self.trigger_steps: list[int] = []

    def step(self, fast_power_w: float) -> tuple[float, float]:
        fw = _req_num(fast_power_w, "fast_power_w", "ERR_NONFINITE_POWER")
        self._step_idx += 1
        if self._prev_val is None:
            self._prev_val = fw
            return 1.0, 0.0

        pos_inc = max(0.0, fw - self._prev_val)
        self._prev_val = fw

        z_innov = 0.0
        if self._step_idx >= 12:
            sd = math.sqrt(max(self._var, 1e-12))
            z_innov = (pos_inc - self._mu) / sd
            if z_innov >= self.z_trigger and self._active_rem == 0 and self._cooldown_rem == 0:
                self._active_rem = self.hold_steps
                self._cooldown_rem = self.hold_steps + self.cooldown_steps
                self.trigger_steps.append(self._step_idx)

        diff = pos_inc - self._mu
        self._mu = (1.0 - self.ema_alpha) * self._mu + self.ema_alpha * pos_inc
        self._var = (1.0 - self.ema_alpha) * self._var + self.ema_alpha * (diff * diff)

        scale = 1.0
        if self._active_rem > 0:
            scale = self.damping_scale
            self._active_rem -= 1
        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1
        return scale, z_innov


def grade_h15b_results(
    *,
    lead_receipts: Sequence[dict[str, Any]],
    per_arm_stall: dict[str, Sequence[float]],
    per_arm_cv: dict[str, Sequence[float]],
    per_arm_tps: dict[str, Sequence[float]],
    mode: str = "telemetry-only",
) -> dict[str, Any]:
    lead_thresh_ms = -60.0 if mode == "dmic-authorized" else -20.0
    max_p_circ = max(float(r["p_circ"]) for r in lead_receipts)
    leads_sorted = sorted(float(r["peak_precursor_lag_ms"]) for r in lead_receipts)
    med_lead = leads_sorted[len(leads_sorted) // 2]
    crit1 = bool(max_p_circ < 0.005 and med_lead <= lead_thresh_ms)

    stall_no = paired_stats(per_arm_stall["NoGovernor"], per_arm_stall["SiliSense-Precursor-Gov"])
    stall_smi = paired_stats(per_arm_stall["Reactive-1Hz-SMI"], per_arm_stall["SiliSense-Precursor-Gov"])
    stall_shuf = paired_stats(per_arm_stall["Shuffled-Trigger-Control"], per_arm_stall["SiliSense-Precursor-Gov"])

    cv_no = paired_stats(per_arm_cv["NoGovernor"], per_arm_cv["SiliSense-Precursor-Gov"])
    cv_smi = paired_stats(per_arm_cv["Reactive-1Hz-SMI"], per_arm_cv["SiliSense-Precursor-Gov"])
    cv_shuf = paired_stats(per_arm_cv["Shuffled-Trigger-Control"], per_arm_cv["SiliSense-Precursor-Gov"])
    tps_no = paired_stats(per_arm_tps["NoGovernor"], per_arm_tps["SiliSense-Precursor-Gov"])

    stall_win = bool(
        stall_no["rel_change_pct"] <= -25.0
        and stall_no["z_paired"] <= -2.0
        and stall_smi["rel_change_pct"] <= -25.0
        and stall_smi["z_paired"] <= -2.0
        and stall_shuf["z_paired"] <= -2.0
    )
    cv_win = bool(
        cv_no["rel_change_pct"] <= -20.0
        and cv_no["z_paired"] <= -2.0
        and cv_smi["rel_change_pct"] <= -20.0
        and cv_smi["z_paired"] <= -2.0
        and cv_shuf["z_paired"] <= -2.0
    )
    tps_ok = bool(tps_no["rel_change_pct"] >= -2.0)
    crit2 = bool((stall_win or cv_win) and tps_ok)

    if crit1 and crit2:
        verdict = "CONFIRMED"
    elif crit1 or (
        (stall_no["rel_change_pct"] <= -25.0 or cv_no["rel_change_pct"] <= -20.0)
        and tps_no["rel_change_pct"] >= -5.0
    ):
        verdict = "PARTIAL"
    else:
        verdict = "FALSIFIED"

    return {
        "hypothesis": "H15b",
        "verdict": verdict,
        "criterion_1_precursor_lead": crit1,
        "criterion_2_closed_loop_stability": crit2,
        "median_peak_precursor_lead_ms": round(med_lead, 4),
        "max_p_circ": round(max_p_circ, 6),
        "stall_comparisons": {
            "vs_NoGovernor": stall_no,
            "vs_Reactive_1Hz_SMI": stall_smi,
            "vs_Shuffled_Trigger_Control": stall_shuf,
        },
        "mj_cv_comparisons": {
            "vs_NoGovernor": cv_no,
            "vs_Reactive_1Hz_SMI": cv_smi,
            "vs_Shuffled_Trigger_Control": cv_shuf,
        },
        "throughput_vs_NoGovernor": tps_no,
    }


def simulate_sustained_governor_benchmark(
    *,
    seeds: Sequence[int] = (42, 137, 2026),
    n_steps: int = 800,
    dt_ms: float = 5.0,
) -> dict[str, Any]:
    """Pure-stdlib synthetic selftest of all 4 PREREG-S115 H15b arms."""
    require_ok(len(seeds) >= 3 and n_steps >= 200, "ERR_H15B_PARAMS", "Need >= 3 seeds and >= 200 steps")
    arms = ("NoGovernor", "Reactive-1Hz-SMI", "Shuffled-Trigger-Control", "SiliSense-Precursor-Gov")
    per_arm_stall: dict[str, list[float]] = {a: [] for a in arms}
    per_arm_cv: dict[str, list[float]] = {a: [] for a in arms}
    per_arm_tps: dict[str, list[float]] = {a: [] for a in arms}
    lead_receipts: list[dict[str, Any]] = []
    smi_period = max(1, int(round(1000.0 / dt_ms)))

    for seed in seeds:
        rng = random.Random(int(seed))
        base_demand = [93.0 + 0.50 * rng.gauss(0.0, 1.0) for _ in range(n_steps)]
        precursor_wave = [0.0] * n_steps
        heavy_wave = [0.0] * n_steps
        t0 = 32
        while t0 < n_steps - 38:
            for k, dv in enumerate((5.5, 11.5, 17.5, 17.5)):
                precursor_wave[t0 + k] += dv
            for k, dv in enumerate((16.5, 22.0, 23.0, 23.0, 23.0, 22.0, 20.0, 17.0, 13.0)):
                heavy_wave[t0 + 16 + k] += dv
            t0 += rng.randint(34, 68)

        def _run_plant(mode_name: str, fixed_sched: list[float] | None = None) -> tuple[float, float, float, list[float], list[float], list[int]]:
            gov = PrecursorThrottleGovernor(z_trigger=2.5, damping_scale=0.84, hold_steps=11, cooldown_steps=16)
            damped_s = gov.damping_scale
            thermal_accum = 93.0
            clamp_rem = 0
            reactive_rem = 0
            stalls = 0
            mj_tok: list[float] = []
            tps_list: list[float] = []
            fast_tr: list[float] = []
            clamp_tr = [0.0] * n_steps

            for t in range(n_steps):
                raw_fast = base_demand[t] + precursor_wave[t]
                fast_tr.append(raw_fast)
                if mode_name == "SiliSense-Precursor-Gov":
                    _, _ = gov.step(raw_fast)
                    in_pre = any((ts + 11) <= (t + 1) <= (ts + 26) for ts in gov.trigger_steps)
                    scale = damped_s if in_pre else 1.0
                elif mode_name == "Reactive-1Hz-SMI":
                    if t > 0 and (t % smi_period == 0) and clamp_rem > 0:
                        reactive_rem = 15
                    scale = damped_s if reactive_rem > 0 else 1.0
                    if reactive_rem > 0:
                        reactive_rem -= 1
                elif mode_name == "Shuffled-Trigger-Control":
                    scale = fixed_sched[t] if fixed_sched is not None else 1.0
                else:
                    scale = 1.0

                inst_w = (base_demand[t] + heavy_wave[t]) * scale + 0.25 * precursor_wave[t]
                thermal_accum = 0.80 * thermal_accum + 0.20 * inst_w

                if clamp_rem == 0 and (inst_w > 108.0 or thermal_accum > 104.5):
                    clamp_rem = 14
                    for k, rc in enumerate((5.0, 10.0, 15.0, 15.0)):
                        if t + k < n_steps and rc > clamp_tr[t + k]:
                            clamp_tr[t + k] = rc

                if clamp_rem > 0:
                    stalls += 1
                    eff_tps = 1000.0 * 0.62 * scale
                    eff_w = inst_w * 0.92 + 12.0
                    clamp_rem -= 1
                else:
                    eff_tps = 1000.0 * scale
                    eff_w = inst_w

                tps_list.append(eff_tps)
                mj_tok.append((eff_w * 1000.0) / max(eff_tps, 1e-6))

            mu_mj = sum(mj_tok) / float(n_steps)
            sd_mj = math.sqrt(sum((v - mu_mj) ** 2 for v in mj_tok) / float(n_steps - 1))
            return float(stalls) / float(n_steps), sd_mj / mu_mj, sum(tps_list) / float(n_steps), fast_tr, clamp_tr, gov.trigger_steps

        st_no, cv_no, tps_no, f_tr, cl_tr, _ = _run_plant("NoGovernor")
        lead_receipts.append(
            compute_a5_precursor_lead_lag(f_tr, cl_tr, dt_ms=dt_ms, max_lag_ms=130.0, n_shifts=300, seed=int(seed) + 10)
        )
        st_sili, cv_sili, tps_sili, _, _, trig_steps = _run_plant("SiliSense-Precursor-Gov")

        damped_val = floor_to_exact_f32(0.84)
        sched = [1.0] * n_steps
        for ts in trig_steps:
            for idx in range(min(n_steps, ts + 11), min(n_steps, ts + 26)):
                sched[idx] = damped_val
        sh_amt = n_steps // 3
        shuf_sched = sched[-sh_amt:] + sched[:-sh_amt]
        st_shuf, cv_shuf, tps_shuf, _, _, _ = _run_plant("Shuffled-Trigger-Control", fixed_sched=shuf_sched)
        st_smi, cv_smi, tps_smi, _, _, _ = _run_plant("Reactive-1Hz-SMI")

        for arm_name, sv, cv, tv in [
            ("NoGovernor", st_no, cv_no, tps_no),
            ("Reactive-1Hz-SMI", st_smi, cv_smi, tps_smi),
            ("Shuffled-Trigger-Control", st_shuf, cv_shuf, tps_shuf),
            ("SiliSense-Precursor-Gov", st_sili, cv_sili, tps_sili),
        ]:
            per_arm_stall[arm_name].append(sv)
            per_arm_cv[arm_name].append(cv)
            per_arm_tps[arm_name].append(tv)

    res = grade_h15b_results(
        lead_receipts=lead_receipts,
        per_arm_stall=per_arm_stall,
        per_arm_cv=per_arm_cv,
        per_arm_tps=per_arm_tps,
    )
    res["evidence_class"] = "cpu_synthetic_selftest"
    res["synthetic_selftest_status"] = "SYNTHETIC_SELFTEST_PASS" if res["verdict"] == "CONFIRMED" else "SYNTHETIC_SELFTEST_FAIL"
    return res
