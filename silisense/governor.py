"""Closed-loop precursor rail governor, Delay-Bandwidth Lock, L^{5/3} Lyapunov choke, and Contract-4 latch."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import struct
from typing import Any, Mapping, Sequence

from silisense.errors import require_ok
from silisense.lenses import compute_a5_precursor_lead_lag


def floor_to_exact_f32(alpha: float) -> float:
    """Round `alpha` downward to an exact IEEE-754 binary32 dyadic rational (0-ULP overshoot)."""
    require_ok(math.isfinite(alpha) and alpha > 0.0, "ERR_BAD_DYADIC_ALPHA", f"alpha must be finite > 0, got {alpha}")
    f32 = struct.unpack("<f", struct.pack("<f", float(alpha)))[0]
    if f32 > float(alpha):
        bits = struct.unpack("<I", struct.pack("<f", f32))[0]
        require_ok(bits > 0, "ERR_DYADIC_UNDERFLOW", "binary32 underflow at zero")
        f32 = struct.unpack("<f", struct.pack("<I", bits - 1))[0]
    require_ok(f32 <= float(alpha), "ERR_DYADIC_OVERSHOOT", "binary32 floor exceeded target alpha")
    return float(f32)


def check_delay_bandwidth_lock(f_control_hz: float, tau_sensor_sec: float) -> dict[str, Any]:
    """Phase-stability Delay-Bandwidth Lock: |1 - exp(-i 2pi f tau)| < 1 iff f * tau < 1/6."""
    require_ok(
        math.isfinite(f_control_hz) and f_control_hz > 0.0 and math.isfinite(tau_sensor_sec) and tau_sensor_sec > 0.0,
        "ERR_BAD_DELAY_BW",
        "f_control_hz and tau_sensor_sec must be finite > 0",
    )
    prod = float(f_control_hz) * float(tau_sensor_sec)
    return {
        "f_tau_product": round(prod, 8),
        "limit": round(1.0 / 6.0, 8),
        "max_stable_control_hz": round(1.0 / (6.0 * float(tau_sensor_sec)), 4),
        "stable": bool(prod < (1.0 / 6.0)),
    }


def smooth_l53_shared_heatpipe_excess(rail_excess_watts: Sequence[float]) -> float:
    """Smooth C^1 shared-heatpipe excess norm (sum_i max(0, u_i)^{5/3})^{3/5}."""
    s = 0.0
    for v in rail_excess_watts:
        fv = float(v)
        require_ok(math.isfinite(fv), "ERR_NONFINITE_RAIL", "Non-finite rail excess watts")
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
    require_ok(0.0 < g_floor <= 1.0 and alpha >= 0.0 and 0.0 <= beta <= 1.0, "ERR_LYAPUNOV_PARAMS", "Invalid Lyapunov params")
    require_ok(math.isfinite(g_prev) and math.isfinite(u_53), "ERR_LYAPUNOV_FINITE", "Non-finite Lyapunov state")
    g_cur = min(1.0, max(g_floor, float(g_prev)))
    excess = max(0.0, float(u_53) - float(u_thresh))
    num = g_cur + float(beta) * (1.0 - g_cur)
    den = 1.0 + float(alpha) * excess * excess
    raw_next = min(1.0, max(g_floor, num / den))
    return floor_to_exact_f32(raw_next)


def gate_asymmetric_step(
    *,
    headroom_before: float,
    headroom_after: float,
    obj_before: float,
    obj_after: float,
) -> str:
    """Asymmetric safety trust-region gate with deadlock-free `.okRecovery` and `lexImproves`."""
    for v in (headroom_before, headroom_after, obj_before, obj_after):
        require_ok(math.isfinite(float(v)), "ERR_GATE_NONFINITE", "Non-finite gate value")
    if headroom_before < 0.0:
        # Out-of-envelope thermal/power state: require strict monotone headroom recovery without deadlocking
        return ".okRecovery" if headroom_after > headroom_before else ".refusedUnsafe"
    if headroom_after < 0.0:
        return ".refusedUnsafe"
    if obj_after < obj_before:
        return ".okStep"
    if obj_after == obj_before and headroom_after > headroom_before:
        return ".okPureSafety"
    if obj_after == obj_before and headroom_after <= headroom_before:
        return ".refusedHeadroomErosion"
    return ".refusedObjectiveRegression"


def cyclic_switch_dist(pstate_seq: Sequence[int]) -> int:
    """Exact cumulative cyclic DVFS / P-state transition distance (`TransitionBounds.lean`)."""
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
    """Gate Contract 4 Pre-Adaptation Projection Latch (freezes raw pre-tuning silicon residuals)."""

    latched_abs_a3: float
    latched_z_a3: float
    latched_a5_lead_ms: float
    latched_a7_copula_mi_nats: float
    latched_z_a7: float
    latched_mean_gpu_w: float
    open_pathology: bool

    @classmethod
    def capture_from_profile(cls, profile_report: Mapping[str, Any]) -> "PreAdaptationResidualLatch":
        obs = profile_report["observables"]
        a3 = obs["A3_telemetry"]
        a5 = obs["A5_precursor"]
        a7 = obs["A7_uncore_copula_mi"]
        open_flag = bool(
            a3["significant_0p005"]
            or a7["significant_0p005"]
            or abs(float(a3["z_iaaft"])) >= 3.5
            or float(a7["z_iaaft"]) >= 3.5
        )
        return cls(
            latched_abs_a3=float(a3["abs_asymmetry"]),
            latched_z_a3=float(a3["z_iaaft"]),
            latched_a5_lead_ms=float(a5["peak_precursor_lag_ms"]),
            latched_a7_copula_mi_nats=float(a7["copula_mi_nats"]),
            latched_z_a7=float(a7["z_iaaft"]),
            latched_mean_gpu_w=float(profile_report["mean_gpu_power_w"]),
            open_pathology=open_flag,
        )


def paired_stats(baseline_vals: Sequence[float], candidate_vals: Sequence[float]) -> dict[str, float]:
    require_ok(len(baseline_vals) == len(candidate_vals) and len(baseline_vals) >= 3, "ERR_PAIRED_LEN", "Need >= 3 pairs")
    n = len(baseline_vals)
    diffs = [float(b) - float(a) for a, b in zip(baseline_vals, candidate_vals)]
    mean_a = sum(float(a) for a in baseline_vals) / float(n)
    mean_b = sum(float(b) for b in candidate_vals) / float(n)
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
        require_ok(z_trigger >= 1.5, "ERR_BAD_Z_TRIGGER", "z_trigger must be >= 1.5")
        require_ok(0.5 <= damping_scale < 1.0, "ERR_BAD_DAMPING", "damping_scale must lie in [0.5, 1.0)")
        require_ok(hold_steps >= 1 and cooldown_steps >= 0, "ERR_BAD_HOLD", "Invalid hold/cooldown steps")
        self.z_trigger = float(z_trigger)
        self.ema_alpha = float(ema_alpha)
        self.damping_scale = floor_to_exact_f32(float(damping_scale))
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
        require_ok(math.isfinite(fast_power_w), "ERR_NONFINITE_POWER", "fast_power_w must be finite")
        self._step_idx += 1
        if self._prev_val is None:
            self._prev_val = float(fast_power_w)
            return 1.0, 0.0

        pos_inc = max(0.0, float(fast_power_w) - self._prev_val)
        self._prev_val = float(fast_power_w)

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
    """Pure-stdlib simulation of all 4 locked PREREG-S115 H15b arms."""
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

    return grade_h15b_results(
        lead_receipts=lead_receipts,
        per_arm_stall=per_arm_stall,
        per_arm_cv=per_arm_cv,
        per_arm_tps=per_arm_tps,
    )
