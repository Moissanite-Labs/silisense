"""PREREG-S115 H15a ABABAB Evaluator, H15c Kernel Autotuner, and 5-Gate Pure-Fraction Ratchet."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from silisense.airlock import AudioAuthorizationGate
from silisense.errors import require_ok
from silisense.governor import PreAdaptationResidualLatch, paired_stats
from silisense.lenses import SiliSenseProfiler

LOCKED_LAMBDA_A3 = 0.15
LOCKED_LAMBDA_A7 = 0.15
LOCKED_LAMBDA_A5 = 0.10
LOCKED_SIGNIFICANCE_HINGE_Z = 2.0


class PairedABABABEvaluator:
    """PREREG-S115 H15a evaluator across interleaved ABABAB blocks + equal-wattage control."""

    def __init__(
        self,
        *,
        mode: str = "telemetry-only",
        auth_gate: AudioAuthorizationGate | None = None,
        n_iaaft: int = 249,
        n_shifts: int = 300,
    ) -> None:
        self.mode = mode
        self.auth_gate = auth_gate
        self.n_iaaft = int(n_iaaft)
        self.n_shifts = int(n_shifts)

    def evaluate_h15a(
        self,
        *,
        standard_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        equal_watt_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        allow_offline_dmic: bool = False,
    ) -> dict[str, Any]:
        require_ok(len(standard_pairs) >= 3, "ERR_H15A_PAIRS", "H15a requires >= 3 paired AB blocks")
        require_ok(len(equal_watt_pairs) >= 3, "ERR_H15A_EQW_PAIRS", "H15a equal-watt gate requires >= 3 pairs")

        eager_a3: list[float] = []
        fused_a3: list[float] = []
        eager_a7: list[float] = []
        fused_a7: list[float] = []
        any_sig = False

        for idx, (tr_e, tr_f) in enumerate(standard_pairs):
            tok_e = int(tr_e.get("tokens_processed", 0))
            tok_f = int(tr_f.get("tokens_processed", 0))
            if tok_e > 0 or tok_f > 0:
                require_ok(tok_e == tok_f, "ERR_CS08_TOKEN_MISMATCH", f"Pair {idx} token mismatch: {tok_e} vs {tok_f}")

            cur_surr = self.n_iaaft if not any_sig else min(self.n_iaaft, 39)
            cur_sh = self.n_shifts if not any_sig else min(self.n_shifts, 60)
            pe = SiliSenseProfiler(
                mode=self.mode, auth_gate=self.auth_gate, n_iaaft=cur_surr, n_shifts=cur_sh, seed=100 + 2 * idx
            ).profile_trace(tr_e, allow_offline_dmic=allow_offline_dmic)
            pf = SiliSenseProfiler(
                mode=self.mode, auth_gate=self.auth_gate, n_iaaft=min(cur_surr, 39) if pe["any_non_textbook_significant_0p005"] else cur_surr, n_shifts=cur_sh, seed=101 + 2 * idx
            ).profile_trace(tr_f, allow_offline_dmic=allow_offline_dmic)

            if pe["any_non_textbook_significant_0p005"] or pf["any_non_textbook_significant_0p005"]:
                any_sig = True
            eager_a3.append(float(pe["observables"]["A3_telemetry"]["abs_asymmetry"]))
            fused_a3.append(float(pf["observables"]["A3_telemetry"]["abs_asymmetry"]))
            eager_a7.append(float(pe["observables"]["A7_uncore_copula_mi"]["copula_mi_nats"]))
            fused_a7.append(float(pf["observables"]["A7_uncore_copula_mi"]["copula_mi_nats"]))

        st_a3 = paired_stats(eager_a3, fused_a3)
        st_a7 = paired_stats(eager_a7, fused_a7)

        eq_e3: list[float] = []
        eq_f3: list[float] = []
        eq_e7: list[float] = []
        eq_f7: list[float] = []
        max_w_diff = 0.0
        eqw_surr = min(self.n_iaaft, 29)
        eqw_sh = min(self.n_shifts, 60)
        for idx, (tr_e, tr_f) in enumerate(equal_watt_pairs):
            pe = SiliSenseProfiler(
                mode=self.mode, auth_gate=self.auth_gate, n_iaaft=eqw_surr, n_shifts=eqw_sh, seed=500 + 2 * idx
            ).profile_trace(tr_e, allow_offline_dmic=allow_offline_dmic)
            pf = SiliSenseProfiler(
                mode=self.mode, auth_gate=self.auth_gate, n_iaaft=eqw_surr, n_shifts=eqw_sh, seed=501 + 2 * idx
            ).profile_trace(tr_f, allow_offline_dmic=allow_offline_dmic)
            we = float(pe["mean_gpu_power_w"])
            wf = float(pf["mean_gpu_power_w"])
            rw = abs(we - wf) / max(wf, 1e-9)
            if rw > max_w_diff:
                max_w_diff = rw
            eq_e3.append(float(pe["observables"]["A3_telemetry"]["abs_asymmetry"]))
            eq_f3.append(float(pf["observables"]["A3_telemetry"]["abs_asymmetry"]))
            eq_e7.append(float(pe["observables"]["A7_uncore_copula_mi"]["copula_mi_nats"]))
            eq_f7.append(float(pf["observables"]["A7_uncore_copula_mi"]["copula_mi_nats"]))

        eq_a3 = paired_stats(eq_e3, eq_f3)
        eq_a7 = paired_stats(eq_e7, eq_f7)

        a3_ok = bool(st_a3["rel_change_pct"] <= -20.0 and st_a3["z_paired"] <= -2.0)
        a7_ok = bool(st_a7["rel_change_pct"] <= -20.0 and st_a7["z_paired"] <= -2.0)
        crit1 = bool(any_sig)
        crit2 = bool(a3_ok and a7_ok)
        crit3 = bool(max_w_diff <= 0.05 and (eq_a3["z_paired"] <= -2.0 or eq_a7["z_paired"] <= -2.0))

        if crit1 and crit2 and crit3:
            verdict = "CONFIRMED"
        elif crit1 and (a3_ok or a7_ok):
            verdict = "PARTIAL"
        else:
            verdict = "FALSIFIED"

        return {
            "hypothesis": "H15a",
            "verdict": verdict,
            "criterion_1_non_gaussian_activity_0p005": crit1,
            "criterion_2_both_a3_and_a7_reduced_20pct": crit2,
            "criterion_3_equal_watt_non_confound": crit3,
            "max_equal_watt_rel_diff": round(max_w_diff, 6),
            "standard_ab_stats": {"abs_A3": st_a3, "A7_copula_mi_nats": st_a7},
            "equal_watt_ab_stats": {"abs_A3": eq_a3, "A7_copula_mi_nats": eq_a7},
        }


@dataclass(frozen=True)
class HoloSphereKernelConfig:
    """Discrete kernel/execution configuration at locked CS-08 effective tokens/step."""

    config_id: str
    warp_tile_t: int
    num_warps: int
    micro_batch: int
    grad_accum: int
    seq_len: int
    grad_ckpt: int

    def effective_tokens_per_step(self) -> int:
        return int(self.micro_batch) * int(self.grad_accum) * int(self.seq_len)


class SiliSenseKernelAutotuner:
    """PREREG-S115 H15c multi-objective kernel autotuner with locked weights."""

    def __init__(
        self,
        *,
        lambda_a3: float = LOCKED_LAMBDA_A3,
        lambda_a7: float = LOCKED_LAMBDA_A7,
        lambda_a5: float = LOCKED_LAMBDA_A5,
        hinge_z: float = LOCKED_SIGNIFICANCE_HINGE_Z,
    ) -> None:
        require_ok(
            abs(lambda_a3 - LOCKED_LAMBDA_A3) <= 1e-12
            and abs(lambda_a7 - LOCKED_LAMBDA_A7) <= 1e-12
            and abs(lambda_a5 - LOCKED_LAMBDA_A5) <= 1e-12
            and abs(hinge_z - LOCKED_SIGNIFICANCE_HINGE_Z) <= 1e-12,
            "ERR_LOCKED_WEIGHT_TAMPER",
            "PREREG-S115 forbids altering locked composite objective weights (0.15, 0.15, 0.10, z=2.0)",
        )
        self.lambda_a3 = float(lambda_a3)
        self.lambda_a7 = float(lambda_a7)
        self.lambda_a5 = float(lambda_a5)
        self.hinge_z = float(hinge_z)

    def select_winners(self, calibration_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        require_ok(len(calibration_rows) >= 2, "ERR_TOO_FEW_CONFIGS", "Need >= 2 candidate configs")
        eff_set = {row["config"].effective_tokens_per_step() for row in calibration_rows}
        require_ok(len(eff_set) == 1, "ERR_CS08_EFFECTIVE_BATCH_MISMATCH", f"CS-08 effective batch mismatch: {sorted(eff_set)}")

        min_edp = min(float(r["cal_edp"]) for r in calibration_rows)
        require_ok(min_edp > 0.0, "ERR_BAD_EDP", "cal_edp must be > 0")
        scored: list[dict[str, Any]] = []
        for row in calibration_rows:
            cfg: HoloSphereKernelConfig = row["config"]
            tps = float(row["cal_tok_per_s"])
            edp = float(row["cal_edp"])
            require_ok(tps > 0.0, "ERR_BAD_TPS", "cal_tok_per_s must be > 0")
            prec = float(row["a5_precursor_rate"])
            require_ok(0.0 <= prec <= 1.0, "ERR_BAD_PREC_RATE", f"Invalid precursor rate {prec}")
            j_val = (
                (edp / min_edp)
                + self.lambda_a3 * max(0.0, abs(float(row["z_a3"])) - self.hinge_z)
                + self.lambda_a7 * max(0.0, float(row["z_a7"]) - self.hinge_z)
                + self.lambda_a5 * prec
            )
            scored.append(
                {
                    "config_id": cfg.config_id,
                    "cal_tok_per_s": round(tps, 4),
                    "cal_edp": round(edp, 6),
                    "z_a3": round(float(row["z_a3"]), 4),
                    "z_a7": round(float(row["z_a7"]), 4),
                    "a5_precursor_rate": round(prec, 4),
                    "composite_j": round(j_val, 6),
                }
            )

        return {
            "Scalar-TokPerSec-Tuner": max(scored, key=lambda r: r["cal_tok_per_s"])["config_id"],
            "Scalar-MeanWatt-EDP-Tuner": min(scored, key=lambda r: r["cal_edp"])["config_id"],
            "SiliSense-Composite-Tuner": min(scored, key=lambda r: r["composite_j"])["config_id"],
            "scored_candidates": scored,
        }

    @staticmethod
    def grade_h15c(
        *,
        tps_sustained_edp: Sequence[float],
        edp0_sustained_edp: Sequence[float],
        sili_sustained_edp: Sequence[float],
        edp0_tail_drift: Sequence[float],
        sili_tail_drift: Sequence[float],
        selection_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        vs_tps = paired_stats(tps_sustained_edp, sili_sustained_edp)
        vs_edp0 = paired_stats(edp0_sustained_edp, sili_sustained_edp)
        drift_vs = paired_stats(edp0_tail_drift, sili_tail_drift)

        crit1 = bool(vs_tps["rel_change_pct"] <= -5.0 and vs_tps["z_paired"] <= -2.0)
        crit2 = bool(
            (vs_edp0["rel_change_pct"] <= -2.0 and vs_edp0["z_paired"] <= -2.0)
            or (drift_vs["rel_change_pct"] <= -15.0 and drift_vs["z_paired"] <= -2.0)
        )
        if crit1 and crit2:
            verdict = "CONFIRMED"
        elif crit1 and abs(vs_edp0["rel_change_pct"]) <= 2.0:
            verdict = "PARTIAL"
        else:
            verdict = "FALSIFIED"

        return {
            "hypothesis": "H15c",
            "verdict": verdict,
            "selected_configs": {
                k: selection_summary[k]
                for k in ("Scalar-TokPerSec-Tuner", "Scalar-MeanWatt-EDP-Tuner", "SiliSense-Composite-Tuner")
            },
            "criterion_1_edp_vs_scalar_tps": crit1,
            "criterion_2_edp_or_drift_vs_mean_watt": crit2,
            "sustained_edp_vs_Scalar_TokPerSec": vs_tps,
            "sustained_edp_vs_Scalar_MeanWatt_EDP": vs_edp0,
            "tail_edp_drift_vs_Scalar_MeanWatt_EDP": drift_vs,
        }


def two_step_mean_watt_quotient_fork_witness() -> dict[str, Any]:
    """Exact rational witness of the 2-step `mean_watt` quotient fork.

    Schedule A = (95 W, 95 W) and Schedule B = (60 W, 130 W) share identical
    2-step mean wattage (95 W), yet Schedule B triggers >= 110 W SwPowerCap
    throttling on step 2, inflating real EDP from 95/1 to 1275/8 (= 159.375).
    """
    sched_steady = (Fraction(95), Fraction(95))
    sched_bursty = (Fraction(60), Fraction(130))
    mean_steady = (sched_steady[0] + sched_steady[1]) / Fraction(2)
    mean_bursty = (sched_bursty[0] + sched_bursty[1]) / Fraction(2)
    require_ok(mean_steady == mean_bursty, "ERR_FORK_MEAN", "Both schedules must share exact mean wattage")

    # Step time (ms/tok): 1 at <= 110 W, 3/2 under SwPowerCap (> 110 W)
    dt_steady = (Fraction(1), Fraction(1))
    dt_bursty = (Fraction(1), Fraction(3, 2))
    energy_steady = sum(w * dt for w, dt in zip(sched_steady, dt_steady)) / Fraction(2)
    energy_bursty = sum(w * dt for w, dt in zip(sched_bursty, dt_bursty)) / Fraction(2)
    edp_steady = energy_steady * (sum(dt_steady) / Fraction(2))
    edp_bursty = energy_bursty * (sum(dt_bursty) / Fraction(2))

    return {
        "shared_mean_watt_q": f"{mean_steady.numerator}/{mean_steady.denominator}",
        "edp_steady_q": f"{edp_steady.numerator}/{edp_steady.denominator}",
        "edp_bursty_q": f"{edp_bursty.numerator}/{edp_bursty.denominator}",
        "quotient_fork_proven": bool(mean_steady == mean_bursty and edp_bursty > edp_steady),
    }


class FiveGateAntiGoodhartRatchet:
    """Pure-Fraction 5-gate anti-Goodhart ratchet preventing tok/s gaming over mJ/tok & thermals."""

    def __init__(
        self,
        *,
        min_edp_rel_gain_q: Fraction = Fraction(1, 50),
        max_tail_drift_q: Fraction = Fraction(1, 10),
        max_shock_z_q: Fraction = Fraction(7, 2),
    ) -> None:
        self.min_edp_rel_gain_q = Fraction(min_edp_rel_gain_q)
        self.max_tail_drift_q = Fraction(max_tail_drift_q)
        self.max_shock_z_q = Fraction(max_shock_z_q)

    def evaluate(
        self,
        *,
        base_effective_tokens: int,
        cand_effective_tokens: int,
        base_edp_q: Fraction,
        cand_edp_q: Fraction,
        base_throttle_frac_q: Fraction,
        cand_throttle_frac_q: Fraction,
        cand_edp_head_q: Fraction,
        cand_edp_tail_q: Fraction,
        pre_adapt_latch_sha256: str,
        cand_max_shock_z_q: Fraction,
        expected_latch: PreAdaptationResidualLatch | None = None,
    ) -> dict[str, Any]:
        for qv in (base_edp_q, cand_edp_q, base_throttle_frac_q, cand_throttle_frac_q, cand_edp_head_q, cand_edp_tail_q, cand_max_shock_z_q):
            require_ok(isinstance(qv, Fraction), "ERR_RATCHET_NON_EXACT", f"Ratchet requires exact Fraction, got {type(qv).__name__}")
        require_ok(base_edp_q > 0 and cand_edp_q > 0 and cand_edp_head_q > 0, "ERR_RATCHET_POS", "EDP values must be > 0")
        require_ok(
            isinstance(expected_latch, PreAdaptationResidualLatch),
            "ERR_RATCHET_LATCH_REQUIRED",
            "G4 anti-Goodhart check requires an actual PreAdaptationResidualLatch instance to verify pre_adapt_latch_sha256",
        )

        g1_cs08 = bool(int(base_effective_tokens) == int(cand_effective_tokens) and int(cand_effective_tokens) > 0)
        rel_gain_q = (base_edp_q - cand_edp_q) / base_edp_q
        g2_edp = bool(rel_gain_q >= self.min_edp_rel_gain_q)
        tail_drift_q = (cand_edp_tail_q - cand_edp_head_q) / cand_edp_head_q
        g3_thermal = bool(tail_drift_q <= self.max_tail_drift_q and cand_throttle_frac_q <= base_throttle_frac_q)
        recomputed_latch_sha = expected_latch.latch_sha256()  # type: ignore[union-attr]
        g4_latch = bool(
            isinstance(pre_adapt_latch_sha256, str)
            and len(pre_adapt_latch_sha256) == 64
            and pre_adapt_latch_sha256 == recomputed_latch_sha
        )
        g5_no_shock_gaming = bool(cand_max_shock_z_q <= self.max_shock_z_q)

        all_pass = bool(g1_cs08 and g2_edp and g3_thermal and g4_latch and g5_no_shock_gaming)
        return {
            "accepted": all_pass,
            "gates": {
                "G1_CS08_effective_batch_parity": g1_cs08,
                "G2_sustained_edp_gain": g2_edp,
                "G3_no_thermal_or_throttle_gaming": g3_thermal,
                "G4_pre_adaptation_latch_pinned": g4_latch,
                "G5_quotient_fork_shock_ceiling": g5_no_shock_gaming,
            },
            "edp_rel_gain_exact_q": f"{rel_gain_q.numerator}/{rel_gain_q.denominator}",
            "tail_drift_exact_q": f"{tail_drift_q.numerator}/{tail_drift_q.denominator}",
        }
