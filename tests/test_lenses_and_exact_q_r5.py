"""R5 tests for S113 lenses (A1/A2/A3/A5/A7), S20c exact-Q operators, and 7-tone Fisher ID."""

from __future__ import annotations

import cmath
from fractions import Fraction
import math
from pathlib import Path
import random
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense.errors import SiliSensePermissionError, SiliSenseValueError
from silisense.lenses import (
    SiliSenseProfiler,
    compute_a1_ultrasonic_bicoherence,
    compute_a2_chiral_vrm_balance,
    compute_a3_time_reversal_asymmetry,
    compute_a5_precursor_lead_lag,
    compute_a7_nonlinear_uncore_copula_mi,
    compute_op1_seam1_sherman_morrison_leverage,
    compute_op2_seam3_ordinal_hodge_curl,
    compute_op3_f3_nonnormal_commutator,
    compute_time_locked_7tone_cole_shunt,
    make_synthetic_workload_trace,
    wht_phase_surrogate,
)


class TestLensesAndExactQR5(unittest.TestCase):
    def test_01_null_calibration_zero_false_positives_colored_ar1_and_linear_lags(self) -> None:
        null_tr = make_synthetic_workload_trace("colored_null", n_samples=256, seed=12345)
        prof = SiliSenseProfiler(mode="telemetry-only", n_iaaft=149, n_shifts=200, seed=12345)
        res = prof.profile_trace(null_tr)
        self.assertFalse(res["any_non_textbook_significant_0p005"])
        self.assertLessEqual(res["observables"]["A7_uncore_copula_mi"]["max_abs_residual_linear_corr"], 1e-9)

        rng = random.Random(54321)
        ch_l = [0.0] * 2048
        ch_r = [0.0] * 2048
        for t in range(1, 2048):
            ch_l[t] = 0.50 * ch_l[t - 1] + rng.gauss(0.0, 1.0)
            ch_r[t] = 0.50 * ch_r[t - 1] + rng.gauss(0.0, 1.0)
        a1_n = compute_a1_ultrasonic_bicoherence(ch_l, n_iaaft=49, seed=54321, allow_offline_array=True)
        a2_n = compute_a2_chiral_vrm_balance(ch_l, ch_r, n_surr=49, seed=54321, allow_offline_array=True)
        self.assertFalse(a1_n["significant_0p005"])
        self.assertFalse(a2_n["significant_0p005"])

        # Mode-1 permission check refuses DMIC without gate or explicit offline flag
        with self.assertRaises(SiliSensePermissionError):
            compute_a1_ultrasonic_bicoherence(ch_l, allow_offline_array=False)

    def test_02_positive_recovery_all_five_s113_lenses_pure_stdlib(self) -> None:
        eager_tr = make_synthetic_workload_trace("eager_microkernel", n_samples=512, dt_ms=5.0, seed=99)
        a3 = compute_a3_time_reversal_asymmetry(eager_tr["gpu_power_w"], dt_ms=5.0, n_iaaft=249, seed=99)
        a5 = compute_a5_precursor_lead_lag(
            eager_tr["fast_power_w"], eager_tr["slow_throttle_or_psys"], dt_ms=5.0, max_lag_ms=130.0, n_shifts=300, seed=99
        )
        a7 = compute_a7_nonlinear_uncore_copula_mi(eager_tr["uncore_w"], eager_tr["gpu_power_w"], n_iaaft=249, seed=99)

        self.assertTrue(a3["significant_0p005"])
        self.assertGreaterEqual(abs(a3["z_iaaft"]), 3.5)
        self.assertTrue(a5["significant_0p005"])
        self.assertAlmostEqual(a5["peak_precursor_lag_ms"], -80.0, delta=10.0)
        self.assertTrue(a7["significant_0p005"])
        self.assertLessEqual(a7["max_abs_residual_linear_corr"], 1e-9)

        # A1: Quadratic phase-coupled ultrasonic triad vs random-phase uncoupled control
        fs = 48000
        n_fft = 128
        n_seg = 24
        rng = random.Random(2026)
        coupled: list[float] = []
        uncoupled: list[float] = []
        f1, f2 = 7500.0, 11250.0
        f3 = f1 + f2
        for _ in range(n_seg):
            p1 = rng.uniform(0.0, 2.0 * math.pi)
            p2 = rng.uniform(0.0, 2.0 * math.pi)
            p3_r = rng.uniform(0.0, 2.0 * math.pi)
            for i in range(n_fft):
                ti = float(i) / float(fs)
                n_val = 0.18 * rng.gauss(0.0, 1.0)
                coupled.append(
                    math.cos(2.0 * math.pi * f1 * ti + p1)
                    + math.cos(2.0 * math.pi * f2 * ti + p2)
                    + 0.95 * math.cos(2.0 * math.pi * f3 * ti + p1 + p2)
                    + n_val
                )
                uncoupled.append(
                    math.cos(2.0 * math.pi * f1 * ti + p1)
                    + math.cos(2.0 * math.pi * f2 * ti + p2)
                    + 0.95 * math.cos(2.0 * math.pi * f3 * ti + p3_r)
                    + n_val
                )
        r_c = compute_a1_ultrasonic_bicoherence(coupled, fs=fs, n_fft=n_fft, n_iaaft=59, seed=2026, allow_offline_array=True)
        r_u = compute_a1_ultrasonic_bicoherence(uncoupled, fs=fs, n_fft=n_fft, n_iaaft=59, seed=2026, allow_offline_array=True)
        self.assertGreater(r_c["max_b2"], 0.65)
        self.assertTrue(r_c["significant_0p005"])
        self.assertLess(r_u["max_b2"], 0.25)

        # A2: Chiral quadrature wave + signed CPU-vs-GPU VRM lateral TDOA (+/- 8 samples = +/- 166.7 us)
        n_a2 = 3072
        delay = 8
        vrm_tones = [12200.0 + 187.5 * float(k) for k in range(32)]
        vrm_phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in vrm_tones]
        vrm_wave = [
            sum(0.22 * math.cos(2.0 * math.pi * fk * (float(t) / float(fs)) + pk) for fk, pk in zip(vrm_tones, vrm_phases))
            for t in range(n_a2)
        ]
        ch_cos = [0.85 * math.cos(2.0 * math.pi * 20250.0 * float(t) / float(fs)) for t in range(n_a2)]
        ch_sin = [0.85 * math.sin(2.0 * math.pi * 20250.0 * float(t) / float(fs)) for t in range(n_a2)]

        cpu_l = [vrm_wave[t] + ch_cos[t] + 0.08 * rng.gauss(0.0, 1.0) for t in range(n_a2)]
        cpu_r = [(vrm_wave[t - delay] if t >= delay else 0.0) + ch_sin[t] + 0.08 * rng.gauss(0.0, 1.0) for t in range(n_a2)]
        res_cpu = compute_a2_chiral_vrm_balance(cpu_l, cpu_r, fs=fs, n_surr=59, seed=11, allow_offline_array=True)
        self.assertTrue(res_cpu["significant_0p005"])
        self.assertEqual(res_cpu["dominant_vrm_rail"], "CPU_VRM_LEFT")
        self.assertGreater(res_cpu["tdoa_left_leads_right_us"], 135.0)

        gpu_l = [(vrm_wave[t - delay] if t >= delay else 0.0) + ch_cos[t] + 0.08 * rng.gauss(0.0, 1.0) for t in range(n_a2)]
        gpu_r = [vrm_wave[t] + ch_sin[t] + 0.08 * rng.gauss(0.0, 1.0) for t in range(n_a2)]
        res_gpu = compute_a2_chiral_vrm_balance(gpu_l, gpu_r, fs=fs, n_surr=59, seed=12, allow_offline_array=True)
        self.assertTrue(res_gpu["significant_0p005"])
        self.assertEqual(res_gpu["dominant_vrm_rail"], "GPU_VRM_RIGHT")
        self.assertLess(res_gpu["tdoa_left_leads_right_us"], -135.0)

    def test_03_exact_q_s20c_trio_monotone_invariance_and_7tone_fisher_id(self) -> None:
        # OP1_SEAM1 exact rational Sherman-Morrison shock detection
        int_seq = [10 + (i % 4) for i in range(44)] + [250, 12, 11, 13]
        op1 = compute_op1_seam1_sherman_morrison_leverage(int_seq, window=32)
        self.assertGreater(op1["max_leverage_decimal"], 0.90)
        self.assertEqual(op1["peak_step"], 45)

        # OP2_SEAM3 exact invariance under strictly monotone nonlinear ADC warping x -> x^3 + 5*x
        raw_saw = [float((i * 7) % 11 - 5) + 0.01 * float(i % 3) for i in range(80)]
        warped_saw = [v * v * v + 5.0 * v for v in raw_saw]
        op2_raw = compute_op2_seam3_ordinal_hodge_curl(raw_saw)
        op2_warp = compute_op2_seam3_ordinal_hodge_curl(warped_saw)
        self.assertEqual(op2_raw["phi_asym_exact_q"], op2_warp["phi_asym_exact_q"])
        self.assertEqual(op2_raw["hvg_beta1_per_n_exact_q"], op2_warp["hvg_beta1_per_n_exact_q"])

        # OP3_F3: Schur-stable upper-triangular T (eigenvalues all 1/2 < 1) with transient overshoot (In(I - T^T T) > 0)
        t_nonnormal = [
            [Fraction(1, 2), Fraction(2, 1), Fraction(0)],
            [Fraction(0), Fraction(1, 2), Fraction(2, 1)],
            [Fraction(0), Fraction(0), Fraction(1, 2)],
        ]
        op3 = compute_op3_f3_nonnormal_commutator(t_nonnormal)
        self.assertTrue(op3["transient_overshoot_risk"])
        self.assertGreater(op3["kappa_f3_decimal"], 0.0)

        t_normal = [
            [Fraction(1, 2), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1, 3), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1, 4)],
        ]
        op3_norm = compute_op3_f3_nonnormal_commutator(t_normal)
        self.assertEqual(op3_norm["kappa_f3_exact_q"], "0/1")
        self.assertFalse(op3_norm["transient_overshoot_risk"])

        # WHT_PHASE preserves L2 energy (Parseval)
        vec64 = [float((i % 7) - 3) for i in range(64)]
        surr64 = wht_phase_surrogate(vec64, seed=9)
        self.assertAlmostEqual(sum(v * v for v in vec64), sum(v * v for v in surr64), places=9)

        # ASKA 7-tone Fisher ID: must refuse None delay; 1-pole Debye vanishes (0/35); Cole+shunt fires (35/35)
        tones = [0.5, 1.2, 2.8, 6.5, 15.0, 35.0, 80.0]
        tau_d = 0.080
        tau_rc = 0.050
        h_1pole = [cmath.rect(1.0, -2.0 * math.pi * f * tau_d) / complex(1.0, 2.0 * math.pi * f * tau_rc) for f in tones]
        with self.assertRaises(SiliSenseValueError) as ctx_free:
            compute_time_locked_7tone_cole_shunt(tones, h_1pole, locked_tau_d_sec=None)
        self.assertEqual(ctx_free.exception.code, "ERR_FREE_DELAY_FORBIDDEN")

        id_1pole = compute_time_locked_7tone_cole_shunt(tones, h_1pole, locked_tau_d_sec=tau_d)
        self.assertEqual(id_1pole["nonzero_collinearity_triples"], 0)
        self.assertFalse(id_1pole["shunt_or_fractional_pole_detected"])

        h_cole_shunt = [
            cmath.rect(1.0, -2.0 * math.pi * f * tau_d)
            * (0.72 / (1.0 + (complex(0.0, 2.0 * math.pi * f * tau_rc) ** 0.75)) + 0.28 / complex(1.0, 2.0 * math.pi * f * 0.004))
            for f in tones
        ]
        id_shunt = compute_time_locked_7tone_cole_shunt(tones, h_cole_shunt, locked_tau_d_sec=tau_d)
        self.assertEqual(id_shunt["nonzero_collinearity_triples"], 35)
        self.assertTrue(id_shunt["shunt_or_fractional_pole_detected"])


if __name__ == "__main__":
    unittest.main()
