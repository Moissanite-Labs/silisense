"""R5 tests for H15a/b/c, Lyapunov/Delay-BW governor, 5-Gate Fraction Ratchet, and HoloSphere parity."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense.autotuner import (
    FiveGateAntiGoodhartRatchet,
    HoloSphereKernelConfig,
    PairedABABABEvaluator,
    SiliSenseKernelAutotuner,
    two_step_mean_watt_quotient_fork_witness,
)
from silisense.benchmarks.prereg_s115_suite import run_cpu_preflight
from silisense.errors import SiliSenseValueError
from silisense.governor import (
    PreAdaptationResidualLatch,
    check_delay_bandwidth_lock,
    cyclic_switch_dist,
    floor_to_exact_f32,
    gate_asymmetric_step,
    grade_h15b_results,
    lyapunov_telescoping_choke,
    simulate_sustained_governor_benchmark,
    smooth_l53_shared_heatpipe_excess,
)
from silisense.holosphere_bridge import (
    HoloSphereStepTelemetry,
    HoloSphereTuningProfile,
    SiliSenseHoloSphereBridge,
    build_step_telemetry,
)
from silisense.lenses import SiliSenseProfiler, make_synthetic_workload_trace


class TestGovernorAutotunerBridgeR5(unittest.TestCase):
    def test_01_governor_math_laws_and_h15b_controls(self) -> None:
        # 0-ULP overshoot downward binary32 rounding across 200 non-dyadic values
        for i in range(1, 201):
            a = 0.50 + 0.0023 * float(i)
            f32 = floor_to_exact_f32(a)
            self.assertLessEqual(f32, a)
            self.assertLess(a - f32, 1e-6)

        # Delay-Bandwidth Lock f * tau < 1/6
        self.assertFalse(check_delay_bandwidth_lock(5.0, 0.080)["stable"])
        self.assertTrue(check_delay_bandwidth_lock(20.0, 0.005)["stable"])

        # Smooth L^{5/3} shared-heatpipe norm & Lyapunov telescoping choke
        self.assertEqual(smooth_l53_shared_heatpipe_excess([0.0, -4.0]), 0.0)
        u53 = smooth_l53_shared_heatpipe_excess([10.0, 15.0])
        self.assertGreater(u53, 15.0)
        g1 = lyapunov_telescoping_choke(1.0, u53, u_thresh=2.0, alpha=0.05, beta=0.10, g_floor=0.70)
        self.assertLess(g1, 1.0)
        self.assertGreaterEqual(g1, floor_to_exact_f32(0.70))

        # gateAsymmetric: deadlock-free .okRecovery when headroom < 0, refusal on tied-obj headroom erosion
        self.assertEqual(gate_asymmetric_step(headroom_before=-5.0, headroom_after=-2.0, obj_before=1.0, obj_after=1.1), ".okRecovery")
        self.assertEqual(gate_asymmetric_step(headroom_before=-5.0, headroom_after=-6.0, obj_before=1.0, obj_after=0.9), ".refusedUnsafe")
        self.assertEqual(gate_asymmetric_step(headroom_before=4.0, headroom_after=3.5, obj_before=1.0, obj_after=1.0), ".refusedHeadroomErosion")
        self.assertEqual(gate_asymmetric_step(headroom_before=4.0, headroom_after=5.0, obj_before=1.0, obj_after=1.0), ".okPureSafety")

        # cyclicSwitchDist
        self.assertEqual(cyclic_switch_dist([0, 2, 1]), 4)

        # H15b full 4-arm benchmark & shuffled-control spoof refusal
        h15b = simulate_sustained_governor_benchmark(seeds=(42, 137, 2026, 909), n_steps=800, dt_ms=5.0)
        self.assertEqual(h15b["verdict"], "CONFIRMED")

        spoof = grade_h15b_results(
            lead_receipts=[{"p_circ": 0.002, "peak_precursor_lag_ms": -80.0} for _ in range(4)],
            per_arm_stall={k: [0.22, 0.23, 0.21, 0.24] if "No" in k or "SMI" in k else [0.06] * 4 for k in ("NoGovernor", "Reactive-1Hz-SMI", "Shuffled-Trigger-Control", "SiliSense-Precursor-Gov")},
            per_arm_cv={k: [0.20, 0.21, 0.19, 0.22] if "No" in k or "SMI" in k else [0.08] * 4 for k in ("NoGovernor", "Reactive-1Hz-SMI", "Shuffled-Trigger-Control", "SiliSense-Precursor-Gov")},
            per_arm_tps={k: [920.0] * 4 for k in ("NoGovernor", "Reactive-1Hz-SMI", "Shuffled-Trigger-Control", "SiliSense-Precursor-Gov")},
        )
        self.assertNotEqual(spoof["verdict"], "CONFIRMED")

    def test_02_h15a_h15c_five_gate_ratchet_and_quotient_fork(self) -> None:
        preflight = run_cpu_preflight()
        self.assertEqual(preflight["H15a_verdict"], "CONFIRMED")
        self.assertEqual(preflight["H15b_verdict"], "CONFIRMED")
        self.assertEqual(preflight["H15c_verdict"], "CONFIRMED")
        self.assertTrue(preflight["ratchet_accepted"])
        self.assertTrue(preflight["quotient_fork_proven"])

        fork = two_step_mean_watt_quotient_fork_witness()
        self.assertTrue(fork["quotient_fork_proven"])
        self.assertEqual(fork["shared_mean_watt_q"], "95/1")

        # Tampering with locked H15c weights MUST fail closed
        with self.assertRaises(SiliSenseValueError) as ctx_w:
            SiliSenseKernelAutotuner(lambda_a3=0.50)
        self.assertEqual(ctx_w.exception.code, "ERR_LOCKED_WEIGHT_TAMPER")

        # 5-gate anti-Goodhart ratchet MUST reject tok/s gaming that breaches thermal/throttle or shock ceiling
        ratchet = FiveGateAntiGoodhartRatchet()
        latch_hex = hashlib.sha256(b"latch").hexdigest()
        rej_shock = ratchet.evaluate(
            base_effective_tokens=8192,
            cand_effective_tokens=8192,
            base_edp_q=Fraction(55, 100),
            cand_edp_q=Fraction(48, 100),
            base_throttle_frac_q=Fraction(10, 100),
            cand_throttle_frac_q=Fraction(8, 100),
            cand_edp_head_q=Fraction(48, 100),
            cand_edp_tail_q=Fraction(49, 100),
            pre_adapt_latch_sha256=latch_hex,
            cand_max_shock_z_q=Fraction(6, 1),
        )
        self.assertFalse(rej_shock["accepted"])
        self.assertFalse(rej_shock["gates"]["G5_quotient_fork_shock_ceiling"])

        # Pre-Adaptation Latch captures open pathology on eager trace
        prof_e = SiliSenseProfiler(mode="telemetry-only", n_iaaft=149, n_shifts=150, seed=55).profile_trace(
            make_synthetic_workload_trace("eager_microkernel", n_samples=256, seed=55)
        )
        self.assertTrue(PreAdaptationResidualLatch.capture_from_profile(prof_e).open_pathology)

        # CS-08 token mismatch in H15a MUST fail closed
        ev = PairedABABABEvaluator(mode="telemetry-only", n_iaaft=99, n_shifts=100)
        t_a = {**make_synthetic_workload_trace("eager_microkernel", n_samples=128, seed=1), "tokens_processed": 4096}
        t_b = {**make_synthetic_workload_trace("fused_holosphere", n_samples=128, seed=2), "tokens_processed": 2048}
        with self.assertRaises(SiliSenseValueError) as ctx_cs:
            ev.evaluate_h15a(standard_pairs=[(t_a, t_b)] * 3, equal_watt_pairs=[(t_a, t_a)] * 3)
        self.assertEqual(ctx_cs.exception.code, "ERR_CS08_TOKEN_MISMATCH")

    def test_03_holosphere_bridge_wire_and_sha256_parity(self) -> None:
        bridge = SiliSenseHoloSphereBridge(
            base_profile=HoloSphereTuningProfile(backend="cuda_sm86", block_t=128, microbatch_size=8, r_loops_override=3)
        )
        for s_idx in range(15):
            tel = build_step_telemetry(
                step_idx=s_idx,
                seq_len=512,
                batch_size=8,
                r_loops=3,
                backend="cuda_sm86",
                block_t=128,
                step_ms=78.068,
                norm_drift_max=1.8e-7,
            )
            tel_rt = HoloSphereStepTelemetry.from_dict(json.loads(tel.to_json()))
            self.assertEqual(tel, tel_rt)
            out_p = bridge.on_step_boundary(tel, fast_power_w=92.0 if s_idx < 14 else 128.0)
            if s_idx == 14:
                self.assertGreater(out_p.inter_step_cooldown_ms, 0.0)

        p_sili = HoloSphereTuningProfile(
            backend="cuda_sm86", block_t=128, microbatch_size=8, inter_step_cooldown_ms=1.25, r_loops_override=3
        )
        self.assertEqual(p_sili, HoloSphereTuningProfile.from_dict(json.loads(p_sili.to_json())))

        # Sibling check: verify byte-exact profile_sha256() with ../holosphere/holosphere/telemetry.py when present
        repo_root = Path(__file__).resolve().parent.parent
        holo_root = repo_root.parent / "holosphere"
        if (holo_root / "holosphere" / "telemetry.py").is_file():
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    f"import sys; sys.path.insert(0, {str(holo_root)!r}); "
                    f"from holosphere.telemetry import HoloSphereTuningProfile as P; "
                    f"print(P.from_dict({p_sili.to_dict()!r}).profile_sha256())",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), p_sili.profile_sha256())


if __name__ == "__main__":
    unittest.main()
