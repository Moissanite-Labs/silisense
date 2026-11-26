#!/usr/bin/env python3
"""Deterministic Pure-Stdlib CPU Synthetic Selftest & Offline Trace Runner."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silisense.airlock import AudioAuthorizationGate, validate_endpoint_path
from silisense.autotuner import (
    FiveGateAntiGoodhartRatchet,
    HoloSphereKernelConfig,
    PairedABABABEvaluator,
    SiliSenseKernelAutotuner,
    two_step_mean_watt_quotient_fork_witness,
)
from silisense.errors import require_ok
from silisense.governor import (
    PreAdaptationResidualLatch,
    check_delay_bandwidth_lock,
    simulate_sustained_governor_benchmark,
)
from silisense.holosphere_bridge import HoloSphereTuningProfile
from silisense.lenses import (
    PREREG_S115_ID,
    SiliSenseProfiler,
    make_synthetic_workload_trace,
)


def _selftest_label(gate_verdict: str) -> str:
    return "SYNTHETIC_SELFTEST_PASS" if gate_verdict == "CONFIRMED" else f"SYNTHETIC_SELFTEST_{gate_verdict}"


def run_cpu_preflight() -> dict[str, Any]:
    std_pairs = [
        (
            {**make_synthetic_workload_trace("eager_microkernel", n_samples=512, seed=10 + s, target_mean_gpu_w=98.0), "tokens_processed": 16384},
            {**make_synthetic_workload_trace("fused_holosphere", n_samples=512, seed=20 + s, target_mean_gpu_w=91.0), "tokens_processed": 16384},
        )
        for s in range(4)
    ]
    eqw_pairs = [
        (
            {**make_synthetic_workload_trace("eager_microkernel", n_samples=512, seed=30 + s, target_mean_gpu_w=92.0), "tokens_processed": 16384},
            {**make_synthetic_workload_trace("fused_holosphere", n_samples=512, seed=40 + s, target_mean_gpu_w=92.0), "tokens_processed": 16384},
        )
        for s in range(4)
    ]
    h15a = PairedABABABEvaluator(mode="telemetry-only", n_iaaft=609, n_shifts=300).evaluate_h15a(
        standard_pairs=std_pairs,
        equal_watt_pairs=eqw_pairs,
    )
    h15b = simulate_sustained_governor_benchmark(seeds=(42, 137, 2026, 777), n_steps=800, dt_ms=5.0)

    configs = [
        HoloSphereKernelConfig("cfg_aggressive_mb16_w8", 32, 8, 16, 1, 512, 0),
        HoloSphereKernelConfig("cfg_bursty_low_edp_cal", 64, 8, 8, 2, 512, 0),
        HoloSphereKernelConfig("cfg_warp_clean_float4", 32, 4, 8, 2, 512, 0),
    ]
    tuner = SiliSenseKernelAutotuner()
    sel = tuner.select_winners(
        [
            {"config": configs[0], "cal_tok_per_s": 14800.0, "cal_edp": 0.505, "z_a3": -6.8, "z_a7": 7.4, "a5_precursor_rate": 0.62},
            {"config": configs[1], "cal_tok_per_s": 14200.0, "cal_edp": 0.472, "z_a3": -5.1, "z_a7": 5.9, "a5_precursor_rate": 0.48},
            {"config": configs[2], "cal_tok_per_s": 14100.0, "cal_edp": 0.484, "z_a3": -0.9, "z_a7": 0.8, "a5_precursor_rate": 0.04},
        ]
    )
    h15c = tuner.grade_h15c(
        tps_sustained_edp=[0.612, 0.619, 0.608, 0.615],
        edp0_sustained_edp=[0.548, 0.553, 0.545, 0.551],
        sili_sustained_edp=[0.491, 0.495, 0.489, 0.493],
        edp0_tail_drift=[0.162, 0.171, 0.158, 0.166],
        sili_tail_drift=[0.028, 0.031, 0.026, 0.030],
        selection_summary=sel,
    )

    pre_prof = SiliSenseProfiler(mode="telemetry-only", n_iaaft=99, n_shifts=120, seed=42).profile_trace(std_pairs[0][0])
    pre_latch = PreAdaptationResidualLatch.capture_from_profile(pre_prof)
    ratchet = FiveGateAntiGoodhartRatchet().evaluate(
        base_effective_tokens=8192,
        cand_effective_tokens=8192,
        base_edp_q=Fraction(550, 1000),
        cand_edp_q=Fraction(492, 1000),
        base_throttle_frac_q=Fraction(18, 100),
        cand_throttle_frac_q=Fraction(5, 100),
        cand_edp_head_q=Fraction(485, 1000),
        cand_edp_tail_q=Fraction(499, 1000),
        pre_adapt_latch_sha256=pre_latch.latch_sha256(),
        cand_max_shock_z_q=Fraction(11, 10),
        expected_latch=pre_latch,
    )

    payload = {
        "schema": "silisense.prereg_s115.cpu_preflight.v2",
        "evidence_class": "cpu_synthetic_preflight",
        "prereg_id": PREREG_S115_ID,
        "default_mode": "telemetry-only",
        "camera_policy": "ERR_CAMERA_FORBIDDEN",
        "H15a_synthetic_selftest": _selftest_label(h15a["verdict"]),
        "H15b_synthetic_selftest": _selftest_label(h15b["verdict"]),
        "H15c_synthetic_selftest": _selftest_label(h15c["verdict"]),
        "H15c_selected": h15c["selected_configs"],
        "ratchet_accepted": ratchet["accepted"],
        "quotient_fork_proven": two_step_mean_watt_quotient_fork_witness()["quotient_fork_proven"],
        "delay_bw_rapl_5ms_at_20hz_stable": check_delay_bandwidth_lock(20.0, 0.005)["stable"],
        "delay_bw_psys_80ms_at_5hz_stable": check_delay_bandwidth_lock(5.0, 0.080)["stable"],
        "default_tuning_profile_sha256": HoloSphereTuningProfile().profile_sha256(),
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SiliSense Synthetic CPU Preflight & Offline Trace Suite")
    parser.add_argument("--cpu-preflight", action="store_true", help="Run deterministic pure-stdlib synthetic CPU selftest")
    parser.add_argument("--profile-trace-json", type=Path, default=None, help="Profile an offline telemetry JSON trace")
    parser.add_argument("--enable-dmic-authorized", action="store_true", help="Opt-in Mode 2 DMIC authorization")
    parser.add_argument("--authorize-audio-window-sec", type=float, default=None, help="Bounded audio window in (0, 300] s")
    parser.add_argument("--pi-authorization-token", type=str, default=None, help="Explicit PI authorization token")
    parser.add_argument("--json", action="store_true", help="Emit indented JSON output")
    args = parser.parse_args(argv)

    gate: AudioAuthorizationGate | None = None
    mode = "telemetry-only"
    if args.enable_dmic_authorized or args.authorize_audio_window_sec is not None or args.pi_authorization_token is not None:
        gate = AudioAuthorizationGate.create(
            enable_dmic_authorized=bool(args.enable_dmic_authorized),
            authorize_audio_window_sec=args.authorize_audio_window_sec,
            pi_authorization_token=args.pi_authorization_token,
        )
        mode = "dmic-authorized"

    if args.profile_trace_json is not None:
        validate_endpoint_path(args.profile_trace_json, auth_gate=gate)
        raw = json.loads(args.profile_trace_json.read_text(encoding="utf-8"))
        out = SiliSenseProfiler(mode=mode, auth_gate=gate).profile_trace(raw, allow_offline_dmic=False)
        sys.stdout.write(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
        return 0

    require_ok(
        bool(args.cpu_preflight) or argv is None or len(argv) == 0,
        "ERR_CLI_MODE_REQUIRED",
        "Specify --cpu-preflight or --profile-trace-json <path>",
    )
    report = run_cpu_preflight()
    sys.stdout.write(json.dumps(report, indent=2 if args.json else None, sort_keys=True, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
