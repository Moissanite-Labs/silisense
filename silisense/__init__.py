"""SiliSense: Fail-closed silicon physics metrology, precursor governor, and EDP autotuner."""

from __future__ import annotations

from silisense.airlock import (
    AudioAuthorizationGate,
    RaplSysfsReader,
    query_nvidia_smi_once,
    query_proc_irq_and_cpu_khz,
    validate_endpoint_path,
)
from silisense.autotuner import (
    FiveGateAntiGoodhartRatchet,
    HoloSphereKernelConfig,
    PairedABABABEvaluator,
    SiliSenseKernelAutotuner,
    two_step_mean_watt_quotient_fork_witness,
)
from silisense.errors import (
    HardwareExitCode,
    SiliSenseError,
    SiliSensePermissionError,
    SiliSenseValueError,
    require_ok,
    require_perm,
)
from silisense.frame64 import (
    TelemetryFrame64,
    canonicalize_zone_canon_v0,
    domain_tree_leaf_sha256,
)
from silisense.governor import (
    PreAdaptationResidualLatch,
    PrecursorThrottleGovernor,
    check_delay_bandwidth_lock,
    cyclic_switch_dist,
    floor_to_exact_f32,
    gate_asymmetric_step,
)
from silisense.holosphere_bridge import (
    HoloSphereStepTelemetry,
    HoloSphereTuningProfile,
    SiliSenseHoloSphereBridge,
    build_step_telemetry,
)
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
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AudioAuthorizationGate",
    "FiveGateAntiGoodhartRatchet",
    "HardwareExitCode",
    "HoloSphereKernelConfig",
    "HoloSphereStepTelemetry",
    "HoloSphereTuningProfile",
    "PairedABABABEvaluator",
    "PreAdaptationResidualLatch",
    "PrecursorThrottleGovernor",
    "RaplSysfsReader",
    "SiliSenseError",
    "SiliSenseHoloSphereBridge",
    "SiliSenseKernelAutotuner",
    "SiliSensePermissionError",
    "SiliSenseProfiler",
    "SiliSenseValueError",
    "TelemetryFrame64",
    "build_step_telemetry",
    "canonicalize_zone_canon_v0",
    "check_delay_bandwidth_lock",
    "compute_a1_ultrasonic_bicoherence",
    "compute_a2_chiral_vrm_balance",
    "compute_a3_time_reversal_asymmetry",
    "compute_a5_precursor_lead_lag",
    "compute_a7_nonlinear_uncore_copula_mi",
    "compute_op1_seam1_sherman_morrison_leverage",
    "compute_op2_seam3_ordinal_hodge_curl",
    "compute_op3_f3_nonnormal_commutator",
    "compute_time_locked_7tone_cole_shunt",
    "cyclic_switch_dist",
    "domain_tree_leaf_sha256",
    "floor_to_exact_f32",
    "gate_asymmetric_step",
    "query_nvidia_smi_once",
    "query_proc_irq_and_cpu_khz",
    "require_ok",
    "require_perm",
    "two_step_mean_watt_quotient_fork_witness",
    "validate_endpoint_path",
]
