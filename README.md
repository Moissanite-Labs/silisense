# SiliSense (`Moissanite-Labs/silisense`)

**Fail-Closed Silicon Physics Metrology, Precursor Rail Governor, and Anti-Goodhart `EDP` Kernel Autotuner (`Track S115` / Sibling to [`Moissanite-Labs/holosphere`](https://github.com/Moissanite-Labs/holosphere)).**

`SiliSense` turns verified microscopic silicon and firmware observables (`Track S113` Amendment-1 non-textbook invariants + `Track S20c` exact-$\mathbb{Q}$ monotone-invariant operators) into closed-loop dependent variables for GPU/CPU kernel optimization — running **100% pure Python stdlib** (`0` mandatory third-party dependencies, `0` bare `assert`s, `0` sensor opens in test/offline mode) behind a strict two-mode privacy airlock.

---

## Strict Two-Mode Privacy & Sensor Airlock

| Mode | Activation | Permitted Endpoints | Fail-Closed Refusals |
| :--- | :--- | :--- | :--- |
| **Mode 1 (`telemetry-only`, Default)** | Default (`0` flags) | `intel-rapl` (`pkg_w`, `core_w`, `uncore_w`, `psys_w`), `nvidia-smi`/NVML (`gpu_power_w`, `sm_clock_mhz`, `SwPowerCap`), `/proc/stat` IRQ/ctxt, `cpufreq` | `/dev/video*`, `/dev/media*`, V4L2, IR LEDs (`ERR_CAMERA_FORBIDDEN`); `/dev/snd/*`, ALSA/PipeWire audio (`ERR_DMIC_DISABLED_DEFAULT`) |
| **Mode 2 (`dmic-authorized`, Opt-In)** | `--enable-dmic-authorized` + `--authorize-audio-window-sec (0, 300]` + `--pi-authorization-token` (monotonic deadline) | Mode 1 + `48 kHz` 4-ch DMIC ultrasonic summary statistics (`A1`, `A2`, `A3-62us`, `A5-80ms`) with `AV=0` (zero waveform persistence) | `/dev/video*`, `/dev/media*`, V4L2, IR LEDs (`ERR_CAMERA_FORBIDDEN`); expired windows (`ERR_AUDIO_WINDOW_EXPIRED`) |

> **Physical Justification for Permanent Camera Ban (`S113` `H3`/`H4`):** Synchronized `141 W` CPU+GPU full-load experiments proved optical RGB/IR camera channels carry zero silicon workload leakage (`<= 2.36 ppm`), so camera devices are permanently refused across every mode.

---

## Core Architecture & One-Way `HoloSphere` Contract

```text
+-------------------------------------------------------------------------+
|  Moissanite-Labs/silisense  (Pure-Stdlib Core, 0 Mandatory Dependencies)|
|  ---------------------------------------------------------------------  |
|  silisense/airlock.py          ERR_CAMERA_FORBIDDEN + Mode 1/2 Airlock  |
|  silisense/frame64.py          64 B <8Q> uint64_ns Cache-Line Frame     |
|                                + zone-canon-v0 + u64_le Tree Digest     |
|  silisense/lenses.py           S113 A3 / A5 / A7 + Opt-In Mode-2 A1 / A2|
|                                + Exact-Q OP1_SEAM1 / OP2_SEAM3 / OP3_F3 |
|                                + A5-Time-Locked 7-Tone Fisher Impedance |
|  silisense/governor.py         f*tau < 1/6 Lock + L^{5/3} Heatpipe Norm |
|                                + Lyapunov Choke (floor_to_exact_f32)    |
|                                + gateAsymmetric (.okRecovery) + Latch   |
|  silisense/autotuner.py        PREREG-S115 H15c Composite EDP Autotuner |
|                                + 5-Gate Pure-Fraction Anti-Goodhart     |
|                                + 2-Step mean_watt Quotient Fork Guard   |
|  silisense/holosphere_bridge.py Wire- & SHA256-Exact One-Way Bridge     |
+------------------------------------+------------------------------------+
                                     |
       HoloSphereStepTelemetry (in)  |  HoloSphereTuningProfile (out, SHA-256)
                                     v
                   Moissanite-Labs/holosphere (0 imports of silisense)
```

---

## Preregistered Falsifiable Hypotheses (`PREREG-S115`, Honest Law-4 Gates)

1. **`H15a` (Fused `HoloSphere` vs. Eager Micro-Kernel Chip Signature):** Interleaved `ABABAB` blocks testing whether `0 B`-spill warp-fused `SU(2)` holonomy kernels reduce `|A3|` third-order shockwave asymmetry and `A7` multi-lag Schur-residualized (`|\rho| <= 1e-9`) `uncore_w` copula MI by `>= 20%` (`z <= -2.0`), including under `<= 5%` equal-wattage pacing.
2. **`H15b` (Sub-Millisecond / High-Rate `A5` Precursor Rail Governor):** Tests causal `A5` fast-rail innovation triggers (`p_circ < 0.005`) + runaway-free $L^{5/3}$ Lyapunov conductance damping against `NoGovernor`, `Reactive-1Hz-SMI`, and duty-cycle-matched `Shuffled-Trigger-Control` (`<= 2%` throughput cost).
3. **`H15c` (Anti-Goodhart Silicon-Observable `EDP` Kernel Autotuner):** Evaluates locked `(0.15, 0.15, 0.10)` `2.0σ`-hinged composite selection + pure-`Fraction` 5-gate anti-Goodhart ratcheting against `Scalar-TokPerSec-Tuner` and `Scalar-MeanWatt-EDP-Tuner` on held-out sustained `EDP = (mJ/tok) × (ms/tok)`.

---

## Quickstart (Stock Python `>= 3.11`, Zero External Packages Required)

```bash
# Run full 3-mode isolated (-I -B, -I -O -B, -I -B -W error) test suite
make test-3mode

# Run deterministic PREREG-S115 CPU preflight benchmark
make bench-cpu
```

## License

Released under the [MIT License](LICENSE).
