# SiliSense (`Moissanite-Labs/silisense`)

**Fail-Closed Silicon Physics Metrology, Precursor Rail Governor, and Anti-Goodhart `EDP` Kernel Autotuner (Sibling to [`Moissanite-Labs/holosphere`](https://github.com/Moissanite-Labs/holosphere)).**

`SiliSense` turns microscopic silicon and firmware observables (non-textbook power/ultrasonic invariants and exact-$\mathbb{Q}$ monotone-invariant operators) into closed-loop dependent variables for GPU/CPU kernel profiling and rail governance — running **100% pure Python stdlib** (`dependencies = []`, `0` bare `assert`s, `0` sensor opens in test/offline mode) behind a strict two-mode privacy airlock.

---

## Strict Two-Mode Privacy & Sensor Airlock

| Mode | Activation | Permitted Endpoints | Fail-Closed Refusals |
| :--- | :--- | :--- | :--- |
| **Mode 1 (`telemetry-only`, Default)** | Default (`0` flags) | `intel-rapl` (`pkg_w`, `core_w`, `uncore_w`, `psys_w` via verified sysfs realpath confinement), `nvidia-smi`/NVML (`gpu_power_w`, `sm_clock_mhz`, `SwPowerCap`), `/proc/stat` IRQ/ctxt, `cpufreq` | `/dev/video*`, `/dev/media*`, `/dev/char/81:*`, V4L2, symlinks, IR LEDs (`ERR_CAMERA_FORBIDDEN`); `/dev/snd/*`, ALSA/PipeWire audio (`ERR_DMIC_DISABLED_DEFAULT`) |
| **Mode 2 (`dmic-authorized`, Opt-In)** | `--enable-dmic-authorized` + `--authorize-audio-window-sec (0, 300]` + `--pi-authorization-token` (monotonic deadline, token redacted from `repr`) | Mode 1 + `48 kHz` 4-ch DMIC ultrasonic summary statistics (`A1`, `A2`, `A3-62us`, `A5-80ms`) with `AV=0` (zero waveform persistence) | `/dev/video*`, `/dev/media*`, `/dev/char/81:*`, V4L2, IR LEDs (`ERR_CAMERA_FORBIDDEN`); expired windows (`ERR_AUDIO_WINDOW_EXPIRED`) |

> **Camera Exclusion Rationale:** Prior externally reported `141 W` CPU+GPU sensor characterization established that optical RGB/IR camera channels carry zero silicon workload signal (`<= 2.36 ppm` floor), so camera and V4L2 endpoints (including unnormalized paths, `..` traversals, symlinks, and major-`81` char nodes) are permanently refused across every mode.

---

## Core Architecture & One-Way `HoloSphere` Contract

```text
+-------------------------------------------------------------------------+
|  Moissanite-Labs/silisense  (Pure-Stdlib Core, dependencies = [])       |
|  ---------------------------------------------------------------------  |
|  silisense/airlock.py          ERR_CAMERA_FORBIDDEN + Mode 1/2 Airlock  |
|  silisense/frame64.py          64 B <8Q> uint64_ns Cache-Line Frame     |
|                                + zone-canon-v0 + u64_le Tree Digest     |
|  silisense/lenses.py           A3 / A5 / A7 + Opt-In Mode-2 A1 / A2     |
|                                + Exact-Q OP1_SEAM1 / OP2_SEAM3 / OP3_F3 |
|                                + A5-Time-Locked 7-Tone Fisher Impedance |
|  silisense/governor.py         f*tau < 1/6 Lock + L^{5/3} Heatpipe Norm |
|                                + Lyapunov Choke (floor_to_exact_f32)    |
|                                + gateAsymmetric (.okRecovery) + Latch   |
|  silisense/autotuner.py        PREREG-S115 H15c Composite EDP Autotuner |
|                                + 5-Gate Pure-Fraction Anti-Goodhart     |
|                                + 2-Step mean_watt Quotient Fork (1275/8)|
|  silisense/holosphere_bridge.py Canonical SHA256 One-Way Step Bridge    |
+------------------------------------+------------------------------------+
                                     |
       HoloSphereStepTelemetry (in)  |  HoloSphereTuningProfile (out, SHA-256)
                                     v
                   Moissanite-Labs/holosphere (0 imports of silisense)
```

---

## Live Mode-1 RTX 3080 (`GA104M` + `i9-11980HK`, `90 W` Cap) Empirical Calibration (`PREREG-S115`)

Evaluated live in Mode 1 (`telemetry-only`, `0` audio/camera/IR opens) across `3` seeds (`[42, 137, 2026]`):
- **`H15a` (`FALSIFIED` under Mode-1 NVML ZOH):** Warp-fused `HoloSphere-V2` (`sm_86`) cut training energy by **`5.97×` (`11.16` vs `66.59 mJ/tok`, `6.30×` faster)** and `OP1_SEAM1` leverage shock by **`-45.7%`**, with non-Gaussian `A7` uncore copula MI confirmed at `z_iaaft = +5.50σ` (`p_Holm = 0.00375 < 0.005`), while `~10 Hz` NVML power-register zero-order hold (`ZOH`) masked sub-millisecond `A3` shockwaves (`A7` `-14.66%` vs `-20%` gate).
- **`H15b` (`FALSIFIED` vs `-25%` gate; `-9.30%` live stall reduction):** Closed-loop `PrecursorThrottleGovernor` (`f*tau = 0.10 < 1/6`, `g_min = 0.72`) reduced `SwPowerCap` stall fraction by **`-9.30%` (`z = -7.922`, `p_t(df=2) = 0.0156`)** and step `mJ/tok` CV by **`-11.41%` (`z = -2.098`)** vs `NoGovernor` (`-5.63%` vs duty-matched `Shuffled-Trigger-Control`), establishing the Mode-1 vs Mode-2 (`48 kHz` DMIC) precursor bandwidth ceiling.
- **`H15c` (`FALSIFIED`; Anti-Goodhart Ratchet Verified Live):** Bare-metal `sm_86` (`B=16`) achieved optimal sustained `EDP` (`0.1885–0.1941 mJ·ms/tok²`); when ZOH-inflated `|z_A3|` perturbed `J(theta)` on `2/3` seeds, `FiveGateAntiGoodhartRatchet` (`G1..G5` over `Fraction`) **rejected `2/2` suboptimal candidates (`accepted = false`)**.

---

## Quickstart (Stock Python `>= 3.11`, Zero External Packages Required)

```bash
# Run full 3-mode isolated (-I -B, -I -O -B, -I -B -W error) pure-stdlib test suite
make test-3mode

# Run deterministic synthetic CPU selftest (evidence_class = cpu_synthetic_preflight)
make bench-cpu
```

## License

Released under the [MIT License](LICENSE).
