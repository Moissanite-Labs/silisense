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
|  silisense/razor.py            30 kHz BER-Canary Razor Controller       |
|                                + N=256 Epoch Hysteresis & 16 B Rollback |
|                                + Non-Scalar Krein-Holonomy Detector K_t |
|  silisense/contention.py       Host Contention Control Sentinel         |
|                                + Pre/Post Process & GPU-App Snapshots   |
|                                + Pure-Q Band Stratification & Spreads   |
|  silisense/holosphere_bridge.py Canonical SHA256 One-Way Step Bridge    |
|  include/razor_abi_4byte_sys.h 64 B Cacheline .sys Canary & 16 B Pi ABI |
+------------------------------------+------------------------------------+
                                     |
       HoloSphereStepTelemetry (in)  |  HoloSphereTuningProfile (out, SHA-256)
                                     v
                   Moissanite-Labs/holosphere (0 imports of silisense)
```

---

## `30 kHz` Razor BER-Canary, `N=256` Epoch Hysteresis, & Krein–Holonomy Detector (`[EXACT_PROVEN / C11+CXX17_ABI_VERIFIED]`)

`silisense/razor.py` and [`include/razor_abi_4byte_sys.h`](include/razor_abi_4byte_sys.h) close the `~10 Hz` NVML zero-order hold (`ZOH`) observability ceiling (`100 ms` blind window) using inline `[16, 11, 4]` Block-H4 SECDED bit-error detection sampled at **`30 kHz` (`33.33 us/tok`, `3,000×` faster than NVML)** over a cacheline-aligned **`64 B` `.sys` D2H canary (`sizeof=64, alignof=64`)** that executes **`0` PCIe atomic writes on clean cycles**:

- **Companion Live RTX 3080 (`GA104M`, `sm_86`) `.sys` Canary Receipts (latency and canary figures pinned to [`Moissanite-Labs/silikernels`](https://github.com/Moissanite-Labs/silikernels) main [`b3712675`](https://github.com/Moissanite-Labs/silikernels/commit/b37126752e948a2dbf5d6275d52c2a5200ec65db), the merge of [`silikernels#4`](https://github.com/Moissanite-Labs/silikernels/pull/4) @ [`437e52c0`](https://github.com/Moissanite-Labs/silikernels/commit/437e52c069fb3fd0042b0b4d137d22189152a481) with the same tree; companion kernel PR [`Moissanite-Labs/holosphere#3`](https://github.com/Moissanite-Labs/holosphere/pull/3) @ `6f75548d`; not derived inside `silisense`):**
  - *V1 Inline SECDED Kernel ([`249.37 us/tok`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_live_rtx3080_zone_b_results.json) baseline, receipt `uniform_cell_16warp_d512.t8192_us_per_tok = 249.3671`):* `0` PCIe atomic writes on clean cycles, **`16/16` single-bit LSB faults healed inline (`max_abs_diff = 0.0`, bit-exact output byte)**, and **`1/1` DED double-bit fault flagged with zero silent miscorrection**.
  - *V2 `16`-Warp `d=512` Fused Kernel ([`26.21 us/tok`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_live_v2_d512_zone_b_results.json) CellOp, receipt `cellop_only_12L_d512.t8192_us_per_tok = 26.2082`; [`27.41 us/tok`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_live_v2_d512_zone_b_results.json) E2E with `0 B` tied `FWHT256` LM head, receipt `full_e2e_fwht256_12L_d512.t8192_us_per_tok = 27.4079`, [`9.097×`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_live_v2_d512_zone_b_results.json) over V1, receipt `speedup_vs_v1_249_32us = 9.0967`, `1.83e-7` parity vs FP64 dense; Zone-B [`receipt_sha256 = 699c3d05…`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/mlos_zone_b_s117_v2_d512.json)):* `0` PCIe atomic writes on clean cycles, **`16/16` SEC `0.0` bit-exact across all `512` hidden floats, `256` logits, and output byte**, and **`1/1` DED flagged with zero silent miscorrection**.
  - *V3 `16`-Warp `d=512` Frontier Kernel ([`20.50 us/tok`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_v3_frontier_live_gpu_results.json) `T=8192` CUDA-graph CellOp, receipt `v3_d512.cellop_graph_us_per_tok_8192 = 20.5038`; [`21.52 us/tok`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/s117_v3_frontier_live_gpu_results.json) `T=16384` E2E, receipt `v3_d512.e2e_graph_us_per_tok_16384 = 21.5184`; Zone-B [`receipt_sha256 = feecc748…`](https://github.com/Moissanite-Labs/silikernels/blob/b37126752e948a2dbf5d6275d52c2a5200ec65db/receipts/mlos_zone_b_s117_v3_frontier.json)):* `0` clean PCIe writes, **`16/16` SEC `0.0` bit-exact**, and **`1/1` DED flagged**.
  - **Energy figures in those receipts are uncontrolled and provisional.** Recent laptop runs were not controlled (the machine was moved, in use, or unplugged). Neither the previous pin nor the replay at `b3712675` is a controlled energy result, and the `mJ/tok` fields are not reported here as a result.
- **Honest `0/48 -> 48/48` Vmin Carry-Path Upper-Bit Closure:** Standard `[16, 11, 4]` LSB parity detects `16/16` LSB flips and `120/120` 2-bit LSB pairs, but detects **`0/48`** single bit flips in signed-nibble upper bits `1..3` (which carry-path timing violations can flip directly). `silisense/razor.py` and `include/razor_abi_4byte_sys.h` close **`48/48` (`100%`)** of upper-bit single faults via:
  1. **Folded `bit1 ^ bit2 ^ bit3` upper-bit parity** (`1` separate side-channel parity bit outside the 16-bit `[16, 11, 4]` codeword, `0` extra DRAM weight bytes when carried in metadata/registers; catches **`48/48` on every epoch including epoch `0`**),
  2. **Epoch-rotated nibble bit-plane checks** (`b = epoch & 3`; catches **`0/48` on epoch `0 mod 4`** where plane `0` is already the `[16, 11, 4]` LSB plane, **`16/48` on each of epochs `1, 2, 3 mod 4`**, and closes **`48/48` union across any `4` consecutive epochs**), and
  3. **Dual-channel `block_crc16_ccitt`** (catches **`48/48` single and multi-bit upper-bit faults on every epoch including epoch `0`**).
- **Exact-$\mathbb{Q}$ `N=256`-Token Anti-Chatter Hysteresis & `16 B` Rollback:** `HoloSphereRazorController.step_epoch()` updates a `Q16`-quantized exact rational EWMA `e <- floor_Q16((1 - 1/32)e + (1/32)(sec_delta / 256))` (matching `ewma_ber_q16_u32` in the C ABI with bounded denominator `<= 65536` across unbounded runtime), holds deadband on isolated `1`-SEC epochs, steps margin up immediately on `sec_delta >= 2` or `e >= 2/256` with a `4`-epoch (`1,024`-token) dwell hold, requires `16` consecutive clean epochs (`4,096` tokens) + healthy `SiliSense` `OP1/OP2/OP3` + `FiveGateAntiGoodhartRatchet` before margin descent, and latches Tier-2 DED / non-LSB faults for `64` clean epochs (`16,384` tokens); single-poll `step()` enforces the same DED latch, min-dwell hold, clean-epoch warmup, and health gates (without advancing `clean_streak`, which increments only on `step_epoch()` boundaries). `WeftHolonomyRollbackBuffer` maintains exact-`Fraction` `QuatQ` + `S_fast` + clean chunk-32 boundary state (`Fraction(0)` residual drift on replay) and packs `Pi_{t_0}` via `pack_checkpoint_pi16()` into the `16 B` little-endian IEEE-754 `HoloSpherePiRollback16` ABI struct.
- **Exact-$\mathbb{Q}$ Non-Scalar Krein–Holonomy Detector (`K_t`):** `nonscalar_krein_holonomy_detector()` computes exact Sylvester inertia of `K_t = diag(G - A^T G A, 3 - |tr(Delta U_t)|^2)` via Descartes' rule over `Fraction`, proving scalar `G = I_3` is blind to non-Abelian operator ordering (`n_-(UL) == n_-(LU) == 1`) whereas non-scalar SPD `G_3 = diag(3, 1, 1)` cleanly separates ordering (`n_-(UL) == 1 != 2 == n_-(LU)`).
- **Multi-Compiler ABI Verification:** `include/razor_abi_4byte_sys.h` enforces little-endian host byte order, uses `__atomic_load_n(..., __ATOMIC_ACQUIRE)` (zero C11/C++17 atomic-pointer type-punning UB, pure read with `_ReadWriteBarrier()` under `_MSC_VER`), emits up to `4` (`2–4`: `1–3` conditional `atom.release.sys.global.add.u32` + `1` `st.release.sys.global.u32`) `.sys` release ops on fault cycles (`0` when clean), and compiles cleanly in CPU CI under `gcc -std=c11` and `g++ -std=c++17` (`-Wall -Wextra -Werror -pedantic -fsanitize=undefined`); on local Ampere CUDA hosts it also compiles cleanly under `nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler -Wall,-Wextra,-Werror` (`12–15` registers depending on probe harness, `0 B` spills).
- **Host Contention Control Sentinel ([`docs/HOST_CONTENTION_CONTROL.md`](docs/HOST_CONTENTION_CONTROL.md), `silisense/contention.py`):** Addresses [`Moissanite-Labs/silikernels#7`](https://github.com/Moissanite-Labs/silikernels/issues/7) by recording privacy-scrubbed pre/post OS process & GPU compute-app snapshots (zero `/proc/<pid>/cmdline`, window-title, or audio/video device opens), sampling external CPU/GPU/AC contention during benchmark windows, binding a canonical `environment` block (`environment_sha256`) into `result_sha256`, rejecting uncontrolled laptop energy/watt claim keys (`energy_power_status == "UNCONTROLLED_LAPTOP_NOT_CLAIMED"`), and enforcing fail-closed pure-$\mathbb{Q}$ contention-band stratification with mandatory `n_repeats >= 3` spread reporting (median, IQR, MAD).

---

## Live Mode-1 RTX 3080 (`GA104M` + `i9-11980HK`, `90 W` Cap) Empirical Calibration (`PREREG-S115`)

Evaluated live in Mode 1 (`telemetry-only`, `0` audio/camera/IR opens) across `3` seeds (`[42, 137, 2026]`):
- **`H15a` (`FALSIFIED` under Mode-1 NVML ZOH):** Warp-fused `HoloSphere-V2` (`sm_86`) cut training energy by **`5.97×` (`11.16` vs `66.59 mJ/tok`, `6.30×` faster)** and `OP1_SEAM1` leverage shock by **`-45.7%`**, with non-Gaussian `A7` uncore copula MI confirmed at `z_iaaft = +5.50σ` (unpooled per-trace `m=3` `p_Holm = 0.00375 < 0.005`; campaign-wide Holm-Bonferroni yields `0` survivors at `α = 0.005` in the companion `PREREG-S115` evidence release @ `00e83534d0f9`), while `~10 Hz` NVML power-register zero-order hold (`ZOH`) masked sub-millisecond `A3` shockwaves (`A7` `-14.66%` vs `-20%` gate).
- **`H15b` (`FALSIFIED` vs `-25%` gate; `-9.30%` live stall reduction):** Closed-loop `PrecursorThrottleGovernor` (`f*tau = 0.10 < 1/6`, `g_min = 0.72`) reduced `SwPowerCap` stall fraction by **`-9.30%` (`z = -7.922`, `p_t(df=2) = 0.0156`)** and step `mJ/tok` CV by **`-11.41%` (`z = -2.098`)** vs `NoGovernor` (`-5.63%` vs duty-matched `Shuffled-Trigger-Control`), establishing the Mode-1 vs Mode-2 (`48 kHz` DMIC) precursor bandwidth ceiling.
- **`H15c` (`FALSIFIED`; Anti-Goodhart Ratchet Verified Live):** Bare-metal `sm_86` (`B=16`) achieved `0.1901–0.1941 mJ·ms/tok²` 3-seed mean sustained `EDP` across scalar baselines (Seed-42 Composite reached `0.1837`, `+14.16%`); `FiveGateAntiGoodhartRatchet` (`G1..G5` over `Fraction`) returned **`accepted = false` on `3/3` seeds** (with `G5` tripping `9/9` under ZOH-inflated `|z_A3|`, while `G2+G3` selectively rejected the degraded Seeds `137`/`2026`).

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
