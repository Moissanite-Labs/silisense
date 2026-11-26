# Host Contention Control Sentinel (`silisense.contention`)

`silisense.contention` implements process, GPU-application, and power-source control-variable telemetry, canonical receipt binding, and pure-$\mathbb{Q}$ contention-band stratification (addressing [`Moissanite-Labs/silikernels#7`](https://github.com/Moissanite-Labs/silikernels/issues/7)).

## Architecture (5-Part Control-Variable Contract)

1. **Pre/Post Host Snapshots (`HostContentionCollector.capture_snapshot`):**
   - Captures `/proc/<pid>/stat` and `/proc/<pid>/comm` top CPU and RSS consumers, `/proc/loadavg`, `/proc/stat` aggregate jiffies, `/sys/class/power_supply/AC0/online`, read-only `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits` (gracefully `unavailable` on CPU-only hosts), and optional scrubbed desktop foreground window class (`hyprctl activewindow -j` `class` / `initialClass` only).
   - **Privacy & Airlock Guarantee:** NEVER reads `/proc/<pid>/cmdline`, full file paths, window titles, user secrets, or any `/dev/snd/*` or `/dev/video*` endpoint. Every process or class identifier passes through `sanitize_proc_name()`.
2. **In-Run Sampling (`collect_in_run_sample`, `run_monitored_arm`):**
   - Polls instantaneous external host CPU utilization (`/proc/stat` busy ticks minus excluded benchmark-PID ticks), 1-minute load ratio, AC mains state, and read-only GPU utilization, SM clock, temperature, and fault throttle mask (`NVML_FAULT_THROTTLE_MASK = 0xFC`, excluding benign `0x1 GPU_IDLE` and `0x2 APP_CLOCKS`).
   - All OS readers are injectable for deterministic offline and CI testing.
3. **Canonical Receipt Binding (`build_environment_block`, `bind_receipt_with_environment`):**
   - Emits a canonical 17-key `environment` block (`schema_version`, `energy_power_status`, `pre_snapshot_sha256`, `post_snapshot_sha256`, `in_run_summary`, `ac_online_all`, `power_band`, `external_gpu_apps_detected`, `cpu_load_max_ratio`, `cpu_load_max_ratio_q`, `bg_cpu_util_max_ratio`, `bg_cpu_util_max_ratio_q`, `foreground_apps`, `thresholds`, `contended`, `contention_reasons`, and `environment_sha256`) folded directly into the enclosing benchmark receipt's `result_sha256`.
   - In Python, `validate_contention_thresholds(thresholds=None)` intentionally falls back to `DEFAULT_CONTENTION_THRESHOLDS`, whereas `verify_environment_block()` strictly requires a JSON object (`dict`) for `environment["thresholds"]` and rejects serialized `null` or non-`dict` values with `SiliSenseValueError("THRESH_TYPE")`.
4. **Fail-Closed Verification & Pure-$\mathbb{Q}$ Band Stratification (`verify_environment_block`, `verify_bound_receipt`, `verify_live_probe`, `compare_bound_receipts`, `stratify_receipts`):**
   - Rejects missing or tampered hashes, unpinned extra keys (`RECEIPT_KEYS`, `PROBE_KEYS`), non-`dict` `metadata` (`RECEIPT_METADATA_TYPE`), `bool`-as-`int` and float-vs-int type confusion, inconsistent contention labels, and cross-threshold or cross-band comparisons (`COMPARE_THRESHOLD_MISMATCH` on mismatched threshold profiles and `COMPARE_BAND_MISMATCH` on `quiet_ac` vs. `contended_ac` or `AC` vs. `battery` by default, or `status = "STRATIFIED_UNCOMPARABLE"` with `ratio_q = null` when `allow_stratified=True`).
   - **Uncontrolled-Laptop Energy Policy:** Enforces `energy_power_status == "UNCONTROLLED_LAPTOP_NOT_CLAIMED"`, strict `dict` `metadata` typing in both bind and verify paths, NFKC+NFKD (`Cf`/`Mn` stripped) + non-ASCII letter homoglyph rejection (excluding micro sign `μ`), and recursive rejection of uncontrolled laptop `mJ/tok`, watt, joule, watthour, `TDP`, and energy-reduction keys, metric labels, unit strings, and nested `metadata` entries (`FORBIDDEN_ENERGY_KEYS`). Non-W/J energy units (`kcal`, `BTU`, `eV`, `hp`), spelled-out numbers, and identifier-like bare tokens (`W10-OBS-61`, `J2000`, `bench w/ perf counters`) are outside the regex scope by design.
   - **CLI Status Labeling:** `--verify-probe` prints `"status": "PASS"` when `band_separation_verified` is `true` and `"status": "VERIFIED_NO_BAND_SEPARATION"` when integrity passes with `band_separation_verified: false`; invalid CLI arguments (e.g., `--repeats 2`) emit structured JSON errors on `stderr` with exit code `2`.
5. **Mandatory Spread Reporting (`compute_spread_stats_q`):**
   - Requires `n_repeats >= 3` per arm and reports exact pure-$\mathbb{Q}$ `min_q`, `q1_q`, `median_q`, `q3_q`, `max_q`, `iqr_q`, `mad_q`, and `rel_iqr_q` alongside exact `non_overlapping_iqr` comparisons within matched bands.

## CLI Usage

```bash
# Run a CPU-only quiet vs. synthetic-contention self-test probe from the repo root
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 -B silisense/contention.py \
  --run-live-probe /tmp/silisense_contention_probe.json --repeats 5

# Verify a saved probe receipt deterministically
PYTHONPATH=. python3 -B silisense/contention.py \
  --verify-probe /tmp/silisense_contention_probe.json
```
