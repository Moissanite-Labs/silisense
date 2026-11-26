"""R1-R5 Test Suite for silisense.contention (HostContentionSentinel).

Targets 100.0% guard/boundary mutation kill rate, 100.0% AST BoolOp coverage,
3-mode isolated -I parity, hermetic execution, zero sensor opens, zero skips,
and zero bare asserts.
"""

from __future__ import annotations

import contextlib
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import silisense
from silisense import contention as hc
from silisense.errors import SiliSenseValueError


def _make_mock_collector(
    *,
    cpu_count: int = 16,
    loadavg: str = "1.60 1.40 1.20 1/500 9999\n",
    stat_series: Optional[List[Tuple[int, int]]] = None,
    self_ticks_series: Optional[List[int]] = None,
    ac_online: Optional[str] = "1\n",
    gpu_apps_rc: int = 0,
    gpu_apps_out: str = "",
    gpu_tel_rc: int = 0,
    gpu_tel_out: str = "2, 210, 55, 0x0000000000000001\n",
    hypr_rc: int = 0,
    hypr_out: str = '{"class": "/usr/bin/foot", "title": "SECRET_TITLE_DO_NOT_RECORD"}\n',
    extra_procs: Optional[Dict[int, Tuple[str, Optional[str]]]] = None,
    enable_gpu_probe: bool = True,
    enable_foreground_probe: bool = True,
) -> hc.HostContentionCollector:
    stat_list = list(stat_series) if stat_series is not None else [(1000, 900), (2000, 1800), (3000, 2700)]
    self_list = list(self_ticks_series) if self_ticks_series is not None else [10, 20, 30]
    stat_idx = [0]
    self_idx = [0]
    procs: Dict[int, Tuple[str, Optional[str]]] = (
        dict(extra_procs)
        if extra_procs is not None
        else {
            200: ("gcc\n", "200 (gcc) R 1 1 1 0 -1 0 0 0 0 0 500 100 0 0 20 0 1 0 1 1 4096 0\n"),
            201: ("python3\n", "201 (python3) S 1 1 1 0 -1 0 0 0 0 0 120 30 0 0 20 0 1 0 1 1 8192 0\n"),
        }
    )

    def _read_text(path: str) -> Optional[str]:
        if path == "/proc/stat":
            idx = min(stat_idx[0], len(stat_list) - 1)
            stat_idx[0] += 1
            tot, idle = stat_list[idx]
            busy = max(0, tot - idle)
            return f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n"
        if path == "/proc/loadavg":
            return loadavg
        if path == "/sys/class/power_supply/AC0/online":
            return ac_online
        if path == f"/proc/{os.getpid()}/stat":
            idx = min(self_idx[0], len(self_list) - 1)
            self_idx[0] += 1
            st = self_list[idx]
            return f"{os.getpid()} (py) R 1 1 1 0 -1 0 0 0 0 0 {st} 0 0 0 20 0 1 0 1 1 100 0\n"
        if path.startswith("/proc/") and path.endswith("/stat"):
            pid = int(path.split("/")[2])
            entry = procs.get(pid)
            return None if entry is None else entry[1]
        if path.startswith("/proc/") and path.endswith("/comm"):
            pid = int(path.split("/")[2])
            entry = procs.get(pid)
            return None if entry is None else entry[0]
        return None

    def _list_pids() -> List[int]:
        return sorted([os.getpid()] + list(procs.keys()))

    def _run_cmd(argv: Sequence[str]) -> Tuple[int, str]:
        if argv[0] == "hyprctl":
            return hypr_rc, hypr_out
        if argv[0] == "nvidia-smi" and "--query-compute-apps" in argv[1]:
            return gpu_apps_rc, gpu_apps_out
        if argv[0] == "nvidia-smi" and "--query-gpu" in argv[1]:
            return gpu_tel_rc, gpu_tel_out
        return 127, ""

    return hc.HostContentionCollector(
        read_text=_read_text,
        list_proc_pids=_list_pids,
        run_cmd=_run_cmd,
        cpu_count=cpu_count,
        sleep=lambda _s: None,
        enable_gpu_probe=enable_gpu_probe,
        enable_foreground_probe=enable_foreground_probe,
    )


def _resign_env(env: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(env)
    out["environment_sha256"] = hc.canonical_sha256(out, exclude_key="environment_sha256")
    return out


def _resign_receipt(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(rec)
    out["result_sha256"] = hc.canonical_sha256(out, exclude_key="result_sha256")
    return out


def _build_sample_receipt(
    *,
    arm_name: str = "quiet_arm",
    values: Optional[List[Fraction]] = None,
    ac_online: Optional[str] = "1\n",
    loadavg: str = "1.60 1.40 1.20 1/500 9999\n",
    stat_series: Optional[List[Tuple[int, int]]] = None,
    self_ticks_series: Optional[List[int]] = None,
    gpu_apps_rc: int = 0,
    gpu_apps_out: str = "",
    gpu_tel_rc: int = 0,
    gpu_tel_out: str = "2, 210, 55, 0x0000000000000001\n",
    thresholds: Optional[Dict[str, Any]] = None,
    metric_name: str = "ops_per_sec",
    metric_unit: str = "ops/s",
) -> Dict[str, Any]:
    col = _make_mock_collector(
        ac_online=ac_online,
        loadavg=loadavg,
        stat_series=stat_series if stat_series is not None else [(1000, 900), (2000, 1800), (3000, 2700)],
        self_ticks_series=self_ticks_series,
        gpu_apps_rc=gpu_apps_rc,
        gpu_apps_out=gpu_apps_out,
        gpu_tel_rc=gpu_tel_rc,
        gpu_tel_out=gpu_tel_out,
    )
    pre = col.capture_snapshot(phase="pre")
    samp, t1, i1, s1 = col.collect_in_run_sample(
        prev_cpu_total=pre["cpu_jiffies_total"],
        prev_cpu_idle=pre["cpu_jiffies_idle"],
        prev_self_ticks=pre["self_cpu_ticks"],
        exclude_pids={os.getpid()},
    )
    _ = (t1, i1, s1)
    post = col.capture_snapshot(phase="post")
    env = hc.build_environment_block(
        pre_snapshot=pre,
        post_snapshot=post,
        in_run_samples=[samp],
        thresholds=thresholds,
    )
    vals = values if values is not None else [Fraction(100, 1), Fraction(102, 1), Fraction(104, 1)]
    return hc.bind_receipt_with_environment(
        arm_name=arm_name,
        metric_name=metric_name,
        metric_unit=metric_unit,
        repeat_values_q=vals,
        environment=env,
    )


class TestContentionSentinelR5(unittest.TestCase):
    def assertSentinelCode(self, code: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        with self.assertRaises(SiliSenseValueError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_public_exports_sanitize_rational_and_energy_policy(self) -> None:
        self.assertIs(silisense.HostContentionCollector, hc.HostContentionCollector)
        self.assertIs(silisense.verify_bound_receipt, hc.verify_bound_receipt)
        self.assertIs(silisense.compare_bound_receipts, hc.compare_bound_receipts)

        self.assertEqual(hc.sanitize_proc_name(123), "unknown")
        self.assertEqual(hc.sanitize_proc_name("   "), "unknown")
        self.assertEqual(hc.sanitize_proc_name("/usr/bin/python3.13"), "python3.13")
        self.assertEqual(hc.sanitize_proc_name("/usr/bin/firefox --profile /tmp/secret"), "firefox")
        self.assertEqual(hc.sanitize_proc_name("C:\\Apps\\my_app@1"), "my_app_1")
        self.assertEqual(hc.sanitize_proc_name("///@@@"), "unknown")
        self.assertEqual(len(hc.sanitize_proc_name("a" * 64)), 32)

        self.assertSentinelCode("BAD_JSON_PAYLOAD", hc.canonical_json_bytes, {"x": float("nan")})

        self.assertEqual(hc.parse_q("3/4"), Fraction(3, 4))
        self.assertEqual(hc.parse_q("-5/2"), Fraction(-5, 2))
        self.assertEqual(hc.parse_q("0/1"), Fraction(0, 1))
        self.assertSentinelCode("BAD_RATIONAL", hc.parse_q, 1)
        self.assertSentinelCode("BAD_RATIONAL", hc.parse_q, "3/0")
        self.assertSentinelCode("BAD_RATIONAL", hc.parse_q, "2/4")
        self.assertSentinelCode("BAD_RATIONAL", hc.parse_q, "+1/2")

        self.assertFalse(
            hc.has_forbidden_energy_key(
                {
                    "energy_power_status": hc.ENERGY_POLICY_STATUS,
                    "power_band": "ac",
                    "contention_reasons": ["not_on_ac_power", "gpu_temp_exceeded"],
                    "metadata": {
                        "workload": "stdlib_sha256_chain",
                        "repeats": 5,
                        "note": "bench w/ perf counters",
                        "cmd": "taskset -c 3 w/ 5 reps",
                        "tag": "W10-OBS-61",
                        "epoch": "J2000.0",
                        "worker": "worker W10",
                        "week": "2024-W05",
                        "temp": "78\u00b0C",
                        "lat": "5 \u03bcs",
                    },
                    "ok": [1, {"safe": "ops/s", "free": "tdp_like_free_text_no"}],
                    "raw": b"123",
                    "tup": ("ops/s",),
                    "fset": frozenset({"ms"}),
                }
            )
        )
        for bad_obj in [
            {"mJ_per_tok": 1.2},
            {"MJ_PER_TOK": 1.2},
            {" mj_per_tok": 1.2},
            {"mJ.per.tok": 1.2},
            {"m\u200bj_per_tok": 1.2},
            {"po\u0301wer": 45},
            {"avgPower": 45.2},
            {"powerMw": 45200},
            {"powermw": 45200},
            {"energymj": 1.2},
            {"pkgW": 35},
            {"energyJ": 0.1},
            {"totalEnergy": 1.2},
            {"max_45w": 1},
            {"45w": 1},
            {"130w_psu": 1},
            {"pwr": 45},
            {"\u051datt": 45},
            {"\u0435nergy": 1.2},
            {"metadata": {"power_band": 45.2}},
            {"metadata": {"energy_power_status": hc.ENERGY_POLICY_STATUS}},
            {"metadata": [["mJ_per_tok", 1.2], ["avgPower", 45.2]]},
            {"metadata": {"nested": [["avgPower", 45.2]]}},
            {"power_band": 45.2},
            {"power_band": "45.2 W"},
            {"power_band": "turbo"},
            {"w": 45.0},
            {"mw": 45000},
            {"kwh": 0.1},
            {"power": 45.2},
            {"avg_power": 45.2},
            {"power_mw": 45200},
            {"tdp_w": 90},
            {"pkg_w": 35},
            {"mwh_per_tok": 0.01},
            {"wh_per_tok": 0.01},
            {"energy_mj": 1.2},
            {"energy_j": 0.01},
            {"watts": 45.0},
            {"watt": 45.0},
            {"power_draw_w": 29.3},
            {"joules_per_tok": 0.01},
            {"energy_reduction_x": 3.3},
            {"nested": [{"cuda_graph_mj_per_tok": 20.0}]},
            {1: {"mean_watts": 30.0}},
            {"tup": ("1.1728 mJ/tok",)},
            {"fset": frozenset({"45.2 W"})},
            {"note": "45.2W"},
            {"note": "45.2 W."},
            {"note": "45.2W!"},
            {"note": "W?"},
            {"note": "[W]"},
            {"note": "45.2\u00d7W"},
            {"note": "3.9e2 W"},
            {"note": "1.5E-3W"},
            {"note": "5 GW"},
            {"note": "5 TW"},
            {"note": "5 PW"},
            {"note": "5 pW"},
            {"note": "5 kilo watts"},
            {"note": "5 kilo-watts"},
            {"note": "5 m W"},
            {"note": "15 watthours"},
            {"note": "15 watt-hour"},
            {"note": "5\u2013W"},
            {"note": "12\u00b7W"},
            {"note": "45.2\u2219W"},
            {"note": "joules per token"},
            {"note": "MJ per token"},
            {"note": "15 TDP"},
            {"note": "TDP 15"},
            {"note": "TDP"},
            {"note": "energy use dropped"},
            {"note": "cap_45_w"},
            {"note": "45_watts"},
            {"note": "limit_45_w"},
            {"note": "draw_45w"},
            {"note": "45 Whr"},
            {"note": "5 \u051d"},
            {"note": "\u0435nergy"},
            {"note": "45.2 \u03c9"},
            {"note": "\u051datt"},
            {"note": "\uff11\uff12\uff37"},
            {"note": "45.2\u200bW"},
            {"note": "45.2 kilowatts."},
            {"note": "45.2-W"},
            {"note": "45.2 [W]"},
            {"note": "W: 45.2"},
            {"note": "45.2 kilowatts"},
            {"note": "45 milliwatts"},
            {"note": "1.2 megawatts"},
            {"note": "12.5 kilojoules"},
            {"note": "12.5 megajoules"},
            {"note": "70% less energy"},
            {"note": "energy saving"},
            {"note": "45200 mW"},
            {"note": "measured 45.2 mW"},
            {"note": "0.0452 kWh"},
            {"note": "1.2 J/tok"},
            {"note": "12.5 joules"},
            {"note": "mJ/token"},
            {"metric_unit": "W"},
            {"note": "3.318x less energy"},
            {"note": "45.2 watts"},
        ]:
            self.assertTrue(hc.has_forbidden_energy_key(bad_obj), f"should reject {bad_obj!r}")

        self.assertTrue(hc._is_strict_finite_num(1))
        self.assertTrue(hc._is_strict_finite_num(1.5))
        self.assertFalse(hc._is_strict_finite_num(True))
        self.assertFalse(hc._is_strict_finite_num(float("nan")))
        self.assertFalse(hc._is_strict_finite_num(float("inf")))

    def test_02_thresholds_and_exact_boundaries(self) -> None:
        cfg = hc.validate_contention_thresholds()
        self.assertEqual(cfg["min_repeats"], 3)
        boundary_ok = dict(cfg)
        boundary_ok["max_cpu_load_ratio_q"] = "4/1"
        boundary_ok["max_bg_cpu_util_ratio_q"] = "1/1"
        boundary_ok["max_external_gpu_util_pct"] = 0
        boundary_ok["max_gpu_temp_c"] = 30
        self.assertEqual(hc.validate_contention_thresholds(boundary_ok)["max_gpu_temp_c"], 30)
        boundary_hi = dict(cfg)
        boundary_hi["max_external_gpu_util_pct"] = 100
        boundary_hi["max_gpu_temp_c"] = 105
        self.assertEqual(hc.validate_contention_thresholds(boundary_hi)["max_gpu_temp_c"], 105)

        self.assertSentinelCode("THRESH_TYPE", hc.validate_contention_thresholds, [["min_repeats", 3]])
        self.assertSentinelCode("THRESH_TYPE", hc.validate_contention_thresholds, "abc")
        bad = dict(cfg)
        bad["extra"] = 1
        self.assertSentinelCode("THRESH_KEYS", hc.validate_contention_thresholds, bad)
        for key, val, code in [
            ("max_cpu_load_ratio_q", "0/1", "THRESH_RANGE"),
            ("max_cpu_load_ratio_q", "1000/1", "THRESH_RANGE"),
            ("max_bg_cpu_util_ratio_q", "0/1", "THRESH_RANGE"),
            ("max_bg_cpu_util_ratio_q", "11/10", "THRESH_RANGE"),
            ("max_external_gpu_util_pct", -1, "THRESH_RANGE"),
            ("max_external_gpu_util_pct", 101, "THRESH_RANGE"),
            ("max_external_gpu_util_pct", True, "THRESH_RANGE"),
            ("max_gpu_temp_c", 29, "THRESH_RANGE"),
            ("max_gpu_temp_c", 106, "THRESH_RANGE"),
            ("max_gpu_temp_c", False, "THRESH_RANGE"),
            ("min_repeats", 2, "THRESH_MIN_REPEATS"),
            ("min_repeats", True, "THRESH_MIN_REPEATS"),
            ("require_ac_online", 1, "THRESH_BOOL"),
            ("allow_external_gpu_apps", 0, "THRESH_BOOL"),
            ("allow_gpu_throttle", "no", "THRESH_BOOL"),
        ]:
            mut = dict(cfg)
            mut[key] = val
            self.assertSentinelCode(code, hc.validate_contention_thresholds, mut)

    def test_03_parsers_and_collector_branches(self) -> None:
        self.assertEqual(hc._parse_proc_stat_cpu("cpu  10 2 3 80 5\n"), (100, 85))
        self.assertEqual(hc._parse_proc_stat_cpu("cpu  10 2 3 80\n"), (95, 80))
        self.assertSentinelCode("PROC_STAT_PARSE", hc._parse_proc_stat_cpu, "cpu  10 2 3\n")
        self.assertSentinelCode("PROC_STAT_PARSE", hc._parse_proc_stat_cpu, "intr 123\n")

        self.assertEqual(hc._parse_loadavg("1.50 2.00 0.25 1/10 99"), (Fraction(3, 2), Fraction(2, 1), Fraction(1, 4)))
        self.assertEqual(hc._parse_loadavg("1.50 2.00 0.25"), (Fraction(3, 2), Fraction(2, 1), Fraction(1, 4)))
        self.assertSentinelCode("LOADAVG_PARSE", hc._parse_loadavg, "1.50 2.00")
        self.assertSentinelCode("LOADAVG_PARSE", hc._parse_loadavg, "NaN 1.0 1.0")

        self.assertIsNone(hc._parse_pid_stat("1 x R " + " ".join(["0"] * 25)))
        self.assertIsNone(hc._parse_pid_stat(") R " + " ".join(["0"] * 25)))
        self.assertIsNone(hc._parse_pid_stat("1 (x) R 1 2 3"))
        self.assertIsNone(hc._parse_pid_stat("1 (x) R " + " ".join(["bad"] * 22)))
        self.assertEqual(
            hc._parse_pid_stat("200 (a b) R 1 1 1 0 -1 0 0 0 0 0 50 10 0 0 20 0 1 0 1 1 2048"),
            (60, 2048),
        )

        apps = hc._parse_gpu_compute_apps(
            "\n10, /opt/bin/python, 512\n11, bad\n12, /bin/x, badmem\n13, /usr/bin/train, 1024\n",
            exclude_pids={10},
        )
        self.assertEqual(apps, [{"pid": 13, "proc": "train", "used_memory_mib": 1024}])

        self.assertEqual(
            hc._parse_gpu_telemetry("\n12, 1500, 72, 0x0000000000000020\n"),
            {
                "gpu_util_pct": 12,
                "gpu_sm_clock_mhz": 1500,
                "gpu_temp_c": 72,
                "gpu_throttle_mask": 0x20,
                "gpu_fault_throttled": True,
            },
        )
        parsed_idle = hc._parse_gpu_telemetry("0, 210, 50, 0x0000000000000003")
        self.assertIsNotNone(parsed_idle)
        if parsed_idle is not None:
            self.assertFalse(parsed_idle["gpu_fault_throttled"])
        self.assertIsNone(hc._parse_gpu_telemetry("   \n"))
        self.assertIsNone(hc._parse_gpu_telemetry("12, 1500"))
        self.assertIsNone(hc._parse_gpu_telemetry("bad, 1500, 72, 0x1"))

        self.assertSentinelCode("BAD_CPU_COUNT", hc.HostContentionCollector, cpu_count=0)
        self.assertSentinelCode("BAD_CPU_COUNT", hc.HostContentionCollector, cpu_count=True)

        self.assertEqual(_make_mock_collector(ac_online="1\n").read_ac_online(), True)
        self.assertEqual(_make_mock_collector(ac_online="0\n").read_ac_online(), False)
        self.assertIsNone(_make_mock_collector(ac_online=None).read_ac_online())
        self.assertIsNone(_make_mock_collector(ac_online="2\n").read_ac_online())

        self.assertEqual(_make_mock_collector(enable_foreground_probe=False).read_foreground_app(), "unavailable")
        self.assertEqual(_make_mock_collector(hypr_rc=1).read_foreground_app(), "unavailable")
        self.assertEqual(_make_mock_collector(hypr_out="not-json").read_foreground_app(), "unavailable")
        self.assertEqual(_make_mock_collector(hypr_out="[]").read_foreground_app(), "unavailable")
        self.assertEqual(_make_mock_collector(hypr_out='{"initialClass": "/bin/alacritty"}').read_foreground_app(), "alacritty")
        self.assertEqual(_make_mock_collector(hypr_out='{"title": "SECRET"}').read_foreground_app(), "unavailable")
        self.assertEqual(_make_mock_collector().read_foreground_app(), "foot")

        self.assertEqual(_make_mock_collector(enable_gpu_probe=False).read_gpu_compute_apps(set()), ("unavailable", []))
        self.assertEqual(_make_mock_collector(gpu_apps_rc=1).read_gpu_compute_apps(set()), ("unavailable", []))
        self.assertEqual(_make_mock_collector(enable_gpu_probe=False).read_gpu_telemetry(), ("unavailable", None))
        self.assertEqual(_make_mock_collector(gpu_tel_rc=1).read_gpu_telemetry(), ("unavailable", None))
        self.assertEqual(_make_mock_collector(gpu_tel_out="bad\n").read_gpu_telemetry(), ("unavailable", None))

        missing_stat = hc.HostContentionCollector(read_text=lambda _p: None, cpu_count=4)
        self.assertSentinelCode("PROC_STAT_MISSING", missing_stat.read_cpu_and_load, set())
        missing_load = hc.HostContentionCollector(
            read_text=lambda p: "cpu  1 0 0 9 0\n" if p == "/proc/stat" else None,
            cpu_count=4,
        )
        self.assertSentinelCode("LOADAVG_MISSING", missing_load.read_cpu_and_load, set())

        col_weird = _make_mock_collector(
            stat_series=[(1000, 900), (1000, 900)],
            extra_procs={
                300: ("gone\n", None),
                301: ("bad\n", "badstat"),
                302: ("ok\n", "302 (ok) S 1 1 1 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 1 1 512 0\n"),
            },
        )
        snap = col_weird.capture_snapshot(phase="pre", top_k=1)
        self.assertEqual(snap["process_count_observed"], 1)
        self.assertEqual(len(snap["top_cpu_processes"]), 1)
        self.assertSentinelCode("SNAPSHOT_PHASE", col_weird.capture_snapshot, phase="mid")
        self.assertSentinelCode("SNAPSHOT_TOPK", col_weird.capture_snapshot, phase="pre", top_k=0)
        self.assertSentinelCode("SNAPSHOT_TOPK", col_weird.capture_snapshot, phase="pre", top_k=21)
        self.assertSentinelCode("SNAPSHOT_TOPK", col_weird.capture_snapshot, phase="pre", top_k=True)
        self.assertEqual(col_weird.read_cpu_and_load({300, 301, 302})["self_cpu_ticks"], 15)
        samp0, _, _, _ = col_weird.collect_in_run_sample(
            prev_cpu_total=1000,
            prev_cpu_idle=900,
            prev_self_ticks=10,
            exclude_pids={os.getpid()},
        )
        self.assertEqual(samp0["bg_cpu_util_ratio_q"], "0/1")

        self.assertIsNone(hc._default_read_text("/proc/0/nonexistent_silisense_probe"))
        with tempfile.TemporaryDirectory() as fake_proc:
            Path(fake_proc, "12").mkdir()
            Path(fake_proc, "self").mkdir()
            self.assertEqual(hc._default_list_proc_pids(fake_proc), [12])
        self.assertEqual(hc._default_list_proc_pids("/nonexistent_silisense_proc"), [])
        self.assertGreaterEqual(hc.HostContentionCollector().cpu_count, 1)
        rc_bad, _ = hc._default_run_cmd(["nonexistent_silisense_binary_xyz"])
        self.assertEqual(rc_bad, 127)
        rc_ok, out_ok = hc._default_run_cmd([sys.executable, "-I", "-B", "-c", "print('ok')"])
        self.assertEqual((rc_ok, out_ok.strip()), (0, "ok"))

    def test_04_spread_and_quantiles_pure_q(self) -> None:
        self.assertSentinelCode("EMPTY_SAMPLES", hc.exact_quantile_q, [], 1, 2)
        self.assertEqual(hc.exact_quantile_q([Fraction(7, 1)], 1, 2), Fraction(7, 1))
        self.assertSentinelCode("SPREAD_MIN_REPEATS", hc.compute_spread_stats_q, [])
        self.assertSentinelCode("SPREAD_MIN_REPEATS", hc.compute_spread_stats_q, [Fraction(1, 1), Fraction(2, 1)])
        self.assertSentinelCode(
            "NONPOSITIVE_MEDIAN",
            hc.compute_spread_stats_q,
            [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
        )
        st3 = hc.compute_spread_stats_q([Fraction(7, 1), Fraction(7, 1), Fraction(7, 1)])
        self.assertEqual(st3["median_q"], "7/1")
        self.assertEqual(st3["iqr_q"], "0/1")

        st4 = hc.compute_spread_stats_q([Fraction(10, 1), Fraction(20, 1), Fraction(30, 1), Fraction(40, 1)])
        self.assertEqual(st4["median_q"], "25/1")
        self.assertEqual(st4["q1_q"], "35/2")
        self.assertEqual(st4["q3_q"], "65/2")
        self.assertEqual(st4["iqr_q"], "15/1")
        self.assertEqual(st4["mad_q"], "10/1")
        self.assertEqual(st4["rel_iqr_q"], "3/5")

        st5 = hc.compute_spread_stats_q([Fraction(2, 1), Fraction(4, 1), Fraction(6, 1), Fraction(8, 1), Fraction(10, 1)])
        self.assertEqual(st5["median_q"], "6/1")
        self.assertEqual(st5["iqr_q"], "4/1")
        self.assertEqual(st5["mad_q"], "2/1")

    def test_05_environment_verification_and_all_contention_triggers(self) -> None:
        rec_q = _build_sample_receipt()
        hc.verify_bound_receipt(rec_q)
        self.assertFalse(rec_q["environment"]["contended"])
        self.assertEqual(rec_q["environment"]["power_band"], "ac")
        self.assertEqual(rec_q["environment"]["contention_reasons"], [])

        rec_edge = _build_sample_receipt(
            loadavg="8.00 8.00 8.00 1/500 9999\n",
            stat_series=[(1000, 900), (2000, 1490), (3000, 2000)],
            gpu_tel_out="5, 1200, 80, 0x0000000000000001\n",
        )
        hc.verify_bound_receipt(rec_edge)
        self.assertFalse(rec_edge["environment"]["contended"])

        # Physical boundary zeros and maxes (0 util, 0 temp, 0 clock, 100 util, 130 temp, 0 load, 1 bg)
        rec_phys_zero = _build_sample_receipt(
            loadavg="0.00 0.00 0.00 1/500 9999\n",
            stat_series=[(1000, 1000), (2000, 2000), (3000, 3000)],
            gpu_tel_out="0, 0, 0, 0x0000000000000001\n",
        )
        hc.verify_bound_receipt(rec_phys_zero)
        rec_phys_hi = _build_sample_receipt(
            stat_series=[(1000, 1000), (2000, 1000), (3000, 1000)],
            self_ticks_series=[0, 0, 0],
            gpu_tel_out="100, 2000, 130, 0x0000000000000001\n",
        )
        hc.verify_bound_receipt(rec_phys_hi)

        rec_all_bad = _build_sample_receipt(
            ac_online="0\n",
            loadavg="9.00 9.00 9.00 1/500 9999\n",
            stat_series=[(1000, 900), (2000, 1400), (3000, 1800)],
            gpu_apps_out="999, /usr/bin/blender, 2048\n",
            gpu_tel_out="40, 900, 85, 0x0000000000000024\n",
        )
        hc.verify_bound_receipt(rec_all_bad)
        self.assertTrue(rec_all_bad["environment"]["contended"])
        self.assertEqual(rec_all_bad["environment"]["power_band"], "battery")
        self.assertEqual(
            rec_all_bad["environment"]["contention_reasons"],
            [
                "ac_power_lost_or_unknown",
                "cpu_load_ratio_exceeded",
                "bg_cpu_util_exceeded",
                "external_gpu_apps_present",
                "external_gpu_util_exceeded",
                "gpu_temp_exceeded",
                "gpu_throttle_active",
            ],
        )

        permissive = dict(hc.DEFAULT_CONTENTION_THRESHOLDS)
        permissive["require_ac_online"] = False
        permissive["allow_external_gpu_apps"] = True
        permissive["allow_gpu_throttle"] = True
        rec_perm = _build_sample_receipt(
            ac_online=None,
            gpu_apps_out="999, /usr/bin/blender, 2048\n",
            gpu_tel_out="2, 1200, 60, 0x0000000000000020\n",
            thresholds=permissive,
        )
        hc.verify_bound_receipt(rec_perm)
        self.assertFalse(rec_perm["environment"]["contended"])
        self.assertEqual(rec_perm["environment"]["power_band"], "unknown")

        rec_cpu_only = _build_sample_receipt(gpu_apps_rc=127, gpu_tel_rc=127)
        hc.verify_bound_receipt(rec_cpu_only)
        self.assertEqual(rec_cpu_only["environment"]["in_run_summary"]["gpu_status"], "unavailable")

        col = _make_mock_collector()
        pre = col.capture_snapshot(phase="pre")
        post = col.capture_snapshot(phase="post")
        self.assertSentinelCode("SNAPSHOT_TYPE", hc.verify_host_snapshot, "bad", expected_phase="pre")
        for code, mut in [
            ("SNAPSHOT_SCHEMA", lambda d: d.__setitem__("schema_version", "bad")),
            ("SNAPSHOT_PHASE", lambda d: d.__setitem__("phase", "post")),
            ("SNAPSHOT_CPU_COUNT", lambda d: d.__setitem__("cpu_count", 0)),
            ("SNAPSHOT_CPU_COUNT", lambda d: d.__setitem__("cpu_count", True)),
            ("SNAPSHOT_SHA_FORMAT", lambda d: d.__setitem__("snapshot_sha256", "xyz")),
            ("SNAPSHOT_SHA_MISMATCH", lambda d: d.__setitem__("process_count_observed", 999999)),
        ]:
            s = copy.deepcopy(pre)
            mut(s)
            self.assertSentinelCode(code, hc.verify_host_snapshot, s, expected_phase="pre")

        post_diff_cpu = _make_mock_collector(cpu_count=8).capture_snapshot(phase="post")
        self.assertSentinelCode(
            "ENV_CPU_COUNT_MISMATCH",
            hc.build_environment_block,
            pre_snapshot=pre,
            post_snapshot=post_diff_cpu,
            in_run_samples=[],
        )
        self.assertSentinelCode(
            "ENV_NO_SAMPLES",
            hc.build_environment_block,
            pre_snapshot=pre,
            post_snapshot=post,
            in_run_samples=[],
        )

        env_ok = rec_q["environment"]
        self.assertSentinelCode("ENV_TYPE", hc.verify_environment_block, [])
        tamper_cases = [
            ("ENV_SCHEMA", False, lambda e: e.__setitem__("schema_version", "bad")),
            ("ENV_ENERGY_POLICY", True, lambda e: e.__setitem__("energy_power_status", "CLAIMED")),
            ("ENV_ENERGY_POLICY", True, lambda e: e.__setitem__("mj_per_tok", 1.2)),
            ("ENV_KEYS", True, lambda e: e.__setitem__("extra_unknown", 1)),
            ("ENV_HASH_FORMAT", False, lambda e: e.__setitem__("pre_snapshot_sha256", "short")),
            ("ENV_HASH_FORMAT", False, lambda e: e.__setitem__("environment_sha256", 123)),
            ("ENV_HASH_MISMATCH", False, lambda e: e.__setitem__("contended", True)),
            ("ENV_STRICT_TYPES", True, lambda e: e.__setitem__("ac_online_all", 1)),
            ("ENV_STRICT_TYPES", True, lambda e: e.__setitem__("external_gpu_apps_detected", 0)),
            ("ENV_STRICT_TYPES", True, lambda e: e.__setitem__("contended", 0)),
            ("ENV_STRICT_TYPES", True, lambda e: e.__setitem__("cpu_load_max_ratio", True)),
            ("ENV_STRICT_TYPES", True, lambda e: e.__setitem__("bg_cpu_util_max_ratio", "0.1")),
            ("ENV_SUMMARY_TYPE", True, lambda e: e.__setitem__("in_run_summary", [])),
            ("ENV_SUMMARY_KEYS", True, lambda e: e["in_run_summary"].__setitem__("extra_field", 1)),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("sample_count", 0)),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("sample_count", True)),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("ac_online_observations", -1)),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("external_gpu_apps_max", False)),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("gpu_throttle_fault_any", 0)),
            ("THRESH_TYPE", True, lambda e: e.__setitem__("thresholds", None)),
            ("THRESH_TYPE", True, lambda e: e.__setitem__("thresholds", [["min_repeats", 3]])),
            ("THRESH_TYPE", True, lambda e: e.__setitem__("thresholds", "abc")),
            ("ENV_SUMMARY_MISMATCH", True, lambda e: e["in_run_summary"].__setitem__("cpu_load_ratio_max_q", "1/100")),
            ("ENV_SUMMARY_MISMATCH", True, lambda e: e["in_run_summary"].__setitem__("bg_cpu_util_max_q", "1/100")),
            (
                "ENV_RANGE",
                True,
                lambda e: (
                    e.__setitem__("cpu_load_max_ratio_q", "-1/1"),
                    e["in_run_summary"].__setitem__("cpu_load_ratio_max_q", "-1/1"),
                    e["in_run_summary"].__setitem__("cpu_load_ratio_median_q", "-1/1"),
                ),
            ),
            ("ENV_RANGE", True, lambda e: e["in_run_summary"].__setitem__("cpu_load_ratio_median_q", "9/1")),
            ("ENV_RANGE", True, lambda e: e["in_run_summary"].__setitem__("bg_cpu_util_median_q", "-1/10")),
            ("ENV_RANGE", True, lambda e: e["in_run_summary"].__setitem__("bg_cpu_util_median_q", "9/10")),
            (
                "ENV_RANGE",
                True,
                lambda e: (
                    e.__setitem__("bg_cpu_util_max_ratio_q", "11/10"),
                    e["in_run_summary"].__setitem__("bg_cpu_util_max_q", "11/10"),
                ),
            ),
            ("ENV_FLOAT_MISMATCH", True, lambda e: e.__setitem__("cpu_load_max_ratio", 0.99)),
            ("ENV_FLOAT_MISMATCH", True, lambda e: e.__setitem__("bg_cpu_util_max_ratio", 0.99)),
            ("ENV_POWER_BAND", True, lambda e: e["in_run_summary"].__setitem__("ac_online_observations", 99)),
            (
                "ENV_AC_ALL_MISMATCH",
                True,
                lambda e: (
                    e["in_run_summary"].__setitem__("ac_online_observations", 2),
                    e["in_run_summary"].__setitem__("ac_offline_observations", 1),
                    e.__setitem__("power_band", "battery"),
                ),
            ),
            ("ENV_POWER_BAND", True, lambda e: e.__setitem__("power_band", "battery")),
            ("ENV_GPU_STATUS", True, lambda e: e["in_run_summary"].__setitem__("gpu_status", "maybe")),
            ("ENV_STRICT_TYPES", True, lambda e: e["in_run_summary"].__setitem__("gpu_temp_max_c", 55.0)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_util_max_pct", -1)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_util_max_pct", 101)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_sm_clock_min_mhz", -1)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_sm_clock_min_mhz", 9999)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_temp_max_c", -500)),
            ("ENV_GPU_RANGE", True, lambda e: e["in_run_summary"].__setitem__("gpu_temp_max_c", 131)),
            ("ENV_GPU_APPS_MISMATCH", True, lambda e: e["in_run_summary"].__setitem__("external_gpu_apps_max", 2)),
            ("ENV_CONTENTION_MISMATCH", True, lambda e: e.__setitem__("contention_reasons", ["fake"])),
            ("ENV_CONTENTION_MISMATCH", True, lambda e: e.__setitem__("contended", True)),
        ]
        for code, resign, mut in tamper_cases:
            e = copy.deepcopy(env_ok)
            mut(e)
            if resign:
                e = _resign_env(e)
            self.assertSentinelCode(code, hc.verify_environment_block, e)

        # Unavailable GPU tamper checks (non-null metrics or throttle=True)
        env_nogpu = rec_cpu_only["environment"]
        for mut in [
            lambda e: e["in_run_summary"].__setitem__("gpu_util_max_pct", 10),
            lambda e: e["in_run_summary"].__setitem__("gpu_sm_clock_min_mhz", 210),
            lambda e: e["in_run_summary"].__setitem__("gpu_sm_clock_max_mhz", 210),
            lambda e: e["in_run_summary"].__setitem__("gpu_temp_max_c", 50),
            lambda e: e["in_run_summary"].__setitem__("gpu_throttle_fault_any", True),
        ]:
            e = _resign_env(copy.deepcopy(env_nogpu))
            mut(e)
            e = _resign_env(e)
            self.assertSentinelCode("ENV_GPU_UNAVAILABLE", hc.verify_environment_block, e)

    def test_06_receipt_binding_comparison_and_stratification(self) -> None:
        base = _build_sample_receipt(arm_name="base", values=[Fraction(100, 1), Fraction(102, 1), Fraction(104, 1)])
        cand = _build_sample_receipt(arm_name="cand", values=[Fraction(200, 1), Fraction(204, 1), Fraction(208, 1)])
        overlap = _build_sample_receipt(arm_name="overlap", values=[Fraction(101, 1), Fraction(103, 1), Fraction(105, 1)])
        contended = _build_sample_receipt(
            arm_name="contended",
            stat_series=[(1000, 900), (2000, 1100), (3000, 1200)],
            values=[Fraction(70, 1), Fraction(75, 1), Fraction(80, 1)],
        )
        batt = _build_sample_receipt(arm_name="batt", ac_online="0\n", values=[Fraction(80, 1), Fraction(82, 1), Fraction(84, 1)])

        # B2 threshold-shopping proof: hot workload with loosened thresholds must NEVER compare as same band!
        loose_thresh = dict(hc.DEFAULT_CONTENTION_THRESHOLDS)
        loose_thresh["max_cpu_load_ratio_q"] = "2/1"
        loose_thresh["max_bg_cpu_util_ratio_q"] = "1/1"
        hot_loose = _build_sample_receipt(
            arm_name="hot_loose",
            loadavg="24.00 24.00 24.00 1/500 9999\n",
            stat_series=[(1000, 900), (2000, 1000), (3000, 1100)],
            thresholds=loose_thresh,
            values=[Fraction(60, 1), Fraction(65, 1), Fraction(70, 1)],
        )
        self.assertFalse(hot_loose["environment"]["contended"])
        self.assertSentinelCode("COMPARE_THRESHOLD_MISMATCH", hc.compare_bound_receipts, base, hot_loose)
        strat_loose = hc.compare_bound_receipts(base, hot_loose, allow_stratified=True)
        self.assertEqual(strat_loose["status"], "STRATIFIED_UNCOMPARABLE")
        self.assertIsNone(strat_loose["ratio_q"])

        # B1 end-to-end bare unit / arm_name / power metric_name / metadata rejections
        for bad_arm, bad_name, bad_unit, bad_meta in [
            ("45.2 W", "ops", "ops/s", {}),
            ("avgPower", "ops", "ops/s", {}),
            ("x", "throughput", "W", {}),
            ("x", "energy", "mJ", {}),
            ("x", "avg_power", "W", {}),
            ("x", "avgPower", "ops/s", {}),
            ("x", "latency", "ms", {"power": 45.2, "note": "45.2 W"}),
            ("x", "latency", "ms", {"po\u0301wer": 45}),
            ("x", "latency", "ms", {"power_band": 45.2}),
            ("x", "eff", "mJ/token", {"kwh": 0.1}),
            ("x", "ops", "ops/s", {"note": ("1.1728 mJ/tok",)}),
            ("x", "45.2 W", "ops/s", {}),
            ("x", "ops", "1.2 J/tok", {}),
        ]:
            self.assertSentinelCode(
                "RECEIPT_ENERGY_POLICY",
                hc.bind_receipt_with_environment,
                arm_name=bad_arm,
                metric_name=bad_name,
                metric_unit=bad_unit,
                repeat_values_q=[Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                environment=base["environment"],
                metadata=bad_meta,
            )

        for bad_arm, bad_name, bad_unit in [
            ("", "ops", "ops/s"),
            (123, "ops", "ops/s"),
            ("x", "", "ops/s"),
            ("x", "ops", ""),
            ("x", 123, "ops/s"),
            ("x", "ops", 123),
        ]:
            self.assertSentinelCode(
                "RECEIPT_METRIC_TYPE",
                hc.bind_receipt_with_environment,
                arm_name=bad_arm,  # type: ignore[arg-type]
                metric_name=bad_name,  # type: ignore[arg-type]
                metric_unit=bad_unit,  # type: ignore[arg-type]
                repeat_values_q=[Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                environment=base["environment"],
            )

        for bad_vals in (None, "123", b"123", 123):
            self.assertSentinelCode(
                "RECEIPT_VALUES_TYPE",
                hc.bind_receipt_with_environment,
                arm_name="x",
                metric_name="ops",
                metric_unit="ops/s",
                repeat_values_q=bad_vals,  # type: ignore[arg-type]
                environment=base["environment"],
            )
        for bad_meta_type in ([1, 2], "ab", 45, [["mJ_per_tok", 1.2]]):
            self.assertSentinelCode(
                "RECEIPT_METADATA_TYPE",
                hc.bind_receipt_with_environment,
                arm_name="x",
                metric_name="ops",
                metric_unit="ops/s",
                repeat_values_q=[Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                environment=base["environment"],
                metadata=bad_meta_type,  # type: ignore[arg-type]
            )
        self.assertSentinelCode(
            "RECEIPT_MIN_REPEATS",
            hc.bind_receipt_with_environment,
            arm_name="x",
            metric_name="ops",
            metric_unit="ops/s",
            repeat_values_q=[Fraction(1, 1), Fraction(2, 1)],
            environment=base["environment"],
        )
        self.assertSentinelCode(
            "RECEIPT_BAD_VALUE",
            hc.bind_receipt_with_environment,
            arm_name="x",
            metric_name="ops",
            metric_unit="ops/s",
            repeat_values_q=[Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)],
            environment=base["environment"],
        )
        self.assertSentinelCode(
            "RECEIPT_BAD_VALUE",
            hc.bind_receipt_with_environment,
            arm_name="x",
            metric_name="ops",
            metric_unit="ops/s",
            repeat_values_q=[1, Fraction(2, 1), Fraction(3, 1)],  # type: ignore[list-item]
            environment=base["environment"],
        )

        self.assertSentinelCode("RECEIPT_TYPE", hc.verify_bound_receipt, "not-a-dict")
        for code, resign, mut in [
            ("RECEIPT_SCHEMA", False, lambda r: r.__setitem__("schema_version", "bad")),
            ("RECEIPT_METADATA_TYPE", True, lambda r: r.__setitem__("metadata", [["mJ_per_tok", 1.2], ["avgPower", 45.2]])),
            ("RECEIPT_METADATA_TYPE", True, lambda r: r.__setitem__("metadata", [["safe", 1]])),
            ("RECEIPT_ENERGY_POLICY", True, lambda r: r.__setitem__("metadata", {"nested": [["mJ_per_tok", 1.2]]})),
            ("RECEIPT_ENERGY_POLICY", True, lambda r: r.__setitem__("energy_power_status", "CLAIMED")),
            ("RECEIPT_ENERGY_POLICY", True, lambda r: r["metadata"].__setitem__("mj_per_tok", 0.5)),
            ("RECEIPT_ENERGY_POLICY", True, lambda r: r.__setitem__("metric_unit", "W")),
            ("RECEIPT_KEYS", True, lambda r: r.__setitem__("unpinned_extra", 1)),
            ("RECEIPT_METRIC_TYPE", True, lambda r: r.__setitem__("metric_name", "")),
            ("RECEIPT_SHA_FORMAT", False, lambda r: r.__setitem__("result_sha256", "bad")),
            ("RECEIPT_SHA_MISMATCH", False, lambda r: r.__setitem__("arm_name", "forged")),
            ("RECEIPT_VALUES_TYPE", True, lambda r: r.__setitem__("repeat_values_q", "1/1")),
            ("RECEIPT_MIN_REPEATS", True, lambda r: r.__setitem__("repeat_values_q", ["1/1", "2/1"])),
            ("RECEIPT_BAD_VALUE", True, lambda r: r.__setitem__("repeat_values_q", ["0/1", "2/1", "3/1"])),
            ("RECEIPT_SPREAD_MISMATCH", True, lambda r: r["spread"].__setitem__("median_q", "999/1")),
        ]:
            r = copy.deepcopy(base)
            mut(r)
            if resign:
                r = _resign_receipt(r)
            self.assertSentinelCode(code, hc.verify_bound_receipt, r)

        cmp_up = hc.compare_bound_receipts(base, cand)
        self.assertEqual(cmp_up["status"], "COMPARABLE_SAME_BAND")
        self.assertEqual(cmp_up["ratio_q"], "2/1")
        self.assertEqual(cmp_up["delta_q"], "102/1")
        self.assertTrue(cmp_up["non_overlapping_iqr"])

        cmp_down = hc.compare_bound_receipts(cand, base)
        self.assertEqual(cmp_down["ratio_q"], "1/2")
        self.assertTrue(cmp_down["non_overlapping_iqr"])

        cmp_ov = hc.compare_bound_receipts(base, overlap)
        self.assertFalse(cmp_ov["non_overlapping_iqr"])

        diff_metric = _build_sample_receipt(metric_name="other_metric")
        diff_unit = _build_sample_receipt(metric_unit="tok/s")
        self.assertSentinelCode("COMPARE_METRIC_MISMATCH", hc.compare_bound_receipts, base, diff_metric)
        self.assertSentinelCode("COMPARE_METRIC_MISMATCH", hc.compare_bound_receipts, base, diff_unit)

        self.assertSentinelCode("COMPARE_BAND_MISMATCH", hc.compare_bound_receipts, base, contended)
        self.assertSentinelCode("COMPARE_BAND_MISMATCH", hc.compare_bound_receipts, base, batt)
        strat_cmp = hc.compare_bound_receipts(base, contended, allow_stratified=True)
        self.assertEqual(strat_cmp["status"], "STRATIFIED_UNCOMPARABLE")
        self.assertIsNone(strat_cmp["ratio_q"])

        bands = hc.stratify_receipts([contended, base, batt, hot_loose])
        self.assertEqual(len(bands), 4)
        self.assertTrue(all(k.startswith(("contended_ac_", "contended_battery_", "quiet_ac_")) for k in bands))

    def test_07_monitored_arm_probe_and_hermetic_cli_roundtrip(self) -> None:
        col = _make_mock_collector(
            stat_series=[(1000, 900), (2000, 1800), (3000, 2700), (4000, 3600)],
        )
        self.assertSentinelCode(
            "ARM_MIN_REPEATS",
            hc.run_monitored_arm,
            arm_name="short",
            metric_name="ops",
            metric_unit="ops/s",
            n_repeats=2,
            workload_fn=lambda: Fraction(10, 1),
            collector=col,
        )
        self.assertSentinelCode(
            "ARM_MIN_REPEATS",
            hc.run_monitored_arm,
            arm_name="boolrep",
            metric_name="ops",
            metric_unit="ops/s",
            n_repeats=True,  # type: ignore[arg-type]
            workload_fn=lambda: Fraction(10, 1),
            collector=col,
        )
        rec = hc.run_monitored_arm(
            arm_name="mock_monitored",
            metric_name="ops",
            metric_unit="ops/s",
            n_repeats=3,
            workload_fn=lambda: Fraction(50, 1),
            collector=col,
            sample_interval_s=0.0,
        )
        hc.verify_bound_receipt(rec)

        slow_col = _make_mock_collector(stat_series=[(1000, 900), (2000, 1800), (3000, 2700)])
        rec_fast = hc.run_monitored_arm(
            arm_name="fast_arm",
            metric_name="ops",
            metric_unit="ops/s",
            n_repeats=3,
            workload_fn=lambda: Fraction(75, 1),
            collector=slow_col,
            sample_interval_s=0.001,
        )
        hc.verify_bound_receipt(rec_fast)

        quiet = _build_sample_receipt(arm_name="quiet", values=[Fraction(100, 1), Fraction(105, 1), Fraction(110, 1)])
        cont = _build_sample_receipt(
            arm_name="cont",
            stat_series=[(1000, 900), (2000, 1100), (3000, 1200)],
            values=[Fraction(60, 1), Fraction(65, 1), Fraction(70, 1)],
        )
        probe: Dict[str, Any] = {
            "schema_version": hc.PROBE_SCHEMA,
            "energy_power_status": hc.ENERGY_POLICY_STATUS,
            "cpu_count": 16,
            "n_repeats_per_band": 3,
            "synthetic_burner_workers": 2,
            "quiet_receipt": quiet,
            "contended_receipt": cont,
            "cross_band_comparison": hc.compare_bound_receipts(quiet, cont, allow_stratified=True),
            "stratified_bands": hc.stratify_receipts([quiet, cont]),
            "band_separation_verified": True,
        }
        probe["probe_sha256"] = hc.canonical_sha256(probe)
        hc.verify_live_probe(probe)

        self.assertSentinelCode("PROBE_TYPE", hc.verify_live_probe, [])
        for code, resign, mut in [
            ("PROBE_SCHEMA", False, lambda p: p.__setitem__("schema_version", "bad")),
            ("PROBE_ENERGY_POLICY", True, lambda p: p.__setitem__("energy_power_status", "CLAIMED")),
            ("PROBE_ENERGY_POLICY", True, lambda p: p.__setitem__("mean_watts", 55.0)),
            ("PROBE_KEYS", True, lambda p: p.__setitem__("unpinned_extra", "extra_text")),
            (
                "PROBE_ENERGY_POLICY",
                True,
                lambda p: p.__setitem__(
                    "quiet_receipt",
                    _resign_receipt({**p["quiet_receipt"], "metadata": [["mJ_per_tok", 1.2]]}),
                ),
            ),
            (
                "RECEIPT_METADATA_TYPE",
                True,
                lambda p: p.__setitem__(
                    "quiet_receipt",
                    _resign_receipt({**p["quiet_receipt"], "metadata": [["safe", 1]]}),
                ),
            ),
            ("PROBE_SHA_FORMAT", False, lambda p: p.__setitem__("probe_sha256", "bad")),
            ("PROBE_SHA_MISMATCH", False, lambda p: p.__setitem__("cpu_count", 99)),
            ("PROBE_META", True, lambda p: p.__setitem__("cpu_count", 0)),
            ("PROBE_META", True, lambda p: p.__setitem__("cpu_count", True)),
            ("PROBE_META", True, lambda p: p.__setitem__("n_repeats_per_band", 4)),
            ("PROBE_META", True, lambda p: p.__setitem__("n_repeats_per_band", True)),
            (
                "PROBE_META",
                True,
                lambda p: p.__setitem__(
                    "contended_receipt",
                    _build_sample_receipt(
                        arm_name="cont4",
                        stat_series=[(1000, 900), (2000, 1100), (3000, 1200)],
                        values=[Fraction(60, 1), Fraction(65, 1), Fraction(70, 1), Fraction(75, 1)],
                    ),
                ),
            ),
            ("PROBE_META", True, lambda p: p.__setitem__("synthetic_burner_workers", -1)),
            ("PROBE_META", True, lambda p: p.__setitem__("synthetic_burner_workers", False)),
            ("PROBE_CMP_MISMATCH", True, lambda p: p.__setitem__("cross_band_comparison", {})),
            ("PROBE_STRAT_MISMATCH", True, lambda p: p.__setitem__("stratified_bands", {})),
            ("PROBE_SEP_MISMATCH", True, lambda p: p.__setitem__("band_separation_verified", False)),
        ]:
            pm = copy.deepcopy(probe)
            mut(pm)
            if resign:
                pm["probe_sha256"] = hc.canonical_sha256(pm, exclude_key="probe_sha256")
            self.assertSentinelCode(code, hc.verify_live_probe, pm)

        quiet2 = _build_sample_receipt(arm_name="quiet2", values=[Fraction(90, 1), Fraction(95, 1), Fraction(100, 1)])
        probe_same: Dict[str, Any] = {
            "schema_version": hc.PROBE_SCHEMA,
            "energy_power_status": hc.ENERGY_POLICY_STATUS,
            "cpu_count": 1,
            "n_repeats_per_band": 3,
            "synthetic_burner_workers": 0,
            "quiet_receipt": quiet,
            "contended_receipt": quiet2,
            "cross_band_comparison": hc.compare_bound_receipts(quiet, quiet2, allow_stratified=True),
            "stratified_bands": hc.stratify_receipts([quiet, quiet2]),
            "band_separation_verified": False,
        }
        probe_same["probe_sha256"] = hc.canonical_sha256(probe_same)
        hc.verify_live_probe(probe_same)

        ops_q = hc._cpu_hash_workload(rounds=10)
        self.assertGreater(ops_q, Fraction(0))
        mock_probe = hc.run_live_contention_probe(
            n_repeats=3,
            rounds_per_repeat=10,
            burner_workers=0,
            collector=_make_mock_collector(),
            workload_fn=lambda: Fraction(100, 1),
        )
        hc.verify_live_probe(mock_probe)

        # Hermetic CLI roundtrip (zero live /proc, nvidia-smi, or hyprctl calls)
        with tempfile.TemporaryDirectory() as td:
            p_path = os.path.join(td, "probe.json")
            with open(p_path, "w", encoding="utf-8") as fh:
                json.dump(probe, fh)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(hc.main(["--verify-probe", p_path]), 0)
            self.assertEqual(json.loads(buf.getvalue())["status"], "PASS")
            out_path = os.path.join(td, "hermetic_out.json")
            buf2 = io.StringIO()
            hermetic_col = _make_mock_collector(
                stat_series=[(1000, 900), (2000, 1800), (3000, 2700), (4000, 3600), (5000, 4500), (6000, 5400)]
            )
            with mock.patch.object(hc, "HostContentionCollector", return_value=hermetic_col):
                with contextlib.redirect_stdout(buf2):
                    self.assertEqual(
                        hc.main(["--run-live-probe", out_path, "--repeats", "3", "--rounds", "20", "--burners", "1"]),
                        0,
                    )
            self.assertEqual(json.loads(buf2.getvalue())["status"], "WROTE")
            buf3 = io.StringIO()
            with contextlib.redirect_stdout(buf3):
                self.assertEqual(hc.main(["--verify-probe", out_path]), 0)
            self.assertEqual(json.loads(buf3.getvalue())["status"], "VERIFIED_NO_BAND_SEPARATION")
            err_buf = io.StringIO()
            with mock.patch.object(hc, "HostContentionCollector", return_value=hermetic_col):
                with contextlib.redirect_stderr(err_buf):
                    self.assertEqual(
                        hc.main(["--run-live-probe", out_path, "--repeats", "2", "--rounds", "20", "--burners", "0"]),
                        2,
                    )
            self.assertEqual(json.loads(err_buf.getvalue())["code"], "ARM_MIN_REPEATS")
            bad_json_path = os.path.join(td, "bad.json")
            with open(bad_json_path, "w", encoding="utf-8") as fh:
                fh.write("{bad")
            bad_utf8_path = os.path.join(td, "bad_utf8.json")
            with open(bad_utf8_path, "wb") as fh:
                fh.write(b"\xff\xfe")
            for bad_cli_args in (
                ["--verify-probe", os.path.join(td, "missing.json")],
                ["--verify-probe", bad_json_path],
                ["--verify-probe", bad_utf8_path],
                ["--run-live-probe", os.path.join(td, "missing_dir", "out.json"), "--repeats", "3", "--rounds", "20", "--burners", "0"],
            ):
                ebuf = io.StringIO()
                with mock.patch.object(hc, "HostContentionCollector", return_value=hermetic_col):
                    with contextlib.redirect_stderr(ebuf):
                        self.assertEqual(hc.main(bad_cli_args), 2)
                self.assertEqual(json.loads(ebuf.getvalue())["code"], "CLI_IO_ERROR")
            buf4 = io.StringIO()
            with contextlib.redirect_stdout(buf4):
                self.assertEqual(hc.main([]), 2)


if __name__ == "__main__":
    unittest.main()
