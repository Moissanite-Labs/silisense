"""R5 tests for Two-Mode Sensor Airlock, 64 B `<8Q>` Frame, and `zone-canon-v0`."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense.airlock import (
    AudioAuthorizationGate,
    RaplSysfsReader,
    query_proc_irq_and_cpu_khz,
    validate_endpoint_path,
)
from silisense.errors import HardwareExitCode, SiliSensePermissionError, SiliSenseValueError
from silisense.frame64 import (
    FRAME64_BYTES,
    TelemetryFrame64,
    canonicalize_zone_canon_v0,
    domain_tree_leaf_sha256,
    pack_flags_temp_mhz,
    unpack_flags_temp_mhz,
)


class TestAirlockAndFrame64R5(unittest.TestCase):
    def test_01_camera_and_ir_unconditional_ban_and_mode2_expiry(self) -> None:
        clock = [500.0]
        gate = AudioAuthorizationGate.create(
            enable_dmic_authorized=True,
            authorize_audio_window_sec=25.0,
            pi_authorization_token="PI-AUDIO-20261125",
            clock_fn=lambda: clock[0],
        )
        gate.verify_active()
        self.assertAlmostEqual(gate.remaining_sec(), 25.0, places=6)

        for forbidden in ("/dev/video0", "/dev/video1", "/dev/media0", "/dev/v4l/by-id/cam", "sys/class/leds/ir_led", "pipewire://camera"):
            with self.assertRaises(SiliSensePermissionError) as ctx:
                validate_endpoint_path(forbidden, auth_gate=gate)
            self.assertEqual(ctx.exception.code, "ERR_CAMERA_FORBIDDEN")
            self.assertEqual(ctx.exception.exit_code, HardwareExitCode.PREFLIGHT_FAIL)

        for dmic_ep in ("/dev/snd/pcmC0D0c", "alsa:hw:0,0", "arecord", "pw-record"):
            with self.assertRaises(SiliSensePermissionError) as ctx_m1:
                validate_endpoint_path(dmic_ep, auth_gate=None)
            self.assertEqual(ctx_m1.exception.code, "ERR_DMIC_DISABLED_DEFAULT")
            validate_endpoint_path(dmic_ep, auth_gate=gate)

        clock[0] = 525.1
        with self.assertRaises(SiliSensePermissionError) as ctx_exp:
            gate.verify_active()
        self.assertEqual(ctx_exp.exception.code, "ERR_AUDIO_WINDOW_EXPIRED")

        for bad in (
            {"enable_dmic_authorized": False, "authorize_audio_window_sec": 10.0, "pi_authorization_token": "TOKEN1234"},
            {"enable_dmic_authorized": True, "authorize_audio_window_sec": 0.0, "pi_authorization_token": "TOKEN1234"},
            {"enable_dmic_authorized": True, "authorize_audio_window_sec": 305.0, "pi_authorization_token": "TOKEN1234"},
            {"enable_dmic_authorized": True, "authorize_audio_window_sec": float("inf"), "pi_authorization_token": "TOKEN1234"},
            {"enable_dmic_authorized": True, "authorize_audio_window_sec": 10.0, "pi_authorization_token": "short"},
        ):
            with self.assertRaises(SiliSensePermissionError):
                AudioAuthorizationGate.create(**bad)  # type: ignore[arg-type]

    def test_02_mocked_rapl_wraparound_and_proc_irq_reader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d_pkg = root / "intel-rapl:0"
            d_pkg.mkdir()
            (d_pkg / "name").write_text("package-0\n", encoding="utf-8")
            (d_pkg / "max_energy_range_uj").write_text("1000000\n", encoding="utf-8")
            (d_pkg / "energy_uj").write_text("950000\n", encoding="utf-8")

            t_box = [10.0]
            reader = RaplSysfsReader(powercap_root=root)
            self.assertEqual(reader.discover(), ["pkg_w"])
            w0 = reader.sample_watts(clock_fn=lambda: t_box[0])
            self.assertEqual(w0["pkg_w"], 0.0)

            # Simulate counter wraparound: 950000 -> 50000 modulo 1000000 => +100000 uJ over 0.01 s = 10.0 W
            (d_pkg / "energy_uj").write_text("50000\n", encoding="utf-8")
            t_box[0] = 10.01
            w1 = reader.sample_watts(clock_fn=lambda: t_box[0])
            self.assertAlmostEqual(w1["pkg_w"], 10.0, places=5)

            proc_f = root / "proc_stat"
            proc_f.write_text("cpu  100 0 50 1000\nintr 98765 1 2\nctxt 543210\n", encoding="utf-8")
            cpu_dir = root / "cpu0" / "cpufreq"
            cpu_dir.mkdir(parents=True)
            (cpu_dir / "scaling_cur_freq").write_text("4200000\n", encoding="utf-8")
            irq_info = query_proc_irq_and_cpu_khz(proc_stat_path=proc_f, cpufreq_glob_root=root)
            self.assertEqual(irq_info["intr_total"], 98765)
            self.assertEqual(irq_info["ctxt_total"], 543210)
            self.assertEqual(irq_info["mean_cpu_clock_khz"], 4200000.0)

    def test_03_frame64_binary_pack_and_zone_canon_v0_refusals(self) -> None:
        packed_flags = pack_flags_temp_mhz(sw_power_cap=True, hw_thermal_slowdown=False, gpu_temp_deci_c=715, sm_clock_mhz=1710)
        unpacked = unpack_flags_temp_mhz(packed_flags)
        self.assertTrue(unpacked["sw_power_cap"])
        self.assertFalse(unpacked["hw_thermal_slowdown"])
        self.assertEqual(unpacked["gpu_temp_deci_c"], 715)
        self.assertEqual(unpacked["sm_clock_mhz"], 1710)

        frame = TelemetryFrame64(
            ts_mono_ns=1732550000123456789,
            pkg_uj=45000000,
            core_uj=31000000,
            uncore_uj=8200000,
            psys_uj=92000000,
            gpu_mj=110500,
            step_us=78068,
            flags_temp_mhz=packed_flags,
        )
        raw = frame.to_bytes()
        self.assertEqual(len(raw), FRAME64_BYTES)
        restored = TelemetryFrame64.from_bytes(raw)
        self.assertEqual(frame, restored)
        self.assertEqual(len(frame.frame_sha256()), 64)

        with self.assertRaises(SiliSenseValueError) as ctx_len:
            TelemetryFrame64.from_bytes(raw[:-1])
        self.assertEqual(ctx_len.exception.code, "ERR_FRAME64_LEN")

        # zone-canon-v0 MUST reject floats and 0x7f DEL bytes
        with self.assertRaises(SiliSenseValueError) as ctx_flt:
            canonicalize_zone_canon_v0({"bad_float": 1.25})
        self.assertEqual(ctx_flt.exception.code, "ERR_ZONE_CANON_FLOAT_FORBIDDEN")

        with self.assertRaises(SiliSenseValueError) as ctx_del:
            canonicalize_zone_canon_v0({"bad_del": "abc\x7fdef"})
        self.assertEqual(ctx_del.exception.code, "ERR_ZONE_CANON_CHAR")

        # Domain-separated u64_le tree hash prevents path||content concatenation ambiguity
        h_empty = hashlib.sha256(b"").hexdigest()
        d1 = domain_tree_leaf_sha256("a/b.json", h_empty)
        d2 = domain_tree_leaf_sha256("a/b.jso", h_empty)
        self.assertNotEqual(d1, d2)


if __name__ == "__main__":
    unittest.main()
