"""R5 tests for Two-Mode Sensor Airlock, Symlink/Normalization Hardening, Frame64, and zone-canon-v0."""

from __future__ import annotations

import hashlib
import os
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
    def test_01_camera_ir_aliases_traversal_symlinks_and_token_redaction(self) -> None:
        clock = [500.0]
        secret_tok = "PI-AUDIO-SECRET-20261125"
        gate = AudioAuthorizationGate.create(
            enable_dmic_authorized=True,
            authorize_audio_window_sec=25.0,
            pi_authorization_token=secret_tok,
            clock_fn=lambda: clock[0],
        )
        # Token MUST NOT leak into repr()
        self.assertNotIn(secret_tok, repr(gate))
        gate.verify_active()
        self.assertAlmostEqual(gate.remaining_sec(), 25.0, places=6)

        # Direct, unnormalized, traversal, and V4L2 char-major-81 aliases MUST ALL fail closed with ERR_CAMERA_FORBIDDEN
        # even when Mode-2 audio authorization is active!
        forbidden_paths = (
            "/dev/video0",
            "/dev/./video0",
            "/dev//video0",
            "/dev/snd/../video0",
            "/dev/char/81:0",
            "/dev/char/81:12",
            "/dev/media0",
            "/dev/v4l/by-id/cam",
            "sys/class/leds/ir_led",
            "pipewire://camera",
        )
        for forbidden in forbidden_paths:
            for g_arg in (None, gate):
                with self.assertRaises(SiliSensePermissionError) as ctx:
                    validate_endpoint_path(forbidden, auth_gate=g_arg)
                self.assertEqual(ctx.exception.code, "ERR_CAMERA_FORBIDDEN")
                self.assertEqual(ctx.exception.exit_code, HardwareExitCode.PREFLIGHT_FAIL)

        # Symlink pointing to /dev/video0 MUST fail closed with ERR_CAMERA_FORBIDDEN
        with tempfile.TemporaryDirectory() as td:
            cam_link = Path(td) / "innocent_sensor_link"
            os.symlink("/dev/video0", cam_link)
            with self.assertRaises(SiliSensePermissionError) as ctx_sym:
                validate_endpoint_path(cam_link, auth_gate=gate)
            self.assertEqual(ctx_sym.exception.code, "ERR_CAMERA_FORBIDDEN")

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

    def test_02_symlinked_sysfs_rapl_discovery_escape_refusal_and_wraparound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            class_powercap = base / "sys_class_powercap"
            virtual_powercap = base / "sys_devices_virtual_powercap"
            outside_dir = base / "outside_escape"
            class_powercap.mkdir()
            virtual_powercap.mkdir()
            outside_dir.mkdir()

            # Simulate real Linux sysfs layout: /sys/class/powercap/intel-rapl:0 -> virtual/powercap/intel-rapl:0
            # with nested child domain intel-rapl:0:1 (uncore)
            real_pkg = virtual_powercap / "intel-rapl:0"
            real_pkg.mkdir()
            (real_pkg / "name").write_text("package-0\n", encoding="utf-8")
            (real_pkg / "max_energy_range_uj").write_text("1000000\n", encoding="utf-8")
            (real_pkg / "energy_uj").write_text("950000\n", encoding="utf-8")

            real_unc = real_pkg / "intel-rapl:0:1"
            real_unc.mkdir()
            (real_unc / "name").write_text("uncore\n", encoding="utf-8")
            (real_unc / "max_energy_range_uj").write_text("1000000\n", encoding="utf-8")
            (real_unc / "energy_uj").write_text("200000\n", encoding="utf-8")

            os.symlink(real_pkg, class_powercap / "intel-rapl:0")

            t_box = [10.0]
            reader = RaplSysfsReader(
                powercap_root=class_powercap,
                allowed_real_roots=(class_powercap, virtual_powercap),
            )
            self.assertEqual(reader.discover(), ["pkg_w", "uncore_w"])
            w0 = reader.sample_watts(clock_fn=lambda: t_box[0])
            self.assertEqual(w0["pkg_w"], 0.0)

            (real_pkg / "energy_uj").write_text("50000\n", encoding="utf-8")
            (real_unc / "energy_uj").write_text("220000\n", encoding="utf-8")
            t_box[0] = 10.01
            w1 = reader.sample_watts(clock_fn=lambda: t_box[0])
            self.assertAlmostEqual(w1["pkg_w"], 10.0, places=5)
            self.assertAlmostEqual(w1["uncore_w"], 2.0, places=5)

            # Escape symlink outside allowed_real_roots MUST fail closed with ERR_SYMLINK_ESCAPE
            os.symlink(outside_dir, class_powercap / "intel-rapl:9")
            escape_reader = RaplSysfsReader(
                powercap_root=class_powercap,
                allowed_real_roots=(class_powercap, virtual_powercap),
            )
            with self.assertRaises(SiliSensePermissionError) as ctx_esc:
                escape_reader.discover()
            self.assertEqual(ctx_esc.exception.code, "ERR_SYMLINK_ESCAPE")

            proc_f = base / "proc_stat"
            proc_f.write_text("cpu  100 0 50 1000\nintr 98765 1 2\nctxt 543210\n", encoding="utf-8")
            cpu_dir = base / "cpu0" / "cpufreq"
            cpu_dir.mkdir(parents=True)
            (cpu_dir / "scaling_cur_freq").write_text("4200000\n", encoding="utf-8")
            irq_info = query_proc_irq_and_cpu_khz(proc_stat_path=proc_f, cpufreq_glob_root=base)
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

        with self.assertRaises(SiliSenseValueError) as ctx_flt:
            canonicalize_zone_canon_v0({"bad_float": 1.25})
        self.assertEqual(ctx_flt.exception.code, "ERR_ZONE_CANON_FLOAT_FORBIDDEN")

        with self.assertRaises(SiliSenseValueError) as ctx_del:
            canonicalize_zone_canon_v0({"bad_del": "abc\x7fdef"})
        self.assertEqual(ctx_del.exception.code, "ERR_ZONE_CANON_CHAR")

        h_empty = hashlib.sha256(b"").hexdigest()
        d1 = domain_tree_leaf_sha256("a/b.json", h_empty)
        d2 = domain_tree_leaf_sha256("a/b.jso", h_empty)
        self.assertNotEqual(d1, d2)


if __name__ == "__main__":
    unittest.main()
