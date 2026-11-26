"""R5 tests for README-vs-receipt pin verification, exact-Q rounding, and uncontrolled-laptop disclaimers."""

from __future__ import annotations

import contextlib
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from silisense import readme_pins as rp
    from silisense.errors import SiliSenseError
except SystemExit as exc:
    raise RuntimeError(f"readme_pins exited on import: {exc.code}") from exc


def _write_mock_silikernels_receipts(root: Path, *, overrides: dict | None = None) -> None:
    """Create a minimal synthetic companion receipt tree matching b3712675 schema."""
    ov = overrides or {}
    v2_zb_sha = ov.get("v2_zb_sha", rp.PINNED_ZONE_B_FULL_SHA256["v2"])
    v3_zb_sha = ov.get("v3_zb_sha", rp.PINNED_ZONE_B_FULL_SHA256["v3"])

    docs = {
        rp.RECEIPT_RELPATHS["v1_results"]: ov.get(
            "v1_results",
            {"bare_metal_rtx3080": {"uniform_cell_16warp_d512": {"t8192_us_per_tok": 249.3671}}},
        ),
        rp.RECEIPT_RELPATHS["v2_results"]: ov.get(
            "v2_results",
            {
                "cellop_only_12L_d512": {"t8192_us_per_tok": 26.2082},
                "full_e2e_fwht256_12L_d512": {
                    "t8192_us_per_tok": 27.4079,
                    "speedup_vs_v1_249_32us": 9.0967,
                },
                "mlos_zone_b_receipt_sha256": v2_zb_sha,
            },
        ),
        rp.RECEIPT_RELPATHS["v3_results"]: ov.get(
            "v3_results",
            {
                "v3_d512": {
                    "cellop_graph_us_per_tok_8192": 20.5038,
                    "e2e_graph_us_per_tok_16384": 21.5184,
                },
                "mlos_zone_b_receipt_sha256": v3_zb_sha,
            },
        ),
        rp.RECEIPT_RELPATHS["v2_zone_b"]: ov.get("v2_zone_b", {"receipt_sha256": v2_zb_sha}),
        rp.RECEIPT_RELPATHS["v3_zone_b"]: ov.get("v3_zone_b", {"receipt_sha256": v3_zb_sha}),
    }
    for rel, doc in docs.items():
        dst = root.joinpath(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TestReadmeReceiptPinsR5(unittest.TestCase):
    def assertErrCode(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(SiliSenseError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_pure_q_decimal_and_rounding_primitives(self) -> None:
        self.assertEqual(rp.decimal_str_to_q("249.3671"), Fraction(2493671, 10000))
        self.assertEqual(rp.decimal_str_to_q("9"), Fraction(9, 1))
        self.assertEqual(rp.decimal_str_to_q("9" * rp.MAX_DEC_DIGITS), Fraction(int("9" * rp.MAX_DEC_DIGITS), 1))
        for bad in (123, "", "-1.5", "1.2.3", "abc", "1" * (rp.MAX_DEC_DIGITS + 1), "1" * 4400):
            self.assertErrCode("ERR_BAD_DECIMAL", rp.decimal_str_to_q, bad)

        self.assertEqual(rp.round_half_up_q(Fraction(2493671, 10000), 2), "249.37")
        self.assertEqual(rp.round_half_up_q(Fraction(205038, 10000), 2), "20.50")
        self.assertEqual(rp.round_half_up_q(Fraction(2493671, 274079), 3), "9.098")
        self.assertEqual(rp.round_half_up_q(Fraction(2493200, 274079), 3), "9.097")
        self.assertEqual(rp.round_half_up_q(Fraction(5, 2), 0), "3")
        self.assertEqual(rp.round_half_up_q(Fraction(0, 1), 6), "0.000000")

        self.assertErrCode("ERR_BAD_Q", rp.round_half_up_q, 1, 2)  # type: ignore[arg-type]
        self.assertErrCode("ERR_BAD_Q", rp.round_half_up_q, Fraction(-1, 2), 2)
        self.assertErrCode("ERR_BAD_PLACES", rp.round_half_up_q, Fraction(1, 2), -1)
        self.assertErrCode("ERR_BAD_PLACES", rp.round_half_up_q, Fraction(1, 2), 7)
        self.assertErrCode("ERR_BAD_PLACES", rp.round_half_up_q, Fraction(1, 2), True)

        self.assertEqual(rp.q_div(Fraction(3, 4), Fraction(2, 5)), Fraction(15, 8))
        self.assertErrCode("ERR_BAD_Q", rp.q_div, 1, Fraction(1, 2))  # type: ignore[arg-type]
        self.assertErrCode("ERR_BAD_Q", rp.q_div, Fraction(1, 2), 1)  # type: ignore[arg-type]
        self.assertErrCode("ERR_ZERO_DIV", rp.q_div, Fraction(1, 2), Fraction(0, 1))

        self.assertEqual(rp._as_decimal_str(0, "x"), "0")
        self.assertEqual(rp._as_decimal_str(42, "x"), "42")
        self.assertErrCode("ERR_RECEIPT_FIELD", rp._as_decimal_str, -1, "x")
        self.assertEqual(rp._as_decimal_str(int("9" * rp.MAX_DEC_DIGITS), "x"), "9" * rp.MAX_DEC_DIGITS)
        self.assertErrCode("ERR_RECEIPT_FIELD", rp._as_decimal_str, int("9" * (rp.MAX_DEC_DIGITS + 1)), "x")
        self.assertEqual(rp._as_decimal_str("27.4079", "x"), "27.4079")
        self.assertEqual(rp._as_decimal_str("9" * rp.MAX_DEC_DIGITS, "x"), "9" * rp.MAX_DEC_DIGITS)
        self.assertErrCode("ERR_RECEIPT_FIELD", rp._as_decimal_str, "9" * (rp.MAX_DEC_DIGITS + 1), "x")
        self.assertErrCode("ERR_RECEIPT_FIELD", rp._as_decimal_str, None, "x")
        self.assertErrCode("ERR_RECEIPT_FIELD", rp._as_decimal_str, "bad", "x")
        self.assertErrCode("ERR_RECEIPT_SHA", rp._require_hex64, 123, "x")
        self.assertErrCode("ERR_RECEIPT_SHA", rp._require_hex64, "z" * 64, "x")
        self.assertEqual(rp._parse_bounded_int("9" * rp.MAX_DEC_DIGITS), int("9" * rp.MAX_DEC_DIGITS))
        self.assertEqual(rp._parse_bounded_int("-" + "9" * rp.MAX_DEC_DIGITS), -int("9" * rp.MAX_DEC_DIGITS))
        self.assertErrCode("ERR_BAD_DECIMAL", rp._parse_bounded_int, "9" * (rp.MAX_DEC_DIGITS + 1))
        self.assertErrCode("ERR_BAD_DECIMAL", rp._parse_bounded_int, "-" + "9" * (rp.MAX_DEC_DIGITS + 1))

    def test_02_live_readme_anchored_mutations_and_energy_policy(self) -> None:
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        report = rp.verify_readme_pins(readme_text)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pins_checked"], 8)
        self.assertEqual(report["v2_speedup_vs_pinned_v1_3dp"], "9.098")
        self.assertEqual(report["v2_speedup_vs_249_32_3dp"], "9.097")
        self.assertFalse(report["silikernels_receipts_read"])

        self.assertErrCode("ERR_BAD_README", rp.verify_readme_pins, 123)  # type: ignore[arg-type]
        self.assertErrCode("ERR_BAD_README", rp.verify_readme_pins, "")
        self.assertErrCode("ERR_BAD_PINS", rp.verify_readme_pins, readme_text, pins=[])  # type: ignore[arg-type]
        self.assertErrCode("ERR_BAD_PINS", rp.verify_readme_pins, readme_text, pins={"extra": "1"})

        for bad_pfx_key in ("v2_zone_b_receipt_sha256_prefix", "v3_zone_b_receipt_sha256_prefix"):
            bad_pins = dict(rp.PINNED_RECEIPT_VALUES)
            bad_pins[bad_pfx_key] = "ZZZZZZZZ"
            self.assertErrCode("ERR_BAD_SHA_PREFIX", rp.verify_readme_pins, readme_text, pins=bad_pins)
            bad_pins[bad_pfx_key] = 12345678  # type: ignore[assignment]
            self.assertErrCode("ERR_BAD_SHA_PREFIX", rp.verify_readme_pins, readme_text, pins=bad_pins)

        bad_sp_pins = dict(rp.PINNED_RECEIPT_VALUES)
        bad_sp_pins["v2_speedup_vs_v1_249_32us"] = "9.5000"
        self.assertErrCode("ERR_SPEEDUP_MATH", rp.verify_readme_pins, readme_text, pins=bad_sp_pins)

        # Exact boundary test: |249.32 / 24.932 - 10.0001| == 1/10000 MUST pass (kills <= -> < mutant!)
        bnd_pins = dict(rp.PINNED_RECEIPT_VALUES)
        bnd_pins["v2_e2e_t8192_us_per_tok"] = "24.932"
        bnd_pins["v2_speedup_vs_v1_249_32us"] = "10.0001"
        bnd_readme = (
            readme_text.replace("27.4079", "24.932")
            .replace("27.41 us/tok", "24.93 us/tok")
            .replace("9.0967", "10.0001")
            .replace("9.098×", "10.002×")
        )
        self.assertEqual(rp.verify_readme_pins(bnd_readme, pins=bnd_pins)["status"], "PASS")

        # Stale commit, missing commit, and uncontrolled laptop energy/watt claims MUST fail closed
        self.assertErrCode("ERR_STALE_SILIKERNELS_PIN", rp.verify_readme_pins, readme_text + "\n44eccad8\n")
        self.assertErrCode(
            "ERR_MISSING_COMMIT_PIN",
            rp.verify_readme_pins,
            readme_text.replace(rp.PINNED_SILIKERNELS_COMMIT, "0" * 40),
        )
        for bad_energy in (
            "11.16 mJ/tok",
            "0.1901 mJ·ms/tok",
            "90 W Cap",
            "141 W CPU+GPU",
            "90 watts",
            "95 W",
            "10 mW",
            "mW",
            "1.2 J/tok",
            "J/tok",
            "50 millijoules",
            "5 MJ",
            "5 J",
            "5 kWh",
            "5e3 mJ",
            "5 μJ",
            "100 W/ token",
            "wattage: 45",
            "5.97×",
            "2.2275",
        ):
            self.assertErrCode("ERR_README_ENERGY_CLAIM", rp.verify_readme_pins, readme_text + f"\n{bad_energy}\n")

        # Linear-time ReDoS check on README energy regex
        t0 = time.monotonic()
        self.assertIsNone(rp._FORBIDDEN_README_ENERGY_RE.search("1" * 8000))
        self.assertIsNone(rp._FORBIDDEN_README_ENERGY_RE.search("1." * 3000))
        self.assertLess(time.monotonic() - t0, 0.25)

        # Anchored digit append/prepend, HTML comment decoy, & V2/V3 cross-section swap MUST ALL fail closed!
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, readme_text.replace("27.4079", "27.4080"))
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, readme_text.replace("= 27.4079", "= 27.40791"))
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, readme_text.replace("249.37 us/tok", "1249.37 us/tok"))
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, readme_text.replace("9.098×", "19.098×"))
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, readme_text.replace("699c3d05", "00000000"))

        swapped = (
            readme_text.replace("26.21 us/tok", "__TMP_SWAP__")
            .replace("20.50 us/tok", "26.21 us/tok")
            .replace("__TMP_SWAP__", "20.50 us/tok")
        )
        self.assertErrCode("ERR_README_PIN_MISMATCH", rp.verify_readme_pins, swapped)
        self.assertErrCode(
            "ERR_README_SECTION",
            rp.verify_readme_pins,
            readme_text.replace("*V1 Inline SECDED Kernel", "*V1 Removed"),
        )
        self.assertErrCode(
            "ERR_README_SECTION",
            rp.verify_readme_pins,
            readme_text + "\n  - *V1 Inline SECDED Kernel decoy\n",
        )
        self.assertErrCode(
            "ERR_README_SECTION",
            rp.verify_readme_pins,
            readme_text.replace("*V1 Inline SECDED Kernel", "<!-- *V1 Inline SECDED Kernel"),
        )
        self.assertErrCode(
            "ERR_README_SECTION",
            rp.verify_readme_pins,
            readme_text.replace("*V1 Inline SECDED Kernel", "*V1 Inline SECDED Kernel -->"),
        )

        # Dropping ANY required disclaimer MUST fail closed
        for phrase in (
            "Energy figures in those receipts are uncontrolled and provisional.",
            "Uncontrolled Laptop & Timing Disclaimer",
            "first committed after the live laptop run ended",
            "paired contrast family uncorrected",
        ):
            self.assertErrCode(
                "ERR_MISSING_UNCONTROLLED_DISCLAIMER",
                rp.verify_readme_pins,
                readme_text.replace(phrase, "removed"),
            )

    def test_03_companion_receipt_loading_symlinks_git_and_cli(self) -> None:
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertErrCode("ERR_BAD_PATH", rp._load_receipt_json_exact, str(root))  # type: ignore[arg-type]
            self.assertErrCode("ERR_MISSING_RECEIPT", rp._load_receipt_json_exact, root / "missing.json")
            self.assertErrCode("ERR_BAD_ROOT", rp.load_silikernels_receipt_pins, str(root))  # type: ignore[arg-type]
            self.assertErrCode("ERR_MISSING_RECEIPT", rp.load_silikernels_receipt_pins, root / "no_such_dir")

            sub = root / "companion"
            sub.mkdir()
            _write_mock_silikernels_receipts(sub)
            loaded = rp.load_silikernels_receipt_pins(sub)
            self.assertEqual(loaded, rp.PINNED_RECEIPT_VALUES)
            rep = rp.verify_readme_pins(readme_text, silikernels_root=sub)
            self.assertTrue(rep["silikernels_receipts_read"])

            # Single missing receipt file fails closed
            (sub / rp.RECEIPT_RELPATHS["v3_zone_b"]).unlink()
            self.assertErrCode("ERR_MISSING_RECEIPT", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(sub)

            # Non-decimal receipt field & bad receipt_sha256 inside companion receipt files fail closed
            _write_mock_silikernels_receipts(
                sub,
                overrides={"v1_results": {"bare_metal_rtx3080": {"uniform_cell_16warp_d512": {"t8192_us_per_tok": "bad"}}}},
            )
            self.assertErrCode("ERR_RECEIPT_FIELD", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(sub, overrides={"v2_zone_b": {"receipt_sha256": "not-hex"}})
            self.assertErrCode("ERR_RECEIPT_SHA", rp.load_silikernels_receipt_pins, sub)

            # Planted receipt drift MUST fail closed
            _write_mock_silikernels_receipts(
                sub,
                overrides={"v1_results": {"bare_metal_rtx3080": {"uniform_cell_16warp_d512": {"t8192_us_per_tok": 249.3200}}}},
            )
            self.assertErrCode("ERR_RECEIPT_DRIFT", rp.verify_readme_pins, readme_text, silikernels_root=sub)

            # Cross-file Zone-B SHA mismatch (both v2 and v3, plus forged matching non-pinned full SHA!)
            _write_mock_silikernels_receipts(
                sub,
                overrides={"v2_zb_sha": "699c3d05" + "0" * 56},
            )
            self.assertErrCode("ERR_RECEIPT_CROSS_SHA", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(
                sub,
                overrides={"v3_zb_sha": "feecc748" + "0" * 56},
            )
            self.assertErrCode("ERR_RECEIPT_CROSS_SHA", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(
                sub,
                overrides={
                    "v2_results": {
                        "cellop_only_12L_d512": {"t8192_us_per_tok": 26.2082},
                        "full_e2e_fwht256_12L_d512": {
                            "t8192_us_per_tok": 27.4079,
                            "speedup_vs_v1_249_32us": 9.0967,
                        },
                        "mlos_zone_b_receipt_sha256": "0" * 64,
                    }
                },
            )
            self.assertErrCode("ERR_RECEIPT_CROSS_SHA", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(
                sub,
                overrides={
                    "v3_results": {
                        "v3_d512": {
                            "cellop_graph_us_per_tok_8192": 20.5038,
                            "e2e_graph_us_per_tok_16384": 21.5184,
                        },
                        "mlos_zone_b_receipt_sha256": "0" * 64,
                    }
                },
            )
            self.assertErrCode("ERR_RECEIPT_CROSS_SHA", rp.load_silikernels_receipt_pins, sub)

            # Missing section shape error (including nested bare_metal_rtx3080 and each v2/v3 sub-dict individually)
            for bad_key in ("v1_results", "v2_results", "v3_results"):
                _write_mock_silikernels_receipts(sub, overrides={bad_key: {}})
                self.assertErrCode("ERR_RECEIPT_SHAPE", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(sub, overrides={"v1_results": {"bare_metal_rtx3080": {}}})
            self.assertErrCode("ERR_RECEIPT_SHAPE", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(
                sub, overrides={"v2_results": {"cellop_only_12L_d512": {}, "full_e2e_fwht256_12L_d512": []}}
            )
            self.assertErrCode("ERR_RECEIPT_SHAPE", rp.load_silikernels_receipt_pins, sub)
            _write_mock_silikernels_receipts(
                sub, overrides={"v2_results": {"cellop_only_12L_d512": [], "full_e2e_fwht256_12L_d512": {}}}
            )
            self.assertErrCode("ERR_RECEIPT_SHAPE", rp.load_silikernels_receipt_pins, sub)

            # Symlink refusal on: README, receipt leaf, receipts/ parent dir, and silikernels_root!
            _write_mock_silikernels_receipts(sub)
            sym_root = root / "sym_companion"
            sym_root.symlink_to(sub)
            self.assertErrCode("ERR_SYMLINK_FORBIDDEN", rp.load_silikernels_receipt_pins, sym_root)

            sym_parent = root / "parent_sym_companion"
            sym_parent.mkdir()
            (sym_parent / "receipts").symlink_to(sub / "receipts")
            self.assertErrCode("ERR_SYMLINK_FORBIDDEN", rp.load_silikernels_receipt_pins, sym_parent)

            sym_leaf_root = root / "leaf_sym_companion"
            sym_leaf_root.mkdir()
            _write_mock_silikernels_receipts(sym_leaf_root)
            v1_leaf = sym_leaf_root / rp.RECEIPT_RELPATHS["v1_results"]
            v1_leaf.unlink()
            v1_leaf.symlink_to(sub / rp.RECEIPT_RELPATHS["v1_results"])
            self.assertErrCode("ERR_SYMLINK_FORBIDDEN", rp.load_silikernels_receipt_pins, sym_leaf_root)

            # Non-directory root, directory-as-leaf, and named FIFO fail closed BEFORE os.open()!
            not_dir = root / "plain_file.txt"
            not_dir.write_text("x", encoding="utf-8")
            self.assertErrCode("ERR_NOT_REGULAR_FILE", rp.load_silikernels_receipt_pins, not_dir)
            dir_leaf_root = root / "dir_leaf_companion"
            dir_leaf_root.mkdir()
            _write_mock_silikernels_receipts(dir_leaf_root)
            dl = dir_leaf_root / rp.RECEIPT_RELPATHS["v1_results"]
            dl.unlink()
            dl.mkdir()
            real_open = os.open
            leaf_open_calls: list[str] = []

            def _track_leaf_open(p, flags, *a, **kw):
                if "s117_live" in str(p):
                    leaf_open_calls.append(str(p))
                return real_open(p, flags, *a, **kw)

            with mock.patch.object(rp.os, "open", side_effect=_track_leaf_open):
                self.assertErrCode("ERR_NOT_REGULAR_FILE", rp.load_silikernels_receipt_pins, dir_leaf_root)
            self.assertEqual(leaf_open_calls, [])

            fifo_leaf_root = root / "fifo_leaf_companion"
            fifo_leaf_root.mkdir()
            _write_mock_silikernels_receipts(fifo_leaf_root)
            fl = fifo_leaf_root / rp.RECEIPT_RELPATHS["v1_results"]
            fl.unlink()
            os.mkfifo(fl)
            with mock.patch.object(rp.os, "open", side_effect=_track_leaf_open):
                self.assertErrCode("ERR_NOT_REGULAR_FILE", rp.load_silikernels_receipt_pins, fifo_leaf_root)
            self.assertEqual(leaf_open_calls, [])

            # PermissionError on os.open (root dir & receipt leaf) cleanly wrapped as ERR_MISSING_RECEIPT
            with mock.patch.object(rp.os, "open", side_effect=PermissionError("denied root")):
                self.assertErrCode("ERR_MISSING_RECEIPT", rp.load_silikernels_receipt_pins, sub)

            def _deny_leaf_open(p, flags, *a, **kw):
                if "s117_live" in str(p):
                    raise PermissionError("denied leaf")
                return real_open(p, flags, *a, **kw)

            with mock.patch.object(rp.os, "open", side_effect=_deny_leaf_open):
                self.assertErrCode("ERR_MISSING_RECEIPT", rp.load_silikernels_receipt_pins, sub)

            # Git commit ancestry, diff, symlink-.git, GIT_* env strip, and OSError/Timeout guards
            sym_git_root = root / "sym_git_companion"
            sym_git_root.mkdir()
            (sym_git_root / ".git").symlink_to(sub)
            self.assertErrCode("ERR_SYMLINK_FORBIDDEN", rp._verify_git_pin_if_repo, sym_git_root)

            git_dir = sub / ".git"
            git_dir.mkdir()
            fake_ok = subprocess.CompletedProcess([], 0, "", "")
            fake_bad = subprocess.CompletedProcess([], 1, "", "")
            seen_envs: list[dict[str, str]] = []

            def _capture_git(*a, **kw):
                seen_envs.append(dict(kw.get("env", {})))
                return fake_ok

            with mock.patch.dict(os.environ, {"GIT_DIR": "/bogus/git_dir"}, clear=False):
                with mock.patch.object(rp.subprocess, "run", side_effect=_capture_git):
                    self.assertTrue(rp._verify_git_pin_if_repo(sub))
            self.assertEqual(len(seen_envs), 2)
            self.assertNotIn("GIT_DIR", seen_envs[0])
            with mock.patch.object(rp.subprocess, "run", side_effect=[fake_bad]):
                self.assertErrCode("ERR_SILIKERNELS_COMMIT", rp._verify_git_pin_if_repo, sub)
            with mock.patch.object(rp.subprocess, "run", side_effect=[fake_ok, fake_bad]):
                self.assertErrCode("ERR_SILIKERNELS_RECEIPT_MODIFIED", rp._verify_git_pin_if_repo, sub)
            with mock.patch.object(rp.subprocess, "run", side_effect=OSError("no git")):
                self.assertErrCode("ERR_SILIKERNELS_GIT", rp._verify_git_pin_if_repo, sub)
            with mock.patch.object(rp.subprocess, "run", side_effect=subprocess.TimeoutExpired(["git"], 5)):
                self.assertErrCode("ERR_SILIKERNELS_GIT", rp._verify_git_pin_if_repo, sub)
            git_dir.rmdir()

            # Duplicate JSON key, non-finite JSON, 4400-digit int, deep recursion, invalid UTF-8, symlink, dir, FIFO, oversized, OSError & non-dict
            dup_p = sub / rp.RECEIPT_RELPATHS["v1_results"]
            dup_p.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            self.assertErrCode("ERR_DUP_JSON_KEY", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text('{"a": NaN}\n', encoding="utf-8")
            self.assertErrCode("ERR_NON_FINITE_JSON", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text('{"a": ' + ("9" * 4400) + '}\n', encoding="utf-8")
            self.assertErrCode("ERR_BAD_DECIMAL", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text(('{"a":' * rp.MAX_JSON_DEPTH) + "{}" + ("}" * rp.MAX_JSON_DEPTH) + "\n", encoding="utf-8")
            self.assertIsInstance(rp._load_receipt_json_exact(dup_p), dict)
            dup_p.write_text(('{"a":' * (rp.MAX_JSON_DEPTH + 1)) + "{}" + ("}" * (rp.MAX_JSON_DEPTH + 1)) + "\n", encoding="utf-8")
            self.assertErrCode("ERR_RECEIPT_DEPTH", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text('{"a":' + ("[" * rp.MAX_JSON_DEPTH) + "{}" + ("]" * rp.MAX_JSON_DEPTH) + "}\n", encoding="utf-8")
            self.assertErrCode("ERR_RECEIPT_DEPTH", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text('{"a":' + ("[" * (rp.MAX_JSON_DEPTH + 1)) + "1" + ("]" * (rp.MAX_JSON_DEPTH + 1)) + "}\n", encoding="utf-8")
            self.assertErrCode("ERR_RECEIPT_DEPTH", rp._load_receipt_json_exact, dup_p)
            with mock.patch.object(rp.json, "loads", side_effect=RecursionError("deep")):
                self.assertErrCode("ERR_RECEIPT_PARSE", rp._load_receipt_json_exact, dup_p)
            dup_p.write_bytes(b"\xff\xfe")
            self.assertErrCode("ERR_RECEIPT_PARSE", rp._load_receipt_json_exact, dup_p)
            dup_p.write_text("[]\n", encoding="utf-8")
            self.assertErrCode("ERR_RECEIPT_TYPE", rp._load_receipt_json_exact, dup_p)

            sym_r = root / "sym_readme.md"
            sym_r.symlink_to(REPO_ROOT / "README.md")
            err_sym = io.StringIO()
            with contextlib.redirect_stderr(err_sym):
                self.assertEqual(rp.main(["--readme", str(sym_r)]), 2)
            self.assertEqual(json.loads(err_sym.getvalue())["code"], "ERR_SYMLINK_FORBIDDEN")

            bad_utf8 = root / "bad_readme.md"
            bad_utf8.write_bytes(b"\xff\xfe")
            err_u = io.StringIO()
            with contextlib.redirect_stderr(err_u):
                self.assertEqual(rp.main(["--readme", str(bad_utf8)]), 2)
            self.assertEqual(json.loads(err_u.getvalue())["code"], "ERR_BAD_README")

            # Pre-open S_ISREG blocks dir & FIFO BEFORE os.open(), and PermissionError on os.open is wrapped cleanly!
            with mock.patch.object(rp.os, "open", side_effect=AssertionError("MUST_NOT_OPEN")):
                self.assertErrCode("ERR_NOT_REGULAR_FILE", rp._load_receipt_json_exact, root)
            fifo_readme = root / "fifo_readme.md"
            os.mkfifo(fifo_readme)
            with mock.patch.object(rp.os, "open", side_effect=AssertionError("MUST_NOT_OPEN")):
                self.assertErrCode("ERR_NOT_REGULAR_FILE", rp._read_regular_file_bytes_nofollow, fifo_readme, missing_code="ERR_MISSING_README")

            big_p = root / "big.json"
            big_p.write_bytes(b"a" * (rp.MAX_FILE_BYTES + 1))
            self.assertErrCode("ERR_FILE_TOO_LARGE", rp._load_receipt_json_exact, big_p)
            exact_p = root / "exact.json"
            exact_p.write_bytes(b" " * (rp.MAX_FILE_BYTES - 2) + b"{}")
            self.assertEqual(rp._load_receipt_json_exact(exact_p), {})
            with mock.patch.object(rp.os, "open", side_effect=PermissionError("denied file")):
                self.assertErrCode("ERR_MISSING_RECEIPT", rp._load_receipt_json_exact, exact_p)

            # OSError during fstat/read and non-regular fstat are wrapped cleanly in SiliSenseError (RC=2)
            with mock.patch.object(rp.os, "fstat", side_effect=OSError("io fail")):
                self.assertErrCode("ERR_MISSING_RECEIPT", rp._load_receipt_json_exact, exact_p)
            fake_dir_st = os.lstat(str(root))
            with mock.patch.object(rp.os, "fstat", return_value=fake_dir_st):
                self.assertErrCode("ERR_NOT_REGULAR_FILE", rp._load_receipt_json_exact, exact_p)

            # CLI tests (normal, --json, missing README, and subprocess __main__ guard)
            _write_mock_silikernels_receipts(sub)
            out1 = io.StringIO()
            with contextlib.redirect_stdout(out1):
                self.assertEqual(rp.main(["--silikernels-root", str(sub)]), 0)
            self.assertTrue(out1.getvalue().startswith("PASS pins=8 summary_sha256="))

            out2 = io.StringIO()
            with contextlib.redirect_stdout(out2):
                self.assertEqual(rp.main(["--json", "--silikernels-root", str(sub)]), 0)
            self.assertEqual(json.loads(out2.getvalue())["status"], "PASS")

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(rp.main(["--readme", str(root / "missing_readme.md")]), 2)
            self.assertEqual(json.loads(err.getvalue())["code"], "ERR_MISSING_README")

            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(REPO_ROOT / "silisense" / "readme_pins.py")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(proc.stdout.startswith("PASS pins=8 summary_sha256="), proc.stdout)


if __name__ == "__main__":
    unittest.main()
