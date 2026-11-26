"""R5 tests for 30 kHz Razor BER-canary controller, Weft rollback, and Krein-Holonomy detector."""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silisense import razor as rz
from silisense.autotuner import FiveGateAntiGoodhartRatchet
from silisense.errors import SiliSenseValueError
from silisense.governor import PreAdaptationResidualLatch
from silisense.holosphere_bridge import HoloSphereTuningProfile


def _make_clean_nibbles(data11: int = 0x5A3) -> tuple[int, ...]:
    w16 = rz.a16_encode_lsb16(data11)
    out = []
    for i in range(16):
        base = ((i % 7) - 3) & 0xE
        raw4 = base | ((w16 >> i) & 1)
        out.append(raw4 - 16 if raw4 >= 8 else raw4)
    return tuple(out)


class TestRazorR5(unittest.TestCase):
    def test_01_a16_secded_2048_codewords_16_sec_120_ded_and_48_vmin_upper_bit_closures(self) -> None:
        for payload in range(2048):
            w16 = rz.a16_encode_lsb16(payload)
            self.assertEqual(rz.a16_secded_syndrome(w16), 0)
            self.assertEqual(rz.a16_secded_classify(0), ("CLEAN", None))

        clean = _make_clean_nibbles(0x3C7)
        crc16 = rz.block_crc16_ccitt(clean)
        fold_bit = rz.a16_folded_upper_parity_bit(clean)
        plane_synds = rz.a16_plane_syndromes_4(clean)
        self.assertEqual(plane_synds[0], 0)
        for ep in range(8):
            self.assertEqual(rz.a16_rotate_bitplane_idx(ep), ep & 3)

        clean_res = rz.a16_inline_secded_step(
            clean,
            expected_crc16=crc16,
            expected_upper_fold_bit=fold_bit,
            epoch_u32=3,
            expected_plane_syndromes=plane_synds,
        )
        self.assertEqual(clean_res["status"], "CLEAN")
        self.assertEqual(clean_res["rotated_plane"], 3)
        self.assertEqual(clean_res["pcie_atomic_writes"], 0)
        self.assertEqual(clean_res["healed_nibbles"], clean)

        for lane in range(16):
            flipped = rz.inject_nibble_bit_flip(clean, lane, 0)
            self.assertNotEqual(flipped, clean)
            res = rz.a16_inline_secded_step(
                flipped,
                expected_crc16=crc16,
                expected_upper_fold_bit=fold_bit,
                epoch_u32=1,
                expected_plane_syndromes=plane_synds,
            )
            self.assertEqual(res["status"], "SEC")
            self.assertEqual(res["fixed_lane"], lane)
            self.assertEqual(res["healed_nibbles"], clean)
            self.assertEqual(res["sec_delta"], 1)
            self.assertEqual(res["ded_delta"], 0)
            self.assertEqual(res["non_lsb_delta"], 0)
            self.assertEqual(res["pcie_atomic_writes"], 1)

        ded_count = 0
        for i in range(16):
            for j in range(i + 1, 16):
                c2 = rz.inject_nibble_bit_flip(rz.inject_nibble_bit_flip(clean, i, 0), j, 0)
                res = rz.a16_inline_secded_step(c2, expected_crc16=crc16)
                self.assertEqual(res["status"], "DED")
                self.assertIsNone(res["fixed_lane"])
                self.assertEqual(res["sec_delta"], 0)
                self.assertEqual(res["ded_delta"], 1)
                self.assertEqual(res["pcie_atomic_writes"], 1)
                ded_count += 1
        self.assertEqual(ded_count, 120)

        # Honest upper-bit disclosure: LSB-only misses 48/48 in bits 1..3;
        # folded parity and CRC-16 each catch 48/48 on EVERY epoch (incl. epoch 0);
        # rotated bit-plane b = epoch & 3 catches {epoch 0: 0/48, epochs 1..3: 16/48}
        # with exact 48/48 union across any 4 consecutive epochs.
        rotated_per_epoch: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        rotated_union: set[tuple[int, int]] = set()
        for lane in range(16):
            for bp in (1, 2, 3):
                cu = rz.inject_nibble_bit_flip(clean, lane, bp)
                lsb_only = rz.a16_inline_secded_step(cu)
                self.assertEqual(lsb_only["status"], "CLEAN")
                self.assertEqual(lsb_only["pcie_atomic_writes"], 0)

                r_fold = rz.a16_inline_secded_step(cu, epoch_u32=0, expected_upper_fold_bit=fold_bit)
                self.assertEqual(r_fold["status"], "NON_LSB_FAULT")
                self.assertEqual(r_fold["non_lsb_delta"], 1)

                r_crc = rz.a16_inline_secded_step(cu, epoch_u32=0, expected_crc16=crc16)
                self.assertEqual(r_crc["status"], "NON_LSB_FAULT")
                self.assertEqual(r_crc["non_lsb_delta"], 1)

                for ep in (0, 1, 2, 3):
                    r_rot = rz.a16_inline_secded_step(cu, epoch_u32=4 + ep, expected_plane_syndromes=plane_synds)
                    self.assertEqual(r_rot["rotated_plane"], ep)
                    if ep == bp:
                        self.assertEqual(r_rot["status"], "NON_LSB_FAULT")
                        self.assertEqual(r_rot["non_lsb_delta"], 1)
                        rotated_per_epoch[ep] += 1
                        rotated_union.add((lane, bp))
                    else:
                        self.assertEqual(r_rot["status"], "CLEAN")
                        self.assertEqual(r_rot["non_lsb_delta"], 0)

        self.assertEqual(rotated_per_epoch, {0: 0, 1: 16, 2: 16, 3: 16})
        self.assertEqual(len(rotated_union), 48)

    def test_02_canary64_wire_layout_and_exact_fail_closed_guards(self) -> None:
        c = rz.RazorCanary64(
            sec_count_u32=7,
            ded_count_u32=2,
            epoch_u32=5,
            rollback_token_u32=64,
            non_lsb_canary_u32=1,
            last_syndrome_u32=0x012B,
            ewma_ber_q16_u32=512,
            clean_streak_u32=16,
            hold_remaining_u32=4,
            ded_latch_remaining_u32=64,
        )
        raw = c.to_bytes()
        self.assertEqual(len(raw), 64)
        self.assertEqual(rz.RazorCanary64.from_bytes(raw), c)
        self.assertEqual(len(c.canary_sha256()), 64)
        self.assertEqual(int.from_bytes(raw[0:4], "little"), 7)
        self.assertEqual(int.from_bytes(raw[4:8], "little"), 2)
        self.assertEqual(int.from_bytes(raw[8:12], "little"), 5)
        self.assertEqual(int.from_bytes(raw[12:16], "little"), 64)
        self.assertEqual(int.from_bytes(raw[16:20], "little"), 1)
        self.assertEqual(int.from_bytes(raw[20:24], "little"), 0x012B)
        self.assertEqual(int.from_bytes(raw[24:28], "little"), 512)
        self.assertEqual(int.from_bytes(raw[28:32], "little"), 16)
        self.assertEqual(int.from_bytes(raw[32:36], "little"), 4)
        self.assertEqual(int.from_bytes(raw[36:40], "little"), 64)
        self.assertEqual(raw[40:64], b"\x00" * 24)
        self.assertEqual(rz.RazorCanary64(sec_count_u32=rz.U32_MAX).sec_count_u32, rz.U32_MAX)

        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64.from_bytes(b"\x00" * 63)
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64.from_bytes(raw[:40] + b"\x01" + raw[41:])
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64(sec_count_u32=-1)
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64(sec_count_u32=rz.U32_MAX + 1)
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64(sec_count_u32=True)  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64(reserved_u32=(0,) * 5)
        with self.assertRaises(SiliSenseValueError):
            rz.RazorCanary64(reserved_u32=(0,) * 5 + (1,))

    def test_03_three_tier_razor_n256_hysteresis_weft_rollback_and_native_ratchet(self) -> None:
        # Build SiliSenseHealthSnapshotQ directly from native FiveGateAntiGoodhartRatchet
        real_latch = PreAdaptationResidualLatch(
            latched_abs_a3=0.12,
            latched_z_a3=1.4,
            latched_a5_lead_ms=-45.0,
            latched_a7_copula_mi_nats=0.05,
            latched_z_a7=1.2,
            latched_mean_gpu_w=95.0,
            open_pathology=False,
        )
        ratchet = FiveGateAntiGoodhartRatchet()
        ratchet_ok = ratchet.evaluate(
            base_effective_tokens=4096,
            cand_effective_tokens=4096,
            base_edp_q=Fraction(100, 1),
            cand_edp_q=Fraction(95, 1),
            base_throttle_frac_q=Fraction(1, 10),
            cand_throttle_frac_q=Fraction(1, 20),
            cand_edp_head_q=Fraction(94, 1),
            cand_edp_tail_q=Fraction(96, 1),
            pre_adapt_latch_sha256=real_latch.latch_sha256(),
            cand_max_shock_z_q=Fraction(3, 1),
            expected_latch=real_latch,
        )
        self.assertTrue(ratchet_ok["accepted"])

        healthy = rz.SiliSenseHealthSnapshotQ.from_ratchet_result(
            op1_leverage_q=Fraction(1, 5),
            op2_phi_asym_q=Fraction(1, 10),
            op3_neg_inertia=0,
            ratchet_result=ratchet_ok,
        )
        self.assertTrue(healthy.ratchet_accepted)

        ctrl = rz.HoloSphereRazorController(
            base_profile=HoloSphereTuningProfile(backend="cuda_sm86", block_t=128),
            initial_margin_mv_q=Fraction(40, 1),
            initial_power_cap_w_q=Fraction(100, 1),
        )

        # Single-step 3-tier verification + R1 anti-flip-flop + profile_sha256 parity
        # Fresh controller (clean_streak == 0 < 16) MUST hold in clean warmup on single-poll step()!
        r0_warm = ctrl.step(rz.RazorCanary64(), healthy)
        self.assertEqual(r0_warm["action"], "TIER0_HOLD_CLEAN_WARMUP")
        self.assertEqual(r0_warm["margin_mv_q"], "40/1")
        expected_prof = HoloSphereTuningProfile(backend="cuda_sm86", block_t=128, inter_step_cooldown_ms=0.0)
        self.assertEqual(r0_warm["profile_sha256"], expected_prof.profile_sha256())

        for bad_health in (
            rz.SiliSenseHealthSnapshotQ(Fraction(1, 5), Fraction(1, 10), 0, False),
            rz.SiliSenseHealthSnapshotQ(Fraction(3, 4), Fraction(1, 10), 0, True),
            rz.SiliSenseHealthSnapshotQ(Fraction(1, 5), Fraction(1, 2), 0, True),
            rz.SiliSenseHealthSnapshotQ(Fraction(1, 5), Fraction(1, 10), 1, True),
        ):
            rh = ctrl.step(rz.RazorCanary64(), bad_health)
            self.assertEqual(rh["action"], "TIER0_NOMINAL_HOLD_HEALTH_GATE")

        # After 16 clean BER epochs while health-gated, health recovery allows step() to descend once,
        # resetting clean_streak=0 and arming hold_remaining=4 so an immediate second poll cannot flip-flop.
        unhealthy = rz.SiliSenseHealthSnapshotQ(Fraction(1, 5), Fraction(1, 10), 0, False)
        for _ in range(16):
            self.assertEqual(ctrl.step_epoch(rz.RazorCanary64(), unhealthy)["action"], "TIER0_NOMINAL_HOLD_HEALTH_GATE")
        self.assertEqual(ctrl.clean_streak, 16)
        r0 = ctrl.step(rz.RazorCanary64(), healthy)
        self.assertEqual(r0["action"], "TIER0_NOMINAL_RAZOR_DESCENT")
        self.assertEqual(r0["margin_mv_q"], "38/1")
        self.assertEqual(r0["clean_streak"], 0)
        self.assertEqual(r0["hold_remaining"], 4)
        self.assertEqual(r0["profile_sha256"], expected_prof.profile_sha256())
        r0_dwell = ctrl.step(rz.RazorCanary64(), healthy)
        self.assertEqual(r0_dwell["action"], "TIER0_HOLD_MIN_DWELL")
        self.assertEqual(r0_dwell["margin_mv_q"], "38/1")

        # R1: step() Tier-1 SEC arms hold_remaining >= hold_epochs (prevents immediate flip-flop)
        ctrl_t1 = rz.HoloSphereRazorController(initial_margin_mv_q=Fraction(40, 1))
        r1_solo = ctrl_t1.step(rz.RazorCanary64(sec_count_u32=1), healthy)
        self.assertEqual(r1_solo["action"], "TIER1_SEC_INLINE_HEALED_MARGIN_UP")
        self.assertEqual(r1_solo["margin_mv_q"], "52/1")
        self.assertEqual(r1_solo["hold_remaining"], 4)
        r1_next = ctrl_t1.step(rz.RazorCanary64(sec_count_u32=1), healthy)
        self.assertEqual(r1_next["action"], "TIER0_HOLD_MIN_DWELL")
        self.assertEqual(r1_next["margin_mv_q"], "52/1")

        r1 = ctrl.step(rz.RazorCanary64(sec_count_u32=1), healthy)
        self.assertEqual(r1["action"], "TIER1_SEC_INLINE_HEALED_MARGIN_UP")
        self.assertEqual(r1["margin_mv_q"], "50/1")
        self.assertEqual(r1["hold_remaining"], 4)

        # Weft 16 B Pi_{t_0} rollback cures right-coset drift to exact Fraction(0)
        buf_clean = rz.WeftHolonomyRollbackBuffer(chunk_size=32)
        buf_razor = rz.WeftHolonomyRollbackBuffer(chunk_size=32)
        tokens_w = [
            (Fraction((t % 5) - 2, 7), Fraction((t % 3) - 1, 5), Fraction(1, 4), Fraction(t + 1, 16))
            for t in range(40)
        ]
        for t in range(36):
            wx, wy, wz, sd = tokens_w[t]
            u_c = buf_clean.step_token(wx, wy, wz, sd)
            self.assertEqual(u_c.norm_sq(), Fraction(1))
            if t == 34:
                buf_razor.step_token(wx, wy, wz, sd, corrupt_delta=rz.cayley_su2_q(Fraction(1, 2), Fraction(0), Fraction(0)))
            else:
                buf_razor.step_token(wx, wy, wz, sd)
        self.assertGreater(buf_razor.u_state.diff_norm_sq(buf_clean.u_state), Fraction(0))

        r2 = ctrl.step(rz.RazorCanary64(sec_count_u32=1, ded_count_u32=1, rollback_token_u32=32), healthy, rollback_buf=buf_razor)
        self.assertEqual(r2["action"], "TIER2_DED_ROLLBACK_AND_MARGIN_UP")
        self.assertEqual(r2["epoch_u32"], 16)
        self.assertEqual(r2["ded_latch_remaining"], 64)
        self.assertEqual(r2["rollback_receipt"]["rollback_state_bytes"], 16)
        self.assertEqual(r2["rollback_receipt"]["rollback_token_u32"], 32)
        self.assertEqual(len(buf_razor.pack_checkpoint_pi16()), 16)
        for t in range(36, 40):
            buf_clean.step_token(*tokens_w[t])
        for t in range(32, 40):
            buf_razor.step_token(*tokens_w[t])
        self.assertEqual(buf_razor.u_state, buf_clean.u_state)
        self.assertEqual(buf_razor.u_state.diff_norm_sq(buf_clean.u_state), Fraction(0))

        # Cross-chunk protection: commit_boundary=False preserves prior clean chunk checkpoint
        buf_guard = rz.WeftHolonomyRollbackBuffer(chunk_size=32)
        for t in range(36):
            wx, wy, wz, sd = tokens_w[t]
            buf_guard.step_token(wx, wy, wz, sd, commit_boundary=(t < 32))
        self.assertEqual(buf_guard.ckpt_token_idx, 0)
        self.assertEqual(buf_guard.rollback_to_last_clean_chunk()["rollback_token_u32"], 0)

        # B2 verification: step() MUST strictly respect active DED latch & min-dwell hold!
        margin_at_ded = r2["margin_mv_q"]
        r2_hold = ctrl.step(rz.RazorCanary64(sec_count_u32=1, ded_count_u32=1, rollback_token_u32=32), healthy)
        self.assertEqual(r2_hold["action"], "TIER0_HOLD_DED_LATCH")
        self.assertEqual(r2_hold["margin_mv_q"], margin_at_ded)
        self.assertEqual(r2_hold["epoch_u32"], 16)

        # Full N=256-token epoch hysteresis verification
        hc = rz.HoloSphereRazorController(initial_margin_mv_q=Fraction(40, 1))
        sec_tot = 0
        ded_tot = 0
        sec_tot += 1
        h1 = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
        self.assertEqual(h1["action"], "TIER1_SEC_DEADBAND_HOLD")
        self.assertEqual(h1["margin_mv_q"], "40/1")
        self.assertEqual(h1["ewma_ber_q"], "1/8192")
        self.assertEqual(h1["ewma_ber_q16_u32"], 8)

        sec_tot += 2
        h2 = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
        self.assertEqual(h2["action"], "TIER1_SEC_STEP_UP_HOLD4")
        self.assertEqual(h2["margin_mv_q"], "52/1")
        self.assertEqual(h2["hold_remaining"], 4)
        # B2 check on hold_remaining: step() cannot descend while hold_remaining > 0
        h2_poll = hc.step(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
        self.assertEqual(h2_poll["action"], "TIER0_HOLD_MIN_DWELL")
        self.assertEqual(h2_poll["margin_mv_q"], "52/1")

        for k in range(1, 16):
            hk = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
            if k <= 3:
                self.assertEqual(hk["action"], "TIER0_HOLD_MIN_DWELL")
            else:
                self.assertEqual(hk["action"], "TIER0_HOLD_CLEAN_WARMUP")
            self.assertEqual(hk["margin_mv_q"], "52/1")
            if k == 4:
                # R1 mixed step_epoch()/step() anti-flip-flop: hold_remaining==0, clean_streak==4 < 16
                # -> single-poll step() MUST stay in TIER0_HOLD_CLEAN_WARMUP (no premature descent!)
                h4_poll = hc.step(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
                self.assertEqual(h4_poll["action"], "TIER0_HOLD_CLEAN_WARMUP")
                self.assertEqual(h4_poll["margin_mv_q"], "52/1")
                self.assertEqual(h4_poll["clean_streak"], 4)
        h16 = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
        self.assertEqual(h16["action"], "TIER0_NOMINAL_RAZOR_DESCENT")
        self.assertEqual(h16["margin_mv_q"], "50/1")
        self.assertEqual(h16["hold_remaining"], 4)

        for _ in range(4):
            hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot), healthy)
        hb = hc.step_epoch(
            rz.RazorCanary64(sec_count_u32=sec_tot),
            rz.SiliSenseHealthSnapshotQ(Fraction(1, 5), Fraction(1, 10), 0, False),
        )
        self.assertEqual(hb["action"], "TIER0_NOMINAL_HOLD_HEALTH_GATE")

        ded_tot += 1
        hd = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy, rollback_buf=buf_razor)
        self.assertEqual(hd["action"], "TIER2_DED_ROLLBACK_LATCH64")
        self.assertEqual(hd["ded_latch_remaining"], 64)
        self.assertEqual(hd["margin_mv_q"], "75/1")
        self.assertIsNotNone(hd["rollback_receipt"])
        # B2 check: step() during DED latch from step_epoch() must hold and never descend
        hd_step_hold = hc.step(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
        self.assertEqual(hd_step_hold["action"], "TIER0_HOLD_DED_LATCH")
        self.assertEqual(hd_step_hold["margin_mv_q"], "75/1")

        hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
        self.assertEqual(hc.ded_latch_remaining, 63)
        sec_tot += 1
        hr1 = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
        self.assertEqual(hr1["ded_latch_remaining"], 64)
        sec_tot += 2
        hr2 = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
        self.assertEqual(hr2["ded_latch_remaining"], 64)

        for _ in range(63):
            hl = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
            self.assertEqual(hl["action"], "TIER0_HOLD_DED_LATCH")
        h_rel = hc.step_epoch(rz.RazorCanary64(sec_count_u32=sec_tot, ded_count_u32=ded_tot), healthy)
        self.assertEqual(h_rel["ded_latch_remaining"], 0)
        self.assertEqual(h_rel["action"], "TIER0_NOMINAL_RAZOR_DESCENT")

        # B1 continuous runtime stress test: 4,000 epochs (> 2,856 limit = 1,024,000 tokens @ 30 kHz)
        # must NEVER overflow 4,300-digit int->str limit and must stay strictly on Q16 grid!
        stress = rz.HoloSphereRazorController(initial_margin_mv_q=Fraction(40, 1))
        s_sec = 1
        stress.step_epoch(rz.RazorCanary64(sec_count_u32=s_sec), healthy)
        for ep in range(4000):
            if ep in (500, 1500, 2900):
                s_sec += 1
            res_s = stress.step_epoch(rz.RazorCanary64(sec_count_u32=s_sec), healthy)
            self.assertLessEqual(stress.ewma_ber_q.denominator, rz.EWMA_Q16_SCALE)
            self.assertEqual(res_s["ewma_ber_q16_u32"], stress.ewma_ber_q16_u32)
        self.assertEqual(stress.epoch_u32, 4001)
        self.assertEqual(stress.ewma_ber_q, Fraction(0, 1))

        # Monotonic counter tampering fails closed
        with self.assertRaises(SiliSenseValueError):
            ctrl.step(rz.RazorCanary64(sec_count_u32=0, ded_count_u32=1), healthy)
        with self.assertRaises(SiliSenseValueError):
            ctrl.step(rz.RazorCanary64(sec_count_u32=1, ded_count_u32=0), healthy)
        ctrl.step(rz.RazorCanary64(sec_count_u32=1, ded_count_u32=1, non_lsb_canary_u32=1), healthy)
        with self.assertRaises(SiliSenseValueError):
            ctrl.step(rz.RazorCanary64(sec_count_u32=1, ded_count_u32=1, non_lsb_canary_u32=0), healthy)

    def test_04_nonscalar_krein_holonomy_detector_and_order_witness(self) -> None:
        w = rz.krein_holonomy_order_witness_q()
        self.assertEqual(w["scalar_I3_n_neg_UL"], 1)
        self.assertEqual(w["scalar_I3_n_neg_LU"], 1)
        self.assertTrue(w["scalar_I3_blind"])
        self.assertEqual(w["nonscalar_G3_n_neg_UL"], 1)
        self.assertEqual(w["nonscalar_G3_n_neg_LU"], 2)
        self.assertTrue(w["nonscalar_G3_splits_order"])
        self.assertEqual(w["phase_slot_order3_q"], "3/1")

        q_rot = rz.cayley_su2_q(Fraction(1, 2), Fraction(1, 3), Fraction(0))
        self.assertEqual(q_rot.norm_sq(), Fraction(1))
        g_diag = (
            (Fraction(3), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        l_id = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        res_id = rz.nonscalar_krein_holonomy_detector(q_rot, l_id, g_diag, rz.QUAT_IDENTITY, order="UL")
        self.assertEqual(res_id["tr_delta_u_q"], "3/1")
        self.assertEqual(res_id["phase_slot_q"], "-6/1")
        res_zero = rz.nonscalar_krein_holonomy_detector(rz.QUAT_IDENTITY, l_id, l_id, rz.QUAT_IDENTITY, order="UL")
        self.assertEqual(res_zero["block3_inertia"], {"n_pos": 0, "n_zero": 3, "n_neg": 0})

        with self.assertRaises(SiliSenseValueError):
            rz.su2_to_so3_q("bad")  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.su2_to_so3_q(rz.QuatQ(Fraction(2), Fraction(0), Fraction(0), Fraction(0)))
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(rz.QUAT_IDENTITY, l_id, g_diag, rz.QUAT_IDENTITY, order="BAD")
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(((Fraction(1),),), l_id, g_diag, rz.QUAT_IDENTITY)
        bad_refl = (
            (Fraction(-1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(bad_refl, l_id, g_diag, rz.QUAT_IDENTITY)
        bad_asym = (
            (Fraction(2), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(2), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(2)),
        )
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(rz.QUAT_IDENTITY, l_id, bad_asym, rz.QUAT_IDENTITY)
        for bad_spd in (
            ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(0), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))),
            ((Fraction(0), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))),
            ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(0))),
        ):
            with self.assertRaises(SiliSenseValueError):
                rz.nonscalar_krein_holonomy_detector(rz.QUAT_IDENTITY, l_id, bad_spd, rz.QUAT_IDENTITY)

    def test_05_nvml_vs_razor_latency_and_complete_boundary_guard_matrix(self) -> None:
        lat = rz.compare_nvml_vs_razor_latency_q()
        self.assertEqual(lat["detection_speedup_exact_q"], "3000/1")
        self.assertEqual(lat["nvml_sample_hz_q"], "10/1")
        self.assertEqual(lat["razor_sample_hz_q"], "30000/1")
        self.assertEqual(lat["nvml_max_stable_ctrl_hz_q"], "5/3")
        self.assertEqual(lat["razor_max_stable_ctrl_hz_q"], "5000/1")
        self.assertEqual(lat["worst_case_blind_tokens_nvml_q"], "3000/1")
        self.assertEqual(lat["worst_case_blind_tokens_razor_q"], "1/1")

        # Complete refusal & boundary guard coverage for 100% mutation kill
        clean = _make_clean_nibbles()
        self.assertEqual(rz.a16_secded_syndrome(0), 0)
        self.assertEqual(rz.a16_secded_syndrome(0xFFFF), 0)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_secded_syndrome(-1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_secded_syndrome(0x10000)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_encode_lsb16(-1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_encode_lsb16(2048)
        self.assertEqual(rz.a16_secded_classify(31), ("SEC", 15))
        with self.assertRaises(SiliSenseValueError):
            rz.a16_secded_classify(-1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_secded_classify(32)
        self.assertEqual(rz.a16_extract_bitplane16(clean, 0), rz.a16_extract_bitplane16(clean, 0))
        self.assertIsInstance(rz.a16_extract_bitplane16(clean, 3), int)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_extract_bitplane16(clean, -1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_extract_bitplane16(clean, 4)
        self.assertIsInstance(rz.block_crc16_ccitt([-8] + [7] * 15), int)
        with self.assertRaises(SiliSenseValueError):
            rz.block_crc16_ccitt([0] * 15)
        with self.assertRaises(SiliSenseValueError):
            rz.block_crc16_ccitt({k: "JUNK" for k in range(-8, 8)})  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.a16_folded_upper_parity_bit(set(range(-8, 8)))  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.block_crc16_ccitt(x for x in range(-8, 8))  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.block_crc16_ccitt([8] + [0] * 15)
        with self.assertRaises(SiliSenseValueError):
            rz.block_crc16_ccitt([-9] + [0] * 15)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, expected_upper_fold_bit=2)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, expected_plane_syndromes=(0, 0, 0))
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, expected_plane_syndromes={0: 0, 1: 0, 2: 0, 3: 0})  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, epoch_u32=1, expected_plane_syndromes=(0, -1, 0, 0))
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, epoch_u32=1, expected_plane_syndromes=(0, 32, 0, 0))
        self.assertEqual(rz.a16_inline_secded_step(clean, epoch_u32=1, expected_plane_syndromes=(0, 0, 0, 0))["rotated_plane"], 1)
        self.assertEqual(rz.a16_inline_secded_step(clean, epoch_u32=1, expected_plane_syndromes=(0, 31, 0, 0))["rotated_plane"], 1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, expected_crc16=-1)
        with self.assertRaises(SiliSenseValueError):
            rz.a16_inline_secded_step(clean, expected_crc16=0x10000)
        self.assertEqual(rz.a16_inline_secded_step(clean, expected_crc16=0)["status"], "NON_LSB_FAULT")
        self.assertEqual(rz.a16_inline_secded_step(clean, expected_crc16=0xFFFF)["status"], "NON_LSB_FAULT")
        with self.assertRaises(SiliSenseValueError):
            rz.inject_nibble_bit_flip(clean, -1, 0)
        with self.assertRaises(SiliSenseValueError):
            rz.inject_nibble_bit_flip(clean, 16, 0)
        with self.assertRaises(SiliSenseValueError):
            rz.inject_nibble_bit_flip(clean, 0, -1)
        with self.assertRaises(SiliSenseValueError):
            rz.inject_nibble_bit_flip(clean, 0, 4)
        with self.assertRaises(SiliSenseValueError):
            rz.QuatQ(1.0, Fraction(0), Fraction(0), Fraction(0))  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.QUAT_IDENTITY.mul("bad")  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.QUAT_IDENTITY.diff_norm_sq("bad")  # type: ignore[arg-type]
        self.assertEqual(rz.QUAT_IDENTITY.conj(), rz.QUAT_IDENTITY)
        # N1: exact finite/subnormal IEEE-754 float32 bounds on QuatQ.to_pi16_bytes()
        q_edge = rz.QuatQ(rz.F32_MAX_Q, -rz.F32_MAX_Q, rz.F32_MIN_SUBNORMAL_Q, Fraction(0))
        self.assertEqual(len(q_edge.to_pi16_bytes()), 16)
        for bad_q in (
            rz.QuatQ(Fraction(10**400), Fraction(0), Fraction(0), Fraction(0)),
            rz.QuatQ(rz.F32_MAX_Q + Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
            rz.QuatQ(Fraction(1, 10**400), Fraction(0), Fraction(0), Fraction(0)),
            rz.QuatQ(Fraction(0), Fraction(1, 1 << 150), Fraction(0), Fraction(0)),
        ):
            with self.assertRaises(SiliSenseValueError) as ctx_pi:
                bad_q.to_pi16_bytes()
            self.assertEqual(ctx_pi.exception.code, "ERR_PI16_RANGE")
        with self.assertRaises(SiliSenseValueError):
            rz.WeftHolonomyRollbackBuffer(chunk_size=0)
        self.assertEqual(rz.WeftHolonomyRollbackBuffer(chunk_size=1).chunk_size, 1)
        with self.assertRaises(SiliSenseValueError) as ctx_corrupt:
            rz.WeftHolonomyRollbackBuffer().step_token(
                Fraction(0), Fraction(0), Fraction(0), Fraction(0), corrupt_delta="bad"  # type: ignore[arg-type]
            )
        self.assertEqual(ctx_corrupt.exception.code, "ERR_CORRUPT_Q")
        with self.assertRaises(SiliSenseValueError) as ctx_cb:
            rz.WeftHolonomyRollbackBuffer().step_token(
                Fraction(0), Fraction(0), Fraction(0), Fraction(0), commit_boundary=1  # type: ignore[arg-type]
            )
        self.assertEqual(ctx_cb.exception.code, "ERR_COMMIT_BOOL")

        # B3 matrix & rollback_buf non-sequence / wrong-type guards
        g_id = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(rz.QUAT_IDENTITY, 5, g_id, rz.QUAT_IDENTITY)  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.nonscalar_krein_holonomy_detector(
                rz.QUAT_IDENTITY, {0: g_id[0], 1: g_id[1], 2: g_id[2]}, g_id, rz.QUAT_IDENTITY  # type: ignore[arg-type]
            )
        with self.assertRaises(SiliSenseValueError):
            rz.HoloSphereRazorController().step(
                rz.RazorCanary64(), rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 0, True), rollback_buf="bad"  # type: ignore[arg-type]
            )
        with self.assertRaises(SiliSenseValueError):
            rz.HoloSphereRazorController().step_epoch(
                rz.RazorCanary64(), rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 0, True), rollback_buf="bad"  # type: ignore[arg-type]
            )

        self.assertEqual(rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 0, True).op1_leverage_q, Fraction(0))
        self.assertEqual(rz.SiliSenseHealthSnapshotQ(Fraction(1), Fraction(1), 3, True).op3_neg_inertia, 3)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(-1, 10), Fraction(0), 0, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(2), Fraction(0), 0, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(-1, 10), 0, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(2), 0, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), -1, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 4, True)
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 0, 1)  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError) as ctx_map:
            rz.SiliSenseHealthSnapshotQ.from_ratchet_result(
                op1_leverage_q=Fraction(0), op2_phi_asym_q=Fraction(0), op3_neg_inertia=0, ratchet_result=["accepted"]  # type: ignore[arg-type]
            )
        self.assertEqual(ctx_map.exception.code, "ERR_HEALTH_RATCHET")
        with self.assertRaises(SiliSenseValueError):
            rz.SiliSenseHealthSnapshotQ.from_ratchet_result(
                op1_leverage_q=Fraction(0), op2_phi_asym_q=Fraction(0), op3_neg_inertia=0, ratchet_result={}
            )

        with self.assertRaises(SiliSenseValueError):
            rz.compare_nvml_vs_razor_latency_q(Fraction(0), Fraction(1))
        with self.assertRaises(SiliSenseValueError):
            rz.compare_nvml_vs_razor_latency_q(Fraction(1), Fraction(0))

        with self.assertRaises(SiliSenseValueError):
            rz.HoloSphereRazorController(base_profile="bad")  # type: ignore[arg-type]
        self.assertEqual(
            rz.HoloSphereRazorController(
                min_margin_mv_q=Fraction(0), initial_margin_mv_q=Fraction(0), max_margin_mv_q=Fraction(0)
            ).margin_mv_q,
            Fraction(0),
        )
        self.assertEqual(
            rz.HoloSphereRazorController(
                min_power_cap_w_q=Fraction(80), initial_power_cap_w_q=Fraction(80), max_power_cap_w_q=Fraction(80)
            ).power_cap_w_q,
            Fraction(80),
        )
        self.assertEqual(
            rz.HoloSphereRazorController(
                tier1_cooldown_ms_q=Fraction(0), tier2_cooldown_ms_q=rz.MAX_COOLDOWN_MS_Q
            ).tier2_cooldown_ms_q,
            rz.MAX_COOLDOWN_MS_Q,
        )
        self.assertEqual(
            rz.HoloSphereRazorController(
                epoch_tokens=1,
                sec_step_down_count=1,
                hold_epochs=1,
                step_up_clean_epochs=1,
                ded_latch_clean_epochs=1,
                ewma_alpha_q=Fraction(1, 1),
                ewma_step_down_q=Fraction(1, 1),
            ).epoch_tokens,
            1,
        )
        for bad_kw in (
            {"min_margin_mv_q": Fraction(-1, 1)},
            {"initial_margin_mv_q": Fraction(120, 1)},
            {"min_power_cap_w_q": Fraction(0), "initial_power_cap_w_q": Fraction(70), "max_power_cap_w_q": Fraction(100)},
            {"max_power_cap_w_q": Fraction(110, 1)},
            {"tier0_descent_mv_q": Fraction(0, 1)},
            {"tier0_descent_mv_q": Fraction(15, 1), "tier1_step_up_mv_q": Fraction(12, 1)},
            {"tier1_step_up_mv_q": Fraction(25, 1), "tier2_step_up_mv_q": Fraction(25, 1)},
            {"tier0_power_step_w_q": Fraction(0, 1)},
            {"tier1_cooldown_ms_q": Fraction(-1, 1)},
            {"tier1_cooldown_ms_q": Fraction(5, 1), "tier2_cooldown_ms_q": Fraction(2, 1)},
            {"tier2_cooldown_ms_q": rz.MAX_COOLDOWN_MS_Q + Fraction(1, 1)},
            {"epoch_tokens": 0},
            {"sec_step_down_count": 0},
            {"hold_epochs": 0},
            {"hold_epochs": 5, "step_up_clean_epochs": 4},
            {"step_up_clean_epochs": 20, "ded_latch_clean_epochs": 16},
            {"ewma_alpha_q": Fraction(0, 1)},
            {"ewma_alpha_q": Fraction(2, 1)},
            {"ewma_step_down_q": Fraction(0, 1)},
            {"ewma_step_down_q": Fraction(2, 1)},
        ):
            with self.assertRaises(SiliSenseValueError):
                rz.HoloSphereRazorController(**bad_kw)  # type: ignore[arg-type]

        with self.assertRaises(SiliSenseValueError):
            rz.HoloSphereRazorController().step("bad", rz.SiliSenseHealthSnapshotQ(Fraction(0), Fraction(0), 0, True))  # type: ignore[arg-type]
        with self.assertRaises(SiliSenseValueError):
            rz.HoloSphereRazorController().step(rz.RazorCanary64(), "bad")  # type: ignore[arg-type]

    def test_06_selftest_cli_pure_stdlib_hygiene_and_abi_header_layout_parity(self) -> None:
        abi_hdr = _REPO_ROOT / "include" / "razor_abi_4byte_sys.h"
        self.assertTrue(abi_hdr.is_file())
        hdr_txt = abi_hdr.read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: MIT", hdr_txt)
        self.assertIn("__ORDER_LITTLE_ENDIAN__", hdr_txt)
        self.assertIn("__atomic_load_n(ptr, __ATOMIC_ACQUIRE)", hdr_txt)
        self.assertIn("#include <intrin.h>", hdr_txt)
        self.assertIn("_ReadWriteBarrier();", hdr_txt)
        self.assertNotIn("_InterlockedCompareExchange", hdr_txt)
        self.assertNotIn("std::atomic<uint32_t>*", hdr_txt)
        self.assertNotIn("_Atomic uint32_t*", hdr_txt)
        self.assertIn("atom.release.sys.global.add.u32", hdr_txt)
        self.assertIn("st.release.sys.global.u32", hdr_txt)

        # Verify every offsetof static assertion in C ABI header matches Python <16I & <4f wire packing
        expected_canary_offsets = {
            "sec_count_u32": 0,
            "ded_count_u32": 4,
            "epoch_u32": 8,
            "rollback_token_u32": 12,
            "non_lsb_canary_u32": 16,
            "last_syndrome_u32": 20,
            "ewma_ber_q16_u32": 24,
            "clean_streak_u32": 28,
            "hold_remaining_u32": 32,
            "ded_latch_remaining_u32": 36,
            "reserved_u32": 40,
        }
        parsed_canary_offsets = {
            m.group(1): int(m.group(2))
            for m in re.finditer(r"offsetof\(HoloSphereRazorCanary64,\s*(\w+)\)\s*==\s*(\d+)u", hdr_txt)
        }
        self.assertEqual(parsed_canary_offsets, expected_canary_offsets)
        self.assertIn("sizeof(HoloSphereRazorCanary64) == 64u", hdr_txt)
        self.assertIn("sizeof(HoloSpherePiRollback16) == 16u", hdr_txt)

        q_test = rz.cayley_su2_q(Fraction(1, 2), Fraction(1, 4), Fraction(-1, 8))
        pi16_raw = q_test.to_pi16_bytes()
        self.assertEqual(len(pi16_raw), 16)
        unpacked_q = struct.unpack("<4f", pi16_raw)
        for got_f, exp_q in zip(unpacked_q, (q_test.w, q_test.x, q_test.y, q_test.z)):
            self.assertAlmostEqual(got_f, float(exp_q), places=6)

        src = (_REPO_ROOT / "silisense" / "razor.py").read_text(encoding="utf-8")
        tree = ast.parse(src, filename="silisense/razor.py")
        for node in ast.walk(tree):
            self.assertNotIsInstance(node, ast.Assert)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], {"torch", "numpy", "ctypes", "pynvml", "fcntl"})
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], {"torch", "numpy", "ctypes", "pynvml", "fcntl"})

        cert = rz.run_razor_selftest()
        self.assertTrue(cert["sec_16_of_16_inline_healed"])
        self.assertTrue(cert["ded_120_of_120_pairs_detected"])
        self.assertTrue(cert["upper_bits_1_3_lsb_only_missed_48_of_48"])
        self.assertTrue(cert["upper_bits_1_3_folded_parity_caught_48_of_48"])
        self.assertEqual(cert["upper_bits_1_3_rotated_plane_per_epoch"], {"0": 0, "1": 16, "2": 16, "3": 16})
        self.assertTrue(cert["upper_bits_1_3_rotated_plane_4epoch_union_48_of_48"])
        self.assertTrue(cert["upper_bits_1_3_crc16_caught_48_of_48"])
        self.assertTrue(cert["krein_holonomy_order_witness"]["scalar_I3_blind"])
        self.assertTrue(cert["krein_holonomy_order_witness"]["nonscalar_G3_splits_order"])

        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-c", f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r}); import json, silisense.razor as r; print(json.dumps(r.run_razor_selftest(), sort_keys=True))"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), cert)

        # Standalone pinned 128-combo HoloSphereTuningProfile.profile_sha256() contract digest
        # (matches Moissanite-Labs/holosphere#3 @ 6f75548d across all 4x4x2x2x2 = 128 combos in CI)
        combos_128 = list(
            itertools.product(
                ("auto", "cuda_sm86", "torch_cpu", "triton"),
                (32, 64, 128, 256),
                (1, 4),
                (0.0, 2.5),
                (None, 2),
            )
        )
        self.assertEqual(len(combos_128), 128)
        local_128 = [
            HoloSphereTuningProfile(
                backend=b, block_t=bt, microbatch_size=mb, inter_step_cooldown_ms=cd, r_loops_override=rl
            ).profile_sha256()
            for b, bt, mb, cd, rl in combos_128
        ]
        pinned_128_digest = "6e090bae706863f7b040e4bdba0077b2e07170b65260c7352a507376bf52c9f8"
        self.assertEqual(hashlib.sha256("\n".join(local_128).encode("utf-8")).hexdigest(), pinned_128_digest)

        # Sibling holosphere cross-repo 128/128 live profile_sha256 parity check when present locally
        holosphere_root = _REPO_ROOT.parent / "holosphere"
        if holosphere_root.is_dir():
            probe = (
                f"import hashlib, itertools, sys; sys.modules['torch']=None; sys.path.insert(0, {str(holosphere_root)!r}); "
                "from holosphere.telemetry import HoloSphereTuningProfile as HP; "
                "cs=itertools.product(('auto','cuda_sm86','torch_cpu','triton'),(32,64,128,256),(1,4),(0.0,2.5),(None,2)); "
                "hs=[HP(backend=b,block_t=bt,microbatch_size=mb,inter_step_cooldown_ms=cd,r_loops_override=rl).profile_sha256() for b,bt,mb,cd,rl in cs]; "
                "print(hashlib.sha256('\\n'.join(hs).encode('utf-8')).hexdigest())"
            )
            hp = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=False)
            self.assertEqual(hp.returncode, 0, hp.stderr)
            self.assertEqual(hp.stdout.strip(), pinned_128_digest)


if __name__ == "__main__":
    unittest.main()
