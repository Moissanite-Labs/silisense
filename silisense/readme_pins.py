#!/usr/bin/env python3
"""Pure-Q README-vs-Receipt Pin & Uncontrolled-Laptop-Energy Hygiene Verifier for SiliSense.

Verifies without floating-point arithmetic:
  1. Per-bullet, boundary-anchored parsing of every companion `Moissanite-Labs/silikernels`
     `b3712675` receipt pin and rounded display figure in `README.md` (rejecting appended/
     prepended digits and cross-section V1/V2/V3 swaps), with exact rational rounding
     (`9.098×` against pinned `249.3671 us/tok` vs `speedup_vs_v1_249_32us = 9.0967`).
  2. When `--silikernels-root` is provided, walks directories and files component-by-component
     with `O_NOFOLLOW` + pre-open `lstat` (blocking symlinked roots, symlinked `receipts/`
     directories, symlinked leaves, and device nodes), verifies full 64-hex Zone-B digests,
     verifies git ancestry & receipt tree match against `b3712675...` when `.git` is present,
     and refuses duplicate JSON keys, non-finite constants, and oversized/4400-digit inputs.
  3. Zero stale `44eccad8` pins and zero uncontrolled laptop energy/power claims (`W`, `mW`,
     `watts`, `J/tok`, `millijoules`, `mJ/tok` magnitudes, `EDP` magnitudes) in `README.md`,
     with explicit uncontrolled-laptop and timing disclaimers on both sections.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silisense.errors import SiliSenseError, require_ok

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_DEC_DIGITS = 64
MAX_JSON_DEPTH = 64

PINNED_SILIKERNELS_COMMIT = "b37126752e948a2dbf5d6275d52c2a5200ec65db"
PINNED_SILIKERNELS_SHORT = "b3712675"
STALE_SILIKERNELS_SHORT = "44eccad8"

PINNED_ZONE_B_FULL_SHA256: dict[str, str] = {
    "v2": "699c3d05958a8c274c04662f3cdcfee59049a0ce1f6ab6647d2183f1bb9923fa",
    "v3": "feecc7487bc03c860148d559f76abd7ea705b6ab323572bfe91be37e0266bdf1",
}

PINNED_RECEIPT_VALUES: dict[str, str] = {
    "v1_t8192_us_per_tok": "249.3671",
    "v2_cellop_t8192_us_per_tok": "26.2082",
    "v2_e2e_t8192_us_per_tok": "27.4079",
    "v2_speedup_vs_v1_249_32us": "9.0967",
    "v2_zone_b_receipt_sha256_prefix": "699c3d05",
    "v3_cellop_graph_us_per_tok_8192": "20.5038",
    "v3_e2e_graph_us_per_tok_16384": "21.5184",
    "v3_zone_b_receipt_sha256_prefix": "feecc748",
}

RECEIPT_RELPATHS: dict[str, str] = {
    "v1_results": "receipts/s117_live_rtx3080_zone_b_results.json",
    "v2_results": "receipts/s117_live_v2_d512_zone_b_results.json",
    "v3_results": "receipts/s117_v3_frontier_live_gpu_results.json",
    "v2_zone_b": "receipts/mlos_zone_b_s117_v2_d512.json",
    "v3_zone_b": "receipts/mlos_zone_b_s117_v3_frontier.json",
}

_V1_OLD_REF_US_Q = Fraction(24932, 100)
_DECIMAL_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_HEX_PREFIX8_RE = re.compile(r"^[0-9a-f]{8}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_NUM_ANCHORED = r"(?<![a-z0-9_.,])\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?"
_FORBIDDEN_README_ENERGY_RE = re.compile(
    "(?:"
    rf"{_NUM_ANCHORED}\s*(?:mJ/tok|mJ·ms/tok|mJ\s*·\s*ms/tok|J/tok|uJ/tok|MJ|kJ|mJ|uJ|μJ|µJ|J|kWh|Wh|W|mW|kW|watts?|joules?|millijoules?)(?![a-z0-9_])"
    r"|(?<![a-z0-9_])(?:watts?|wattage|joules?|millijoules?|kilowatts?|milliwatts?|mW|kW|J/tok)(?![a-z0-9_])"
    r"|\b5\.97×|\b11\.16\b|\b66\.59\b|\b0\.989\b|\b1\.3076\b|\b2\.2275\b|\b0\.1901\b|\b0\.1837\b"
    ")",
    re.IGNORECASE,
)


def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in pairs:
        require_ok(key not in out, "ERR_DUP_JSON_KEY", f"duplicate JSON key: {key!r}")
        out[key] = val
    return out


def _reject_non_finite(token: str) -> Any:
    raise SiliSenseError("ERR_NON_FINITE_JSON", f"non-finite JSON constant: {token!r}")


def _read_fd_regular_bounded(fd: int, label: str, *, io_code: str) -> bytes:
    """Validate an opened fd is a regular file and read up to MAX_FILE_BYTES."""
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "ERR_NOT_REGULAR_FILE", f"not a regular file: {label}")
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read(MAX_FILE_BYTES + 1)
    except SiliSenseError:
        raise
    except OSError as exc:
        raise SiliSenseError(io_code, f"I/O failure reading {label}") from exc
    require_ok(len(data) <= MAX_FILE_BYTES, "ERR_FILE_TOO_LARGE", f"file exceeds {MAX_FILE_BYTES} B")
    return data


def _read_regular_file_bytes_nofollow(path: Path, *, missing_code: str) -> bytes:
    """Pre-open lstat + O_NOFOLLOW + S_ISREG read refusing symlinks and device nodes."""
    require_ok(isinstance(path, Path), "ERR_BAD_PATH", "path must be Path")
    try:
        lst = os.lstat(str(path))
    except OSError as exc:
        raise SiliSenseError(missing_code, f"cannot stat file: {path.name}") from exc
    if stat.S_ISLNK(lst.st_mode):
        raise SiliSenseError("ERR_SYMLINK_FORBIDDEN", f"symlink forbidden: {path.name}")
    require_ok(stat.S_ISREG(lst.st_mode), "ERR_NOT_REGULAR_FILE", f"not a regular file: {path.name}")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise SiliSenseError(missing_code, f"cannot open file: {path.name}") from exc
    try:
        return _read_fd_regular_bounded(fd, path.name, io_code=missing_code)
    finally:
        os.close(fd)


def _read_rel_bytes_under_root_nofollow(root: Path, rel_path: str, *, missing_code: str) -> bytes:
    """Walk every path segment under `root` with O_DIRECTORY | O_NOFOLLOW so parent symlinks fail closed."""
    require_ok(isinstance(root, Path), "ERR_BAD_ROOT", "silikernels_root must be Path")
    try:
        rst = os.lstat(str(root))
    except OSError as exc:
        raise SiliSenseError(missing_code, f"cannot stat root: {root.name}") from exc
    if stat.S_ISLNK(rst.st_mode):
        raise SiliSenseError("ERR_SYMLINK_FORBIDDEN", f"symlink root forbidden: {root.name}")
    require_ok(stat.S_ISDIR(rst.st_mode), "ERR_NOT_REGULAR_FILE", f"root must be directory: {root.name}")

    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        cur_fd = os.open(str(root), dir_flags)
    except OSError as exc:
        raise SiliSenseError(missing_code, f"cannot open root: {root.name}") from exc

    parts = Path(rel_path).parts
    try:
        for seg in parts[:-1]:
            try:
                nxt_fd = os.open(seg, dir_flags, dir_fd=cur_fd)
            except OSError as exc:
                try:
                    if stat.S_ISLNK(os.lstat(seg, dir_fd=cur_fd).st_mode):
                        raise SiliSenseError("ERR_SYMLINK_FORBIDDEN", f"symlink dir forbidden: {seg}") from exc
                except OSError:
                    pass
                raise SiliSenseError(missing_code, f"cannot open subdir: {seg}") from exc
            os.close(cur_fd)
            cur_fd = nxt_fd

        leaf = parts[-1]
        try:
            lst = os.lstat(leaf, dir_fd=cur_fd)
        except OSError as exc:
            raise SiliSenseError(missing_code, f"missing receipt: {leaf}") from exc
        if stat.S_ISLNK(lst.st_mode):
            raise SiliSenseError("ERR_SYMLINK_FORBIDDEN", f"symlink receipt forbidden: {leaf}")
        require_ok(stat.S_ISREG(lst.st_mode), "ERR_NOT_REGULAR_FILE", f"not a regular file: {leaf}")

        try:
            leaf_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=cur_fd)
        except OSError as exc:
            raise SiliSenseError(missing_code, f"cannot open receipt: {leaf}") from exc
        try:
            return _read_fd_regular_bounded(leaf_fd, leaf, io_code=missing_code)
        finally:
            os.close(leaf_fd)
    finally:
        os.close(cur_fd)


def _parse_bounded_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    require_ok(len(digits) <= MAX_DEC_DIGITS, "ERR_BAD_DECIMAL", "JSON integer exceeds digit cap")
    return int(token)


def _check_json_depth(obj: Any) -> None:
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        require_ok(depth <= MAX_JSON_DEPTH, "ERR_RECEIPT_DEPTH", f"JSON nesting exceeds {MAX_JSON_DEPTH}")
        if type(cur) is dict:
            for val in cur.values():
                if type(val) is dict or type(val) is list:
                    stack.append((val, depth + 1))
        elif type(cur) is list:
            for item in cur:
                if type(item) is dict or type(item) is list:
                    stack.append((item, depth + 1))


def _parse_receipt_json_bytes(raw_bytes: bytes, label: str) -> dict[str, Any]:
    try:
        raw_text = raw_bytes.decode("utf-8")
        parsed = json.loads(
            raw_text,
            object_pairs_hook=_no_dup_pairs,
            parse_int=_parse_bounded_int,
            parse_float=str,
            parse_constant=_reject_non_finite,
        )
    except SiliSenseError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise SiliSenseError("ERR_RECEIPT_PARSE", str(exc)) from exc
    require_ok(type(parsed) is dict, "ERR_RECEIPT_TYPE", f"receipt must be dict: {label}")
    _check_json_depth(parsed)
    return parsed


def _load_receipt_json_exact(path: Path) -> dict[str, Any]:
    raw_bytes = _read_regular_file_bytes_nofollow(path, missing_code="ERR_MISSING_RECEIPT")
    return _parse_receipt_json_bytes(raw_bytes, path.name)


def decimal_str_to_q(text: Any) -> Fraction:
    """Parse a bounded non-negative decimal string into an exact Fraction without floats."""
    require_ok(
        type(text) is str and len(text) <= MAX_DEC_DIGITS and bool(_DECIMAL_RE.match(text)),
        "ERR_BAD_DECIMAL",
        f"bad decimal: {text!r}",
    )
    if "." in text:
        whole, frac = text.split(".", 1)
        scale = 10 ** len(frac)
        return Fraction(int(whole) * scale + int(frac), scale)
    return Fraction(int(text), 1)


def round_half_up_q(val: Fraction, places: int) -> str:
    """Exact pure-Q half-up rounding of a non-negative Fraction to `places` decimals."""
    require_ok(isinstance(val, Fraction) and val >= Fraction(0, 1), "ERR_BAD_Q", "val must be non-negative Fraction")
    require_ok(type(places) is int and 0 <= places <= 6, "ERR_BAD_PLACES", f"bad places: {places!r}")
    scale = 10 ** places
    scaled = (2 * val.numerator * scale + val.denominator) // (2 * val.denominator)
    if places == 0:
        return str(scaled)
    whole, frac = divmod(scaled, scale)
    return f"{whole}.{frac:0{places}d}"


def q_div(a: Fraction, b: Fraction) -> Fraction:
    """Exact rational division without ast.Div."""
    require_ok(isinstance(a, Fraction) and isinstance(b, Fraction), "ERR_BAD_Q", "a and b must be Fraction")
    require_ok(b != Fraction(0, 1), "ERR_ZERO_DIV", "division by zero")
    return Fraction(a.numerator * b.denominator, a.denominator * b.numerator)


def _as_decimal_str(raw: Any, name: str) -> str:
    if type(raw) is int:
        require_ok(raw >= 0, "ERR_RECEIPT_FIELD", f"{name} must be non-negative")
        s = str(raw)
        require_ok(len(s) <= MAX_DEC_DIGITS, "ERR_RECEIPT_FIELD", f"{name} exceeds digit cap")
        return s
    require_ok(
        type(raw) is str and len(raw) <= MAX_DEC_DIGITS and bool(_DECIMAL_RE.match(raw)),
        "ERR_RECEIPT_FIELD",
        f"{name} must be decimal",
    )
    return raw


def _require_hex64(raw: Any, name: str) -> str:
    require_ok(type(raw) is str and bool(_HEX64_RE.match(raw)), "ERR_RECEIPT_SHA", f"{name} must be 64-hex")
    return raw


def _run_git(silikernels_root: Path, git_args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        return subprocess.run(
            ["git", "-C", str(silikernels_root), *git_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=clean_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SiliSenseError("ERR_SILIKERNELS_GIT", f"git verification failed: {exc}") from exc


def _verify_git_pin_if_repo(silikernels_root: Path) -> bool:
    """If silikernels_root is a git checkout, verify b3712675 ancestry and unmodified receipts."""
    git_marker = silikernels_root.joinpath(".git")
    try:
        gst = os.lstat(str(git_marker))
    except OSError:
        return False
    if stat.S_ISLNK(gst.st_mode):
        raise SiliSenseError("ERR_SYMLINK_FORBIDDEN", "symlink .git forbidden")
    anc = _run_git(silikernels_root, ["merge-base", "--is-ancestor", PINNED_SILIKERNELS_COMMIT, "HEAD"])
    require_ok(anc.returncode == 0, "ERR_SILIKERNELS_COMMIT", "silikernels checkout does not contain pinned commit")
    diff = _run_git(
        silikernels_root,
        ["diff", "--quiet", PINNED_SILIKERNELS_COMMIT, "--", *RECEIPT_RELPATHS.values()],
    )
    require_ok(diff.returncode == 0, "ERR_SILIKERNELS_RECEIPT_MODIFIED", "silikernels receipts differ from b3712675")
    return True


def load_silikernels_receipt_pins(silikernels_root: Path) -> dict[str, str]:
    """Read and verify the 8 pinned values directly from companion silikernels receipts."""
    v1 = _parse_receipt_json_bytes(
        _read_rel_bytes_under_root_nofollow(silikernels_root, RECEIPT_RELPATHS["v1_results"], missing_code="ERR_MISSING_RECEIPT"),
        "v1_results",
    )
    v2 = _parse_receipt_json_bytes(
        _read_rel_bytes_under_root_nofollow(silikernels_root, RECEIPT_RELPATHS["v2_results"], missing_code="ERR_MISSING_RECEIPT"),
        "v2_results",
    )
    v3 = _parse_receipt_json_bytes(
        _read_rel_bytes_under_root_nofollow(silikernels_root, RECEIPT_RELPATHS["v3_results"], missing_code="ERR_MISSING_RECEIPT"),
        "v3_results",
    )
    zb2 = _parse_receipt_json_bytes(
        _read_rel_bytes_under_root_nofollow(silikernels_root, RECEIPT_RELPATHS["v2_zone_b"], missing_code="ERR_MISSING_RECEIPT"),
        "v2_zone_b",
    )
    zb3 = _parse_receipt_json_bytes(
        _read_rel_bytes_under_root_nofollow(silikernels_root, RECEIPT_RELPATHS["v3_zone_b"], missing_code="ERR_MISSING_RECEIPT"),
        "v3_zone_b",
    )
    _verify_git_pin_if_repo(silikernels_root)

    v1_bm = v1.get("bare_metal_rtx3080")
    v1_blk = v1_bm.get("uniform_cell_16warp_d512") if type(v1_bm) is dict else None
    v2_cell = v2.get("cellop_only_12L_d512")
    v2_e2e = v2.get("full_e2e_fwht256_12L_d512")
    v3_blk = v3.get("v3_d512")
    require_ok(
        type(v1_blk) is dict and type(v2_cell) is dict and type(v2_e2e) is dict and type(v3_blk) is dict,
        "ERR_RECEIPT_SHAPE",
        "missing expected section in silikernels receipts",
    )

    v2_zb_sha = _require_hex64(zb2.get("receipt_sha256"), "zb2.receipt_sha256")
    v3_zb_sha = _require_hex64(zb3.get("receipt_sha256"), "zb3.receipt_sha256")
    require_ok(
        v2.get("mlos_zone_b_receipt_sha256") == v2_zb_sha and v2_zb_sha == PINNED_ZONE_B_FULL_SHA256["v2"],
        "ERR_RECEIPT_CROSS_SHA",
        "v2 Zone-B sha mismatch",
    )
    require_ok(
        v3.get("mlos_zone_b_receipt_sha256") == v3_zb_sha and v3_zb_sha == PINNED_ZONE_B_FULL_SHA256["v3"],
        "ERR_RECEIPT_CROSS_SHA",
        "v3 Zone-B sha mismatch",
    )

    return {
        "v1_t8192_us_per_tok": _as_decimal_str(v1_blk.get("t8192_us_per_tok"), "v1_t8192_us_per_tok"),
        "v2_cellop_t8192_us_per_tok": _as_decimal_str(v2_cell.get("t8192_us_per_tok"), "v2_cellop_t8192_us_per_tok"),
        "v2_e2e_t8192_us_per_tok": _as_decimal_str(v2_e2e.get("t8192_us_per_tok"), "v2_e2e_t8192_us_per_tok"),
        "v2_speedup_vs_v1_249_32us": _as_decimal_str(v2_e2e.get("speedup_vs_v1_249_32us"), "v2_speedup_vs_v1_249_32us"),
        "v2_zone_b_receipt_sha256_prefix": v2_zb_sha[:8],
        "v3_cellop_graph_us_per_tok_8192": _as_decimal_str(
            v3_blk.get("cellop_graph_us_per_tok_8192"), "v3_cellop_graph_us_per_tok_8192"
        ),
        "v3_e2e_graph_us_per_tok_16384": _as_decimal_str(
            v3_blk.get("e2e_graph_us_per_tok_16384"), "v3_e2e_graph_us_per_tok_16384"
        ),
        "v3_zone_b_receipt_sha256_prefix": v3_zb_sha[:8],
    }


def _extract_unique_bullet(readme_text: str, marker: str) -> str:
    matches = [ln for ln in readme_text.splitlines() if marker in ln]
    require_ok(len(matches) == 1, "ERR_README_SECTION", f"expected unique bullet for {marker!r}")
    line = matches[0]
    require_ok("<!--" not in line and "-->" not in line, "ERR_README_SECTION", f"HTML comment forbidden in {marker!r}")
    return line


def _require_anchored_pattern(line: str, pattern: str, label: str) -> None:
    require_ok(re.search(pattern, line) is not None, "ERR_README_PIN_MISMATCH", f"mismatch for {label}")


def verify_readme_pins(
    readme_text: str,
    *,
    pins: dict[str, str] | None = None,
    silikernels_root: Path | None = None,
) -> dict[str, Any]:
    """Verify README.md per-bullet anchored pins, exact pure-Q rounding, disclaimers, & zero laptop energy claims."""
    require_ok(type(readme_text) is str and len(readme_text) > 0, "ERR_BAD_README", "readme_text must be non-empty str")

    active_pins = dict(PINNED_RECEIPT_VALUES)
    if silikernels_root is not None:
        loaded = load_silikernels_receipt_pins(silikernels_root)
        require_ok(loaded == PINNED_RECEIPT_VALUES, "ERR_RECEIPT_DRIFT", "silikernels receipts diverged from b3712675 pins")
        active_pins = loaded
    elif pins is not None:
        require_ok(type(pins) is dict and set(pins.keys()) == set(PINNED_RECEIPT_VALUES.keys()), "ERR_BAD_PINS", "bad pins keys")
        active_pins = dict(pins)

    v1_q = decimal_str_to_q(active_pins["v1_t8192_us_per_tok"])
    v2_cell_q = decimal_str_to_q(active_pins["v2_cellop_t8192_us_per_tok"])
    v2_e2e_q = decimal_str_to_q(active_pins["v2_e2e_t8192_us_per_tok"])
    v2_old_sp_q = decimal_str_to_q(active_pins["v2_speedup_vs_v1_249_32us"])
    v3_cell_q = decimal_str_to_q(active_pins["v3_cellop_graph_us_per_tok_8192"])
    v3_e2e_q = decimal_str_to_q(active_pins["v3_e2e_graph_us_per_tok_16384"])

    v2_zb_pfx = active_pins["v2_zone_b_receipt_sha256_prefix"]
    v3_zb_pfx = active_pins["v3_zone_b_receipt_sha256_prefix"]
    require_ok(
        type(v2_zb_pfx) is str and bool(_HEX_PREFIX8_RE.match(v2_zb_pfx))
        and type(v3_zb_pfx) is str and bool(_HEX_PREFIX8_RE.match(v3_zb_pfx)),
        "ERR_BAD_SHA_PREFIX",
        "Zone-B digest prefixes must be 8 lowercase hex chars",
    )

    speedup_pinned_q = q_div(v1_q, v2_e2e_q)
    speedup_old_ref_q = q_div(_V1_OLD_REF_US_Q, v2_e2e_q)
    require_ok(
        abs(speedup_old_ref_q - v2_old_sp_q) <= Fraction(1, 10000),
        "ERR_SPEEDUP_MATH",
        "receipt speedup_vs_v1_249_32us inconsistent with 249.32 / v2_e2e",
    )

    v1_2dp = round_half_up_q(v1_q, 2)
    v2_cell_2dp = round_half_up_q(v2_cell_q, 2)
    v2_e2e_2dp = round_half_up_q(v2_e2e_q, 2)
    v3_cell_2dp = round_half_up_q(v3_cell_q, 2)
    v3_e2e_2dp = round_half_up_q(v3_e2e_q, 2)
    sp_pinned_3dp = round_half_up_q(speedup_pinned_q, 3)
    sp_old_3dp = round_half_up_q(speedup_old_ref_q, 3)

    require_ok(STALE_SILIKERNELS_SHORT not in readme_text, "ERR_STALE_SILIKERNELS_PIN", "stale 44eccad8 pin in README")
    require_ok(PINNED_SILIKERNELS_COMMIT in readme_text, "ERR_MISSING_COMMIT_PIN", "missing b3712675 full commit pin")
    require_ok(
        _FORBIDDEN_README_ENERGY_RE.search(readme_text) is None,
        "ERR_README_ENERGY_CLAIM",
        "uncontrolled laptop energy/watt claim detected in README",
    )

    # Strict per-bullet boundary-anchored verification (blocks digit appends/prepends and V1/V2/V3 swaps!)
    v1_line = _extract_unique_bullet(readme_text, "*V1 Inline SECDED Kernel")
    v2_line = _extract_unique_bullet(readme_text, "*V2 `16`-Warp `d=512` Fused Kernel")
    v3_line = _extract_unique_bullet(readme_text, "*V3 `16`-Warp `d=512` Frontier Kernel")

    _require_anchored_pattern(v1_line, rf"`{re.escape(v1_2dp)} us/tok`", "v1_2dp")
    _require_anchored_pattern(
        v1_line,
        rf"`uniform_cell_16warp_d512\.t8192_us_per_tok = {re.escape(active_pins['v1_t8192_us_per_tok'])}`",
        "v1_raw",
    )

    _require_anchored_pattern(v2_line, rf"`{re.escape(v2_cell_2dp)} us/tok`\]\([^\)]+\) CellOp", "v2_cell_2dp")
    _require_anchored_pattern(
        v2_line,
        rf"`cellop_only_12L_d512\.t8192_us_per_tok = {re.escape(active_pins['v2_cellop_t8192_us_per_tok'])}`",
        "v2_cell_raw",
    )
    _require_anchored_pattern(v2_line, rf"`{re.escape(v2_e2e_2dp)} us/tok`\]\([^\)]+\) E2E", "v2_e2e_2dp")
    _require_anchored_pattern(
        v2_line,
        rf"`full_e2e_fwht256_12L_d512\.t8192_us_per_tok = {re.escape(active_pins['v2_e2e_t8192_us_per_tok'])}`",
        "v2_e2e_raw",
    )
    _require_anchored_pattern(v2_line, rf"`{re.escape(sp_pinned_3dp)}×`", "v2_speedup_pinned")
    _require_anchored_pattern(
        v2_line,
        rf"`speedup_vs_v1_249_32us = {re.escape(active_pins['v2_speedup_vs_v1_249_32us'])}`",
        "v2_speedup_raw",
    )
    _require_anchored_pattern(v2_line, rf"`receipt_sha256 = {re.escape(v2_zb_pfx)}…`", "v2_zb_pfx")

    _require_anchored_pattern(v3_line, rf"`{re.escape(v3_cell_2dp)} us/tok`\]\([^\)]+\) `T=8192` CUDA-graph CellOp", "v3_cell_2dp")
    _require_anchored_pattern(
        v3_line,
        rf"`v3_d512\.cellop_graph_us_per_tok_8192 = {re.escape(active_pins['v3_cellop_graph_us_per_tok_8192'])}`",
        "v3_cell_raw",
    )
    _require_anchored_pattern(v3_line, rf"`{re.escape(v3_e2e_2dp)} us/tok`\]\([^\)]+\) `T=16384` E2E", "v3_e2e_2dp")
    _require_anchored_pattern(
        v3_line,
        rf"`v3_d512\.e2e_graph_us_per_tok_16384 = {re.escape(active_pins['v3_e2e_graph_us_per_tok_16384'])}`",
        "v3_e2e_raw",
    )
    _require_anchored_pattern(v3_line, rf"`receipt_sha256 = {re.escape(v3_zb_pfx)}…`", "v3_zb_pfx")

    require_ok(
        "Energy figures in those receipts are uncontrolled and provisional." in readme_text
        and "Uncontrolled Laptop & Timing Disclaimer" in readme_text
        and "first committed after the live laptop run ended" in readme_text
        and "paired contrast family uncorrected" in readme_text,
        "ERR_MISSING_UNCONTROLLED_DISCLAIMER",
        "missing required uncontrolled-laptop or statistical scope disclaimers in README",
    )

    summary: dict[str, Any] = {
        "schema": "silisense.readme_pins_check.v1",
        "status": "PASS",
        "silikernels_commit": PINNED_SILIKERNELS_COMMIT,
        "pins_checked": len(active_pins),
        "silikernels_receipts_read": silikernels_root is not None,
        "v2_speedup_vs_pinned_v1_3dp": sp_pinned_3dp,
        "v2_speedup_vs_249_32_3dp": sp_old_3dp,
        "pins": active_pins,
    }
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    summary["summary_sha256"] = hashlib.sha256(canon).hexdigest()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify SiliSense README pins against companion silikernels receipts.")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT.joinpath("README.md"), help="Path to README.md")
    parser.add_argument("--silikernels-root", type=Path, default=None, help="Optional companion silikernels checkout")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON report")
    args = parser.parse_args(argv)

    try:
        raw_readme = _read_regular_file_bytes_nofollow(args.readme, missing_code="ERR_MISSING_README")
        try:
            readme_text = raw_readme.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SiliSenseError("ERR_BAD_README", str(exc)) from exc
        report = verify_readme_pins(readme_text, silikernels_root=args.silikernels_root)
    except SiliSenseError as exc:
        sys.stderr.write(json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    else:
        sys.stdout.write(f"PASS pins={report['pins_checked']} summary_sha256={report['summary_sha256']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
