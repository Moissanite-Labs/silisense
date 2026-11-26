"""AST & Repository Hygiene Gate for SiliSense (0 bare asserts, 0 unseeded RNG, 0 leaks)."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_FORBIDDEN_EXTS = {
    ".so",
    ".o",
    ".wav",
    ".s32le",
    ".yuyv",
    ".grey",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
}
_AUDIT_TAG_RE = re.compile(r"\b" + "AU" + r"DIT-[A-Za-z0-9_-]+")
_PRIVATE_SLUG_RE = re.compile(r"\b(" + "expe" + r"dition|" + "alet" + r"heia)\b", re.IGNORECASE)


def _repo_files() -> list[Path]:
    if (REPO_ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc.returncode == 0:
            return [REPO_ROOT / ln for ln in proc.stdout.splitlines() if ln.strip() and (REPO_ROOT / ln).is_file()]
    return sorted(p for p in REPO_ROOT.rglob("*") if ".git" not in p.parts and p.is_file())


class TestRepoHygieneR5(unittest.TestCase):
    def test_01_ast_zero_bare_asserts_zero_unseeded_rng_and_zero_shell_true(self) -> None:
        py_files = [p for p in _repo_files() if p.suffix == ".py"]
        self.assertGreaterEqual(len(py_files), 12)

        for py_path in py_files:
            src = py_path.read_text(encoding="utf-8")
            self.assertNotIn("\r\n", src, f"CRLF line ending in {py_path.relative_to(REPO_ROOT)}")
            tree = ast.parse(src, filename=str(py_path))
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert, f"Bare assert forbidden in {py_path.name}")
                if isinstance(node, ast.keyword) and node.arg == "shell":
                    self.assertFalse(
                        isinstance(node.value, ast.Constant) and node.value.value is True,
                        f"shell=True forbidden in {py_path.name}",
                    )
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    elif isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    if func_name in ("Random", "default_rng", "seed"):
                        has_seed_arg = (len(node.args) >= 1) or (len(node.keywords) >= 1)
                        self.assertTrue(has_seed_arg, f"Unseeded RNG call {func_name}() in {py_path.name}")

    def test_02_zero_path_leaks_zero_audit_tags_zero_private_slugs_and_zero_binaries(self) -> None:
        home_unix = "/" + "home/"
        home_mac = "/" + "Users/"
        for d in REPO_ROOT.rglob("__pycache__"):
            if ".git" not in d.parts and ".venv" not in d.parts:
                self.fail(f"Bytecode cache found: {d.relative_to(REPO_ROOT)}")

        for p in _repo_files():
            rel = str(p.relative_to(REPO_ROOT))
            self.assertNotIn(p.suffix.lower(), _FORBIDDEN_EXTS, f"Forbidden artifact: {rel}")
            raw_bytes = p.read_bytes()
            self.assertFalse(raw_bytes.startswith(b"\x7fELF"), f"ELF binary forbidden: {rel}")
            txt = raw_bytes.decode("utf-8")
            self.assertNotIn(home_unix, txt, f"Unix home path leak in {rel}")
            self.assertNotIn(home_mac, txt, f"macOS home path leak in {rel}")
            self.assertIsNone(_AUDIT_TAG_RE.search(txt), f"Internal audit tag in {rel}")
            self.assertIsNone(_PRIVATE_SLUG_RE.search(txt), f"Private repo slug in {rel}")


if __name__ == "__main__":
    unittest.main()
