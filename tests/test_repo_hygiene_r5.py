"""AST & Repository Hygiene Gate for SiliSense (0 bare asserts, 0 unseeded RNG, 0 leaks)."""

from __future__ import annotations

import ast
from pathlib import Path
import re
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


class TestRepoHygieneR5(unittest.TestCase):
    def test_01_ast_zero_bare_asserts_zero_unseeded_rng_and_zero_shell_true(self) -> None:
        py_files = sorted(p for p in REPO_ROOT.rglob("*.py") if ".git" not in p.parts)
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

    def test_02_zero_path_leaks_zero_audit_tags_and_zero_binary_artifacts(self) -> None:
        home_tok = "/" + "home/"
        for p in sorted(REPO_ROOT.rglob("*")):
            if ".git" in p.parts:
                continue
            self.assertNotEqual(p.name, "__pycache__", f"Bytecode cache found: {p}")
            if p.is_file():
                self.assertNotIn(p.suffix.lower(), _FORBIDDEN_EXTS, f"Forbidden artifact: {p.name}")
                txt = p.read_text(encoding="utf-8")
                self.assertNotIn(home_tok, txt, f"Home path leak in {p.relative_to(REPO_ROOT)}")
                self.assertIsNone(_AUDIT_TAG_RE.search(txt), f"Internal audit tag in {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    unittest.main()
