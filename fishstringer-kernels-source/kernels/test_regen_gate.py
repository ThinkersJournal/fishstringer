#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for regen_gate: the regeneration gate a consumer drops in. Uses the spv_equiv fixtures as `.spvasm` text, so
no slangc or SDK tool is needed. The toolchain pin check is exercised with a stubbed `compile.sh --print-toolchain`.

Run:  python3 -m unittest test_regen_gate   (from this directory)
"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import regen_gate as gate

FIX = Path(__file__).resolve().parent / "tests" / "spv_equiv"
EQUIVALENT = ["arg_reduce_last_dim_bf16", "arg_reduce_last_dim_f16", "arg_reduce_last_dim_f32",
              "arg_reduce_last_dim_f64", "flash_attention", "matmul", "silu_backward"]


def stage(root: Path, which: str, names, suffix=".spvasm") -> Path:
    d = root / which
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = "old" if which == "committed" else "new"
        (d / f"{n}{suffix}").write_text((FIX / f"{n}.{src}.spvasm").read_text(encoding="utf-8"), encoding="utf-8")
    return d


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = gate.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


class Compare(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_reordered_kernels_pass(self):
        c, r = stage(self.root, "committed", EQUIVALENT), stage(self.root, "regen", EQUIVALENT)
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 0, out)
        self.assertIn("equivalent: 7", out)

    def test_byte_identical_is_reported_identical(self):
        c = stage(self.root, "committed", ["matmul"])
        r = self.root / "regen"
        r.mkdir()
        (r / "matmul.spvasm").write_bytes((c / "matmul.spvasm").read_bytes())
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual((rc, "identical: 1" in out, "equivalent: 0" in out), (0, True, True), out)

    def test_a_real_difference_fails_and_is_named(self):
        names = EQUIVALENT + ["dequant_q4_km"]
        c, r = stage(self.root, "committed", names), stage(self.root, "regen", names)
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 1, out)
        self.assertIn("different: dequant_q4_km", out)

    def test_regenerated_kernel_with_nothing_committed_fails(self):
        c = stage(self.root, "committed", ["matmul"])
        r = stage(self.root, "regen", ["matmul", "silu_backward"])
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 1, out)
        self.assertIn("missing: silu_backward", out)

    def test_committed_kernel_no_longer_regenerated_fails(self):
        c = stage(self.root, "committed", ["matmul", "silu_backward"])
        r = stage(self.root, "regen", ["matmul"])
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 1, out)
        self.assertIn("extra: silu_backward", out)

    def test_undecidable_is_refused_not_passed(self):
        c, r = stage(self.root, "committed", ["silu_backward"]), stage(self.root, "regen", ["silu_backward"])
        p = r / "silu_backward.spvasm"
        p.write_text(p.read_text(encoding="utf-8").replace("OpReturn", "%p = OpPhi %uint %uint_0 %4\n OpReturn"), encoding="utf-8")
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 2, out)
        self.assertIn("undecidable: silu_backward", out)

    def test_report_file_is_written(self):
        c, r = stage(self.root, "committed", ["matmul"]), stage(self.root, "regen", ["matmul"])
        rep = self.root / "report.txt"
        run_cli("--committed", c, "--regen-dir", r, "--report", rep)
        self.assertIn("compared: 1", rep.read_text(encoding="utf-8"))

    def test_spv_and_spvasm_names_match_by_stem(self):
        c = stage(self.root, "committed", ["matmul"], suffix=".spv")
        r = stage(self.root, "regen", ["matmul"], suffix=".spvasm")
        # a .spv holding text cannot be disassembled: refused (2), never silently passed
        rc, out, _ = run_cli("--committed", c, "--regen-dir", r)
        self.assertEqual(rc, 2, out)


class Pin(unittest.TestCase):
    def stub(self, stdout, rc=0):
        return mock.patch.object(gate.subprocess, "run", return_value=SimpleNamespace(returncode=rc, stdout=stdout, stderr=""))

    def test_matching_version_passes(self):
        with self.stub("slangc: 2026.21\npython: Python 3.15.2\n"):
            self.assertIsNone(gate.check_pin("2026.21"))

    def test_other_version_is_refused(self):
        with self.stub("slangc: 2026.8\npython: Python 3.15.2\n"):
            self.assertIn("pinned", gate.check_pin("2026.21"))

    def test_prefix_of_a_version_is_not_a_match(self):
        with self.stub("slangc: 2026.21.1\n"):
            self.assertIsNotNone(gate.check_pin("2026.21"))

    def test_failing_toolchain_query_is_refused(self):
        with self.stub("", rc=1):
            self.assertIn("failed", gate.check_pin("2026.21"))


class Cli(unittest.TestCase):
    def test_toolchain_run_requires_a_pin(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = run_cli("--committed", d)
        self.assertEqual(rc, 2)
        self.assertIn("--slangc-version", err)

    def test_pin_mismatch_exits_2_without_compiling(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(gate, "check_pin", return_value="slangc is '2026.8', pinned '2026.21'"):
            rc, _, err = run_cli("--committed", d, "--slangc-version", "2026.21")
        self.assertEqual(rc, 2)
        self.assertIn("not pinned", err)

    def test_missing_committed_dir_exits_2(self):
        rc, _, _ = run_cli("--committed", "no/such/dir", "--regen-dir", ".")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
