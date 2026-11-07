#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for no_contraction._checked_spv: the output directory is the caller's, so it may be anywhere.

A consumer (Fuel, Vulkane) runs compile.sh from the published crate with --out-dir in its own tree,
which is never under this script's repository root. 0.0.12 refused that ("is outside the repository")
and compile.sh exited 1 at unary_f64. Text/filesystem only: no SDK tools are needed.

Run:  python3 -m unittest test_no_contraction   (from this directory)
"""

import tempfile
import unittest
from pathlib import Path

import no_contraction as nc


class CheckedSpv(unittest.TestCase):
    def test_spv_outside_the_script_tree_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            spv = Path(d) / "unary_f64.spv"
            spv.write_bytes(b"\x03\x02\x23\x07")
            self.assertEqual(nc._checked_spv(str(spv)), spv.resolve())

    def test_non_spv_suffix_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "notes.txt"
            other.write_text("x")
            with self.assertRaises(SystemExit):
                nc._checked_spv(str(other))

    def test_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                nc._checked_spv(str(Path(d) / "absent.spv"))


if __name__ == "__main__":
    unittest.main()
