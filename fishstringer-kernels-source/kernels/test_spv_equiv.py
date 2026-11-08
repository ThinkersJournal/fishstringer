#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for spv_equiv: the SPIR-V equivalence check that replaces a byte-compare gate.

Fixtures (tests/spv_equiv/*.spvasm) are `spirv-dis` text of the 8 kernels whose regenerated SPIR-V differs from
Fuel's committed copy in instruction order / id numbering (fishstringer 0.0.12 vs Fuel e1288c006, both slangc 2026.8):
`<kernel>.old.spvasm` is Fuel's committed file, `<kernel>.new.spvasm` is compile.sh's output. Text in, no SDK tool
is needed. One of the 8 (dequant_q4_km) differs in its arithmetic, not only in order; see the test for it.

Run:  python3 -m unittest test_spv_equiv   (from this directory)
"""

import re
import tempfile
import unittest
from pathlib import Path

import spv_equiv as eq

FIX = Path(__file__).resolve().parent / "tests" / "spv_equiv"
EQUIVALENT = ["arg_reduce_last_dim_bf16", "arg_reduce_last_dim_f16", "arg_reduce_last_dim_f32",
              "arg_reduce_last_dim_f64", "flash_attention", "matmul", "silu_backward"]


def fx(name: str, which: str) -> str:
    return (FIX / f"{name}.{which}.spvasm").read_text(encoding="utf-8")


class ReorderedKernels(unittest.TestCase):
    def test_each_fixture_is_equivalent_to_itself(self):
        for name in EQUIVALENT + ["dequant_q4_km"]:
            for which in ("old", "new"):
                with self.subTest(name=name, which=which):
                    self.assertTrue(eq.equivalent(fx(name, which), fx(name, which)))

    def test_text_compare_fails_on_these_pairs_which_is_why_the_tool_exists(self):
        for name in EQUIVALENT:
            with self.subTest(name=name):
                self.assertNotEqual(fx(name, "old"), fx(name, "new"))

    def test_reordered_kernels_are_equivalent(self):
        for name in EQUIVALENT:
            with self.subTest(name=name):
                self.assertTrue(eq.equivalent(fx(name, "old"), fx(name, "new")), eq.explain(fx(name, "old"), fx(name, "new")))

    def test_dequant_q4_km_is_not_shown_equivalent(self):
        # Fuel's committed file and the 0.0.13 output differ in the address arithmetic (constants 4 vs 9, a
        # different grouping of adds): not an order difference. The tool does no algebra, so it must say "different".
        self.assertFalse(eq.equivalent(fx("dequant_q4_km", "old"), fx("dequant_q4_km", "new")))


def renumber(text: str) -> str:
    """Rename every %id to a fresh %rN, in order of first appearance, and drop all debug records."""
    kept = [ln for ln in text.splitlines() if not re.match(r"\s*Op(Name|MemberName|Source\w*|String|Line)\b", ln)]
    ids = {}
    def sub(m):
        return ids.setdefault(m.group(0), f"%r{len(ids)}")
    return "\n".join(re.sub(r"%[A-Za-z0-9_]+", sub, ln) for ln in kept) + "\n"


class IgnoredDifferences(unittest.TestCase):
    def test_renamed_ids_and_dropped_debug_records(self):
        for name in EQUIVALENT:
            with self.subTest(name=name):
                self.assertTrue(eq.equivalent(fx(name, "new"), renumber(fx(name, "new"))))

    def test_reversed_declaration_order_of_decorations(self):
        text = fx("matmul", "new")
        lines = text.splitlines()
        deco = [i for i, ln in enumerate(lines) if re.match(r"\s*OpDecorate\b", ln)]
        self.assertGreater(len(deco), 2)
        shuffled = lines[:]
        for i, j in zip(deco, reversed(deco)):
            shuffled[i] = lines[j]
        self.assertNotEqual(lines, shuffled)
        self.assertTrue(eq.equivalent(text, "\n".join(shuffled) + "\n"))


def mutate(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    assert n == 1, f"mutation anchor not found: {pattern}"
    return out


class RealDifferencesAreCaught(unittest.TestCase):
    base = fx("silu_backward", "new")

    def test_changed_opcode(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"OpFSub", "OpFAdd")))

    def test_changed_constant(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"(%float_1 = OpConstant %float) 1", r"\1 2")))

    def test_changed_operand(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"(OpFMul %float) %35 %187", r"\1 %187 %184")))

    def test_changed_local_size(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"LocalSize 256 1 1", "LocalSize 128 1 1")))

    def test_changed_entry_point_name(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r'"main"', '"other"')))

    def test_dropped_binding_decoration(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"^\s*OpDecorate %bwd_x Binding \d+\n", "")))

    def test_changed_branch_target(self):
        self.assertFalse(eq.equivalent(self.base, mutate(self.base, r"OpBranchConditional %26 %9 %10", "OpBranchConditional %26 %10 %9")))

    def test_swapped_stores_in_one_block(self):
        text = fx("arg_reduce_last_dim_bf16", "new")
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            if re.match(r"\s*OpStore\b", lines[i]) and re.match(r"\s*OpStore\b", lines[i + 1]) and lines[i] != lines[i + 1]:
                swapped = lines[:]
                swapped[i], swapped[i + 1] = lines[i + 1], lines[i]
                self.assertFalse(eq.equivalent(text, "\n".join(swapped) + "\n"))
                return
        self.fail("no two adjacent different OpStore lines in the fixture")

    def test_load_moved_across_a_store(self):
        text = fx("arg_reduce_last_dim_bf16", "new")
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            if re.match(r"\s*OpStore\b", lines[i]) and re.search(r"= OpLoad\b", lines[i + 1]):
                moved = lines[:]
                moved[i], moved[i + 1] = lines[i + 1], lines[i]
                self.assertFalse(eq.equivalent(text, "\n".join(moved) + "\n"))
                return
        self.fail("no store directly followed by a load in the fixture")


class Refusals(unittest.TestCase):
    def test_phi_is_refused_not_passed(self):
        text = fx("silu_backward", "new").replace("OpReturn", "%p = OpPhi %uint %uint_0 %4\n               OpReturn")
        with self.assertRaises(eq.Undecidable):
            eq.equivalent(text, text)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.spvasm"
            b = Path(d) / "b.spvasm"
            a.write_text(fx("silu_backward", "old"), encoding="utf-8")
            b.write_text(fx("silu_backward", "new"), encoding="utf-8")
            self.assertEqual(eq.main([str(a), str(b)]), 0)
            b.write_text(mutate(fx("silu_backward", "new"), r"OpFSub", "OpFAdd"), encoding="utf-8")
            self.assertEqual(eq.main([str(a), str(b)]), 1)
            self.assertEqual(eq.main([str(a), str(Path(d) / "missing.spvasm")]), 2)
            self.assertEqual(eq.main([str(a)]), 2)


if __name__ == "__main__":
    unittest.main()
