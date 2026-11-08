# fishstringer-kernels-source

Slang GPU kernel sources and the script that compiles them to SPIR-V.

History: imported with its git history from Fuel's `fuel-kernels-source` (Fuel `5aa295281`), then
reconciled with Vulkane's `vulkane-kernels-source` (Vulkane `9cdc2d4`), which was itself copied from Fuel on
2026-10-30. How each file was chosen: `docs/SLANG_INVENTORY.md` and `docs/KERNEL_RECONCILIATION.md` at the repository root.

Compile: `kernels/compile.sh [--out-dir DIR] [--src-dir DIR] [files...]`; needs `slangc`
(Vulkan SDK or the Slang releases). GLSL is not in this crate: it is Vulkane's (`vulkane-kernels-source`).
Default output is `spv/` next to `kernels/` (git-ignored); the default source directory is the script's own.
A missing tool, an unknown option or a failed compile exits non-zero.

- `compile.sh --print-manifest` prints `<source><TAB><kind><TAB><output.spv>` for every output (a source with a
  `// entries:` directive gives several), using the same rules as the compile step and no tools. A consumer can check
  that every committed `.spv` has a source and every source has its `.spv`.
- `compile.sh --print-toolchain` prints the `slangc` and Python versions, for a provenance record
  (SPIR-V depends on the compiler: pin an exact toolchain).

## Checking regenerated SPIR-V: `kernels/spv_equiv.py`

SPIR-V is not byte-stable across `slangc` builds (debug records, id numbering and the order of independent
instructions change), so a byte-compare of a regenerated `.spv` against a committed one fails even when nothing that
matters changed. `spv_equiv.py` answers the question a gate actually has:

    python spv_equiv.py A.spv B.spv            # needs spirv-dis (Vulkan SDK) on PATH
    python spv_equiv.py A.spvasm B.spvasm      # spirv-dis text, no tools needed
    python spv_equiv.py --explain A.spv B.spv  # on a difference, which part (module-level / functions)

Exit 0: equivalent. 1: different, meaning *not shown equivalent*. 2: cannot decide (a module with `OpPhi`,
`OpFunctionCall` or decoration groups, a missing file or tool); a refusal is never a pass. It ignores debug records,
id numbers and names, the order of instructions that cannot interact, and repeated identical pure instructions. It
keeps every opcode, type, constant, decoration, binding, execution mode, entry point, the control-flow graph, the order
of stores, atomics and barriers within a block, and which store a load can observe. It is structural, not a proof of
semantic equality, and runs nothing: two commutative operands swapped, or a different grouping of adds, are reported
different, and numeric agreement stays the consumer's live test. Measured 2026-11-07 on fishstringer 0.0.13 output
against Fuel's previously committed files: 152 of 153 equivalent, 1 different (`dequant_q4_km`, whose committed file is
not what `slangc` 2026.8 makes from Fuel's own source). Tests: `python3 -m unittest test_spv_equiv` from `kernels/`.

Third-party origins: `THIRD-PARTY-NOTICES.md`.
