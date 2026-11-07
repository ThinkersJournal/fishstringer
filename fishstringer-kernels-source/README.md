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

Third-party origins: `THIRD-PARTY-NOTICES.md`.
