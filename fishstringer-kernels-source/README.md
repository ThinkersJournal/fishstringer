# fishstringer-kernels-source

Slang GPU kernel sources and the script that compiles them to SPIR-V.

History: imported with its git history from Fuel's `fuel-kernels-source` (Fuel `5aa295281`), then
reconciled with Vulkane's `vulkane-kernels-source` (Vulkane `9cdc2d4`), which was itself copied from Fuel on
2026-10-30. How each file was chosen: `docs/SLANG_INVENTORY.md` and `docs/KERNEL_RECONCILIATION.md` at the repository root.

Compile: `kernels/compile.sh [--out-dir DIR] [files...]`; needs `slangc` and `glslangValidator`
(Vulkan SDK). Default output is `spv/` next to `kernels/` (git-ignored).

Third-party origins: `THIRD-PARTY-NOTICES.md`.
