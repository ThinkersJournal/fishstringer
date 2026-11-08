# Kernel reconciliation (Fuel and Vulkane copies)

Dated record, written 2026-11-07, of how `fishstringer-kernels-source/kernels/` was assembled. Refs: Fuel at the
copy `750cd2fc3`, Fuel now `5aa295281`, Vulkane import `460634e`, Vulkane now `9cdc2d4`. Classification method and
counts: `SLANG_INVENTORY.md` section 2. Ruling (PM 2026-11-06): Fuel-newer files come from Fuel, Vulkane-newer files
from Vulkane, and files both sides changed are reconciled so that both sides' tests would pass.

| Group | Files | Taken from | Check |
|---|---|---|---|
| Identical everywhere | 144 | Fuel (history imported) | blob ids equal Fuel `5aa295281` for all 180 imported files (0 mismatches) |
| Fuel-changed since copy | 26 | Fuel | as above |
| Fuel-added since copy | 4 | Fuel | as above |
| Vulkane-newer, comment only | `dequant_q4_km.slang`, `matmul.slang` | Fuel's file plus Vulkane's origin-comment hunks (6 and 2 lines), applied as a patch; Vulkane's CRLF line endings not carried | the patch applied without rejects; `THIRD-PARTY-NOTICES.md` refers to these comments |
| Vulkane-newer | `hoist_extinst_imports.py` and `test_hoist_extinst_imports.py` | Vulkane (LF line endings) | Vulkane's 9 characterisation tests pass against both Vulkane's version and Fuel's original (`python -m unittest`, 9 tests each), so the refactor is behaviour-preserving as far as those tests see |
| Both changed | `compile.sh` | three-way merge of the two edits (`git merge-file`, 0 conflicts): Vulkane's `--out-dir` and Fuel's `no_contraction_if_requested` step | `bash -n`; `compile.sh --out-dir <inside repo> unary_f64.slang softmax_f64.slang` built both; `spirv-val --target-env vulkan1.2` passes; `unary_f64.spv` carries 1543 `NoContraction` decorations |
| Both changed | `unary_f64.slang`, `softmax_f64.slang` | Fuel's versions (shared `f64_dd_exp.slangh`) | see below |

## Why Fuel's exp wins for `unary_f64` and `softmax_f64`

Both sides fixed the same two defects, differently; Fuel's version covers both:

1. **Subnormal results of `exp` (Vulkane #132 / #106):** `pow2_int(k)` is valid only for `k` in [-1022, 1023]; below that
   the result was garbage. Vulkane guards `k < -1021` and scales in two steps through 2^100; Fuel guards `k < -1022`
   (the exact validity bound) and scales through 2^200 (`f64_dd_exp.slangh`). Checked on the scaling step alone: for
   `k` from -1075 to -1022 and 2000 random mantissas in [0.7071, 1.4142] per `k` (108,000 cases), both shifts equal the
   correctly rounded `ldexp` result (0 mismatches each, Python doubles). This does not run the Slang kernel.
2. **Dropped subgroup partials on narrow subgroups (Vulkane #133 and the `softmax_f64` part of #106):** Vulkane combines the
   partials serially on thread 0; Fuel strides `for (i = tid; i < n_sg; i += lane_count)` then one `WaveActive*`. Both are
   correct on an 8-lane device by construction; Fuel's pattern is the one used by the other 26 Fuel-changed kernels, so
   keeping it keeps the family uniform.

Fuel's `softmax_f64` also carries a tighter live test (tolerance 1e-13, a `[3, 1024]` row), and `unary_f64` carries
NoContraction, which Vulkane's copy lacks. **Not run here:** no GPU. `slangc` compiled both files and `spirv-val`
passes (above). The kernels' live tests (Fuel's `vulkan_dispatch_softmax_last_dim_f64` and `unary_f64` tests, Vulkane's
`tests/softmax_f64_live.rs`) have not been run against this crate; the Fuel and Vulkane lanes confirm them when they
switch to it. **This reconciliation is therefore "Fuel's version, shown to subsume Vulkane's two fixes by reading and
by the arithmetic check", not "both test suites passed".**

## Open

- Vulkane's `softmax_f64_live.rs` expects `exp` below 2^-1022 and a Lavapipe row sum; if it fails against Fuel's kernel
  it is a finding to bring back here, not to patch in Vulkane.
- **The 26 Fuel-changed diffs (read 2026-11-07, Fuel at the copy `750cd2fc3` against Fuel `5aa295281`):** 322 changed lines in
  total (at most 28 per file). Six were read in full (`layer_norm_last_dim_backward`, `rms_norm_last_dim_f16`,
  `reduce_last_dim`, `softmax`, `matmul_q4_0_tiled`, `qmatvec_q4_0`); all 26 were then checked line by line against one
  pattern by script. **320 of 322 changed lines fit it**: a `groupshared` array is enlarged (32 to 64, 16 to 32), and the
  clamped single read of the cross-subgroup partials `(tid < n_sg) ? p[tid] : identity` becomes a strided loop
  `for (sgi = tid; sgi < n_sg; sgi += WaveGetLaneCount())` accumulating with the kernel's own combine (sum, max or the
  `op` switch) before one `WaveActive*`; comments are updated to match. The other 2 lines are the array-size parameter of a
  helper in `rms_norm_last_dim_backward` (`[32]` to `[64]`), the same resize. So this is Fuel's row-134 fix (Vulkane #133's
  pattern) and nothing else. Compiled: `compile.sh` built all 26 and `spirv-val --target-env vulkan1.2` passes on all 26
  (0 failures; local slangc 2026.8, not the CI-pinned 2026.20). **Not shown:** that the kernels are numerically right on a
  narrow-subgroup device; that is Fuel's Lavapipe leg and live tests.
- Per-file provenance beyond `THIRD-PARTY-NOTICES.md`: **partly done 2026-11-07.** Same normalised-line method as
  Vulkane's (trimmed lines over 25 characters, comments dropped) over all 177 `.slang`, `.slangh` and `.glsl` files here
  (so including the 26 Fuel-changed and 4 Fuel-added), against llama.cpp `ggml/src/ggml-quants.c` and
  `ggml/src/ggml-impl.h` at `9d57ce456c` (1,784 distinct reference lines): **0 shared lines in every file.** Control: a
  50-line slice of the reference compared with itself finds 50 of 50. Limits: it finds copied lines, not translation
  (the Q6_K and Q4_K kernels are translations and are dispositioned in `THIRD-PARTY-NOTICES.md`); it does not cover
  llama.cpp's `ggml-vulkan/vulkan-shaders/` (where Vulkane found the one sigmoid line in `unary.slang`).
  **Candle, re-run 2026-11-07 on all 150 `.slang`/`.slangh` files (so including the 30 Fuel added or changed after
  the copy), same method:** against candle `cfcbd7186d` (176 `.cu`, `.cuh`, `.metal` and `.wgsl` files, 11,214 distinct
  reference lines): **0 shared lines in every file.** Control: 50 reference lines compared with themselves find 50 of 50.
  It cannot see translation (a line rewritten into Slang syntax shares no text), only copied lines.
