# fishstringer roadmap

Owner: the `fishstringer` lane. Priorities across the portfolio live in `C:\Projects\PORTFOLIO-ROADMAP.md`; this file is
this repo's own queue. Scope and rules: `docs/CHARTER.md`. Updated 2026-11-10 (main + 0.0.17); the shipped `.spv` stay at slangc 2026.8 (PM ruling 2026-11-10).

## Done

| Row | What | Where |
|---|---|---|
| 1 | Charter and the inventory of every Slang file in the portfolio | `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md` (#1) |
| 2 | Kernel sources moved here with history; Fuel and Vulkane copies reconciled | `fishstringer-kernels-source`, `docs/KERNEL_RECONCILIATION.md` (#2, #7) |
| 3 | Parser and emitter crates moved here | `fishstringer-slang-parse` (#4), `fishstringer-slang-emit` (#5) |
| 4 | Publish plan; CI on the whole workspace, pinned `slangc` job, manifest and script checks; branch protection | `docs/PUBLISH_PLAN.md`, `.github/workflows/ci.yml` (#3, #6) |
| 5 | `compile.sh` flags for consumers (`--src-dir`, `--print-manifest`, `--print-toolchain`, hard fail on a missing tool); the 27 `.glsl` removed (GLSL is Vulkane's) | #8, #9 |
| 6 | Published 0.0.12 and 0.0.13 (the PM publishes); `no_contraction.py` accepts an `--out-dir` outside the repo | #10 |
| 7 | Consumers switched: Vulkane on `~0.0.13` and deleting its copies (vulkane#148, #149); Fuel regenerated its 153 `.spv` from 0.0.13 (fuel#636, byte-equal to our tree) | consumer repos |
| 8 | Provenance: line overlap against llama.cpp (0 shared lines) and candle (0 shared lines in 150 files) | `docs/KERNEL_RECONCILIATION.md` |
| 9 | `spv_equiv.py`: a SPIR-V equivalence check that ignores debug records, id numbering and the order of instructions that cannot interact, so a consumer can gate a regeneration without a byte-compare | `kernels/spv_equiv.py`, `kernels/test_spv_equiv.py` (0.0.14) |
| 15 | Unpopped's three Slang sizing docs and the corpus harness moved here (`fishstringer-slang-parse/examples/slang_corpus_report.rs`); `regen_gate.py` (pins slangc, regenerates, compares) | #12, #13 |
| 16 | NaN guards: `isnan` select in `unary*` (relu), `reduce*`, `reduce_last_dim*` (incl. the Wave path), because slangc 2026.8's `FMax`/`FMin` drop NaN (Fuel measured 72 of 72 cases on Vulkan) and Fuel pins NaN-propagating; regenerated at 2026.8, 12 kernels differ on purpose (0.0.17) | PR for 0.0.17 |

## Next

| Row | What | Blocked on |
|---|---|---|
| 10 | Consumers wire `kernels/regen_gate.py` (pins slangc, regenerates, `spv_equiv.py` against committed `.spv`, exits non-zero on a difference) into their regeneration gate | Fuel's lane (Vulkane commits no `.spv`); the script is here (#12) |
| 11 | Handle live-test failures on kernels Fuel or Vulkane report (unary_f64, softmax_f64, the 26 stride-fix kernels) | a consumer report |
| 12 | `vulkane-slang-*` redirect release: Vulkane prepares 0.22.x re-exporting `fishstringer-*`, the PM publishes (`unpopped-slang` 0.35.4 redirect is published) | Vulkane's PR, the PM |
| 13 | Re-verify the 17 `n_sg` kernels on a subgroup-size-8 device (fishstringer#14; the source already has the stride fix) | Vulkane's Lavapipe test |
| 14 | `dequant_q4_km`: Fuel's committed `.spv` is not what `slangc` 2026.8 makes from Fuel's own source (see `docs/KERNEL_RECONCILIATION.md`); Fuel's regeneration replaced it, nothing to do here unless a live test objects | Fuel's live test |

## Open decision (PM)

slangc 2026.21 (newest, 2026-11-04) is NOT adopted: against the 2026.8 output, 93 of 153 kernels identical, 39 equivalent,
21 different, all only `FMax`/`FMin` -> `NMax`/`NMin` (NaN suppressed; Fuel needs NaN-propagating, so the guards above come
first). `NMax`/`NMin` is right only for a NaN-suppressing max/min, and no kernel needs that (Fuel decomposes it). Re-measure
with `regen_gate.py` before any bump; the shipped `.spv` stay at 2026.8 (PM 2026-11-10).

## Not doing

- **Triton and Mojo.** Parked: there is no way to test either yet (PORTFOLIO-ROADMAP L2/L3). Reopen when there is.
- **GLSL, HLSL, WGSL, SPIR-V as languages** (Vulkane's), **CUDA** (Baracuda's), **Unpopped's IR**, and Fuel's and Vulkane's
  dispatch and runtime code. `spv_equiv.py` only compares the SPIR-V that our `compile.sh` produces.
- **Proving kernels numerically right.** `spv_equiv.py` is a structural check, not a proof, and runs nothing; numeric
  agreement is the consumer's live test on a device.
- **Algebraic equivalence** (reassociating adds, folding constants, swapping commutative operands). Reported as
  "different"; a consumer who cares runs its live test.
- **Publishing.** The PM publishes; this lane never does.
