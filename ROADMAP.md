# fishstringer roadmap

Owner: the `fishstringer` lane. Priorities across the portfolio live in `C:\Projects\PORTFOLIO-ROADMAP.md`; this file is
this repo's own queue. Scope and rules: `docs/CHARTER.md`. Updated 2026-11-07 (main + 0.0.14).

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

## Next

| Row | What | Blocked on |
|---|---|---|
| 10 | Consumers wire `kernels/regen_gate.py` (pins slangc, regenerates, `spv_equiv.py` against committed `.spv`, exits non-zero on a difference) into their regeneration gate | Fuel's and Vulkane's lanes; the script is here |
| 11 | Handle live-test failures on kernels Fuel or Vulkane report (unary_f64, softmax_f64, the 26 stride-fix kernels) | a consumer report |
| 12 | Old crates `vulkane-slang-*` and `unpopped-slang`: the final redirect release | the PM and the owning lanes |
| 13 | Unpopped's Slang docs and corpus example move here or stay | Unpopped's call |
| 14 | `dequant_q4_km`: Fuel's committed `.spv` is not what `slangc` 2026.8 makes from Fuel's own source (see `docs/KERNEL_RECONCILIATION.md`); Fuel's regeneration replaced it, nothing to do here unless a live test objects | Fuel's live test |

## Open decision (PM)

slangc 2026.21 (newest, 2026-11-04) vs the 2026.8 the shipped `.spv` come from: of 153 kernels 93 identical, 39
equivalent, 21 different; every one of the 21 differs only in `OpExtInst` `FMax`/`FMin` becoming `NMax`/`NMin` (the
NaN-propagation behaviour of `max`/`min` changes; `spirv-val` passes). Not adopted for the shipped output until the PM
decides the version and the consumers' live tests agree. Measured 2026-11-09, `compile.sh` + `regen_gate.py`.

## Not doing

- **Triton and Mojo.** Parked: there is no way to test either yet (PORTFOLIO-ROADMAP L2/L3). Reopen when there is.
- **GLSL, HLSL, WGSL, SPIR-V as languages** (Vulkane's), **CUDA** (Baracuda's), **Unpopped's IR**, and Fuel's and Vulkane's
  dispatch and runtime code. `spv_equiv.py` only compares the SPIR-V that our `compile.sh` produces.
- **Proving kernels numerically right.** `spv_equiv.py` is a structural check, not a proof, and runs nothing; numeric
  agreement is the consumer's live test on a device.
- **Algebraic equivalence** (reassociating adds, folding constants, swapping commutative operands). Reported as
  "different"; a consumer who cares runs its live test.
- **Publishing.** The PM publishes; this lane never does.
