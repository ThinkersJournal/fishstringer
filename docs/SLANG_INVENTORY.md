# Slang inventory

Written 2026-11-06 by the fishstringer lane. Every figure was read from `git` objects at these refs, after a fetch:
**Fuel `5aa295281`, Vulkane `9cdc2d4`, Unpopped `d62fa45`** (each `origin/main`). Nothing here comes from a working tree.
Counts are of files at those refs; they are not claims about the present.

## 1. Where Slang lives

| Where | What | Size / version |
|---|---|---|
| Fuel `fuel-kernels-source/` | kernel sources + `compile.sh`, `hoist_extinst_imports.py`, `no_contraction.py`, `f64_dd_exp.slangh` | `kernels/` 180 files: 149 `.slang`, 1 `.slangh`, 27 `.glsl`, 2 `.py`, 1 `.sh`. Crate version follows Fuel's workspace (`0.41.6` in Fuel's `Cargo.toml`) |
| Vulkane `vulkane-kernels-source/` | the same kernels, copied from Fuel 2026-10-30 (Vulkane #105, commit `460634e`) | `kernels/` 177 files: 147 `.slang`, 27 `.glsl`, 2 `.py` (one is a test), 1 `.sh`. `0.21.5` on main; `0.21.1` latest on crates.io (per Vulkane's notice, not re-read from the registry by me) |
| Vulkane `vulkane-slang-emit/` | IR to Slang text, 1 `lib.rs`, 3 conformance tests; imported from Unpopped's `unpopped-slang` | `0.21.5` on main; `0.21.1` on crates.io |
| Vulkane `vulkane-slang-parse/` | `tree-sitter-slang` front end (`SLANG` `Frontend`, residue list); `tests/corpus.rs` pins 147 files | `0.21.5` on main; `0.21.1` on crates.io |
| Vulkane `docs/design/slang-parser-emitter-handover.md` | design note for the Unpopped-to-Vulkane move (9,849 B) | superseded in part by this move |
| Unpopped | `docs/ir-union-surveys/slang.md`, `docs/slang-corpus-sizing.md`, `docs/slang-naming-convention-note.md`, `docs/slang-second-blocker-sizing.md`, `crates/unpopped/examples/slang_corpus_report.rs`; in `crates/unpopped/src/convert.rs`, `parse_slang` / `SLANG` exist **only in the `#[cfg(test)]` module** with `tree-sitter-slang` as a dev-dependency (the shipped crate dropped Slang in 0.37.0) | no `crates/unpopped-slang` at this ref |
| Fuel docs | `docs/SLANG-SEAM-REQUIREMENTS.md` and Slang mentions in 50+ other files (kernel contracts, ROADMAP, CHANGELOG) | prose, not code |

**The PM's earlier brief named `crates/unpopped-slang` and `parse_slang` in Unpopped as live code. At `d62fa45` neither is
shipped code: the emitter and the parser are already Vulkane's.** The Unpopped pieces left are docs, an example, and a
test-only frontend.

## 2. Kernels: Fuel against Vulkane

Method: git blob ids of every file under `kernels/` at four refs: Fuel at the copy (`750cd2fc3`, the last Fuel
`origin/main` commit before Vulkane's import `460634e`; **assumed** to be the source ref, the import commit does not
name one), Fuel now (`5aa295281`), Vulkane's import (`460634e`) and Vulkane now (`9cdc2d4`). Fuel-at-copy against
Vulkane's import differs in exactly 3 files (`compile.sh`, `dequant_q4_km.slang`, `matmul.slang`: the import's own
edits, per its commit message: `--out-dir`, origin comments), so the import is not a clean base and Fuel's pre-copy
state is used. Positive control: the classes below sum to 181 = Fuel's 180 + Vulkane's 1 extra.

| Class | Count | Files |
|---|---|---|
| IDENTICAL at all refs | 144 | includes all 27 `.glsl` |
| FUEL-CHANGED since copy, Vulkane untouched | 26 | `layer_norm_last_dim*` (8), `rms_norm_last_dim*` (5), `reduce_last_dim*` (4), `softmax`, `softmax_bf16`, `softmax_f16`, `softmax_last_dim_backward*` (4), `matmul_q4_0_tiled`, `qmatvec_q4_0` |
| FUEL-ADDED since copy | 4 | `f64_dd_exp.slangh`, `no_contraction.py`, `qmatvec_q6_k.slang`, `qmatvec_q8_0.slang` |
| VULKANE-CHANGED since copy, Fuel untouched | 3 | `dequant_q4_km.slang`, `matmul.slang` (origin comments added by the import), `hoist_extinst_imports.py` (Vulkane #123) |
| VULKANE-ONLY | 1 | `test_hoist_extinst_imports.py` (Vulkane #116) |
| **BOTH CHANGED, DIFFERENT** | **3** | **`compile.sh`, `softmax_f64.slang`, `unary_f64.slang`** |

- **The 26 Fuel-changed** are, by Fuel's own log, the subgroup-combine stride fix (#554, row 134) plus later work; this is
  the fix for **Vulkane #133** (16 kernels with the `tid < n_sg` pattern), so Fuel's copy already carries a fix Vulkane lists
  as open. I have not yet read the 26 diffs line by line; the classification says only which side changed.
- **The 3 divergent:** `softmax_f64` and `unary_f64` were both fixed for `exp` on `double` independently (Fuel #550,
  #557, #568: shared double-double `poly_exp_d`, NoContraction; Vulkane #119, #132: `exp` garbage below 2^-1022).
  `compile.sh` differs because Vulkane added `--out-dir` and Fuel added the `no_contraction.py` step. These need a
  human decision; I will not merge them: `[ASK]` to the PM with both sides' diffs against the copy.

## 3. Consumers (who uses Slang today)

- **Fuel:** `fuel-vulkan-kernels` holds committed SPIR-V compiled from `fuel-kernels-source` by
  `compile.sh` and consumes it with `include_bytes!`; `fuel-vulkan-backend` and `fuel-vulkan-dispatch` use that SPIR-V
  and cite the `.slang` paths in 12+ kernel-contract files (`*.fkc.md`), `scripts/provenance.py`, `fuel-ir/tests/*`
  and `.github/workflows/lavapipe-vulkan.yml` (path filter). **Fuel does not depend on `fuel-kernels-source` as a
  crate: nothing lists it as a dependency except the workspace member list and its own version pin.** The coupling is
  by source path, so the switch to a published crate means rewiring the compile step, not just a `Cargo.toml` line.
- **Vulkane:** `vulkane-spirv-emit` (dev-dep on `vulkane-slang-emit`, `oracle_live.rs`); `vulkane-slang-emit` dev-dep on
  `vulkane-slang-parse`; `vulkane/tests/softmax_f64_live.rs` needs `slangc`; `ci.yml` has a pinned-`slangc` job.
  A planned `vulkane-hlsl-emit` will be a thin wrapper over `vulkane-slang-emit` (byte-identical bodies) and wants
  the speller functions kept `pub`; the list of what is API is to be agreed with Vulkane.
- **Unpopped:** `unpopped-conformance` no longer depends on a Slang crate (the three cells moved to
  `vulkane-slang-emit/tests/`).

## 4. Provenance and licences

- Fuel's `fuel-kernels-source/THIRD-PARTY-NOTICES.md` carries the llama.cpp / ggml licence (and candle) texts, ref
  `9d57ce456c`; Vulkane's copy exists too. `docs/provenance/FILES.tsv` in Fuel classifies 185 files under
  `fuel-kernels-source` against candle (the rows I sampled read "no lineage; 0 of N lines verbatim in candle"). `scripts/provenance.py` in Fuel
  names `index_select.slang` explicitly. **Not yet done:** tracing each file's origin and diffing each kernel family against
  its upstream (the quantized kernels `qmatvec_*`, `dequant_*`, `matmul_q4_0_tiled` derive from ggml; the flash-attention kernels are
  CireSnave's first-party). Every moved file keeps these notices and its SPDX header; the full per-file provenance
  table ships in the move PR, not here.
- The `tree-sitter-slang` grammar (MIT) is the only third-party code in the parser path.

## 5. The `.glsl` files (not ours)

27 files in each of Fuel's `fuel-kernels-source/kernels/` and Vulkane's `vulkane-kernels-source/kernels/`, identical on
both sides. **GLSL is Vulkane's; they are listed here and not moved.** Fuel's copy is therefore a GLSL source that
Fuel must get from Vulkane's crate when Fuel's `fuel-kernels-source` goes away: that is a Fuel-and-Vulkane
decision (`[ASK]`, section 7).

## 6. Move plan (steps; each its own PR; nothing deleted from Fuel or Vulkane by this lane)

1. **This PR:** charter + inventory (docs only).
2. **Receive the kernels.** Import the newer side per file, with history (path-limited import, blob-identity check
   against the source commit). Fuel's `5aa295281` for the 26 + 4 + the identical 144; Vulkane's for `dequant_q4_km`, `matmul`,
   `hoist_extinst_imports.py` and its test. The 3 divergent files wait on the `[ASK]`; Vulkane's comment-only edits to `dequant_q4_km`/`matmul` are kept.
3. **Kernel crate** at 0.0.0: name `[ASK]`, PM claims and publishes.
4. **Emitter and parser:** import `vulkane-slang-emit` and `vulkane-slang-parse` with history, adapt manifests, keep the
   conformance cells; PM publishes. Vulkane's `vulkane-hlsl-emit` then depends on the published emitter.
5. **Fuel and Vulkane** switch to the published crates and delete their copies (their lanes). Unpopped's Slang
   docs and corpus example: Unpopped's call whether they move; I propose they move with the corpus data.
6. **CI and protection:** the repo has `ci.yml`; branch protection waits on the first green CI run on `main`, which the
   PM sets and I report.

## 7. Open questions for the PM

1. Crate names (kernel, emitter, parser) and whether `fishstringer` becomes an umbrella.
2. The 3 divergent files (`compile.sh`, `softmax_f64`, `unary_f64`): which exp wins, and how are both compile.sh changes combined? Both sides' diffs against base available.
3. Who holds the 27 GLSL files after Fuel's `fuel-kernels-source` is gone: Vulkane's kernel crate, presumably.
4. Fuel's coupling is by source path, not by crate dependency (section 3): Fuel's lane needs to say how its
   `compile.sh` pipeline will consume a published source crate.
5. The Fuel-side fix for Vulkane #133 is in the 26 Fuel-changed files; Vulkane should close or re-scope #133 when it takes our crate.
