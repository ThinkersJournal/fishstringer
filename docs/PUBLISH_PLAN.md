# Publish plan for the fishstringer crates

**Status 2026-11-07 (after #2, #4, #5 landed): all three crates are in `main` at workspace version 0.0.9, with their history
(#2 squash-merged, history restored by #7; #4 and #5 merge commits). Nothing is published yet; publishing is the PM's.**

Draft 2026-11-07 for the PM, who publishes every release (lanes never publish). Facts read on 2026-11-07: crates.io
API (max version per crate) and Vulkane `origin/main` `9cdc2d4` manifests. Versions and dependency lines below are
those of the sources being moved, not yet of this repo.

## Crates

| Crate | crates.io today | Holds | Source (moves with history) | Needs |
|---|---|---|---|---|
| `fishstringer-kernels-source` | 0.0.0 (claimed) | Slang kernels, `compile.sh`, hoist/no-contraction scripts | `fuel-kernels-source` (+ Vulkane's copy), #2 and #7 | nothing: no dependencies, no Rust code |
| `fishstringer-slang-parse` | 0.0.0 (claimed) | tree-sitter-slang front end: `SLANG` `Frontend`, residue list | `vulkane-slang-parse` | `unpopped` `0.43` with `convert`, `tree-sitter` `0.27`, `tree-sitter-slang` `0.3`; dev: `unpopped-vocab` |
| `fishstringer-slang-emit` | 0.0.0 (claimed) | IR to Slang text | `vulkane-slang-emit` | `unpopped`, `unpopped-vocab`; feature `convert` = `unpopped/convert`; dev: `unpopped-cpu-c`, `tempfile`, `fishstringer-slang-parse` |
| `fishstringer`, `fishstringer-sys` | 0.0.0 (claimed) | reserved | this repo's stub | stay reserved, `publish = false` until they have an API |

Unpopped crates were all 0.43.2 on crates.io on 2026-11-07 (`unpopped`, `unpopped-vocab`, `unpopped-cpu-c`), so the
moved crates can depend on published versions with no path or git dependencies (CireSnave 2026-10-07).

## Order

1. `fishstringer-kernels-source` (no dependencies). Publish after PR #2 merges and the Fuel and Vulkane lanes have
   confirmed their live tests against it (`KERNEL_RECONCILIATION.md`, "Open"). Publishing earlier freezes an
   unconfirmed reconciliation, which crates.io cannot take back (yank only hides).
2. `fishstringer-slang-parse` (needs published `unpopped` only).
3. `fishstringer-slang-emit` (its dev-dependency on `fishstringer-slang-parse` must resolve from crates.io, so parse
   goes first; a `path` dev-dependency is stripped at publish but the round-trip test then cannot run from the package).

Steps 2 and 3 are separate PRs (each: history-preserving import, manifest adapted, tests, notices).

## Per-crate checklist before the PM publishes

- Version: one series across the repo (CireSnave: one version number per project); the PM allocates. Pre-1.0, a
  breaking change is the second number.
- Manifest: `repository`/`homepage` = `https://github.com/ThinkersJournal/fishstringer`; `include` lists
  `src/**`, `README.md`, `LICENSE-*`, `THIRD-PARTY-NOTICES.md`; `cargo package -p <crate> --list` read whole (positive
  control: it lists a file known to exist). `tree-sitter-language` raises the parse crate's `rust-version` to 1.90
  (Vulkane's manifest, measured 2026-10-31); the workspace `rust-version` is 1.88 today, so either the parse crate carries
  its own or the workspace rises (CireSnave 2026-10-08 allows raising it).
- `tree-sitter` must be the same minor `unpopped` uses (`Frontend::parse` returns a `tree_sitter::Tree`, one
  `links = "tree-sitter"` per build). Every `unpopped` release that moves it forces a same-day release of parse and emit
  (Vulkane's handover note D5, accepted by this move).
- Notices and licences ship in the package; a moved file keeps its SPDX header and origin comment.
- Tag: annotated `v<version>` on the exact commit, pushed with the publish (`tools/tag-release.sh`).
- The slangc-dependent tests (Vulkane `ci.yml` has a pinned-`slangc` job) need that job here before the emitter ships;
  without it the emitter's conformance tests that call `slangc` are not run in CI.

## Consumers that wait on the first publish

- Fuel: replaces `fuel-kernels-source` with the published kernels crate; its `compile.sh` pipeline is coupled by source path
  (`fuel-vulkan-kernels` builds SPIR-V from `../fuel-kernels-source/kernels`), so Fuel's lane decides how it consumes a
  package (vendored copy from `cargo package`, `cargo vendor`, or a fetch step). Not mine to design.
- Vulkane: `vulkane-spirv-emit` (dev-dependency on `vulkane-slang-emit`) and the planned `vulkane-hlsl-emit` (thin wrapper
  over the Slang emitter) switch to the published crates; the 27 `.glsl` files stay Vulkane's.
- **Public API of the emitter (Vulkane's answer, 2026-11-07, read of `hlsl-emit/src/lib.rs` and `spirv-emit/tests` at Vulkane
  `f7fa204`): exactly two items, no new `pub` function.** `Slang` (unit struct, `impl unpopped::backend::Backend`:
  `supports_dtype`, `lower`; also used through `unpopped::try_generate`) and `required_slangc_args(ElementKind) ->
  &'static [&'static str]`. No speller is `pub`. Output-shape promises Vulkane's HLSL wrapper asserts (it panics if they
  change, so they are API in effect; keep them or tell Vulkane first): (1) `GeneratedKernel.name` starts
  `unpopped_slang_`; (2) the source has exactly one `// Generated by unpopped (slang backend)` line; (3) at most one line
  starting `// slangc:`; (4) otherwise the body is byte-identical, entry `[numthreads(256, 1, 1)]`, no `register(..)`.
  The workgroup size 256 is also relied on by Vulkane's SPIR-V oracle harness.
- Old crates: `vulkane-kernels-source`, `vulkane-slang-emit`, `vulkane-slang-parse` (0.21.1 on crates.io) and Unpopped's
  `unpopped-slang` are left published; a final release of each saying where it went is the PM's and the owning lane's call
  (Vulkane's handover note D2).
