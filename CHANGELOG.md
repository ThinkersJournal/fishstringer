# Changelog

Entries are written by `tools/stamp-version.py` from the fragments in `changelog.d/` (see `changelog.d/README.md`).
History before this file existed is the ROADMAP's Done table and the merged PRs (#1 to #15); versions 0.0.12 to 0.0.17
were published to crates.io.

## 0.1.4 — Slang `lower_kernel` first slice (launched elementwise kernels)

`Slang::lower_kernel` now lowers a launched kernel with rank-1 `F32`/`I32`/`U32` buffer parameters, `Let`/`Load`/`Assign`/`Store` and `Local`/`Const`/`ConstInt`/`Exec`/`Add`/`Sub`/`Mul`, and returns a `KernelInterface` (entry symbol, one binding per buffer in order, the `slangc` denorm arguments). No parameter block yet: strided views and scalar parameters are doc row E2, so the layout test against slangc reflection comes with it. Anything else is `UnsupportedKernel` ("not lowered yet"); the gap and patent declines run first. Verified: 3 new tests (lowers with its interface, `slangc` compiles the output to SPIR-V, what stays declined), workspace tests and clippy `-D warnings` pass. Not run on a device. ROADMAP row 17, doc row E1 in part.

## 0.1.3 — E3 re-scoped to harness 0.0.17; ROADMAP row 19 done, row 20 added

Docs only. `docs/SLANG_EMISSION_COVERAGE.md` E3 now says what `unpopped-harness` 0.0.17 provides (exhaustive unary-f32 plan-path compare on Vulkan) and what it does not (launch kernels, views, launch shapes); the harness is to be used from a separate test crate, never as a dev-dependency of the emit crate. ROADMAP row 19 moves to Done; new row 20 records the request to the Unpopped lane.

## 0.1.2 — Slang declines peer, cluster and multicast places and multi-device launches

Completes ROADMAP row 19 / doc E18: `Slang::lower_kernel` declines `Place::ClusterPeer`, `Place::Peer`, `Place::Multicast`, `MulticastLoadReduce`, `devices > 1` and cluster launches with the `slang target gap: ` prefix. Tests: 3 new in `slang_target_gaps_are_typed_declines.rs` (8 pass), including the one-device control. Not covered: `Mma` sparse/scaled arms (no test builds an atom). Stacked on #23 (unpopped 0.45.12).

## 0.1.1 — patent sweep note for async-copy lowering (docs)

Adds `docs/PATENT_SWEEP.md`: the PM's notice on NVIDIA patents US12118382B2 / US11294713B2 (claim 1 quoted), and what this repo's only emitter (Slang to SPIR-V) does about `CopyAsync`/`AsyncCommit`/`AsyncWait`: row E14 lowers them to plain synchronous loads and stores, so no async-scheduling instruction is emitted. The shipped `.spv` and competitors' instruction sets were not examined. Also records the PM's barrier-aggregation notice (US8539204B2): E8 subgroup Reduce/Scan lower to one wave intrinsic (possible exposure, undecided), E9 block collectives use groupshared plus barriers; nothing is emitted today. Docs only.

## 0.1.0 — **BREAKING**: repin unpopped to 0.45.12; patent-flagged statements decline by name

`fishstringer-slang-emit` and `fishstringer-slang-parse` move from `unpopped` 0.43.2 to 0.45.12 (the release with `KernelInterface`, unblocking ROADMAP row 17); `cargo test --workspace` passes unchanged. `Slang::lower_kernel` now declines `CopyAsync`, `AsyncCommit`, `AsyncWait`, `Collective` and `CollectiveTuple` by name (board 434 ruling C: Slang to SPIR-V is non-NVIDIA), reason naming `docs/PATENT_SWEEP.md`. Tested for `AsyncCommit` and `AsyncWait`; the `CopyAsync` and `Collective` arms are untested (no test builds one).

## 0.0.22 — Slang target gaps are named typed declines

`Slang::lower_kernel` now declines dynamic shared extent, device/cluster-scope barriers, cluster-scope fence/atomic and sparse or block-scaled `Mma` with `UnsupportedKernel` whose reason starts with the new `GAP_PREFIX`; listed in the crate README (doc row E18, ROADMAP row 19, first part). Verified by `tests/slang_target_gaps_are_typed_declines.rs` (5 tests, including the no-gap control). Not covered: `Mma` atoms (no test builds one), peer/cluster places and `devices > 1` (need unpopped newer than 0.43.2).

## 0.0.21 — refresh the changelog tools to kit v3

Re-copies `tools/stamp-version.py` and `tools/check-fragment.py` from the portfolio kit (refactored after Codacy flagged the first copies: bandit subprocess use, cyclomatic complexity). No change in behaviour. Tooling only; no crate code changes.

Also adds `docs/SLANG_EMISSION_COVERAGE.md` (PR #20, merged unstamped; folded here at the PM's request): what `fishstringer-slang-emit` lowers today (scalar contiguous Elementwise only; `lower_kernel` is the typed decline) and one row per construct family it would need to emit everything Slang can express, with owner, size, the test that will show it and the typed decline for what Slang cannot express. ROADMAP rows 17 to 19. Docs only; spellings marked probe are unverified until slangc and a device run.

## 0.0.19 — ROADMAP row 12 done (redirect crates published)

Moves ROADMAP row 12 to Done: `vulkane-slang-emit` and `vulkane-slang-parse` 0.22.0 are on crates.io over `fishstringer-slang` 0.0.17 (PM notice 2026-11-11; crates.io max_version 0.22.0 for both). Docs only.

## 0.0.18 — changelog fragments and `tools/stamp-version.py`

The repository adopts the portfolio's fragment-plus-stamp flow (PM task 2026-11-12): a PR adds one `changelog.d/<topic>.md`
and edits no version line, `Cargo.lock` or `CHANGELOG.md`; the PM stamps the number at merge. Adds `tools/stamp-version.py`,
`tools/check-fragment.py`, the `Changelog fragment` workflow and `CHANGELOG.md`. No crate code changes.

## 0.0.17 — NaN-propagating max/min

NaN guards in 12 kernels: an `isnan` select in `unary*` (relu), `reduce*` and `reduce_last_dim*` (including the Wave path),
because slangc 2026.8's `FMax`/`FMin` drop NaN. Regenerated at slangc 2026.8; the 12 changed `.spv` differ on purpose
(PR #15). Fuel measured 72 of 72 cases per adapter (AMD 610M, RTX 4070) propagating NaN.
