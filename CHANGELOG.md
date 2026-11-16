# Changelog

Entries are written by `tools/stamp-version.py` from the fragments in `changelog.d/` (see `changelog.d/README.md`).
History before this file existed is the ROADMAP's Done table and the merged PRs (#1 to #15); versions 0.0.12 to 0.0.17
were published to crates.io.

## 0.3.10 — `lower_kernel` declines a dense `Mma` by name (row 17 E15 route b)

A dense `Mma` atom now declines with `slang: Mma: no cooperative-matrix shape for this atom `<name>` (kernel `<k>`)` instead of the generic "this statement is not lowered yet". It is not a target gap (no `slang target gap: ` prefix): mapping an atom to a `CoopMat` shape is route (a) of the coverage doc and needs the device's shape list (row E19) and a device measurement. Sparse and block-scaled atoms stay target gaps. Tested with a real `nvidia_m16n8k16_f16_f32` atom that passes `check_kernel`, with the control that the reason carries no gap prefix. Declines only; no kernel that lowered before changes.

## 0.3.9 — CI moves the CodeRipper portfolio workflow to the 0.47.5 pin

The `portfolio` job now calls the shared workflow at `c72a6e03479d914de8ea8a73609ba73e93a0de92` (installs CodeRipper 0.47.5). This repo has no `.coderipper.toml` and no `--skip-rule` or `check <rule>` use, so the rule-id migration tool had nothing to rewrite (`migrate-rule-ids.py --dry-run` exits 2: no `.coderipper.toml`; a search of tracked files for the 14 old names finds none). CI configuration only; no code or version change.

## 0.3.8 — every recorded Slang target gap has its own decline test; the README lists the whole boundary (row 17 E18)

Added the decline tests the gap list was missing: cluster-scope fence and atomic, `CopyAsync` into a cluster-peer place, `MulticastLoadReduce`, and the sparse (2:4) and block-scaled `Mma` atom arms (built from real `MmaAtom`s, with a dense atom as the control that is not a gap). Each asserts the `slang target gap: ` prefix and the construct named. The crate README now lists the full boundary in one place (dynamic shared extent, device and cluster barriers/fences/atomics, sparse and scaled `Mma`, peer/cluster/multicast places, multi-device and cluster launches) and drops the stale "needs unpopped 0.45" line. Tests and docs only; no behaviour change.

## 0.3.7 — `lower_kernel` lowers 64-bit locals and Cast; a paged gather compiles (row 17 E13)

`Slang::lower_kernel` now accepts `I64` locals and constants (`int64_t`) and `ScalarExpr::Cast` for the conversions a C cast does exactly: integer to integer with `Saturation::Wrap`, and integer or float to `F32`/`F16` with round-to-nearest-even and `Wrap`. Float to integer, any other rounding or saturation, and `Bf16` on either side decline by name. With the existing nested `For`, `If`, `I32` buffers and strided views, a FlashInfer-shaped paged gather written by hand in the IR now lowers: the page table (`indptr`, `indices`, `last_page_len` as `I32` buffers) is read into locals, the KV row offset is computed in `int64_t`, and the kernel compiles to SPIR-V with `slangc` (required, not skipped). 64-bit offsets need `shaderInt64`, the same feature the params block already records. Integer `Div`/`Rem` are not in the IR, so the gather loops over pages and rows instead of dividing. A buffer index is still 32-bit in Slang: a KV cache past 2^32 elements is the launcher's to avoid (physical-storage-buffer addressing is not used). Not run on a device, and not yet compared against the oracle or the CUDA launcher. Additive.

## 0.3.6 — `lower_kernel` lowers f16 and bf16 buffers, locals and arithmetic (row 17 E11)

`Slang::lower_kernel` now accepts `F16` and `Bf16` kernels. `F16` is Slang's `half` throughout (constants as `((half)x)`). `Bf16` has no Slang scalar, so a bf16 buffer is `StructuredBuffer<uint16_t>` read through `bf16_to_f32_` and written through `f32_to_bf16_` (round to nearest even, NaN stays a quiet NaN), bf16 locals and shared/private arrays are `float` holding a bf16-representable value, and every bf16 `Add`/`Sub`/`Mul` is rounded back with `bf16r_`; the generated source carries the three helpers and the float `-denorm-mode-fp32 preserve` argument. The interface gains a fidelity note that the kernel needs `shaderFloat16` and 16-bit storage (the capability vocabulary is row E19). A 16-bit float scalar parameter, packed (multi-lane) locals and fp8 still decline. An f16 and a bf16 `x*2+y` kernel compile to SPIR-V with `slangc` (required, not skipped). Not run on a device, so the bf16 rounding is not yet compared exhaustively against the oracle. Additive.

## 0.3.5 — `lower_kernel` lowers integer atomics (row 17 E10)

`Slang::lower_kernel` now lowers `AtomicRmw` (`Add`, `Max`, `Min`, `And`, `Or`, `Xor`, `Exchange`) as `InterlockedAdd/Max/Min/And/Or/Xor/Exchange(dest, value[, old])` and `AtomicCas` as `InterlockedCompareExchange(dest, expected, desired[, old])`, on an `I32`/`U32` buffer element or `groupshared` element, with the optional old value in a local. Slang takes neither a scope nor a memory order (Vulkan fixes the scope by the storage), so both are dropped and the atomic is at least as strong as asked. Float atomics decline by name (`a float atomic (needs VK_EXT_shader_atomic_float ...)`), as do 64-bit ones: the emitter cannot yet say the device has the feature (row E19). A cluster scope or a peer place is a named target gap. A histogram with a shared max and a CAS lock, and every integer combine, compile to SPIR-V with `slangc`; the float and cluster declines are tested. Additive. Not run on a device.

## 0.3.4 — `lower_kernel` lowers lane, subgroup and device ids (row 17 E12)

`Slang::lower_kernel` now lowers the remaining execution ids: `LaneId` is `WaveGetLaneIndex()`, `SubgroupSize` is `WaveGetLaneCount()`, `SubgroupId` is derived as `SV_GroupIndex / WaveGetLaneCount()` (the entry point gains an `SV_GroupIndex` argument), and `DeviceRank`/`DeviceCount` are the constants 0 and 1 (a multi-device launch is already a named gap). The subgroup ids read the device's subgroup size, so the sizes a kernel is valid for stay `LaunchShape::subgroup`, which the core-derived half of the `KernelInterface` already carries. A 3-D copy over an odd grid (block 4x2x2, grid 3x5x7) and every id compile to SPIR-V with `slangc`. Vulkan workgroup-size and dispatch limits are not checked by the emitter (they are device properties; row E19). Verified: workspace tests and `clippy -D warnings` pass. Not run on a device.

## 0.3.3 — `lower_kernel` lowers barriers and fences (row 17 E7)

`Slang::lower_kernel` now lowers `Barrier{Block}` to `GroupMemoryBarrierWithGroupSync()`, `Barrier{Subgroup}` to `AllMemoryBarrierWithWaveSync()`, and `Fence` at block, subgroup and device scope to `GroupMemoryBarrier()`, `AllMemoryBarrier()` and `DeviceMemoryBarrier()`. Slang's barriers take no memory order, so every `MemoryOrder` of a fence maps to the same full barrier: stronger, so correct, and the order distinction is lost (recorded in the module docs). Device and cluster barriers and cluster fences stay the named target gaps. With E5 this makes a cross-thread shared-memory kernel lowerable: a block reversal through a shared tile compiles to SPIR-V with `slangc` and its barrier sits between the write and the read (tested). Verified: workspace tests and `clippy -D warnings` pass. Not run on a device, so the reversal is not yet compared against the oracle.

## 0.3.2 — `lower_kernel` lowers static shared and private allocations (row 17 E5)

`Slang::lower_kernel` now lowers `Place::Alloc` on static allocations: a `Shared` one is a module-scope `groupshared T a<k>[N]` (an entry-local `groupshared` is not allowed in Slang), a `Private` one a function-local array. A `Constant` or `Global` allocation declines, because the IR gives a `Constant` allocation no initializer to emit; a launch-parameter extent stays the named target gap. Without the barriers of row E7 a shared array is only usable per thread, so the tile-transpose check waits for E7. Verified: a kernel using both spaces compiles to SPIR-V with `slangc`; both declines are tested; workspace tests and `clippy -D warnings` pass. Not run on a device.

## 0.3.1 — `lower_kernel` lowers control flow, masked access and comparisons (row 17 E4)

`Slang::lower_kernel` now lowers `If`, `While`, `For`, `Break`, `Continue`, `Return`, `MaskedLoad` and `MaskedStore`, with `CmpEq/Ne/Lt/Le/Gt/Ge` as conditions and masks (a non-comparison is tested `!= 0`). A masked access sits inside the branch, so memory is untouched when the mask is false. A `For` lowers only with a nonzero constant step, because the oracle traps on a zero step and a GPU cannot; any other step declines `For: step not a nonzero constant`. Locals are hoisted to one declaration at the top of the entry. Verified: a kernel using every one of these statements compiles to SPIR-V with `slangc`, the step declines are tested, workspace tests and `clippy -D warnings` pass. Not run on a device, so the loop and branch semantics are not yet compared against the oracle.

## 0.3.0 — **BREAKING**: `lower_kernel` buffers are strided views with a parameter block (row 17 E2)

`Slang::lower_kernel` now lowers scalar parameters (`F32`/`I32`/`U32`) and buffers of any declared rank as strided views: element offset `base + sum(coord_d * stride_d)` in `int64_t`, with every stride, base and scalar in one `ConstantBuffer` bound after the buffers. **Breaking for consumers of the 0.2.0 interface:** a rank-1 buffer is no longer a flat index, so every lowered kernel now carries a `ParamsBlock` (fields `b<i>_stride<d>`, `b<i>_base`, `p<i>`, `int64_t` strides/bases) and buffer binding slots are dense in buffer order. A buffer with no declared rank is declined. No shape field and no bounds check is emitted; the block needs `shaderInt64` (fidelity note on the interface). Verified: a new test compares the stated `ParamsBlock` offsets and size with `slangc -reflection-json`, and fails when the sizes are mutated; slangc compiles both kernels to SPIR-V; workspace tests and `clippy -D warnings` pass. Not run on a device. **Who is affected:** baracuda's Slang emitter and any consumer of `fishstringer-slang-emit` that launches a kernel from `Slang::lower_kernel`'s `KernelInterface`. **Migration:** read `interface.params_block` and upload each field at its stated offset (`b<i>_stride<d>` and `b<i>_base` as little-endian `i64`, `p<i>` as the scalar's type), bind the block at the slot after the last buffer binding, and read buffer slots from `interface.bindings` rather than assuming slot = parameter index.

## 0.2.0 — **BREAKING**: repin to unpopped 0.47.0 (sk8); `unpopped-convert` is its own crate

`fishstringer-slang-emit` and `fishstringer-slang-parse` now pin `unpopped`, `unpopped-vocab` and `unpopped-cpu-c` at 0.47.0 (structure_key sk8). The Unpopped `convert` feature became the crate `unpopped-convert`: the parse crate depends on it directly and imports `unpopped_convert::`; the emit crate's `convert` feature is now a marker feature that gates the round-trip unit test, with `unpopped-convert` as a dev-dependency. Verified: workspace build, `clippy -D warnings` and all workspace tests pass with `--all-features`; the structure-keyed conformance cells passed unchanged.

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
