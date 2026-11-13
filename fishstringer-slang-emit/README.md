# fishstringer-slang-emit

The Slang compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped): it lowers Unpopped's
language-neutral IR (`ExprDag`, `KernelPlan`) to Slang source, using HLSL overloaded intrinsics,
`StructuredBuffer<T>` / `RWStructuredBuffer<T>` operands and a `[numthreads]` entry point.

Moved here from `unpopped-slang` (Unpopped `213df4f`, v0.34.11) with its git history, because Vulkane owns Slang
in the CireSnave ecosystem: the language owner writes the parser and the emitter. The crate is otherwise unchanged;
it depends on a published `unpopped` / `unpopped-vocab`.

Not yet on crates.io (first publish is the PM's); once published, add it with `cargo add fishstringer-slang-emit`.

Feature `convert` additionally lifts emitted Slang back to the IR (pulls tree-sitter grammars, which compile C).

Scope today: scalar contiguous elementwise kernels; every other schedule is a typed decline. Float kernels need
`slangc -denorm-mode-fp32 preserve` (see `required_slangc_args` and the crate docs).

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.

History: this crate was `unpopped-slang` in Unpopped, then `vulkane-slang-emit` in Vulkane (same owner, same licence);
its git history is preserved. It moved to FishStringer, which owns Slang, on 2026-11-07. The names and URLs changed.

## Target boundary: kernels Slang declines

`Slang::lower_kernel` refuses these with `LowerError::UnsupportedKernel`, reason starting `fishstringer_slang_emit::GAP_PREFIX`
(`"slang target gap: "`), because Vulkan compute cannot express them:

- run-time dynamic shared extent (`Extent::Param`)
- device-scope barrier (grid sync) and cluster-scope barrier, fence or atomic
- sparse (2:4) and block-scaled `Mma` atoms

(Peer/cluster places and multi-device launches are covered below.)
Today every other kernel also declines, with the generic reason (row 17 adds the lowering).

Also declined by name (patent risk, board 434 ruling C; `docs/PATENT_SWEEP.md`): `CopyAsync`, `AsyncCommit`, `AsyncWait`, `Collective`, `CollectiveTuple`. Slang to SPIR-V is a non-NVIDIA target.

Also declined as target gaps (needs `unpopped` 0.45): cluster-peer, peer-device and multicast places, `MulticastLoadReduce`, launches with `devices > 1` or cluster dimensions.
