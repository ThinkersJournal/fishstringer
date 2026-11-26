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
- cluster-peer, peer-device and multicast places, `MulticastLoadReduce`, launches with `devices > 1` or cluster dimensions

Each has its own test in `tests/slang_target_gaps_are_typed_declines.rs`. A kernel outside the list that `lower_kernel` does not lower yet declines with the generic "... is not lowered yet" reason instead, so the prefix tells "outside the target" from "not taught yet".

<<<<<<< HEAD
Also declined by name (patent risk, board 434 ruling C; `docs/PATENT_SWEEP.md`): `CopyAsync`, `AsyncCommit`, `AsyncWait`, `Collective`, `CollectiveTuple`. Exception (board 506): `Slang::lower_kernel_hlsl` / `lower_module_hlsl` lower subgroup `Reduce` (Sum/Prod/Max/Min/And/Or/Xor; float Max/Min decline), `Shuffle`, `Vote` All/Any and `Broadcast` over `I32`/`U32`/`F32` as HLSL wave intrinsics inside a `__target_switch { case hlsl: }`, so a SPIR-V or GLSL compile of that text fails (the launch must state one subgroup size, `with_subgroup(n, n)`, spelled as `[WaveSize(n)]`; the text is Slang: `slangc -target hlsl`, then dxc; `lower_kernel_hlsl_plain` returns guard-free plain HLSL (minimum shader model 6.6, because of `[WaveSize]`) for a caller that adds `#ifdef __spirv__`/`__SLANG__` `#error` guards itself); `Scan`, `ReduceWith`/`ScanWith`, tuples, Ballot and block scope stay declined. Slang to SPIR-V is a non-NVIDIA target. Also declined: possible exposure, board 434 C extended (NVIDIA US11803380B2; CireSnave 2026-11-16): `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait` and `CopyBulk` on an mbarrier. These statements are not in the IR of unpopped 0.47.0 (the newest on crates.io), so there is no arm or test yet; the arm and a test per statement land with the Unpopped release that adds them, and until then such a kernel cannot be built. Also declined, same ruling (NVIDIA US12204897B2 and continuation US12710961B2, a wait on a count of pending matrix-multiply operations; CireSnave 2026-11-16): `WgmmaWait` and `Tcgen05Commit`. Not lowered to a hardware operation on this non-NVIDIA target; the by-name arms and tests land at the repin to the first Unpopped version that has the variants.
=======
Also declined by name (patent risk, board 434 ruling C; `docs/PATENT_SWEEP.md`): `CopyAsync`, `AsyncCommit`, `AsyncWait`, `Collective`, `CollectiveTuple`. Exception (board 506): `Slang::lower_kernel_hlsl` / `lower_module_hlsl` lower subgroup `Reduce` (Sum/Prod/Max/Min/And/Or/Xor; float Max/Min decline), `Shuffle`, `Vote` All/Any and `Broadcast` over `I32`/`U32`/`F32` as HLSL wave intrinsics inside a `__target_switch { case hlsl: }`, so a SPIR-V or GLSL compile of that text fails (the launch must state one subgroup size, `with_subgroup(n, n)`, spelled as `[WaveSize(n)]`; the text is Slang: `slangc -target hlsl`, then dxc; `lower_kernel_hlsl_plain` returns guard-free plain HLSL for a caller that adds `#ifdef __spirv__`/`__SLANG__` `#error` guards itself); `Scan`, `ReduceWith`/`ScanWith`, tuples, Ballot and block scope stay declined. Slang to SPIR-V is a non-NVIDIA target. Also declined: possible exposure, board 434 C extended (NVIDIA US11803380B2; CireSnave 2026-11-16): `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait` and `CopyBulk` on an mbarrier. These statements are not in the IR of unpopped 0.47.0 (the newest on crates.io), so there is no arm or test yet; the arm and a test per statement land with the Unpopped release that adds them, and until then such a kernel cannot be built. Also declined, same ruling (NVIDIA US12204897B2 and continuation US12710961B2, a wait on a count of pending matrix-multiply operations; CireSnave 2026-11-16): `WgmmaWait` and `Tcgen05Commit`. Not lowered to a hardware operation on this non-NVIDIA target; the by-name arms and tests land at the repin to the first Unpopped version that has the variants.
>>>>>>> origin/main

## What `lower_kernel` lowers today

A launched kernel with `F32`/`I32`/`U32` buffer parameters of known rank and `F32`/`I32`/`U32` scalar parameters, the statements
`Let`, `Load`, `Assign`, `Store`, and the expressions `Local`, `Const` (f32 kernels), `ConstInt`, `Exec` (`ThreadIdx`, `BlockIdx`,
`BlockDim`, `GridDim`), `Add`, `Sub`, `Mul`. `Place::Logical` goes through the view: offset = `base + sum(coord_d * stride_d)` in
`int64_t`. Strides, bases and scalars are one `ConstantBuffer` bound after the buffers (buffers take dense slots in parameter
order); the `KernelInterface` carries the entry symbol, the bindings and a `ParamsBlock` whose offsets a test checks against what
`slangc` reflects. Field names: scalar `p<i>`; buffer `b<i>_stride<d>` then `b<i>_base`. **No shape field and no bounds check**
(a GPU cannot trap where the oracle errors): the launcher passes views that keep every access in range. The block holds `int64_t`,
so the device needs `shaderInt64` (capability vocabulary: doc row E19; the interface carries a fidelity note until then). Control flow
(`If`, `While`, `For` with a nonzero constant step, `Break`, `Continue`, `Return`), `MaskedLoad`/`MaskedStore` and the six comparisons
are lowered too (doc row E4; a `For` with any other step declines `For: step not a nonzero constant`). Static `Shared` allocations (module-scope `groupshared`) and `Private` ones (a local array) lower too (doc row E5); `Constant`
and `Global` allocations decline, and a launch-parameter extent is a target gap. `Barrier` (block, subgroup) and `Fence` (block, subgroup, device) lower to Slang's barrier intrinsics (doc row E7); the memory order of a
fence is not expressible, so every order maps to the full barrier (stronger, so correct). The execution ids lower too (doc row E12): `LaneId`, `SubgroupSize` and a derived `SubgroupId` through `Wave*` intrinsics, `DeviceRank`/`DeviceCount` as
the single-device constants. Atomics lower too (doc row E10): `Interlocked*` on `I32`/`U32` buffer and shared elements (scope and order are dropped, the atomic is at least as strong); float and 64-bit atomics decline by name until the capability vocabulary (E19) exists. `F16` is `half`; `Bf16` is `uint16_t` buffer storage with float arithmetic rounded to bf16 after each operation (doc row E11; 16-bit scalar parameters, packed locals and fp8 decline). `I64` locals and `Cast` (int to int Wrap, to-float NearestEven) lower, so a FlashInfer-shaped paged gather written in the IR compiles (doc row E13; a buffer index stays 32-bit in Slang). The interface lists what the source needs as `vulkan:` `<arith>` tokens (`requires`: `i64`, `f16`, `st16`, `i16`; atomics tokens later, doc row E19); `unmet_requirements(interface, target_arith)` is the consumer-side gate. A dense `Mma` declines by name ("no cooperative-matrix shape for this atom", doc row E15 route b; not a target gap). `Slang::lower_module(&Module, entry)` lowers calls (doc row E16, first slice): scalar-argument `Call` becomes a Slang function and an `ExternCall` with a `reference` kernel emits the reference body; an extern with no reference, buffer arguments, `CallTuple` and a callee that reads a launch id decline by name. Everything
else declines with `UnsupportedKernel` ("... is not lowered yet"). Compiled by `slangc` in tests; not yet run on a device.
