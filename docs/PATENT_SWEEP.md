# Patent sweep: asynchronous-scheduling instructions in what FishStringer emits

Recorded 2026-11-13 on the PM's notice (CireSnave ruling, board 431 hit 22). Not legal advice; no attorney is involved, this is a self-serve record.

## The risk (as relayed)

CireSnave: *"it applies to any code created for non-Nvidia hardware that executes a hardware instruction that causes async scheduling of instructions in the hardware itself. Without knowing what Nividia's competitors' hardware have for instructions, I don't know whether any of them contain instructions that infringe on Nvidia's claims."*

NVIDIA patents US12118382B2 and US11294713B2. Claim 1 (verbatim): *"prevent one or more first instructions of a first thread from being performed while one or more second instructions of the first thread are being performed"*; claims 9, 17 and 23 repeat it as method, system and processor. The dependents add that the wait is on the output of other threads in shared memory. The exposure is in code an emitter generates: lowering `Stmt::CopyAsync` / `AsyncCommit` / `AsyncWait` to a device instruction or runtime call that holds back a thread's instructions while others of that thread run.

## What FishStringer emits

| Target | Hardware or runtime async-copy-with-wait instruction emitted? | Basis |
|---|---|---|
| Slang to SPIR-V (Vulkan compute), the only emitter here | **No.** Doc row E14 lowers `CopyAsync` to plain synchronous loads and stores in a loop; `AsyncCommit` and `AsyncWait` emit nothing; the IR's `Barrier` publishes the data. Nothing in the output asks the hardware to overlap a thread's copy with that thread's later instructions. | `docs/SLANG_EMISSION_COVERAGE.md` E14. Today the emitter lowers none of these statements at all (`lower_kernel` is the typed decline). |
| The 153 shipped `.spv` kernels (`fishstringer-kernels-source`) | Not examined for this. They are hand-written Slang compute kernels, none a `CopyAsync` lowering. | Not measured: no grep of the SPIR-V for async-copy instructions was run. |

Typed decline by name: **not needed**, because there is no target instruction to decline. If an instruction ever appears, decline first (`UnsupportedKernel`, reason prefix `slang target gap: `, see `fishstringer-slang-emit/src/gaps.rs`).

## Barrier aggregation: `Stmt::Collective` (PM notice 2026-11-13, CireSnave hit 23 ruling A)

NVIDIA US8539204B2, claim 1 as relayed: receive a barrier aggregation instruction for a first thread; combine its value with an aggregation result; communicate the result to each thread; wait until all threads have received it before executing more instructions. Claim 13 adds a hardware "barrier instruction execution unit".

| Doc row | Lowering | Single barrier-aggregation instruction? |
|---|---|---|
| E8, subgroup `Reduce` / `Scan` (also `Vote`, `Shuffle`, `Broadcast`) | One HLSL wave intrinsic each: `WaveActiveSum/Product/Min/Max/Bit*`, `WavePrefixSum/Product`, `WaveActiveAllTrue/AnyTrue/Ballot` (SPIR-V group-non-uniform operations). | **Possibly, and not for this lane to rule.** One instruction combines every lane's value and returns the result to each lane. Whether the hardware needs the "wait until all threads have received it" step is the device's, not ours to read from SPIR-V. |
| E9, block `Reduce` / `Scan` / `ReduceWith` / tuple | Derived: per-subgroup wave intrinsic, then a `groupshared` combine and two barriers. | **No** for the block step (ordinary loads, stores and barriers); the per-subgroup step is the E8 case. |

**Today nothing is emitted:** the emitter lowers no `Collective` (`lower_kernel` is the typed decline), so no output carries the instruction. The exposure begins when row 18 lands E8/E9.

**RULED, board 434 = C (CireSnave 2026-11-13, via the PM): NVIDIA devices may be lowered with a risk note; non-NVIDIA targets are declined by name. The PM states Slang to SPIR-V is non-NVIDIA, so E8/E9 stay typed declines here. The hold text that follows is the earlier interim state.**

**HELD on board 434 (PM notice 2026-11-13): until CireSnave rules (A emit with note, B decline by name, C emit for NVIDIA only and decline non-NVIDIA; the PM recommends C), the emitter emits nothing for `AsyncWait`/`CopyAsync`/`AsyncCommit`/`Collective`, and E8/E9 stay held; interim behaviour is to decline by name.** Earlier text: decision needed before row 18 lowers E8 `Reduce`/`Scan`:** decline them by name (`UnsupportedKernel`, reason `slang target gap: subgroup reduction/scan (patent risk, see docs/PATENT_SWEEP.md)`) or emit with the risk recorded. Not decided here; asked of the PM. `Shuffle`/`Vote`/`Broadcast` do not combine values with an aggregation result, so they are outside the quoted claim 1, which is a reading, not a finding.

## Rule for this repo

Row E14 stays synchronous. Emitting any instruction or Vulkan extension feature that schedules a thread's copy asynchronously and holds that thread's later instructions back (for example a vendor async-copy or DMA-with-tag-wait extension) needs the PM and CireSnave first.

## Not checked

Competitors' instruction sets, and the Vulkan extension list for such a feature. Only what this repo emits was read.
