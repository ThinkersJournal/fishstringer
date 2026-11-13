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

## Rule for this repo

Row E14 stays synchronous. Emitting any instruction or Vulkan extension feature that schedules a thread's copy asynchronously and holds that thread's later instructions back (for example a vendor async-copy or DMA-with-tag-wait extension) needs the PM and CireSnave first.

## Not checked

Competitors' instruction sets, and the Vulkan extension list for such a feature. Only what this repo emits was read.
