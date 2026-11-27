# Board 506: what an HLSL consumer (vulkane `hlsl-emit`) gets from `lower_kernel_hlsl_plain`, and the 6.21 rows for what Slang cannot do

Status: doc only. Measured at fishstringer main 46988a7 (code read; nothing was run for this note). Spec sections are cited
as pointers and marked **UNVERIFIED** where this lane has not read them; the measured facts are vulkane's (dxc 1.9.0.5347,
as recorded in the doc comments of `Slang::lower_kernel_hlsl_plain`).

## 1. The entry the consumer calls

`Slang::lower_kernel_hlsl_plain(&KernelFn) -> Result<GeneratedKernel, LowerError>` (and `lower_module_hlsl` for a module).
It emits the collectives as direct HLSL wave intrinsics with no `__target_switch`, so `dxc` can read the text.
`lower_kernel_hlsl` is the same lowering behind `__target_switch { case hlsl }` for slangc to read.

| Need | Statement |
|---|---|
| Intrinsics emitted | `WaveActiveSum`, `WaveActiveProduct`, `WaveActiveMax`/`Min` (int/uint only), `WaveActiveBitAnd`/`BitOr`/`BitXor`, `WaveReadLaneAt` (Idx, Broadcast, and Up/Down/Xor with the out-of-range select), `WaveActiveAllTrue`, `WaveActiveAnyTrue`, `WaveGetLaneIndex` |
| Types | `I32`, `U32`, `F32` values; a vote is a `U32` 0/1 |
| Shader model | **6.6 or higher**: the entry carries `[WaveSize(n)]` (dxc rejects it at `cs_6_0`..`cs_6_5`; accepted at `cs_6_6`/`cs_6_7` and under `-spirv`; vulkane's measurement). The wave intrinsics alone are SM 6.0 (**UNVERIFIED**: HLSL "Shader Model 6.0 wave intrinsics" on learn.microsoft.com not read here) |
| Guards the caller adds | `#ifdef __spirv__` then `#error`, and `#ifdef __SLANG__` then `#error` (vulkane measured: `__spirv__` only under `dxc -spirv`; slangc defines `__SLANG__`; plain dxc neither) |
| Launch | the launch must state its subgroup size (`LaunchShape::with_subgroup(n, n)`); else a named decline |
| Scope (board 506 A) | reductions, shuffles, votes, broadcasts at **Subgroup** scope only. **No scans** (`WavePrefixSum`/`WavePrefixProduct` are not emitted), no `ReduceWith`/`ScanWith`/`CollectiveTuple`, no Block scope |
| Declined by name | float `Reduce` Max/Min (HLSL's NaN rule UNMEASURED; the IR's is NaN-propagating); `Vote` Ballot (`WaveActiveBallot` returns `uint4`, the IR's Ballot is one 32-bit mask) |
| Cast detail | `WaveActiveBit*` take `uint` only (dxc refuses `int` at every SM 6_0..6_9, vulkane): an `I32` reduction is emitted `(int)WaveActiveBitOr((uint)x)` |

Spec pointers (all **UNVERIFIED**, unread this session): HLSL wave intrinsics on Microsoft Learn ("Wave Intrinsics",
DirectXShaderCompiler `docs/SPIR-V.md` "Wave intrinsics" for the `-spirv` mapping to `OpGroupNonUniform*`), and the
DirectXShaderCompiler wiki page "Wave Intrinsics". The consumer should cite the section it reads, not these names.

Still UNVERIFIED on a device: `WaveReadLaneAt` with a non-uniform lane argument (the intrinsic is documented for a
dynamically uniform lane in some profiles; ours is `operand mod n` of a possibly per-lane value); driver acceptance of
`[WaveSize]`; no device run of any of it.

## 2. The 6.21 rows: what Slang cannot express (positions, not declines)

CIRESNAVE-EXPECTATIONS 6.21: nothing is skipped, only scheduled; a row carries a position. These are declined by name today
(`tests/new_dtype_rows_decline_by_name.rs`, slangc 2026.8 measurement); each row says what would reopen it.

| Row | Construct | Position | Reopens when |
|---|---|---|---|
| S1 | `F128`, `I128`, `U128`, `I256`, `U256` | No Slang scalar (`float128_t`, `int128_t`, `uint128_t` are undefined identifiers). Scheduled as an emulation row: a struct of two or four `uint` limbs with explicit add/mul/compare, exact against the oracle, value-equality only | A consumer needs one of them on a Vulkan device, or Slang adds the types. Size M per dtype; `F128` is L (rounding) |
| S2 | `F8E3M4`, `F8UE4M3` | No E3M4 or unsigned-E4M3 type; `FloatE4M3` is a different format, so it is not mapped. Scheduled as storage `uint8_t` plus an explicit decode to `float`, bit-exact against the oracle's table | A consumer needs them; size S each, exhaustive over 2^8 inputs |
| S3 | Multi-output elementwise round trip (lifter) | Blocked on unpopped U67 (lifter multi-output and u8-output round trip, docs branch `lane/consumer-redisposition-r2@e3f9ab81` per pm-kernels notice 2026-11-27; second-hand, not read here) | U67 merges; then the E17 round-trip tests for multi-output are added |

— lane: fishstringer
