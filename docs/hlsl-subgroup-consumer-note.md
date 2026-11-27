# Board 506: what an HLSL consumer (vulkane `hlsl-emit`) gets from `lower_kernel_hlsl_plain`, and the 6.21 rows for what Slang cannot do

Status: doc only. Measured at fishstringer main 46988a7 (code read; nothing was run for this note). Spec sections are cited
as quotes in section 1a, read 2026-11-27, and marked UNVERIFIED only where still unread; the measured facts are vulkane's (dxc 1.9.0.5347,
as recorded in the doc comments of `Slang::lower_kernel_hlsl_plain`).

## 1. The entry the consumer calls

`Slang::lower_kernel_hlsl_plain(&KernelFn) -> Result<GeneratedKernel, LowerError>` (and `lower_module_hlsl` for a module).
It emits the collectives as direct HLSL wave intrinsics with no `__target_switch`, so `dxc` can read the text.
`lower_kernel_hlsl` is the same lowering behind `__target_switch { case hlsl }` for slangc to read.

| Need | Statement |
|---|---|
| Intrinsics emitted | `WaveActiveSum`, `WaveActiveProduct`, `WaveActiveMax`/`Min` (int/uint only), `WaveActiveBitAnd`/`BitOr`/`BitXor`, `WaveReadLaneAt` (Idx, Broadcast, and Up/Down/Xor with the out-of-range select), `WaveActiveAllTrue`, `WaveActiveAnyTrue`, `WaveGetLaneIndex` |
| Types | `I32`, `U32`, `F32` values; a vote is a `U32` 0/1 |
| Shader model | **6.6 or higher**: the entry carries `[WaveSize(n)]` (dxc rejects it at `cs_6_0`..`cs_6_5`; accepted at `cs_6_6`/`cs_6_7` and under `-spirv`; vulkane's measurement). The wave intrinsics alone are SM 6.0 (verified, section 1a) |
| Guards the caller adds | `#ifdef __spirv__` then `#error`, and `#ifdef __SLANG__` then `#error` (vulkane measured: `__spirv__` only under `dxc -spirv`; slangc defines `__SLANG__`; plain dxc neither) |
| Launch | the launch must state its subgroup size (`LaunchShape::with_subgroup(n, n)`); else a named decline |
| Scope (board 506 A) | reductions, shuffles, votes, broadcasts at **Subgroup** scope only. **No scans** (`WavePrefixSum`/`WavePrefixProduct` are not emitted), no `ReduceWith`/`ScanWith`/`CollectiveTuple`, no Block scope |
| Declined by name | float `Reduce` Max/Min (HLSL's NaN rule UNMEASURED; the IR's is NaN-propagating); `Vote` Ballot (`WaveActiveBallot` returns `uint4`, the IR's Ballot is one 32-bit mask) |
| Cast detail | `WaveActiveBit*` take `uint` only (dxc refuses `int` at every SM 6_0..6_9, vulkane): an `I32` reduction is emitted `(int)WaveActiveBitOr((uint)x)` |

## 1a. What the specs say (read 2026-11-27)

Read in full, text downloaded to a non-repo dir: (W) DXC wiki "Wave Intrinsics" (`raw.githubusercontent.com/wiki/microsoft/DirectXShaderCompiler/Wave-Intrinsics.md`, section names below); (Z) DirectX-Specs `d3d/HLSL_SM_6_6_WaveSize.md` v1.00 2021-04-20; (S) DXC `docs/SPIR-V.rst`, only the "Shader Model 6.0+ Wave Intrinsics" table and the "Note on NMax,Nmin,FMax & FMin" paragraph were read (the rest of that file was not). The Microsoft Learn pages were not read; (W) is the DXC copy of the same spec. All quotes verbatim.

| Intrinsic | What the spec states (verbatim) | Gap for our lowering |
|---|---|---|
| Inactive / helper lanes, all ops | (W, "Shading Language Intrinsics") "All wave operations with the exception of Wave Query Intrinsics and Quad-Wide Shuffle Operations are disabled on helper lanes. They are treated as if flow control excludes these operations on helper lanes, therefore values read from or returned to helper lanes by these operations are undefined." (W, "Operation") "implementations must enforce that the number of active lanes exactly corresponds to the programmer's view of flow control." | Our emitted collectives sit at the top level of straight-line code in a compute entry: no helper lanes. A collective under divergent `If` reduces over the active lanes only; the IR oracle's rule for that is unpopped's, not stated here |
| `WaveActiveSum/Product/Min/Max` | (W) "Sums up the value of <expr> across all active non-helper lanes in the current wave, and replicates it to all lanes in said wave. The order of operations is undefined." Same "order of operations is undefined" for Product/Min/Max. Types: (W, "Types") "half, half2, half3, half4 / float... / double... / int... / uint... / short... / ushort... / uint64_t..." "that are also present in the target shader model" | **Verified:** int, uint, float are covered. **Float Sum/Product order is undefined**: results are not bit-reproducible against the oracle's left-to-right fold; compare within a band. **No NaN rule is stated for Min/Max** in (W): UNSPECIFIED. (S) states the SPIR-V path's scalar `min`/`max` "by default generate `NMin` & `NMax` instructions" (NaN-suppressing), which is the opposite of the IR's NaN-propagating rule; that sentence is about the `min`/`max` intrinsics, not about `WaveActiveMin/Max`, so it is evidence only that the float decline is right. Float Max/Min stays declined |
| `WaveActiveBitAnd/Or/Xor` | (W) "Returns the bitwise AND of all the values of <expr> across all active non-helper lanes"; "<int_type>"; "some operations (bitwise operators) only support the integer types." | Verified integers only. The spec's type list includes `int`; vulkane measured that dxc 1.9.0.5347 refuses `int` at SM 6_0..6_9, so the `(int)f((uint)x)` cast stays. The spec text and the compiler disagree here; the measurement wins. (S) table lists the row as `WaveActiveBitAdd()` -> `OpGroupNonUniformBitwiseAnd`: a typo in (S), not ours |
| `WaveReadLaneAt(expr, laneIndex)` | (W) "Returns the value of expr for the given lane index within the current wave. If laneIndex is uniform, this is effectively a broadcast operation, otherwise it is a shuffle operation. The result is undefined on a helper lane, or if the lane referred to by laneIndex is inactive or a helper lane." | **Non-uniform `laneIndex` is allowed (verified, (W)).** **Out-of-range `laneIndex` (>= `WaveGetLaneCount`): the spec states no result**: UNSPECIFIED, so the IR's `Idx = operand mod n` needs the explicit `% n` we emit, and Up/Down/Xor's own-value edge rule needs the explicit select we emit (as MSL needs). (S) maps it to `OpGroupNonUniformBroadcast`, whose SPIR-V uniformity rule for the lane Id is in the SPIR-V spec: UNVERIFIED, unread; a device run on a non-uniform index is still owed |
| `WaveReadLaneFirst` | (W) "Returns the value of expr for the active non-helper lane of the current wave with the smallest index. The resulting value is thus uniform across the wave" | Verified; not emitted (the IR's Broadcast names a lane) |
| `WaveActiveAllTrue/AnyTrue` | (W) "Returns true if <expr> is true in all active non-helper lanes in the current wave." / "...in any active non-helper lane..." | Verified. Inactive lanes do not vote, so they cannot veto All |
| `WaveActiveBallot` | (W) "Returns a uint4 containing a bitmask ... The least-significant bit corresponds to the lane with index zero. The bits corresponding to inactive or helper lanes will be zero. The bits that are greater than or equal to WaveGetLaneCount will be zero." | Verified uint4. The IR's one 32-bit mask is `.x` only for n <= 32; Ballot stays declined, and the decline text stays true |
| `WaveGetLaneCount` | (W) "Returns the number of lanes in a wave on this architecture. The result will be between 4 and 128. Includes all lanes (active, inactive and/or helper lanes)." | Verified. Our lowering uses the launch's literal `n` and enforces it with `[WaveSize(n)]` |
| `WaveGetLaneIndex` | (W) "The result must be in the [0, WaveGetLaneCount) range." | Verified |
| `[WaveSize(n)]` | (Z, "Allowed Wave Sizes") "the powers of 2 between 4 and 128, inclusive" `[4, 8, 16, 32, 64, 128]`; (Z, "HLSL Attribute") "compute shader entry points"; "`<numLanes>` must be an immediate integer value"; (Z, "Device Behavior") "The device must run the shader at the wave size specified in the DXIL metadata, if any." (Z, "Compiler Warning") use "will result in a compiler warning" | **Verified:** a launch subgroup size outside {4,8,16,32,64,128} is not expressible: our lowering should decline it by name (not done: **open item**). Shader model 6.6 floor is stated by (Z)'s title section ("Shader Model 6.6 introduces a new option") and matches vulkane's measurement. Driver acceptance on a real device: still UNVERIFIED |
| Wave intrinsics floor | (W, "Shading Language Intrinsics") "added to HLSL for use in shader model 6 and higher" | Verified: SM 6.0. `-spirv`: (S) "Wave intrinsics requires SPIR-V 1.3, which is supported by Vulkan 1.1 ... -fspv-target-env=vulkan1.1" |
| Quad ops | excluded by the task | not read beyond the heading |

Open item found by this read: the emitter does not check that the launch's subgroup size is a power of two in 4..128. A launch with `with_subgroup(2, 2)` or `(3, 3)` would emit `[WaveSize(2)]`, which (Z) says is not an allowed size. Not fixed here (docs-only branch); proposed for the next PR as a named decline.

Still UNVERIFIED on a device: the lowering has not run anywhere.

## 2. The 6.21 rows: what Slang cannot express (positions, not declines)

CIRESNAVE-EXPECTATIONS 6.21: nothing is skipped, only scheduled; a row carries a position. These are declined by name today
(`tests/new_dtype_rows_decline_by_name.rs`, slangc 2026.8 measurement); each row says what would reopen it.

| Row | Construct | Position | Reopens when |
|---|---|---|---|
| S1 | `F128`, `I128`, `U128`, `I256`, `U256` | No Slang scalar (`float128_t`, `int128_t`, `uint128_t` are undefined identifiers). Scheduled as an emulation row: a struct of two or four `uint` limbs with explicit add/mul/compare, exact against the oracle, value-equality only | A consumer needs one of them on a Vulkan device, or Slang adds the types. Size M per dtype; `F128` is L (rounding) |
| S2 | `F8E3M4`, `F8UE4M3` | No E3M4 or unsigned-E4M3 type; `FloatE4M3` is a different format, so it is not mapped. Scheduled as storage `uint8_t` plus an explicit decode to `float`, bit-exact against the oracle's table | A consumer needs them; size S each, exhaustive over 2^8 inputs |
| S3 | Multi-output elementwise round trip (lifter) | Blocked on unpopped U67 (lifter multi-output and u8-output round trip, docs branch `lane/consumer-redisposition-r2@e3f9ab81` per pm-kernels notice 2026-11-27; second-hand, not read here) | U67 merges; then the E17 round-trip tests for multi-output are added |
| S4 | A memory access marked `volatile`/`coherent` (Slang/HLSL `globallycoherent`, GLSL `coherent`/`volatile`) together with memory-barrier semantics | held: US9324175B2 ("Memory coherency in graphics command streams and shaders"; PM working hold, needs more, not a clearance; pm-kernels NOTICE 2026-11-27). No emitter feature is added; shipped output unchanged | The PM lifts the hold |
| S5 | A host-API request-driven bind of a texture object to an image unit | held: US10134169B2 host-API limb; no such API without a re-read (pm-kernels NOTICE 2026-11-27) | A re-read clears it |

— lane: fishstringer
