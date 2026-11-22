# E17 slice 3: what an `n` guard, `cbuffer` params and multi-output do to the consumer ABI

Status: options for pm-kernels and Unpopped to rule on. **No code until the ack** (pm-kernels TASK 2026-11-22).
Measured at fishstringer main 8c203f2 (+ #45 tip a1d0126), unpopped 0.56.1 (the pinned crate) and Unpopped `origin/main`
(`unpopped-vocab` 0.56.6, `ParamsBlock::of_struct` landed in 0.56.7 per pm-kernels; I have not read 0.56.7).

## 1. The ABI today (`Backend::lower`, the plan path)

`Slang::lower` returns `GeneratedKernel::new(name, source)` with **no `KernelInterface`** (`interface: None`). The contract is
the source text itself, which `fishstringer-slang-parse` re-lifts byte-for-byte:

- Resources: `StructuredBuffer<T> input0..input{n_inputs-1}`, then `RWStructuredBuffer<T> output`; **no
  `[[vk::binding]]`**, so slots are whatever `slangc` assigns (declaration order). One output, same dtype as the compute dtype.
- Entry `void <name>(uint3 tid : SV_DispatchThreadID)`, `[numthreads(256,1,1)]`, body `uint i = tid.x; ... output[i] = ...;`.
- No element count in the kernel: the launcher must dispatch exactly `ceil(n/256)` groups **and** `n` must be a multiple of 256,
  or the tail threads write out of range. No parameters: a body with `Param` is refused (`UnsupportedPlanShape`).
- Compare CUDA/CpuC on the same plan: `(in0.., out, long long n, <ctype> p<i>...)`, params f32-only, one `p<i>` per used `Param`.

`Backend::lower_kernel` (the launched-kernel path) is different and already carries an interface: `bindings` (dense
`[[vk::binding(slot,0)]]`), `params_block` (one `ConstantBuffer<Params_> pb_` after the buffers, fields `p<i>` /
`b<i>_stride<d>` / `b<i>_base`, offsets cross-checked against `slangc` reflection in `tests/lower_kernel_params_block_layout.rs`),
`requires`, `compiler_args`. **Slice 3 is about the plan path only**; nothing below touches `lower_kernel`.

## 2. What each item adds

| Item | What the consumer sees change | Existing fields that can carry it |
|---|---|---|
| `n` bounds guard | A new runtime value (`n`) the kernel reads, plus `if (i >= n) return;`. Either a new `ConstantBuffer` resource (shifts nothing if bound LAST, but there is no binding attribute today, so "last" is only declaration order) or a specialization constant (not applicable: `n` varies per launch). | `KernelInterface::params_block` (a field `n: U32/I64`), `bindings` (slot of the block). Both exist; the plan path sets neither today. |
| `cbuffer` params | One `ConstantBuffer<Params_>` holding `p<i>` for each used `Param` (f32), the same shape `lower_kernel` already uses. | `params_block` (fields `p<i>`, offsets from `ParamsBlock::of_struct`), `bindings`. |
| multi-output | `n_outputs` `RWStructuredBuffer`s `output0..output{m-1}` instead of one `output`; a name change for m=1 would break the lifter and Baracuda's binding order, so m=1 must keep `output`. | `bindings` per buffer; `params` only exist on `KernelFn`, not on a `KernelPlan`, so the buffer list has no field to live in for a plan kernel. |
| hetero out-dtype | `output{j}` typed `out_dtype_of(j)` (e.g. a `u8` keep-mask) while inputs are the compute dtype; storage width per output. | None for a plan kernel (same reason). The plan's own `out_dtype_of` is the source of truth the consumer already has. |

The honest summary: `params_block` and `bindings` already express the guard and the params block. The gap is that **a plan
kernel has no `interface` at all today**, so the consumer would have to start reading one it has never read.

## 3. Option A, additive (recommended): new behaviour only when asked, old bytes untouched

1. `Slang::lower` keeps emitting today's exact source and `interface: None` for every plan that needs none of the three
   (parameterless, single uniform-dtype output, exact dispatch). Byte-identical output is asserted in a test (the lifter
   round-trip already pins it).
2. A plan that **needs** one of the three gets an `interface: Some(..)` and a new source shape:
   - explicit `[[vk::binding(k, 0)]]` on every resource, buffers first in order (inputs, then outputs), then one
     `ConstantBuffer<Params_> pb_` last, so the buffer slots equal today's declaration order;
   - `Params_ { uint n; float p<i>...; }` with `n` first when a guard is emitted; `params_block` states the offsets
     (`ParamsBlock::of_struct`, checked against `slangc` reflection as in the E2 test);
   - the guard `if (i >= pb_.n) { return; }` as the first statement;
   - m>1 outputs: `output0..`, hetero dtype per `out_dtype_of`; m=1 keeps the name `output`.
3. The `n` guard is **opt-in by need, not by default**: a parameterless plan keeps exact dispatch. (A default-on guard would
   change every kernel's binding layout; see Option B.)
4. Consumers detect the new shape by `interface.is_some()` (and `params_block`), never by parsing. A consumer that ignores
   `interface` keeps working on every plan it works on today, and gets the existing typed decline-shaped failure (a missing
   buffer) rather than a silent mismatch on the new ones only if it launches them blind; so we also keep refusing a
   needs-params plan from a consumer that has not opted in, **if** Unpopped gives us a way to know (see question 2).
5. The lifter (`fishstringer-slang-parse`) gets the new shapes as additional recognised forms; the residue-round-trip test
   is extended, the old forms stay.

## 4. Option B, breaking: every plan kernel gets `n` and a binding block

Uniform shape: always `ConstantBuffer<Params_>` (`n` first), always explicit bindings, always the guard. Simpler to reason
about and removes the "n must be a multiple of 256" trap, but it changes **every** existing kernel's resources (one more
binding), its bytes (the lifter and Baracuda/Fuel golden sources), and requires every consumer to bind the block. Per
CireSnave's versioning rule this is a breaking change (0.5.0), and it needs Unpopped, Baracuda and Fuel in lockstep.
I do not recommend it unless Unpopped wants the uniform shape for its own reasons.

## 5. Questions for Unpopped (rdlbx3us) before pm-kernels rules

1. Will Unpopped accept `GeneratedKernel::interface` (`bindings`, `params_block`) on a **plan** kernel from `Backend::lower`,
   i.e. does `generate()` pass it through unchanged, and does any consumer today assert `interface.is_none()` on a plan kernel?
2. Can a backend learn whether the caller can bind a params block (a capability or a `Backend` hook), so Slang can refuse a
   needs-params plan instead of emitting something an old launcher binds wrongly? If not, Option A degrades to "refuse
   params/guard/multi-output as today" for old consumers, i.e. the work only ships when the consumer opts in by version.
3. For multi-output/hetero: should the per-output dtype and role live in `KernelInterface::params` (building a synthetic
   `Param` list from the plan), or is `out_dtype_of` on the plan enough for the consumer?
4. Is `ParamsBlock::of_struct` (0.56.7) the intended way to produce the block, and does it do the std140/scalar-layout offsets
   that `slangc` reflects for a `ConstantBuffer` (our E2 test compares against reflection and will run against it)?
5. Is a 32-bit `n` acceptable (a buffer index is 32-bit in Slang), or does Unpopped want `int64_t n` with `shaderInt64` in `requires`?

## 6. What I need back

pm-kernels: A or B (or neither yet). Unpopped: answers to 1, 2 and 4. Then I build slice 3 behind a changelog fragment, with
a seeded mutant for each of: guard off by one, binding order swapped, `n` offset wrong.

Not covered here: f64 erf family (needs a >f64 reference), and `Symbol` extern bodies (E16).

— lane: fishstringer
