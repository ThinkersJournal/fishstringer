kind: Slang `lower`: multi-output and hetero (u8 keep-mask) out-dtype plans, for interface callers
---
`Slang::lower_with` no longer refuses a plan with several outputs or an output whose dtype differs from the compute dtype (the u8
keep-mask of a dropout-class op). Like a runtime `Param`, such a plan changes the resource list the caller binds, so a legacy
caller is still refused (`UnsupportedPlanShape`, naming `CallerCaps::with_interface()`), and an interface caller gets explicit
`[[vk::binding]]` slots (inputs, then `output0..`, then the `Params_` block last), the `n` field and bounds guard, and a
`KernelInterface` whose per-output buffer dtype is `out_dtype_of(j)`. A single uniform output keeps its historical text and name
`output`. The bodies share one DAG (a value used by two outputs is computed once); the u8 conversion is applied at its own store,
never inside the shared node. A u8 output is declined with a typed error on a target that advertises no 8-bit arithmetic.
Verified: `tests/lower_multi_output.rs`, including slangc compiling both shapes and its reflected binding slots and `Params_`
offsets equalling the stated interface. NOT verified: a device run; the `fishstringer-slang-parse` lifter does not yet read the
multi-output form.
