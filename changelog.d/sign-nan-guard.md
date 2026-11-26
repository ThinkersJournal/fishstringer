kind: sign(NaN)=0 guard in the four unary sign kernels
---
`OP_SIGN` in unary, unary_f16, unary_bf16 and unary_f64 is now `isnan(x) ? 0 : sign(x)`. Measured by fuel (nan-sign-step): the unguarded SPIR-V FSign gives sign(NaN)=1.0 on the Radeon 610M (f32/f16/bf16; f64 ok) and 0 on the RTX 4070; KISS 6.2-0009 pins 0. Device run by fuel on branch bytes (ad5b324, slangc 2026.8): Radeon 610M sign(NaN)=0 on f32/f16/bf16/f64 with sign(+x)=1, sign(-x)=-1, sign(0)=0 intact; the published 0.6.0 bytes (byte-identical to a guard-removed mutant) give 1.0 on f32/f16/bf16. The RTX 4070 gives 0 either way (the driver lowers FSign NaN to 0), so it cannot tell guarded from unguarded.

NaN status of the other ops, by static read of the emitted text only (no device run, none claimed safe): the emitter's Sign/Step/Relu/Max/Min are written as compare-selects, but the measured Sign result shows a static read does not predict device behaviour, so all are UNMEASURED. Also UNMEASURED: Floor, Ceil, Round, Trunc, Pow, Atan2, FmaxIeee/FminIeee, FClamp, SmoothStep (raw GLSL.std.450 calls, NaN result implementation-defined).

— lane: fishstringer
