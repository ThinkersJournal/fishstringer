kind: sign(NaN)=0 guard in the four unary sign kernels
---
`OP_SIGN` in unary, unary_f16, unary_bf16 and unary_f64 is now `isnan(x) ? 0 : sign(x)`. Measured by fuel (nan-sign-step): the unguarded SPIR-V FSign gives sign(NaN)=1.0 on the Radeon 610M (f32/f16/bf16; f64 ok) and 0 on the RTX 4070; KISS 6.2-0009 pins 0. This change is slangc-compiled only; its device run (610M and 4070) is fuel's and UNMEASURED here.

NaN status of the other ops, by static read of the emitted text only (no device run, none claimed safe): the emitter's Sign/Step/Relu/Max/Min are written as compare-selects, but the measured Sign result shows a static read does not predict device behaviour, so all are UNMEASURED. Also UNMEASURED: Floor, Ceil, Round, Trunc, Pow, Atan2, FmaxIeee/FminIeee, FClamp, SmoothStep (raw GLSL.std.450 calls, NaN result implementation-defined).

— lane: fishstringer
