- kernels-source: `OP_SIGN` in unary, unary_f16, unary_bf16 and unary_f64 is now `isnan(x) ? 0 : sign(x)`. SPIR-V FSign returns 1.0 for NaN on the Radeon 610M (fuel nan-sign-step); KISS 6.2-0009 pins sign(NaN)=0. slangc-compiled; device red/green on the 610M not yet run by this lane.

— lane: fishstringer
