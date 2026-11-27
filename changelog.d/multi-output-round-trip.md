kind: added: multi-output and u8 round-trip tests through lift_elementwise_multi
---
`tests/lower_multi_output_round_trip.rs` lifts the emitted two-output and hetero-u8 kernels back with unpopped-convert 0.60.23 `lift_elementwise_multi`. Output 1 and the u8 dtype return exactly; output 0 of the hetero kernel returns as the literal `tmp0 != 0.0` form (pinned, not claimed equal). Vacuity: wrong output counts refuse and a swapped-output source lifts to a different body. No device run.

— lane: fishstringer
