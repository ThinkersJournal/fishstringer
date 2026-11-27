kind: docs: the HLSL consumer note for board 506 (spec read) and the 6.21 rows
---
`docs/hlsl-subgroup-consumer-note.md` states what `lower_kernel_hlsl_plain` needs and emits, quotes the DXC wave-intrinsics reference and the SM 6.6 `[WaveSize]` spec verbatim per intrinsic (inactive-lane, out-of-range-lane and type rules; where the spec states no result it says so), records one open item (a launch subgroup size outside 4..128 powers of two is not declined yet), and lists the 128/256-bit and E3M4/UE4M3 rows as scheduled emulation rows. Docs only; nothing was run.
