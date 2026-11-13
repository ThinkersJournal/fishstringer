kind: Slang target gaps are named typed declines
---
`Slang::lower_kernel` now declines dynamic shared extent, device/cluster-scope barriers, cluster-scope fence/atomic and sparse or block-scaled `Mma` with `UnsupportedKernel` whose reason starts with the new `GAP_PREFIX`; listed in the crate README (doc row E18, ROADMAP row 19, first part). Verified by `tests/slang_target_gaps_are_typed_declines.rs` (5 tests, including the no-gap control). Not covered: `Mma` atoms (no test builds one), peer/cluster places and `devices > 1` (need unpopped newer than 0.43.2).
