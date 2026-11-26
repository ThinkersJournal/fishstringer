kind: U8c declines are reached and carry GAP_PREFIX; Tensor allocations decline by name
---
`Stmt::ProxyFence`, `Wgmma`/`WgmmaCommit`, `Tcgen05Mma`/`Tcgen05Ld` and `MemorySpace::Tensor` allocations are now recorded in `gaps.rs`, so they decline with `GAP_PREFIX` and a name (they are outside the Vulkan target for good). Before, a kernel carrying `Tcgen05Mma`/`Tcgen05Ld` (which unpopped requires to use a `Tensor` allocation) was declined as "a Constant or Global allocation" and the `stmts()` arms for them could never fire (aMusedPip review of #51, Medium). `lower_entry` also names `Tensor` in its allocation match. The `stmts()` arms for these variants stay as defence in depth, now commented as unreachable. `GridDepWait`/`GridDepLaunchDependents`/`MatrixLoad`/`MatrixStore` are NOT gaps (not taught yet) and still decline from `stmts()`.

The by-name test now builds checker-conforming kernels (32-lane subgroup, block 128, Shared/Private/Tensor allocations) and asserts the decline is not the unpopped checker's: the earlier test passed vacuously for `Wgmma`/`Tcgen05` because the checker's `WgmmaUnsupported`/`Tcgen05Unsupported` text also contains the name. Mutation check: with the `Tensor` allocation gap disabled the test fails on the Tcgen05 cases.

— lane: fishstringer
