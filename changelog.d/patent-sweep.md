kind: patent sweep note for async-copy lowering (docs)
---
Adds `docs/PATENT_SWEEP.md`: the PM's notice on NVIDIA patents US12118382B2 / US11294713B2 (claim 1 quoted), and what this repo's only emitter (Slang to SPIR-V) does about `CopyAsync`/`AsyncCommit`/`AsyncWait`: row E14 lowers them to plain synchronous loads and stores, so no async-scheduling instruction is emitted. The shipped `.spv` and competitors' instruction sets were not examined. Docs only.
