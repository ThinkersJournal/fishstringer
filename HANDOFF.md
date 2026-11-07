# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Plan: `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
#2 squash-merged by mistake; repaired by #7 (merged by hand, main 3f95e41; 69 Fuel commits now ancestors). Rule: PM merges #4 and #5 BY HAND with --merge (his gate tool squashes: never let it touch them); #3 goes through the gate (squash ok).
- #4 parse-import (0.0.8, head 222150b), #5 emit-import (0.0.9, head a7a773f; carries #4's crate until #4 lands), #3 publish-plan (0.0.10, 94ea8c9). All three [READY] sent; all based on main.
- PM adds the slang-emit CI job to required checks after #5 lands. Publishing is the PM's (names claimed at 0.0.0).

## Next
1. Watch CI on #3/#4/#5 (run only on PRs to main: first run after retarget); fix reds.
2. After publish of the three crates (PM): tell fuel and vulkane (Synapse) the versions; they switch and delete their copies; Vulkane runs softmax_f64_live/unary against our kernels; hlsl-emit depends on the emitter.
3. Per-file provenance measurement (line-overlap) for the 26 Fuel-changed + 4 added kernels (see KERNEL_RECONCILIATION Open).
4. Ask PM to add the new `slang-emit` CI job to required checks and set branch protection.
5. The 27 .glsl stay Vulkane's; ask where they live once Fuel's crate goes. Unpopped's Slang docs/corpus example: Unpopped's call.

WAIT: PM merges of #4, #5, #3 and their CI | check: gh pr list --repo ThinkersJournal/fishstringer --state all --json number,state,mergedAt | wake: 2026-11-07T15:10:23Z
