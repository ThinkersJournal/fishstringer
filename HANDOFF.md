# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Plan: `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
PM error: #2 was SQUASH-merged (main 60ceecf); the 69 Fuel commits are not ancestors of main. Repair = #7 (history-import, `git merge -s ours --no-ff d19e1d2`, 0.0.7, worktree ../fishstringer-wt-history); PM merges #7, #4 and #5 BY HAND with --merge (his gate tool ignores holds and squashes: never let it touch them).
- #6 merged (ci --workspace). #3 publish-plan (main-based, 0.0.3 in main now; it needs the next free patch).
- #4 parse-import (retargeted to main by the PM's tool; reads wrong until #7 lands: merge-base problem), #5 emit-import (base parse-import).
- After #7 merges: merge main into #4, #5, #3; renumber #4 0.0.8, #5 0.0.9, #3 0.0.10 (confirm with PM); retarget #5 to main; [READY] each; PM adds the slang-emit CI job to required checks after #5.
- Publishing is the PM's (names claimed at 0.0.0). Vulkane confirmed emitter API = Slang + required_slangc_args (PUBLISH_PLAN).

## Next
1. When #2 merges: do the #3 bump (0.0.3) and retarget #4/#5; send [READY]s as needed.
2. After publish of the three crates (PM): tell fuel and vulkane (Synapse) the versions; they switch and delete their copies; Vulkane runs softmax_f64_live/unary against our kernels; hlsl-emit depends on the emitter.
3. Per-file provenance measurement (line-overlap) for the 26 Fuel-changed + 4 added kernels (see KERNEL_RECONCILIATION Open).
4. Ask PM to add the new `slang-emit` CI job to required checks and set branch protection.
5. The 27 .glsl stay Vulkane's; ask where they live once Fuel's crate goes. Unpopped's Slang docs/corpus example: Unpopped's call.

WAIT: PM merge of #7 by hand (then merge main into #4/#5/#3, renumber, retarget #5) | check: gh pr view 7 --repo ThinkersJournal/fishstringer --json state,mergedAt | wake: 2026-11-07T14:40:49Z
