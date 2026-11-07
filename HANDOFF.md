# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Plan: `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
#6 (ci --workspace, 0.0.2) MERGED (main febfb87). Done: main merged into #2 (0.0.3, head d19e1d2, 11/11 green, log shows the kernels crate tested), #4 = 0.0.5 (d7c0a03), #5 = 0.0.6 (734ba5e), CI pending on #4/#5. TODO: after the PM merges #2 (--merge): #3 merge main + version 0.0.4 (resolve Cargo.toml to the workspace form), retarget #4 and #5 to main. PM adds the slang-emit job to required checks after #5.
Stacked PRs, PM merges in this order (#2 with --merge, keeps 69 Fuel commits; the others are history imports too, ask for --merge):
- #2 kernels-import (0.0.2): fishstringer-kernels-source. 11/11 green. PM merges it himself.
- #3 publish-plan: docs/PUBLISH_PLAN.md. PM ruled: after #2 merges, merge main in and bump workspace version to 0.0.3. Worktree ../fishstringer-wt-publishplan.
- #4 parse-import (base kernels-import, 0.0.4): fishstringer-slang-parse. Worktree ../fishstringer-wt-parse.
- #5 emit-import (base parse-import, 0.0.5): fishstringer-slang-emit + ci.yml fix (--workspace; pinned slangc job). Worktree ../fishstringer-wt-emit.
After #2 merges: retarget #4 to main (base kernels-import goes away), merge main into each branch so only own commits remain; then #5.
Publishing is the PM's (claims at 0.0.0 done: all 5 names exist). Vulkane confirmed the emitter's public API = Slang + required_slangc_args (in PUBLISH_PLAN).
Scratch imports (blob-identical to Vulkane f7fa204) are in /tmp/fsimport (imp-slang-parse, imp-slang-emit), already merged into #4/#5.

## Next
1. When #2 merges: do the #3 bump (0.0.3) and retarget #4/#5; send [READY]s as needed.
2. After publish of the three crates (PM): tell fuel and vulkane (Synapse) the versions; they switch and delete their copies; Vulkane runs softmax_f64_live/unary against our kernels; hlsl-emit depends on the emitter.
3. Per-file provenance measurement (line-overlap) for the 26 Fuel-changed + 4 added kernels (see KERNEL_RECONCILIATION Open).
4. Ask PM to add the new `slang-emit` CI job to required checks and set branch protection.
5. The 27 .glsl stay Vulkane's; ask where they live once Fuel's crate goes. Unpopped's Slang docs/corpus example: Unpopped's call.

WAIT: PM merge of #2 (then #3 bump 0.0.4, retarget #4/#5) | check: gh pr view 2 --repo ThinkersJournal/fishstringer --json state,mergedAt | wake: 2026-11-07T13:43:06Z
