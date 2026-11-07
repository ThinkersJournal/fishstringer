# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang
(CireSnave 2026-11-06 via PM). Read `docs/CHARTER.md` and `docs/SLANG_INVENTORY.md` first; they hold the plan.

## State (2026-11-07)
- PR #1 (charter + inventory) merged (main 2326f66). Rulings from PM 2026-11-06: crate names `fishstringer-kernels-source`,
  `fishstringer-slang-emit`, `fishstringer-slang-parse` (PM claims at 0.0.0 and publishes; lanes never publish); the 3
  divergent kernels are reconciled by me (record: `docs/KERNEL_RECONCILIATION.md`); Fuel and Vulkane confirm in one round.
- Branch `kernels-import` (worktree `C:\Projectsishstringer-wt-kernels`): Fuel history import + reconciliation + crate
  scaffold, workspace version 0.0.2. PR opened; [READY] sent. Local fmt/clippy/test/doc/deny green.
- Not run: GPU live tests (no GPU lane here); Fuel and Vulkane run theirs when they switch.

## Next
1. After the kernels PR merges: tell `fuel` and `vulkane` (Synapse) the crate name + the reconciliation doc; they switch and delete their copies.
2. Emitter + parser import: `vulkane-slang-emit`, `vulkane-slang-parse` (history-preserving, same fast-export method: python
   rewrite of M/D lines only, ref renamed to refs/heads/main, blob-identity check), rename to `fishstringer-slang-*`.
   Vulkane's hlsl-emit needs the list of public speller functions (ask Vulkane which, document as API).
3. Per-file provenance table for moved kernels (line-overlap measurement on the 26+4 files changed after the copy).
4. The 27 .glsl stay Vulkane's: ask Vulkane/PM where they live after Fuel's crate goes.
5. Tell PM when the first CI run on main has finished (done for #1: all green; branch protection is the PM's).

WAIT: kernels PR checks and merge | check: gh pr list --repo ThinkersJournal/fishstringer --head kernels-import --json number,state,statusCheckRollup | wake: 2026-11-07T12:31:49Z
