# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang
(CireSnave 2026-11-06 via PM). Read `docs/CHARTER.md` and `docs/SLANG_INVENTORY.md` first; they hold the plan.

## State (2026-11-06)
- PR #1 (charter + inventory, docs only) open, head f5703db. Branch protection not set; PM sets it after the first green CI on main.
- Sent PM [READY] for #1 and [ASK] (3 divergent kernels: compile.sh, softmax_f64, unary_f64; crate names for kernel/emitter/parser).
- Publishing is the PM's (onboarding); I prepare crates at 0.0.0 and ask. Do not delete anything in Fuel or Vulkane.

## Next
1. Read PM's answer; then step 2 of the plan: history-preserving kernel import (Fuel 5aa295281 newer for 26+4; Vulkane 9cdc2d4 for dequant_q4_km, matmul, hoist_extinst_imports.py + test); the 3 divergent files only as the PM rules.
2. Tell Fuel and Vulkane (Synapse roles `fuel`, `vulkane`) the crate names once claimed; Vulkane's hlsl-emit needs the emitter's public speller API list.
3. Then emitter + parser import (vulkane-slang-emit/-parse), provenance table per file, 27 .glsl stay (Vulkane's).

WAIT: PM answer to [ASK] kernel-conflicts+names | check: gh pr view 1 --repo ThinkersJournal/fishstringer --json state,comments (and Synapse fetch) | wake: 2026-11-06T17:37:45Z
WAIT: PR #1 checks and merge | check: gh pr checks 1 --repo ThinkersJournal/fishstringer; gh pr view 1 --repo ThinkersJournal/fishstringer --json state | wake: 2026-11-06T17:37:45Z
