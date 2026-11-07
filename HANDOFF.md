# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Read `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
In main (workspace 0.0.9): fishstringer-kernels-source, fishstringer-slang-parse, fishstringer-slang-emit, all with history
(#2 squashed by the PM's gate tool, repaired by #7; #4, #5 merge commits). Not published: the PM publishes (names claimed at 0.0.0),
order kernels-source, slang-parse, slang-emit. Open: #3 (docs: publish plan + KERNEL_RECONCILIATION addendum + this HANDOFF; 0.0.10; PM gates).
Rules: PM merges history imports BY HAND with --merge (his gate tool ignores holds and squashes); every pushed change takes a version
(PM allocates); lanes never publish; CI only runs on PRs to main. Required check to add (PM): "fishstringer-slang-emit (pinned slangc)".
Fuel and Vulkane notified 2026-11-07 (Synapse roles fuel, vulkane): they switch after publish, run their live tests (softmax_f64, unary_f64,
subgroup-stride kernels) against the crate, and delete their copies. Questions out: Fuel how it consumes a package + where the 27 .glsl go;
Vulkane whether the .glsl are removed from our kernels crate.

## Next
1. Watch #3 CI, then gate. Answer Fuel/Vulkane replies; act on glsl decision (removing .glsl from the crate = a PR + version).
2. After publish: confirm versions with the PM and tell fuel/vulkane the exact versions; handle any live-test failure they report.
3. Read the 26 Fuel-changed kernel diffs line by line (only classified so far); measure candle line-overlap on the 30 post-copy files.
4. Roadmap: Triton/Mojo parked (PORTFOLIO-ROADMAP L2/L3: no way to test yet); Unpopped Slang docs/corpus example move is Unpopped's call.
5. Ask PM about branch protection when the first publish is done.

WAIT: PM gate/merge of #3 and publish of the three crates | check: gh pr view 3 --repo ThinkersJournal/fishstringer --json state,mergedAt; curl crates.io API for fishstringer-slang-emit max_version | wake: 2026-11-07T15:48:50Z
