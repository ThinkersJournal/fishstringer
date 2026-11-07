# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Read `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
In main (0.0.10): fishstringer-kernels-source, -slang-parse, -slang-emit with history (#2 squashed by the PM's gate tool, repaired by #7), publish plan and
reconciliation docs (#3). Branch protection on. Worktrees/branches of merged PRs #1-#7 removed; primary checkout is on main, fast-forwarded.
#9 (drop 27 .glsl) MERGED (main 6cb870d, 0.0.11). Open: #8 compile.sh flags for Fuel, main merged in and glsl leftovers dropped, head 7077931, green, [READY] sent, PM gates (squash ok): #8
(--src-dir, --print-manifest, --print-toolchain, hard fail on a missing tool; 0.0.12, ../fishstringer-wt-compile; its new job "fishstringer-kernels-source (manifest, scripts)" is for the PM to add
to required checks). 
Then the PM publishes kernels-source, slang-parse, slang-emit (PUBLISH_PLAN), only after #9 and #8 are in (Fuel needs #8's flags in the package).
Rules: every pushed change takes a version (PM allocates); lanes never publish; CI only runs on PRs to main; the PM's gate tool squashes (history imports are done).
Fuel: pinned regen step, byte-compare of regenerated vs committed .spv. Vulkane: switches after publish (hlsl-emit, spirv-emit dev-dep, ci slangc job).

## Next
1. Watch CI on #3/#8/#9; after #9 merges rebase #8 (remove glsl/glslang bits it still carries: manifest CI loop `*.glsl`, toolchain print, README lines) and bump if the PM says.
2. After publish: tell fuel/vulkane the exact versions; handle any live-test failure they report (softmax_f64, unary_f64, 26 subgroup kernels).
3. DONE: the 26 Fuel-changed diffs read and pattern-checked (KERNEL_RECONCILIATION). Left: candle line-overlap on the 30 post-copy files.
4. Triton/Mojo stay parked (PORTFOLIO-ROADMAP L2/L3). Unpopped's Slang docs/corpus example move is Unpopped's call.

WAIT: PM gate of #8 then publish of the three crates | check: gh pr view 8 --repo ThinkersJournal/fishstringer --json state,mergedAt; curl crates.io API max_version fishstringer-slang-emit | wake: 2026-11-07T17:35:14Z
