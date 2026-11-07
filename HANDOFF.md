# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Read `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
In main (0.0.10): fishstringer-kernels-source, -slang-parse, -slang-emit with history (#2 squashed by the PM's gate tool, repaired by #7), publish plan and
reconciliation docs (#3). Branch protection on. Worktrees/branches of merged PRs #1-#7 removed; primary checkout is on main, fast-forwarded.
Open, both green, PM gates (squash ok): #9 drop 27 .glsl (0.0.11, worktree ../fishstringer-wt-glsl; PM ruling, Vulkane holds all 27) and #8 compile.sh flags for Fuel
(--src-dir, --print-manifest, --print-toolchain, hard fail on a missing tool; 0.0.12, ../fishstringer-wt-compile; its new job "fishstringer-kernels-source (manifest, scripts)" is for the PM to add
to required checks). Both edit compile.sh: when #9 merges, merge main into #8 and drop what #8 still carries about glsl/glslang (manifest CI loop `*.glsl`, toolchain print, README).
Then the PM publishes kernels-source, slang-parse, slang-emit (PUBLISH_PLAN), only after #9 and #8 are in (Fuel needs #8's flags in the package).
Rules: every pushed change takes a version (PM allocates); lanes never publish; CI only runs on PRs to main; the PM's gate tool squashes (history imports are done).
Fuel: pinned regen step, byte-compare of regenerated vs committed .spv. Vulkane: switches after publish (hlsl-emit, spirv-emit dev-dep, ci slangc job).

## Next
1. Watch CI on #3/#8/#9; after #9 merges rebase #8 (remove glsl/glslang bits it still carries: manifest CI loop `*.glsl`, toolchain print, README lines) and bump if the PM says.
2. After publish: tell fuel/vulkane the exact versions; handle any live-test failure they report (softmax_f64, unary_f64, 26 subgroup kernels).
3. DONE: the 26 Fuel-changed diffs read and pattern-checked (KERNEL_RECONCILIATION). Left: candle line-overlap on the 30 post-copy files.
4. Triton/Mojo stay parked (PORTFOLIO-ROADMAP L2/L3). Unpopped's Slang docs/corpus example move is Unpopped's call.

WAIT: PM gate of #9 and #8 then publish of the three crates | check: gh pr list --repo ThinkersJournal/fishstringer --state all --json number,state; curl crates.io API max_version fishstringer-slang-emit | wake: 2026-11-07T16:46:00Z
