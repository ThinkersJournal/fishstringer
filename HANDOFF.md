# fishstringer HANDOFF

You are the `fishstringer` lane (cwd `C:\Projects\fishstringer`, repo `ThinkersJournal/fishstringer`). Owner of Slang (CireSnave
2026-11-06 via PM). Read `docs/CHARTER.md`, `docs/SLANG_INVENTORY.md`, `docs/KERNEL_RECONCILIATION.md`, `docs/PUBLISH_PLAN.md`.

## State (2026-11-07)
In main (0.0.9): fishstringer-kernels-source, -slang-parse, -slang-emit with history (#2 squashed by the PM's gate tool, repaired by #7). Branch protection is on (8 required checks incl. "fishstringer-slang-emit (pinned slangc)").
Open PRs, all based on main, PM gates (squash ok): #3 docs + this HANDOFF (0.0.10, worktree ../fishstringer-wt-publishplan), #9 drop 27 .glsl (0.0.11, ../fishstringer-wt-glsl; PM ruling, Vulkane confirmed it holds them), #8 compile.sh flags for Fuel (--src-dir, --print-manifest, --print-toolchain, hard fail on missing tool; 0.0.12, ../fishstringer-wt-compile; ask PM to add job "fishstringer-kernels-source (manifest, scripts)" to required checks).
Order: #9 first, then rebase #8 over it (both edit compile.sh; #8 drops glsl in manifest loop and toolchain print), then PM publishes kernels-source, slang-parse, slang-emit (PUBLISH_PLAN). Publish only after #9, #8, #3 are in: Fuel needs #8's flags in the package.
Rules: PM merges history imports by hand --merge (n/a now); every pushed change takes a version (PM allocates); lanes never publish.
Fuel: wants a pinned regen step, will byte-compare regenerated vs committed .spv; asked for the flags (done in #8). Vulkane: switches after publish (hlsl-emit, spirv-emit dev-dep, ci slangc job).

## Next
1. Watch CI on #3/#8/#9; after #9 merges rebase #8 (remove glsl/glslang bits it still carries: manifest CI loop `*.glsl`, toolchain print, README lines) and bump if the PM says.
2. After publish: tell fuel/vulkane the exact versions; handle any live-test failure they report (softmax_f64, unary_f64, 26 subgroup kernels).
3. Read the 26 Fuel-changed kernel diffs line by line (only classified); candle line-overlap on the 30 post-copy files.
4. Triton/Mojo stay parked (PORTFOLIO-ROADMAP L2/L3). Unpopped's Slang docs/corpus example move is Unpopped's call.

WAIT: PM gate of #9, #8, #3 then publish of the three crates | check: gh pr list --repo ThinkersJournal/fishstringer --state all --json number,state; curl crates.io API max_version fishstringer-slang-emit | wake: 2026-11-07T15:50:27Z
