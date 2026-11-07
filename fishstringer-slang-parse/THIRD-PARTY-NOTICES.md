# Third-party notices

**No third-party source is copied into this crate.**

## Origin

- `parse_slang`, `SLANG_RESIDUE` and the `SLANG` frontend are the Slang-specific pieces of Unpopped's
  `crates/unpopped/src/convert.rs` at commit `213df4f896b5e0a4dd0db4ef84bce54ab18f9654` (Unpopped v0.34.11, same
  owner, `MIT OR Apache-2.0`). They were re-created here, not file-copied, because `convert.rs` also holds the
  language-neutral walkers that stay in Unpopped; so unlike the emitter this crate does not carry the
  history of those lines. Checked 2026-10-30 against that commit: `parse_slang` is the same body, the seven
  `SLANG_RESIDUE` strings are the same in the same order, and `SLANG` has the same five values (`"slang"`,
  `"numthreads"`, `SLANG_RESIDUE`, `"output"`, `"input"`, plus `parse_slang`), built through `Frontend::new`
  (the public constructor of the `#[non_exhaustive]` struct) instead of a struct literal.
- The unit tests are adapted from the Slang tests in that file's `tests` module; `tests/corpus.rs` repeats the
  method of Unpopped's `docs/slang-corpus-sizing.md` over `fishstringer-kernels-source`'s `.slang` kernels.

## Dependencies (not copied or redistributed here)

- `tree-sitter-slang` 0.3 (MIT, by the repository owner of https://github.com/theHamsta/tree-sitter-slang).
- `tree-sitter` (MIT) and `unpopped` (`MIT OR Apache-2.0`).

Every source file declares `SPDX-License-Identifier: MIT OR Apache-2.0`.

## Move history

This crate was `vulkane-slang-parse` in Vulkane (`https://github.com/ciresnave/vulkane`, same owner, same licence) until 2026-11-07;
its git history was imported unchanged (26 commits, blob-identical to Vulkane `f7fa204`). The Cargo package name, the
repository URL and the `vulkane`-to-`fishstringer` names in docs and tests were changed; no code was.
