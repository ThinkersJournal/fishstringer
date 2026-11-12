# Changelog

Entries are written by `tools/stamp-version.py` from the fragments in `changelog.d/` (see `changelog.d/README.md`).
History before this file existed is the ROADMAP's Done table and the merged PRs (#1 to #15); versions 0.0.12 to 0.0.17
were published to crates.io.

## 0.0.20 — Slang emission coverage plan (docs)

Adds `docs/SLANG_EMISSION_COVERAGE.md`: what `fishstringer-slang-emit` lowers today (scalar contiguous Elementwise only; `lower_kernel` is the typed decline) and one row per construct family it would need to emit everything Slang can express, with owner, size, the test that will show it and the typed decline for what Slang cannot express. ROADMAP rows 17 to 19. Docs only; spellings marked probe are unverified until slangc and a device run.

## 0.0.19 — ROADMAP row 12 done (redirect crates published)

Moves ROADMAP row 12 to Done: `vulkane-slang-emit` and `vulkane-slang-parse` 0.22.0 are on crates.io over `fishstringer-slang` 0.0.17 (PM notice 2026-11-11; crates.io max_version 0.22.0 for both). Docs only.

## 0.0.18 — changelog fragments and `tools/stamp-version.py`

The repository adopts the portfolio's fragment-plus-stamp flow (PM task 2026-11-12): a PR adds one `changelog.d/<topic>.md`
and edits no version line, `Cargo.lock` or `CHANGELOG.md`; the PM stamps the number at merge. Adds `tools/stamp-version.py`,
`tools/check-fragment.py`, the `Changelog fragment` workflow and `CHANGELOG.md`. No crate code changes.

## 0.0.17 — NaN-propagating max/min

NaN guards in 12 kernels: an `isnan` select in `unary*` (relu), `reduce*` and `reduce_last_dim*` (including the Wave path),
because slangc 2026.8's `FMax`/`FMin` drop NaN. Regenerated at slangc 2026.8; the 12 changed `.spv` differ on purpose
(PR #15). Fuel measured 72 of 72 cases per adapter (AMD 610M, RTX 4070) propagating NaN.
