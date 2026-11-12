kind: refresh the changelog tools to kit v3
---
Re-copies `tools/stamp-version.py` and `tools/check-fragment.py` from the portfolio kit (refactored after Codacy flagged the first copies: bandit subprocess use, cyclomatic complexity). No change in behaviour. Tooling only; no crate code changes.
