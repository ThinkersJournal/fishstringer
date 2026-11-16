kind: CI moves the CodeRipper portfolio workflow to the 0.47.5 pin
---
The `portfolio` job now calls the shared workflow at `c72a6e03479d914de8ea8a73609ba73e93a0de92` (installs CodeRipper 0.47.5). This repo has no `.coderipper.toml` and no `--skip-rule` or `check <rule>` use, so the rule-id migration tool had nothing to rewrite (`migrate-rule-ids.py --dry-run` exits 2: no `.coderipper.toml`; a search of tracked files for the 14 old names finds none). CI configuration only; no code or version change.
