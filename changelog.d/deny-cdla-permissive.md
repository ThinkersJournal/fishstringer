kind: `deny.toml` allows CDLA-Permissive-2.0 (the portfolio template carries it)
---
Adds `CDLA-Permissive-2.0` (the webpki-roots data licence, allowed portfolio-wide) to the licence allow list of `deny.toml`, which stays `all-features = true` (PM ruling 2026-11-18). No crate of ours has it in its lockfile today (`webpki-roots` is absent from `Cargo.lock`), so this changes no check result; it keeps the template identical across the portfolio. Verified with `cargo deny check licenses`. No code change.
