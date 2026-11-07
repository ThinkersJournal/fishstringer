// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fishstringer-kernels-source
//!
//! Home for the Slang GPU kernel sources and the `compile.sh` script that turns them into SPIR-V.
//! FishStringer owns Slang (CireSnave, 2026-11-06); the sources were
//! moved here with their git history from Fuel's `fuel-kernels-source` and Vulkane's
//! `vulkane-kernels-source` (Slang parts). See `README.md` and `THIRD-PARTY-NOTICES.md`.
//!
//! This crate intentionally has no Rust code. Consumers read the sources from the published package
//! (`kernels/`) and build or embed SPIR-V themselves.
//!
//! ## Layout
//!
//! - `kernels/*.slang`, `kernels/*.slangh`: Slang sources and shared headers.
//!
//! GLSL sources are not in this crate: GLSL is Vulkane's (`vulkane-kernels-source`). The 27 that sat beside the
//! Slang files in Fuel were removed before the first publish; their history is in git.
//!
//! - `kernels/compile.sh [--out-dir DIR] [files...]`: ahead-of-time compile; SPIR-V goes to `DIR`
//!   (default: `spv/` next to `kernels/`).
//! - `kernels/hoist_extinst_imports.py` (tested by `test_hoist_extinst_imports.py`): hoists
//!   `OpExtInstImport` out of function bodies in `slangc` output.
//! - `kernels/no_contraction.py`: decorates a kernel's float arithmetic `NoContraction` when the source
//!   carries a `// no-contraction` comment.
