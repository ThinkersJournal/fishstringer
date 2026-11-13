// SPDX-License-Identifier: MIT OR Apache-2.0
//! The recorded gaps of the Slang target (doc row E18): constructs Vulkan compute cannot express, each a typed
//! `UnsupportedKernel` decline whose reason starts with [`GAP_PREFIX`] and names the construct.
//! Peer/cluster *places* and `devices > 1` are not scanned yet: they need Unpopped variants crates.io lacks.

use unpopped::exec::{Extent, Scope};
use unpopped::kernel::{Block, KernelFn, Stmt};

/// Every gap decline's reason starts with this, so a consumer can tell "outside the target" from "not taught yet".
pub const GAP_PREFIX: &str = "slang target gap: ";

/// The first gap `kernel` uses, as a reason (without the prefix), or `None`.
pub(crate) fn first_gap(kernel: &KernelFn) -> Option<String> {
    for a in &kernel.allocs {
        if matches!(a.extent, Extent::Param(_)) {
            return Some(format!(
                "run-time dynamic shared extent (`{}`): the extent must be a compile-time constant",
                a.name
            ));
        }
    }
    block_gap(&kernel.body)
}

fn scope_gap(what: &str, scope: Scope) -> Option<String> {
    match scope {
        Scope::Cluster => Some(format!("{what} at cluster scope: Vulkan compute has no thread-block clusters")),
        _ => None,
    }
}

fn block_gap(b: &Block) -> Option<String> {
    b.0.iter().find_map(stmt_gap)
}

fn stmt_gap(s: &Stmt) -> Option<String> {
    match s {
        Stmt::Barrier { scope, .. } => {
            if *scope == Scope::Device {
                return Some("device-scope barrier (grid sync): Vulkan compute has no grid-wide barrier".into());
            }
            scope_gap("barrier", *scope)
        }
        Stmt::Fence { scope, .. } => scope_gap("fence", *scope),
        Stmt::AtomicRmw { scope, .. } | Stmt::AtomicCas { scope, .. } => scope_gap("atomic", *scope),
        Stmt::Mma { atom, .. } => {
            if atom.is_sparse() {
                Some(format!("sparse (2:4) Mma atom `{}`: no Slang spelling", atom.name()))
            } else if atom.block().is_some() {
                Some(format!("block-scaled Mma atom `{}`: no Slang spelling", atom.name()))
            } else {
                None
            }
        }
        Stmt::If { then, els, .. } => block_gap(then).or_else(|| block_gap(els)),
        Stmt::For { body, .. } | Stmt::While { body, .. } => block_gap(body),
        _ => None,
    }
}
