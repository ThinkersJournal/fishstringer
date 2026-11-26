// SPDX-License-Identifier: MIT OR Apache-2.0
//! The recorded gaps of the Slang target (doc row E18): constructs Vulkan compute cannot express, each a typed
//! `UnsupportedKernel` decline whose reason starts with [`GAP_PREFIX`] and names the construct.
//! Peer/cluster *places* and `devices > 1` are not scanned yet: they need Unpopped variants crates.io lacks.

use unpopped::exec::{Extent, MemorySpace, Scope};
use unpopped::kernel::{Block, KernelFn, Place, Stmt};

/// Every gap decline's reason starts with this, so a consumer can tell "outside the target" from "not taught yet".
pub const GAP_PREFIX: &str = "slang target gap: ";

/// The first gap `kernel` uses, as a reason (without the prefix), or `None`.
pub(crate) fn first_gap(kernel: &KernelFn) -> Option<String> {
    for a in &kernel.allocs {
        if a.space == MemorySpace::Tensor {
            return Some(format!(
                "tensor-memory allocation (`{}`, tcgen05): NVIDIA tensor memory has no Vulkan spelling",
                a.name
            ));
        }
        if matches!(a.extent, Extent::Param(_)) {
            return Some(format!(
                "run-time dynamic shared extent (`{}`): the extent must be a compile-time constant",
                a.name
            ));
        }
    }
    if let Some(l) = &kernel.launch {
        if l.devices > 1 {
            return Some(format!(
                "multi-device launch ({} devices): Vulkan compute launches on one device",
                l.devices
            ));
        }
        if l.cluster.is_some() {
            return Some("cluster launch: Vulkan compute has no thread-block clusters".into());
        }
    }
    block_gap(&kernel.body)
}

fn place_gap(p: &Place) -> Option<String> {
    match p {
        Place::ClusterPeer { .. } => {
            Some("cluster-peer place (distributed shared memory): Vulkan compute has none".into())
        }
        Place::Peer { .. } => {
            Some("peer-device place: Vulkan compute has no peer-device addressing".into())
        }
        Place::Multicast { .. } => Some("multicast place: Vulkan compute has no multimem".into()),
        _ => None,
    }
}

fn scope_gap(what: &str, scope: Scope) -> Option<String> {
    match scope {
        Scope::Cluster => Some(format!(
            "{what} at cluster scope: Vulkan compute has no thread-block clusters"
        )),
        _ => None,
    }
}

fn block_gap(b: &Block) -> Option<String> {
    b.0.iter().find_map(stmt_gap)
}

fn stmt_gap(s: &Stmt) -> Option<String> {
    match s {
        // Held on board 434 = C (CireSnave 2026-11-13): Slang to SPIR-V is non-NVIDIA, so these four are declined by
        // name. See docs/PATENT_SWEEP.md. A rule for what this target may ever emit, not a missing feature.
        Stmt::CopyAsync { .. } | Stmt::AsyncCommit | Stmt::AsyncWait { .. } => Some(
            "async copy statement (CopyAsync/AsyncCommit/AsyncWait): declined by name, patent risk, see docs/PATENT_SWEEP.md".into(),
        ),
        // Board 434 C extended (CireSnave 2026-11-16): NVIDIA US11803380B2 (mbarrier) and US12204897B2 / US12710961B2
        // (a wait on a count of pending matrix-multiply operations). Possible exposure: never lowered to a hardware
        // operation on this non-NVIDIA target. See docs/PATENT_SWEEP.md.
        Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::MbarWait { .. }
        | Stmt::CopyBulk { .. } => Some(
            "mbarrier statement (MbarInit/MbarArrive/MbarCompleteTx/MbarWait/CopyBulk): declined by name, possible exposure, board 434 C extended, see docs/PATENT_SWEEP.md".into(),
        ),
        // CopyUnpack (unpopped 0.58.0): NOT a docs/PATENT_SWEEP.md row by name. It completes on an mbarrier, and unpopped's
        // own doc note says the CopyBulk patent note applies and emitters for other manufacturers decline it. Declined by
        // name on that basis; the PM may reclassify.
        Stmt::CopyUnpack { .. } => Some(
            "CopyUnpack statement: FP4 unpack copy completing on an mbarrier; declined by name, unpopped 0.58.0 applies the CopyBulk patent note to it (not a docs/PATENT_SWEEP.md row by name)".into(),
        ),
        Stmt::WgmmaWait { .. } | Stmt::Tcgen05Commit { .. } => Some(
            "pending-matrix-multiply wait (WgmmaWait/Tcgen05Commit): declined by name, possible exposure, board 434 C extended, see docs/PATENT_SWEEP.md".into(),
        ),
        // U8c (unpopped 0.57.0): NVIDIA-only constructs with no Vulkan spelling, outside the target for good.
        Stmt::ProxyFence { .. } => Some(
            "ProxyFence statement: Vulkan compute has a single memory proxy and no async-proxy operation to order".into(),
        ),
        Stmt::Wgmma { .. } | Stmt::WgmmaCommit => Some(
            "Wgmma statement (Wgmma/WgmmaCommit): NVIDIA warpgroup matrix multiply, no Vulkan spelling".into(),
        ),
        Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Tcgen05Alloc { .. }
        | Stmt::Tcgen05Dealloc { .. }
        | Stmt::Tcgen05RelinquishPermit => Some(
            "Tcgen05 statement (Tcgen05Mma/Tcgen05Ld/Tcgen05Alloc/Tcgen05Dealloc/Tcgen05RelinquishPermit): NVIDIA tensor-memory matrix multiply, no Vulkan spelling".into(),
        ),
        Stmt::Collective { .. } | Stmt::CollectiveTuple { .. } => Some(
            "collective statement (Reduce/Scan/Shuffle/Vote/Broadcast, tuple forms): declined by name, patent risk, see docs/PATENT_SWEEP.md".into(),
        ),
        Stmt::Barrier { scope, .. } => {
            if *scope == Scope::Device {
                return Some(
                    "device-scope barrier (grid sync): Vulkan compute has no grid-wide barrier"
                        .into(),
                );
            }
            scope_gap("barrier", *scope)
        }
        Stmt::Load { place, .. }
        | Stmt::Store { place, .. }
        | Stmt::MaskedLoad { place, .. }
        | Stmt::MaskedStore { place, .. }
        | Stmt::AtomicRmw { place, .. }
        | Stmt::AtomicCas { place, .. }
            if place_gap(place).is_some() =>
        {
            place_gap(place)
        }
        Stmt::MulticastLoadReduce { .. } => Some("multicast load-reduce: Vulkan compute has no multimem".into()),
        Stmt::Fence { scope, .. } => scope_gap("fence", *scope),
        Stmt::AtomicRmw { scope, .. } | Stmt::AtomicCas { scope, .. } => {
            scope_gap("atomic", *scope)
        }
        Stmt::Mma { atom, .. } => {
            if atom.is_sparse() {
                Some(format!(
                    "sparse (2:4) Mma atom `{}`: no Slang spelling",
                    atom.name()
                ))
            } else if atom.block().is_some() {
                Some(format!(
                    "block-scaled Mma atom `{}`: no Slang spelling",
                    atom.name()
                ))
            } else {
                None
            }
        }
        Stmt::If { then, els, .. } => block_gap(then).or_else(|| block_gap(els)),
        Stmt::For { body, .. } | Stmt::While { body, .. } => block_gap(body),
        _ => None,
    }
}
