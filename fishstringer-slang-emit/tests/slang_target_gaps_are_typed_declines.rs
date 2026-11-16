// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row E18: each recorded gap of the Slang target is a typed `UnsupportedKernel` decline whose reason starts with
//! `GAP_PREFIX` and names the construct. A kernel with no gap gets the generic "does not lower" decline instead,
//! so the prefix really discriminates (the control).

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, MemorySpace, Scope};
use unpopped::kernel::{Block, KernelFn, Stmt};
use unpopped_vocab::ElementKind;

fn empty() -> KernelFn {
    KernelFn::new("k", ElementKind::F32, vec![], vec![], Block(vec![]))
}

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

fn gap(k: &KernelFn, names: &str) {
    let d = decline(k);
    assert!(d.starts_with(GAP_PREFIX), "{d}");
    assert!(d.contains(names), "{d}");
}

#[test]
fn a_kernel_without_a_gap_gets_the_generic_decline_not_the_gap_prefix() {
    assert!(!decline(&empty()).starts_with(GAP_PREFIX));
}

#[test]
fn dynamic_shared_extent_declines() {
    let mut k = empty();
    k.allocs.push(AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Param(0),
    ));
    gap(&k, "dynamic shared extent");
}

#[test]
fn a_static_shared_extent_is_not_a_gap() {
    let mut k = empty();
    k.allocs.push(AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(64),
    ));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}

#[test]
fn device_and_cluster_barriers_decline() {
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Device));
    gap(&k, "grid");
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Cluster));
    gap(&k, "cluster");
}

#[test]
fn a_block_barrier_is_not_a_gap() {
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Block));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}

#[test]
fn the_patent_flagged_async_statements_decline_by_name() {
    for st in [Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }] {
        let mut k = empty();
        k.body.0.push(st);
        gap(&k, "async copy statement");
        gap(&k, "PATENT_SWEEP");
    }
}

#[test]
fn peer_cluster_and_multicast_places_decline() {
    use unpopped::ir::ScalarExpr;
    use unpopped::kernel::Place;
    let zero = || ScalarExpr::Const(0.0);
    let places = [
        (
            Place::Peer {
                buffer: 0,
                device: zero(),
                coords: vec![zero()],
            },
            "peer-device",
        ),
        (
            Place::ClusterPeer {
                alloc: 0,
                block: zero(),
                index: zero(),
            },
            "cluster-peer",
        ),
        (
            Place::Multicast {
                buffer: 0,
                coords: vec![zero()],
            },
            "multicast",
        ),
    ];
    for (place, what) in places {
        let mut k = empty();
        k.body.0.push(Stmt::Store {
            place,
            value: zero(),
        });
        gap(&k, what);
    }
}

#[test]
fn multi_device_and_cluster_launches_decline_and_one_device_does_not() {
    use unpopped::exec::LaunchShape;
    let mut l = LaunchShape::new([1, 1, 1], [1, 1, 1]);
    let one = empty().with_launch(l);
    // An empty launched kernel lowers now (E1); the control is only that it is not a gap decline.
    match Slang.lower_kernel(&one) {
        Ok(_) => {}
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(!detail.starts_with(GAP_PREFIX), "{detail}")
        }
        Err(other) => panic!("{other:?}"),
    }
    l.devices = 2;
    gap(&empty().with_launch(l), "multi-device");
    l.devices = 1;
    l.cluster = Some([2, 1, 1]);
    gap(&empty().with_launch(l), "cluster launch");
}

// --- E18 completion: the gaps the list names that had no test of their own ---

fn zero() -> unpopped::ir::ScalarExpr {
    unpopped::ir::ScalarExpr::Const(0.0)
}

#[test]
fn a_cluster_scope_fence_and_atomic_decline() {
    use unpopped::kernel::{AtomicOp, MemoryOrder, Place};
    let mut k = empty();
    k.body.0.push(Stmt::Fence {
        scope: Scope::Cluster,
        order: MemoryOrder::AcqRel,
    });
    gap(&k, "fence at cluster scope");
    let mut k = empty();
    k.body.0.push(Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: Place::Alloc {
            alloc: 0,
            index: zero(),
        },
        value: zero(),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Cluster,
    });
    gap(&k, "atomic at cluster scope");
}

#[test]
fn a_copy_async_into_a_peer_place_declines_as_the_async_gap() {
    use unpopped::kernel::Place;
    let mut k = empty();
    k.body.0.push(Stmt::CopyAsync {
        dst: Place::ClusterPeer {
            alloc: 0,
            block: zero(),
            index: zero(),
        },
        src: Place::Alloc {
            alloc: 1,
            index: zero(),
        },
        elems: 4,
    });
    // Whichever gap the scan meets first, the decline is a named gap, never a silent lowering.
    let d = decline(&k);
    assert!(d.starts_with(GAP_PREFIX), "{d}");
}

#[test]
fn multicast_load_reduce_declines() {
    use unpopped::kernel::{AtomicOp, MemoryOrder, Place};
    let mut k = empty();
    k.body.0.push(Stmt::MulticastLoadReduce {
        local: 0,
        place: Place::Multicast {
            buffer: 0,
            coords: vec![zero()],
        },
        op: AtomicOp::Add,
        order: MemoryOrder::Relaxed,
    });
    gap(&k, "multicast");
}

#[test]
fn sparse_and_block_scaled_mma_atoms_decline_and_a_dense_one_does_not() {
    use unpopped::mma::MmaAtom;
    use unpopped_vocab::layout_algebra::Layout;
    let mma = |atom: MmaAtom| Stmt::Mma {
        atom,
        a: 0,
        b: 1,
        c: 2,
        d: 3,
        extra: vec![],
    };
    let leaf = |n: i64| Layout::leaf(n, 1).expect("a leaf layout");
    let dense = MmaAtom::nvidia_m16n8k16_f16_f32();

    let sparse = dense
        .clone()
        .with_sparse_24(&leaf(128), (ElementKind::U32, &leaf(64)))
        .expect("a sparse atom");
    let scaled = dense
        .clone()
        .with_block_scale(4, ElementKind::F32, (&leaf(64), &leaf(32)))
        .expect("a block-scaled atom");

    let mut k = empty();
    k.body.0.push(mma(sparse));
    gap(&k, "sparse (2:4) Mma atom");
    let mut k = empty();
    k.body.0.push(mma(scaled));
    gap(&k, "block-scaled Mma atom");

    // Control: the dense atom is not a gap (it is just not lowered yet).
    let mut k = empty();
    k.body.0.push(mma(dense));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}
