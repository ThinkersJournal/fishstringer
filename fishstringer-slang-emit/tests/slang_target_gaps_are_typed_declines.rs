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
    assert!(!decline(&one).starts_with(GAP_PREFIX));
    l.devices = 2;
    gap(&empty().with_launch(l), "multi-device");
    l.devices = 1;
    l.cluster = Some([2, 1, 1]);
    gap(&empty().with_launch(l), "cluster launch");
}
