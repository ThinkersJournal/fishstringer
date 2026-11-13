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
    k.allocs.push(AllocDecl::new("tile", MemorySpace::Shared, ElementKind::F32, Extent::Param(0)));
    gap(&k, "dynamic shared extent");
}

#[test]
fn a_static_shared_extent_is_not_a_gap() {
    let mut k = empty();
    k.allocs.push(AllocDecl::new("tile", MemorySpace::Shared, ElementKind::F32, Extent::Static(64)));
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
