// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E7: barriers and fences. A block-reversal through shared memory needs a barrier between the write and the
//! cross-thread read; the emitted source compiles with `slangc`. Device and cluster scopes stay typed declines.

use std::process::Command;

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// `out[tid] = a[63 - tid]`, staged through a shared tile; `extra` statements go between the barrier and the read.
fn reverse(extra: Vec<Stmt>) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    let tile = |i| Place::Alloc { alloc: 0, index: i };
    let mut body = vec![
        Stmt::Load {
            local: 0,
            place: buf(0),
        },
        Stmt::Store {
            place: tile(tid()),
            value: ScalarExpr::Local(0),
        },
        Stmt::barrier(Scope::Block),
    ];
    body.extend(extra);
    body.push(Stmt::Load {
        local: 1,
        place: tile(ScalarExpr::Sub(
            Box::new(ScalarExpr::ConstInt(63, ElementKind::U32)),
            Box::new(tid()),
        )),
    });
    body.push(Stmt::Store {
        place: buf(1),
        value: ScalarExpr::Local(1),
    });
    KernelFn::new(
        "reverse",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
        ],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(64),
    )])
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

fn slangc_compiles(source: &str, entry: &str) {
    if Command::new("slangc").arg("-v").output().is_err() {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_SLANGC").is_none(),
            "FISHSTRINGER_REQUIRE_SLANGC is set but `slangc` could not be run"
        );
        eprintln!("skipping: `slangc` not on PATH");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    std::fs::write(&src, source).expect("write");
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", entry, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "slangc:\n{}\n{source}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_block_barrier_separates_the_shared_write_from_the_cross_thread_read() {
    let g = Slang.lower_kernel(&reverse(vec![])).expect("lowers");
    let w = g.source.find("a0[tid_.x] = ").expect("write");
    let b = g
        .source
        .find("GroupMemoryBarrierWithGroupSync();")
        .expect("barrier");
    let r = g.source.find("l1 = a0[").expect("read");
    assert!(w < b && b < r, "{}", g.source);
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn subgroup_barrier_and_every_fence_scope_compile() {
    let extra = vec![
        Stmt::barrier(Scope::Subgroup),
        Stmt::Fence {
            scope: Scope::Block,
            order: MemoryOrder::AcqRel,
        },
        Stmt::Fence {
            scope: Scope::Subgroup,
            order: MemoryOrder::Release,
        },
        Stmt::Fence {
            scope: Scope::Device,
            order: MemoryOrder::SeqCst,
        },
    ];
    let g = Slang.lower_kernel(&reverse(extra)).expect("lowers");
    for want in [
        "AllMemoryBarrierWithWaveSync();",
        "GroupMemoryBarrier();",
        "AllMemoryBarrier();",
        "DeviceMemoryBarrier();",
    ] {
        assert!(g.source.contains(want), "{want} in\n{}", g.source);
    }
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn device_and_cluster_barriers_stay_named_gaps() {
    for scope in [Scope::Device, Scope::Cluster] {
        let k = reverse(vec![Stmt::barrier(scope)]);
        match Slang.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.starts_with(GAP_PREFIX), "{scope:?}: {detail}");
            }
            other => panic!("{scope:?}: expected a gap decline, got {other:?}"),
        }
    }
}
