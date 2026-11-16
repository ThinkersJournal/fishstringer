// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E10: atomics. A histogram (`InterlockedAdd` on a buffer), a shared-memory max and a compare-and-swap spin
//! lock compile with `slangc`; a float atomic and a cluster-scope atomic stay typed declines.

use std::process::Command;

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}

fn buf(b: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer: b,
        coords: vec![i],
    }
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

fn u32_locals(n: usize) -> Vec<LocalDecl> {
    (0..n)
        .map(|_| LocalDecl::scalar(ElementKind::U32))
        .collect()
}

/// `hist[data[tid]] += 1` (old value kept), `peak = max(peak, data[tid])` in shared, and a CAS on `lock[0]`.
fn kernel(elem: ElementKind, scope: Scope) -> KernelFn {
    let body = vec![
        Stmt::Load {
            local: 0,
            place: buf(0, tid()),
        },
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: buf(1, ScalarExpr::Local(0)),
            value: cu(1),
            old: Some(1),
            order: MemoryOrder::Relaxed,
            scope,
        },
        Stmt::AtomicRmw {
            op: AtomicOp::Max,
            place: Place::Alloc {
                alloc: 0,
                index: cu(0),
            },
            value: ScalarExpr::Local(0),
            old: None,
            order: MemoryOrder::AcqRel,
            scope: Scope::Block,
        },
        Stmt::AtomicCas {
            place: buf(2, cu(0)),
            expected: cu(0),
            desired: cu(1),
            old: Some(2),
            order: MemoryOrder::Acquire,
            scope: Scope::Device,
        },
    ];
    KernelFn::new(
        "hist",
        ElementKind::U32,
        vec![
            Param::buffer("data", ElementKind::U32, BufferRole::Read, Some(1)),
            Param::buffer("hist", elem, BufferRole::ReadWrite, Some(1)),
            Param::buffer("lock", ElementKind::U32, BufferRole::ReadWrite, Some(1)),
        ],
        u32_locals(3),
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "peak",
        MemorySpace::Shared,
        ElementKind::U32,
        Extent::Static(1),
    )])
    .with_launch(LaunchShape::new([64, 1, 1], [4, 1, 1]))
}

#[test]
fn histogram_shared_max_and_cas_compile() {
    let g = Slang
        .lower_kernel(&kernel(ElementKind::U32, Scope::Device))
        .expect("lowers");
    for want in [
        "InterlockedAdd(b1[",
        ", l1);",
        "InterlockedMax(a0[((uint)0)], l0);",
        "InterlockedCompareExchange(b2[",
        ", l2);",
    ] {
        assert!(g.source.contains(want), "{want} in\n{}", g.source);
    }
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn every_integer_combine_compiles() {
    for (op, name) in [
        (AtomicOp::Min, "InterlockedMin"),
        (AtomicOp::And, "InterlockedAnd"),
        (AtomicOp::Or, "InterlockedOr"),
        (AtomicOp::Xor, "InterlockedXor"),
        (AtomicOp::Exchange, "InterlockedExchange"),
    ] {
        let mut k = kernel(ElementKind::U32, Scope::Device);
        k.body = Block(vec![Stmt::AtomicRmw {
            op,
            place: buf(1, tid()),
            value: cu(3),
            old: None,
            order: MemoryOrder::SeqCst,
            scope: Scope::Device,
        }]);
        let g = Slang.lower_kernel(&k).expect("lowers");
        assert!(g.source.contains(&format!("{name}(b1[")), "{}", g.source);
        slangc_compiles(&g.source, &g.name);
    }
}

#[test]
fn a_float_atomic_declines_by_name() {
    let mut k = kernel(ElementKind::F32, Scope::Device);
    k.compute = ElementKind::F32;
    k.body = Block(vec![Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: buf(1, cu(0)),
        value: ScalarExpr::Const(1.0),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }]);
    match Slang.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("float atomic"), "{detail}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}

#[test]
fn a_cluster_scope_atomic_is_a_named_gap() {
    match Slang.lower_kernel(&kernel(ElementKind::U32, Scope::Cluster)) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.starts_with(GAP_PREFIX), "{detail}");
        }
        other => panic!("expected a gap decline, got {other:?}"),
    }
}
