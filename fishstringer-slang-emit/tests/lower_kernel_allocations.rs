// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E5: static allocations. A `Shared` allocation is a module-scope `groupshared` array and a `Private` one a
//! function-local array; `slangc` compiles the result. What has no honest lowering declines by name.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn alloc(a: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc: a, index }
}

/// Each thread stages `a[tid]` in its own shared slot and a private array, then writes the sum to `out[tid]`.
/// (Per-thread only: reading another thread's slot needs the barriers of row E7.)
fn staged(space: MemorySpace, extent: Extent) -> KernelFn {
    let l = ScalarExpr::Local;
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    KernelFn::new(
        "staged",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: buf(0),
            },
            Stmt::Store {
                place: alloc(0, tid()),
                value: l(0),
            },
            Stmt::Store {
                place: alloc(1, ScalarExpr::ConstInt(2, ElementKind::U32)),
                value: l(0),
            },
            Stmt::Load {
                local: 1,
                place: alloc(0, tid()),
            },
            Stmt::Load {
                local: 2,
                place: alloc(1, ScalarExpr::ConstInt(2, ElementKind::U32)),
            },
            Stmt::Store {
                place: buf(1),
                value: ScalarExpr::Add(Box::new(l(1)), Box::new(l(2))),
            },
        ]),
    )
    .with_allocs(vec![
        AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            ElementKind::F32,
            Extent::Static(64),
        ),
        AllocDecl::new("scratch", space, ElementKind::F32, extent),
    ])
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

#[test]
fn shared_and_private_arrays_lower_and_slangc_compiles_them() {
    let g = Slang
        .lower_kernel(&staged(MemorySpace::Private, Extent::Static(4)))
        .expect("lowers");
    // module scope, before the entry point; the private array is inside it
    let shared = g.source.find("groupshared float a0[64];").expect("shared");
    let entry = g.source.find("[shader(\"compute\")]").expect("entry");
    let private = g.source.find("float a1[4];").expect("private");
    assert!(shared < entry && entry < private, "{}", g.source);
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
    std::fs::write(&src, &g.source).expect("write");
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", &g.name, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "slangc:\n{}\n{}",
        String::from_utf8_lossy(&out.stderr),
        g.source
    );
}

#[test]
fn allocations_with_no_honest_lowering_decline() {
    // a Constant allocation has no initializer in the IR
    // (read-only, so the kernel must not store to it: the checker refuses that before the emitter sees it)
    let mut k = staged(MemorySpace::Constant, Extent::Static(4));
    k.body.0.remove(2);
    let d = decline(&k);
    assert!(d.contains("Constant or Global allocation"), "{d}");
    // a launch-parameter extent is a target gap, named as one
    let d = decline(&staged(MemorySpace::Shared, Extent::Param(0)));
    assert!(d.starts_with(fishstringer_slang_emit::GAP_PREFIX), "{d}");
}
