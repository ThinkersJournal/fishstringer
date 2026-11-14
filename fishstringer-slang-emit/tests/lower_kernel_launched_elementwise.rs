// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E1, first slice: a launched elementwise kernel lowers to Slang that `slangc` compiles to SPIR-V, with the
//! `KernelInterface` the emitter owns (entry symbol, one binding per buffer, in order). What stays declined is checked too.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

/// `out[tid] = a[tid] * 2` on 64 threads.
fn doubler() -> KernelFn {
    KernelFn::new(
        "double it",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ElementKind::F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Mul(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Const(2.0)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

#[test]
fn a_launched_elementwise_kernel_lowers_with_its_interface() {
    let g = Slang.lower_kernel(&doubler()).expect("lowers");
    assert_eq!(g.name, "double_it");
    assert!(g.source.contains("[numthreads(64, 1, 1)]"), "{}", g.source);
    assert!(
        g.source.contains("StructuredBuffer<float> b0"),
        "{}",
        g.source
    );
    assert!(
        g.source.contains("RWStructuredBuffer<float> b1"),
        "{}",
        g.source
    );
    let i = g.interface().expect("the emitter states an interface");
    assert_eq!(i.entry_symbol.as_deref(), Some("double_it"));
    let slots: Vec<(usize, u32, u32)> = i
        .bindings
        .iter()
        .map(|b| (b.param, b.space, b.slot))
        .collect();
    assert_eq!(slots, vec![(0, 0, 0), (1, 0, 1)]);
    assert!(
        i.params_block.is_none(),
        "no views or scalars yet, so no parameter block"
    );
}

#[test]
fn the_emitted_source_compiles_with_slangc() {
    if Command::new("slangc").arg("-v").output().is_err() {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_SLANGC").is_none(),
            "FISHSTRINGER_REQUIRE_SLANGC is set but `slangc` could not be run"
        );
        eprintln!("skipping: `slangc` not on PATH");
        return;
    }
    let g = Slang.lower_kernel(&doubler()).expect("lowers");
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

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

#[test]
fn what_is_not_lowered_yet_is_a_named_decline() {
    // no launch
    let mut k = doubler();
    k.launch = None;
    assert!(!decline(&k).is_empty());
    // an expression outside the slice
    let mut k = doubler();
    k.body.0[1] = Stmt::Store {
        place: at(1),
        value: ScalarExpr::Div(
            Box::new(ScalarExpr::Local(0)),
            Box::new(ScalarExpr::Const(2.0)),
        ),
    };
    let d = decline(&k);
    assert!(d.contains("not lowered yet"), "{d}");
}
