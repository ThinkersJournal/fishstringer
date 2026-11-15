// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E4: control flow and masked access. A kernel with a counted loop, a masked load, a branch, `Continue`
//! and `Return` lowers to Slang that `slangc` compiles to SPIR-V; the cases the target cannot honour decline by name.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn bx(a: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(a)
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(a), bx(b))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// `out[0] = sum of the positive a[i], i in 0..n` (one thread), written to exercise every E4 statement.
fn positive_sum(step: ScalarExpr) -> KernelFn {
    let l = ScalarExpr::Local;
    KernelFn::new(
        "positive_sum",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
            Param::scalar("n", ElementKind::U32, 2),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32), // 0 acc
            LocalDecl::scalar(ElementKind::U32), // 1 i
            LocalDecl::scalar(ElementKind::U32), // 2 n
            LocalDecl::scalar(ElementKind::F32), // 3 v
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(0.0),
            },
            Stmt::For {
                local: 1,
                start: ScalarExpr::ConstInt(0, ElementKind::U32),
                end: l(2),
                step,
                body: Block(vec![
                    Stmt::MaskedLoad {
                        local: 3,
                        place: at(0, l(1)),
                        mask: cmp(BinaryOp::CmpLt, l(1), l(2)),
                        fill: ScalarExpr::Const(0.0),
                    },
                    Stmt::If {
                        cond: cmp(BinaryOp::CmpGt, l(3), ScalarExpr::Const(0.0)),
                        then: Block(vec![Stmt::Assign {
                            local: 0,
                            value: ScalarExpr::Add(bx(l(0)), bx(l(3))),
                        }]),
                        els: Block(vec![Stmt::Continue]),
                    },
                ]),
            },
            Stmt::MaskedStore {
                place: at(1, ScalarExpr::Exec(ExecId::ThreadIdx(0))),
                value: l(0),
                mask: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
            },
            Stmt::Store {
                place: at(1, ScalarExpr::ConstInt(0, ElementKind::U32)),
                value: l(0),
            },
            Stmt::Return,
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn one() -> ScalarExpr {
    ScalarExpr::ConstInt(1, ElementKind::U32)
}

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

#[test]
fn control_flow_and_masked_access_lower_to_source_slangc_compiles() {
    let g = Slang.lower_kernel(&positive_sum(one())).expect("lowers");
    for want in ["for (l1 = ", "if (", "continue;", "return;", "while"] {
        // `while` is absent on purpose: this kernel has no While.
        assert_eq!(
            g.source.contains(want),
            want != "while",
            "{want} in\n{}",
            g.source
        );
    }
    // a mask is tested != 0 unless it is already a comparison
    assert!(g.source.contains("(tid_.x != 0)"), "{}", g.source);
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
fn a_for_step_that_is_not_a_nonzero_constant_declines_by_name() {
    // a runtime step (the oracle traps on zero; a GPU cannot)
    let d = decline(&positive_sum(ScalarExpr::Local(2)));
    assert!(d.contains("For: step not a nonzero constant"), "{d}");
    // a constant zero step
    let d = decline(&positive_sum(ScalarExpr::ConstInt(0, ElementKind::U32)));
    assert!(d.contains("For: step not a nonzero constant"), "{d}");
}
