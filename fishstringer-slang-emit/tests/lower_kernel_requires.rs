// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E11: 16-bit floats. An f16 and a bf16 kernel (`out = x * 2 + y`) compile with `slangc`; a bf16 buffer is stored as
//! `uint16_t` and every access converts; a 16-bit scalar parameter stays a typed decline.

use fishstringer_slang_emit::{Slang, VULKAN_ARITH_TOKENS, unmet_requirements};
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn buf(b: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer: b,
        coords: vec![i],
    }
}

fn axpy(dt: ElementKind) -> KernelFn {
    let l = |i| ScalarExpr::Local(i);
    let body = vec![
        Stmt::Load {
            local: 0,
            place: buf(0, tid()),
        },
        Stmt::Load {
            local: 1,
            place: buf(1, tid()),
        },
        Stmt::Let {
            local: 2,
            init: ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(
                    Box::new(l(0)),
                    Box::new(ScalarExpr::Const(2.0)),
                )),
                Box::new(l(1)),
            ),
        },
        Stmt::Store {
            place: buf(2, tid()),
            value: l(2),
        },
    ];
    KernelFn::new(
        "axpy16",
        dt,
        vec![
            Param::buffer("x", dt, BufferRole::Read, Some(1)),
            Param::buffer("y", dt, BufferRole::Read, Some(1)),
            Param::buffer("out", dt, BufferRole::Write, Some(1)),
        ],
        (0..3).map(|_| LocalDecl::scalar(dt)).collect(),
        Block(body),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [4, 1, 1]))
}

fn requires(dt: ElementKind) -> Vec<String> {
    let g = Slang.lower_kernel(&axpy(dt)).expect("lowers");
    g.interface().expect("interface").requires.clone()
}

#[test]
fn the_tokens_a_kernel_needs_are_declared_sorted() {
    assert_eq!(requires(ElementKind::F32), ["i64"]);
    assert_eq!(requires(ElementKind::F16), ["f16", "i64", "st16"]);
    assert_eq!(requires(ElementKind::Bf16), ["i16", "i64", "st16"]);
}

#[test]
fn every_emitted_token_is_a_vocabulary_name() {
    // Control: the list is not empty and knows a real name, so the subset check below can fail.
    assert!(VULKAN_ARITH_TOKENS.contains(&"i64"));
    assert!(!VULKAN_ARITH_TOKENS.contains(&"int64"));
    for dt in [ElementKind::F32, ElementKind::F16, ElementKind::Bf16] {
        for t in requires(dt) {
            assert!(
                VULKAN_ARITH_TOKENS.contains(&t.as_str()),
                "`{t}` is not a vulkan <arith> name"
            );
        }
    }
}

#[test]
fn the_gate_names_the_token_a_target_lacks() {
    let g = Slang.lower_kernel(&axpy(ElementKind::F16)).expect("lowers");
    let iface = g.interface().expect("interface");
    assert_eq!(unmet_requirements(iface, &["f16", "st16"]), ["i64"]);
    assert_eq!(unmet_requirements(iface, &[]), ["f16", "i64", "st16"]);
    // Control: a target that offers everything has nothing unmet.
    assert!(unmet_requirements(iface, &["f16", "i64", "st16"]).is_empty());
}
