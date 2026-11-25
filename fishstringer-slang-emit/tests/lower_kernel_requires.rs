// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E11: 16-bit floats. An f16 and a bf16 kernel (`out = x * 2 + y`) compile with `slangc`; a bf16 buffer is stored as
//! `uint16_t` and every access converts; a 16-bit scalar parameter stays a typed decline.

use fishstringer_slang_emit::{Slang, unmet_requirements, vulkan_arith_tokens};
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
    let names = vulkan_arith_tokens();
    assert!(names.contains(&"i64"));
    assert!(!names.contains(&"int64"));
    for dt in [ElementKind::F32, ElementKind::F16, ElementKind::Bf16] {
        for t in requires(dt) {
            assert!(
                names.contains(&t.as_str()),
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

#[test]
fn the_token_list_is_the_published_vocabulary_not_a_copy() {
    // Vocabulary 7 (kiss-vulkan-vocab 0.6.0) names three atomics tokens the old hand copy (vocabulary 6) lacked.
    let names = vulkan_arith_tokens();
    for t in ["a64", "af32", "af64"] {
        assert!(names.contains(&t), "`{t}` must come from the crate");
    }
    assert_eq!(names, kiss_vulkan_vocab::Arith::alphabet());
    assert_eq!(kiss_vulkan_vocab::VOCABULARY_VERSION, 7);
}
