// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.64.0 U50: `Stmt::Assume` is a pure value fact. Slang emission OMITS it (a spec-permitted ignore, as
//! `unpopped-cpu-c` does) so it never changes the text of the rest of the kernel; a lowering to an optimiser hint is a
//! later row.

use fishstringer_slang_emit::Slang;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// `out[tid] = a[tid]`, with `extra` between the load and the store.
fn copy(extra: Vec<Stmt>) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    let mut body = vec![Stmt::Load {
        local: 0,
        place: buf(0),
    }];
    body.extend(extra);
    body.push(Stmt::Store {
        place: buf(1),
        value: ScalarExpr::Local(0),
    });
    KernelFn::new(
        "copy",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ElementKind::F32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

#[test]
fn an_assume_never_changes_the_emitted_text_of_the_rest_of_the_kernel() {
    let without = Slang.lower_kernel(&copy(vec![])).expect("lowers");
    let facts = vec![
        Stmt::Assume {
            cond: ScalarExpr::ConstInt(1, ElementKind::U32),
        },
        Stmt::Assume {
            cond: ScalarExpr::Local(0),
        },
    ];
    let with = Slang.lower_kernel(&copy(facts)).expect("an Assume lowers");
    assert_eq!(with.source, without.source);
    // Positive control: a statement that is not an Assume does change the text.
    let other = Slang
        .lower_kernel(&copy(vec![Stmt::Assign {
            local: 0,
            value: ScalarExpr::Add(
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::Local(0)),
            ),
        }]))
        .expect("lowers");
    assert_ne!(other.source, without.source);
}
