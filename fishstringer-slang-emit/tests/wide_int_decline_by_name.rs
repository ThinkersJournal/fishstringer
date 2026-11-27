// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.61.3 made `check_kernel` accept `I128`/`U128`/`I256`/`U256` (IR and oracle compute them). This emitter spells
//! none of them (slangc 2026.8 has no `int128_t`), and the carry-flag emit-lowering stays held, so each must be declined **by
//! name before emitting**, never emitted and never panicking. `new_dtype_rows_decline_by_name.rs` covers the Load/Store copy;
//! this file covers the two other paths: operators on a wide value, and `lower_module` with a wide parameter, result and
//! `CallTuple`.

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

const WIDE: [ElementKind; 4] = [
    ElementKind::I128,
    ElementKind::U128,
    ElementKind::I256,
    ElementKind::U256,
];

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn at_tid(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn launch(k: KernelFn) -> KernelFn {
    k.with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]))
}

/// `x = a[tid]; y = x * x + x; out[tid] = y; if (y < x) out[tid] = x` over `dt`: Add, Mul and a comparison.
fn arithmetic(dt: ElementKind) -> KernelFn {
    launch(KernelFn::new(
        "wide_arith",
        dt,
        vec![
            Param::buffer("a", dt, BufferRole::Read, Some(1)),
            Param::buffer("out", dt, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(dt); 2],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at_tid(0),
            },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Add(bx(ScalarExpr::Mul(bx(l(0)), bx(l(0)))), bx(l(0))),
            },
            Stmt::Store {
                place: at_tid(1),
                value: l(1),
            },
            Stmt::If {
                cond: ScalarExpr::Binary(BinaryOp::CmpLt, bx(l(1)), bx(l(0))),
                then: Block(vec![Stmt::Store {
                    place: at_tid(1),
                    value: l(0),
                }]),
                els: Block(vec![]),
            },
        ]),
    ))
}

/// `fn(a, b) -> (a * b, a + b)` over `dt`; result 2 and extra result 3.
fn pair_helper(dt: ElementKind) -> KernelFn {
    KernelFn::new(
        "pair",
        dt,
        vec![Param::scalar("a", dt, 0), Param::scalar("b", dt, 1)],
        vec![LocalDecl::scalar(dt); 4],
        Block(vec![
            Stmt::Let {
                local: 2,
                init: ScalarExpr::Mul(bx(l(0)), bx(l(1))),
            },
            Stmt::Let {
                local: 3,
                init: ScalarExpr::Add(bx(l(0)), bx(l(1))),
            },
        ]),
    )
    .with_result(2)
    .with_results([3])
}

fn module(dt: ElementKind, tuple: bool) -> Module {
    let args = vec![
        CallArg::Scalar(ScalarExpr::ConstInt(2, dt)),
        CallArg::Scalar(ScalarExpr::ConstInt(3, dt)),
    ];
    let call = if tuple {
        Stmt::CallTuple {
            callee: 1,
            args,
            results: vec![0, 1],
        }
    } else {
        Stmt::Call {
            callee: 1,
            args,
            result: Some(0),
        }
    };
    let entry = launch(KernelFn::new(
        "entry",
        dt,
        vec![Param::buffer("out", dt, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(dt); 2],
        Block(vec![
            call,
            Stmt::Store {
                place: at_tid(0),
                value: l(0),
            },
        ]),
    ));
    Module::new(vec![entry, pair_helper(dt)])
}

fn names_dtype(r: Result<impl std::fmt::Debug, LowerError>, dt: ElementKind, what: &str) {
    match r {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                detail.contains(&format!("{dt:?}")),
                "{what} {dt:?}: {detail}"
            );
        }
        other => panic!("{what} {dt:?}: expected a typed decline naming it, got {other:?}"),
    }
}

#[test]
fn operators_on_a_wide_value_decline_by_name() {
    // Control: the same shape over I32 lowers, so the declines below are about the dtype.
    Slang
        .lower_kernel(&arithmetic(ElementKind::I32))
        .expect("control lowers");
    for dt in WIDE {
        names_dtype(
            Slang.lower_kernel(&arithmetic(dt)).map(|g| g.source),
            dt,
            "kernel",
        );
    }
}

#[test]
fn a_module_with_a_wide_call_or_calltuple_declines_by_name() {
    // Control: the same module shapes over I32 lower.
    for tuple in [false, true] {
        Slang
            .lower_module(&module(ElementKind::I32, tuple), 0)
            .expect("control lowers");
    }
    for dt in WIDE {
        for tuple in [false, true] {
            names_dtype(
                Slang.lower_module(&module(dt, tuple), 0).map(|g| g.source),
                dt,
                if tuple {
                    "CallTuple module"
                } else {
                    "Call module"
                },
            );
        }
    }
}
