// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E16: `Slang::lower_module` lowers a module whose entry calls helper functions and externs. A scalar-argument
//! `Call` with a result becomes a Slang function, and an `ExternCall` with a `reference` kernel emits the reference body.
//! A buffer argument is passed by alias (binding, strides, base). A callee with several results returns a struct and a `CallTuple` unpacks it. An extern with no reference, an allocation argument and a launch-id read in a callee stay typed declines.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::LowerError;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CallArg, Effects, ExternDecl, KernelFn, LocalDecl, Module, Param, Place,
    Stmt,
};
use unpopped_vocab::ElementKind;

const F: ElementKind = ElementKind::F32;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn out(i: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
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
    let res = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", entry, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .output()
        .expect("run slangc");
    assert!(
        res.status.success(),
        "slangc:\n{}\n{source}",
        String::from_utf8_lossy(&res.stderr)
    );
}

/// `fn(a, b) { r = a * b + a }` returning `r` (local 2).
fn helper() -> KernelFn {
    KernelFn::new(
        "mad",
        F,
        vec![Param::scalar("a", F, 0), Param::scalar("b", F, 1)],
        vec![LocalDecl::scalar(F); 3],
        Block(vec![Stmt::Let {
            local: 2,
            init: ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(Box::new(l(0)), Box::new(l(1)))),
                Box::new(l(0)),
            ),
        }]),
    )
    .with_result(2)
}

/// The entry: `out[tid] = <call>` where the call's result lands in local 0 and the stores read it.
fn entry(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "entry",
        F,
        vec![Param::buffer("out", F, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(F); 4],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.locals[0] = LocalDecl::scalar(F);
    k
}

fn store_local0() -> Stmt {
    Stmt::Store {
        place: out(tid()),
        value: l(0),
    }
}

fn decline(m: &Module) -> String {
    match Slang.lower_module(m, 0) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

#[test]
fn a_scalar_call_with_a_result_lowers_and_compiles() {
    let m = Module::new(vec![
        entry(vec![
            Stmt::Call {
                callee: 1,
                args: vec![
                    CallArg::Scalar(ScalarExpr::Const(2.0)),
                    CallArg::Scalar(ScalarExpr::Const(3.0)),
                ],
                result: Some(0),
            },
            store_local0(),
        ]),
        helper(),
    ]);
    let g = Slang.lower_module(&m, 0).expect("lowers");
    assert!(g.source.contains("f1("), "{}", g.source);
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn an_extern_with_a_reference_kernel_emits_the_reference_body() {
    let decl = ExternDecl::new("fast_mad", vec![F, F], vec![F], Effects::pure()).with_reference(1);
    let m = Module::new(vec![
        entry(vec![
            Stmt::ExternCall {
                decl: 0,
                args: vec![ScalarExpr::Const(2.0), ScalarExpr::Const(3.0)],
                results: vec![0],
            },
            store_local0(),
        ]),
        helper(),
    ])
    .with_externs(vec![decl]);
    let g = Slang.lower_module(&m, 0).expect("lowers");
    assert!(g.source.contains("f1("), "{}", g.source);
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn an_extern_with_no_reference_is_a_named_decline_and_the_control_lowers() {
    let mk = |with_ref: bool| {
        let mut decl = ExternDecl::new("mystery", vec![F, F], vec![F], Effects::pure());
        if with_ref {
            decl = decl.with_reference(1);
        }
        Module::new(vec![
            entry(vec![
                Stmt::ExternCall {
                    decl: 0,
                    args: vec![ScalarExpr::Const(2.0), ScalarExpr::Const(3.0)],
                    results: vec![0],
                },
                store_local0(),
            ]),
            helper(),
        ])
        .with_externs(vec![decl])
    };
    let d = decline(&mk(false));
    assert!(
        d.contains("slang: ExternCall: no Slang body for extern `mystery`"),
        "{d}"
    );
    assert!(
        Slang.lower_module(&mk(true), 0).is_ok(),
        "the control lowers"
    );
}

/// `fn(buf, i) { buf[i] = 7 }`: a callee with a buffer parameter (rank 1) and an integer index, no result.
fn writer(role: BufferRole) -> KernelFn {
    KernelFn::new(
        "writer",
        F,
        vec![
            Param::buffer("buf", F, role, Some(1)),
            Param::scalar("i", ElementKind::U32, 0),
        ],
        vec![LocalDecl::scalar(ElementKind::U32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![l(0)],
            },
            value: ScalarExpr::Const(7.0),
        }]),
    )
}

#[test]
fn a_buffer_argument_is_passed_by_alias_with_its_strides_and_compiles() {
    let m = Module::new(vec![
        entry(vec![Stmt::Call {
            callee: 1,
            args: vec![CallArg::Buffer(0), CallArg::Scalar(tid())],
            result: None,
        }]),
        writer(BufferRole::ReadWrite),
    ]);
    let g = Slang.lower_module(&m, 0).expect("lowers");
    // The call passes the binding, its stride and its base; the callee reads them as parameters, not from `pb_`.
    assert!(
        g.source.contains("f1(b0, pb_.b0_stride0, pb_.b0_base,"),
        "{}",
        g.source
    );
    assert!(
        g.source
            .contains("RWStructuredBuffer<float> b0, int64_t b0_stride0, int64_t b0_base"),
        "{}",
        g.source
    );
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn an_allocation_argument_is_a_named_decline() {
    let m = Module::new(vec![
        entry(vec![Stmt::Call {
            callee: 1,
            args: vec![CallArg::Alloc(0), CallArg::Scalar(tid())],
            result: None,
        }]),
        writer(BufferRole::ReadWrite),
    ]);
    assert!(decline(&m).contains("allocation"), "{}", decline(&m));
}

#[test]
fn a_callee_that_reads_a_launch_id_is_a_named_decline() {
    let mut callee = helper();
    callee.locals.push(LocalDecl::scalar(F));
    callee.body.0.push(Stmt::Let {
        local: 3,
        init: ScalarExpr::Cast(
            Box::new(tid()),
            CastSpec {
                to: F,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        ),
    });
    let m = Module::new(vec![
        entry(vec![
            Stmt::Call {
                callee: 1,
                args: vec![
                    CallArg::Scalar(ScalarExpr::Const(2.0)),
                    CallArg::Scalar(ScalarExpr::Const(3.0)),
                ],
                result: Some(0),
            },
            store_local0(),
        ]),
        callee,
    ]);
    assert!(decline(&m).contains("launch id"), "{}", decline(&m));
}

#[test]
fn a_cyclic_module_is_refused_by_the_checker() {
    let mut a = helper();
    a.body.0.push(Stmt::Call {
        callee: 1,
        args: vec![
            CallArg::Scalar(ScalarExpr::Const(1.0)),
            CallArg::Scalar(ScalarExpr::Const(1.0)),
        ],
        result: Some(2),
    });
    let m = Module::new(vec![
        entry(vec![Stmt::Call {
            callee: 1,
            args: vec![
                CallArg::Scalar(ScalarExpr::Const(1.0)),
                CallArg::Scalar(ScalarExpr::Const(1.0)),
            ],
            result: Some(0),
        }]),
        a,
    ]);
    assert!(decline(&m).contains("checker"), "{}", decline(&m));
}

/// `fn(a, b) -> (a * b, a + b)`: result local 2, extra result local 3.
fn pair_helper() -> KernelFn {
    KernelFn::new(
        "pair",
        F,
        vec![Param::scalar("a", F, 0), Param::scalar("b", F, 1)],
        vec![LocalDecl::scalar(F); 4],
        Block(vec![
            Stmt::Let {
                local: 2,
                init: ScalarExpr::Mul(Box::new(l(0)), Box::new(l(1))),
            },
            Stmt::Let {
                local: 3,
                init: ScalarExpr::Add(Box::new(l(0)), Box::new(l(1))),
            },
        ]),
    )
    .with_result(2)
    .with_results([3])
}

fn two_args() -> Vec<CallArg> {
    vec![
        CallArg::Scalar(ScalarExpr::Const(2.0)),
        CallArg::Scalar(ScalarExpr::Const(3.0)),
    ]
}

#[test]
fn a_tuple_call_unpacks_a_struct_result_and_compiles() {
    let m = Module::new(vec![
        entry(vec![
            Stmt::Call {
                callee: 1,
                args: two_args(),
                result: Some(2),
            },
            Stmt::CallTuple {
                callee: 1,
                args: two_args(),
                results: vec![0, 1],
            },
            Stmt::Store {
                place: out(tid()),
                value: ScalarExpr::Add(
                    Box::new(ScalarExpr::Add(Box::new(l(0)), Box::new(l(1)))),
                    Box::new(l(2)),
                ),
            },
        ]),
        pair_helper(),
    ]);
    let g = Slang.lower_module(&m, 0).expect("lowers");
    assert!(g.source.contains("struct R1"), "{}", g.source);
    assert!(
        g.source.contains(".v0") && g.source.contains(".v1"),
        "{}",
        g.source
    );
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn a_tuple_call_with_the_wrong_number_of_results_is_refused_by_the_checker() {
    let m = Module::new(vec![
        entry(vec![
            Stmt::CallTuple {
                callee: 1,
                args: two_args(),
                results: vec![0, 1, 2],
            },
            store_local0(),
        ]),
        pair_helper(),
    ]);
    assert!(
        matches!(
            Slang.lower_module(&m, 0),
            Err(LowerError::UnsupportedKernel { .. })
        ),
        "a mismatched results list must be a typed decline"
    );
}
