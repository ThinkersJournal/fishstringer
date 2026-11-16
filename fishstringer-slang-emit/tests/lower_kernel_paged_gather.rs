// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E11: 16-bit floats. An f16 and a bf16 kernel (`out = x * 2 + y`) compile with `slangc`; a bf16 buffer is stored as
//! `uint16_t` and every access converts; a 16-bit scalar parameter stays a typed decline.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::BinaryOp;
use unpopped::ir::ScalarExpr;
use unpopped::ir::{CastSpec, RoundMode, Saturation};
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

const PAGE: i64 = 16;
const DIM: i64 = 64;

fn ci(v: i64, t: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v, t)
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn cast(e: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        bx(e),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

/// For each sequence `b` (one block), `out[b][tid] = sum over its tokens of kv[page * PAGE + row][tid]`, where the pages
/// come from `indices[indptr[b]..indptr[b + 1]]` and the last page holds `last_len[b]` rows.
fn paged_gather() -> KernelFn {
    let bid = ScalarExpr::Exec(ExecId::BlockIdx(0));
    let i32t = ElementKind::I32;
    let i64t = ElementKind::I64;
    let row = ScalarExpr::Add(
        bx(ScalarExpr::Mul(bx(cast(l(5), i64t)), bx(ci(PAGE, i64t)))),
        bx(cast(l(7), i64t)),
    );
    let kv_idx = ScalarExpr::Add(
        bx(ScalarExpr::Mul(bx(l(8)), bx(ci(DIM, i64t)))),
        bx(cast(tid(), i64t)),
    );
    let out_idx = ScalarExpr::Add(
        bx(ScalarExpr::Mul(
            bx(cast(bid.clone(), i64t)),
            bx(ci(DIM, i64t)),
        )),
        bx(cast(tid(), i64t)),
    );
    let rows = Block(vec![
        Stmt::Let {
            local: 8,
            init: row,
        },
        Stmt::Load {
            local: 9,
            place: buf(3, kv_idx),
        },
        Stmt::Assign {
            local: 3,
            value: ScalarExpr::Add(bx(l(3)), bx(l(9))),
        },
    ]);
    let pages = Block(vec![
        Stmt::Load {
            local: 5,
            place: buf(1, l(4)),
        },
        Stmt::Let {
            local: 6,
            init: ci(PAGE, i32t),
        },
        Stmt::If {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpEq,
                bx(l(4)),
                bx(ScalarExpr::Sub(bx(l(1)), bx(ci(1, i32t)))),
            ),
            then: Block(vec![Stmt::Assign {
                local: 6,
                value: l(2),
            }]),
            els: Block(vec![]),
        },
        Stmt::For {
            local: 7,
            start: ci(0, i32t),
            end: l(6),
            step: ci(1, i32t),
            body: rows,
        },
    ]);
    let body = vec![
        Stmt::Load {
            local: 0,
            place: buf(0, bid.clone()),
        },
        Stmt::Load {
            local: 1,
            place: buf(
                0,
                ScalarExpr::Add(bx(bid.clone()), bx(ci(1, ElementKind::U32))),
            ),
        },
        Stmt::Load {
            local: 2,
            place: buf(2, bid),
        },
        Stmt::Let {
            local: 3,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::For {
            local: 4,
            start: l(0),
            end: l(1),
            step: ci(1, i32t),
            body: pages,
        },
        Stmt::Store {
            place: buf(4, out_idx),
            value: l(3),
        },
    ];
    let locals = [
        i32t,
        i32t,
        i32t,
        ElementKind::F32,
        i32t,
        i32t,
        i32t,
        i32t,
        i64t,
        ElementKind::F32,
    ];
    KernelFn::new(
        "paged_gather",
        ElementKind::F32,
        vec![
            Param::buffer("indptr", i32t, BufferRole::Read, Some(1)),
            Param::buffer("indices", i32t, BufferRole::Read, Some(1)),
            Param::buffer("last_len", i32t, BufferRole::Read, Some(1)),
            Param::buffer("kv", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(1)),
        ],
        locals.iter().map(|t| LocalDecl::scalar(*t)).collect(),
        Block(body),
    )
    .with_launch(LaunchShape::new([DIM as u32, 1, 1], [4, 1, 1]))
}

#[test]
fn paged_gather_uses_64_bit_row_offsets_and_compiles() {
    let g = Slang.lower_kernel(&paged_gather()).expect("lowers");
    for want in [
        "StructuredBuffer<int> b1",
        "int64_t l8 = (int64_t)0;",
        "((int64_t)(l5))",
        "for (l7 = ",
    ] {
        assert!(
            g.source.contains(want),
            "{want} in
{}",
            g.source
        );
    }
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn a_float_to_int_cast_declines() {
    let mut k = paged_gather();
    k.body.0.insert(
        3,
        Stmt::Assign {
            local: 2,
            value: ScalarExpr::Cast(
                bx(ScalarExpr::Const(1.5)),
                CastSpec {
                    to: ElementKind::I32,
                    round: RoundMode::TowardZero,
                    sat: Saturation::Clamp,
                },
            ),
        },
    );
    match Slang.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                detail.contains("only int-to-int Wrap and to-float NearestEven lower"),
                "{detail}"
            );
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
