// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E11: 16-bit floats. An f16 and a bf16 kernel (`out = x * 2 + y`) compile with `slangc`; a bf16 buffer is stored as
//! `uint16_t` and every access converts; a 16-bit scalar parameter stays a typed decline.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, ParamKind, Place, Stmt};
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

#[test]
fn f16_axpy_uses_half_and_compiles() {
    let g = Slang.lower_kernel(&axpy(ElementKind::F16)).expect("lowers");
    assert!(
        g.source.contains("StructuredBuffer<half> b0"),
        "{}",
        g.source
    );
    assert!(g.source.contains("((half)2.0"), "{}", g.source);
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn bf16_is_uint16_storage_with_rounded_float_arithmetic() {
    let g = Slang
        .lower_kernel(&axpy(ElementKind::Bf16))
        .expect("lowers");
    for want in [
        "StructuredBuffer<uint16_t> b0",
        "bf16_to_f32_(b0[",
        "= f32_to_bf16_(l2)",
        "bf16r_(",
    ] {
        assert!(
            g.source.contains(want),
            "{want} in
{}",
            g.source
        );
    }
    assert!(
        g.interface()
            .expect("interface")
            .compiler_args
            .iter()
            .any(|a| a.contains("denorm"))
    );
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn a_16_bit_scalar_parameter_declines() {
    let mut k = axpy(ElementKind::F16);
    k.params.push(Param::scalar("s", ElementKind::F16, 3));
    k.locals.push(LocalDecl::scalar(ElementKind::F16));
    assert!(matches!(&k.params[3].kind, ParamKind::Scalar { .. }));
    match Slang.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("16-bit float scalar"), "{detail}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
