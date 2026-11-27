// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.61.0 (BREAKING) added `UnaryOp::{Popc, Clz, Brev}`, integer bit operations that exist in the kernel layer.
//! HLSL has `countbits`, `firstbithigh` and `reversebits`, but none is spelled or measured here yet (`firstbithigh` returns
//! -1 for 0, unlike the IR's `Clz(0) == w`, so a lowering needs an edge rule), so each op is declined **by name** at every
//! integer dtype the kernel path lowers (I32/U32), and the exhaustive `UnaryOp` match in the speller cannot compile without them.

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// `out[tid] = op(in[tid])` over `dt` (`None`: plain copy, the control).
fn kernel(dt: ElementKind, op: Option<UnaryOp>) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    let v = match op {
        Some(op) => ScalarExpr::Unary(op, Box::new(ScalarExpr::Local(0))),
        None => ScalarExpr::Local(0),
    };
    KernelFn::new(
        "bitop",
        dt,
        vec![
            Param::buffer("a", dt, BufferRole::Read, Some(1)),
            Param::buffer("out", dt, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(dt)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: buf(0),
            },
            Stmt::Store {
                place: buf(1),
                value: v,
            },
        ]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

#[test]
fn popc_clz_brev_decline_by_name_at_each_integer_dtype() {
    let mut n = 0;
    // 64-bit rows are not lowered by the kernel path at all (the control below would decline), so U32/I32 only.
    for dt in [ElementKind::U32, ElementKind::I32] {
        // Control: the same kernel without the op lowers, so the decline below is about the op alone.
        Slang
            .lower_kernel(&kernel(dt, None))
            .unwrap_or_else(|e| panic!("control {dt:?}: {e:?}"));
        for op in [UnaryOp::Popc, UnaryOp::Clz, UnaryOp::Brev] {
            match Slang.lower_kernel(&kernel(dt, Some(op))) {
                Err(LowerError::UnsupportedKernel { detail })
                | Err(LowerError::UnsupportedDtype { detail, .. }) => {
                    assert!(
                        detail.contains(&format!("{op:?}")),
                        "{op:?} at {dt:?}: decline must name the op: {detail}"
                    );
                }
                other => panic!(
                    "{op:?} at {dt:?}: expected a typed decline, got {:?}",
                    other.map(|g| g.source)
                ),
            }
            n += 1;
        }
    }
    assert_eq!(n, 6, "no op/dtype pair was skipped");
}
