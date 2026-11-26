// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.60.0 (E1a, KISS 0.24.0) added seven `ElementKind` rows: `F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4` and
//! `F8UE4M3`. This emitter spells none of them, and each must be declined **by name**, never lowered and never panicking.
//!
//! What Slang offers (measured with slangc 2026.8 by compiling `RWStructuredBuffer<T>` for `-target hlsl`; the Slang docs
//! were not read): `half`, `float`, `double`, `int64_t` and the `FloatE4M3`/`FloatE5M2` types compile; `float128_t`,
//! `int128_t`, `uint128_t` are undefined identifiers, and there is no E3M4 or unsigned-E4M3 type. So the three 128-bit and
//! two 256-bit rows and `F8E3M4`/`F8UE4M3` have no spelling here. (`FloatE4M3` exists, but the older FP8 rows are not lowered
//! either, and `F8UE4M3` is a different format: not mapped.)

use fishstringer_slang_emit::Slang;
use unpopped::backend::{Backend, DtypeSupport, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::{ElementKind, TargetId};

const NEW: [ElementKind; 7] = [
    ElementKind::F128,
    ElementKind::I128,
    ElementKind::U128,
    ElementKind::I256,
    ElementKind::U256,
    ElementKind::F8E3M4,
    ElementKind::F8UE4M3,
];

/// `out[tid] = in[tid]` over `dt`.
fn copy(dt: ElementKind) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "copy",
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
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

#[test]
fn the_seven_new_rows_are_unsupported_on_every_target() {
    let mut n = 0;
    for dt in NEW {
        for tok in ["vulkan:sg32.arith-f16-i8", "vulkan:sg32", "cuda:sm_90"] {
            let t = TargetId::parse(tok).expect("valid token");
            assert_eq!(
                Slang.supports_dtype(dt, t),
                DtypeSupport::Unsupported,
                "{dt:?} on {tok}"
            );
            n += 1;
        }
    }
    assert_eq!(n, 21, "no row was skipped");
}

#[test]
fn a_kernel_over_each_new_row_declines_by_name() {
    // Control: the same kernel over F32 lowers, so the declines below are about the dtype and nothing else.
    Slang
        .lower_kernel(&copy(ElementKind::F32))
        .expect("control lowers");
    let mut n = 0;
    for dt in NEW {
        match Slang.lower_kernel(&copy(dt)) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains(&format!("{dt:?}")), "{dt:?}: {detail}");
            }
            other => panic!(
                "{dt:?}: expected a typed decline naming it, got {:?}",
                other.map(|g| g.source)
            ),
        }
        n += 1;
    }
    assert_eq!(n, 7, "no row was skipped");
}
