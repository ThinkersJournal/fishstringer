// SPDX-License-Identifier: MIT OR Apache-2.0
//! Board 506: `Slang::lower_kernel_hlsl` lowers subgroup collectives to HLSL wave intrinsics, and ONLY for HLSL. The emitted
//! text is target-agnostic Slang, so each collective sits in a `__target_switch { case hlsl: .. }` with no default, and
//! `slangc -target spirv` (and glsl) must FAIL on it while `-target hlsl` succeeds. The default `lower_kernel` still declines.
//! Measured with slangc 2026.8 (local) and the pinned 2026.21 (CI, `FISHSTRINGER_REQUIRE_SLANGC`); without slangc the
//! compile assertions are skipped (UNMEASURED), the text assertions still run.

use std::process::Command;

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped_vocab::ElementKind;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// `out[tid] = collective(a[tid])` over one subgroup-sized block; `ty` is the element type of both buffers.
fn kernel(ty: ElementKind, kind: CollectiveKind, scope: Scope, result_ty: ElementKind) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    KernelFn::new(
        "coll",
        ty,
        vec![
            Param::buffer("a", ty, BufferRole::Read, Some(1)),
            Param::buffer("out", result_ty, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ty), LocalDecl::scalar(result_ty)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: buf(0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
            },
            Stmt::Store {
                place: buf(1),
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

fn slangc_ok() -> bool {
    if Command::new("slangc").arg("-v").output().is_err() {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_SLANGC").is_none(),
            "FISHSTRINGER_REQUIRE_SLANGC is set but `slangc` could not be run"
        );
        eprintln!("skipping the compile half: `slangc` not on PATH (UNMEASURED)");
        return false;
    }
    true
}

/// Whether slangc compiles `source` for `target`.
fn compiles(source: &str, entry: &str, target: &str) -> bool {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    std::fs::write(&src, source).expect("write");
    Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", entry, "-stage", "compute", "-target", target, "-o",
        ])
        .arg(dir.path().join("k.out"))
        .output()
        .expect("run slangc")
        .status
        .success()
}

/// Every lowered form: its text uses the intrinsic inside the guard, HLSL compiles, and every other target is refused.
#[test]
fn each_lowered_collective_compiles_for_hlsl_and_is_refused_for_every_other_target() {
    use CollectiveKind as C;
    let f = ElementKind::F32;
    let i = ElementKind::I32;
    let u = ElementKind::U32;
    let cases: Vec<(&str, KernelFn)> = vec![
        (
            "WaveActiveSum",
            kernel(f, C::Reduce(ReduceOp::Sum), Scope::Subgroup, f),
        ),
        (
            "WaveActiveProduct",
            kernel(f, C::Reduce(ReduceOp::Prod), Scope::Subgroup, f),
        ),
        (
            "WaveActiveMax",
            kernel(i, C::Reduce(ReduceOp::Max), Scope::Subgroup, i),
        ),
        (
            "WaveActiveMin",
            kernel(u, C::Reduce(ReduceOp::Min), Scope::Subgroup, u),
        ),
        (
            "WaveActiveBitAnd",
            kernel(u, C::Reduce(ReduceOp::And), Scope::Subgroup, u),
        ),
        (
            "WaveActiveBitOr",
            kernel(i, C::Reduce(ReduceOp::Or), Scope::Subgroup, i),
        ),
        (
            "WaveActiveBitXor",
            kernel(u, C::Reduce(ReduceOp::Xor), Scope::Subgroup, u),
        ),
        (
            "WaveActiveAllTrue",
            kernel(f, C::Vote(VoteKind::All), Scope::Subgroup, u),
        ),
        (
            "WaveActiveAnyTrue",
            kernel(i, C::Vote(VoteKind::Any), Scope::Subgroup, u),
        ),
        (
            "WaveReadLaneAt",
            kernel(
                f,
                C::Broadcast {
                    lane: ScalarExpr::ConstInt(3, u),
                },
                Scope::Subgroup,
                f,
            ),
        ),
        (
            "WaveReadLaneAt",
            kernel(
                f,
                C::Shuffle {
                    mode: ShuffleMode::Idx,
                    operand: ScalarExpr::ConstInt(5, u),
                },
                Scope::Subgroup,
                f,
            ),
        ),
        (
            "WaveReadLaneAt",
            kernel(
                f,
                C::Shuffle {
                    mode: ShuffleMode::Up,
                    operand: ScalarExpr::ConstInt(1, u),
                },
                Scope::Subgroup,
                f,
            ),
        ),
        (
            "WaveReadLaneAt",
            kernel(
                i,
                C::Shuffle {
                    mode: ShuffleMode::Down,
                    operand: ScalarExpr::ConstInt(2, u),
                },
                Scope::Subgroup,
                i,
            ),
        ),
        (
            "WaveReadLaneAt",
            kernel(
                u,
                C::Shuffle {
                    mode: ShuffleMode::Xor,
                    operand: ScalarExpr::ConstInt(1, u),
                },
                Scope::Subgroup,
                u,
            ),
        ),
    ];
    let have = slangc_ok();
    let mut n = 0;
    for (intrinsic, k) in &cases {
        let g = Slang
            .lower_kernel_hlsl(k)
            .unwrap_or_else(|e| panic!("{intrinsic}: {e:?}"));
        assert!(g.source.contains(intrinsic), "{intrinsic}: {}", g.source);
        assert!(
            g.source.contains("__target_switch") && g.source.contains("case hlsl:"),
            "{intrinsic}: the guard is missing:\n{}",
            g.source
        );
        // The guard has no default arm: nothing but `case hlsl` inside the switch.
        assert_eq!(g.source.matches("case ").count(), 1, "{}", g.source);
        assert!(!g.source.contains("default:"), "{}", g.source);
        if have {
            assert!(
                compiles(&g.source, &g.name, "hlsl"),
                "{intrinsic} must compile for hlsl:\n{}",
                g.source
            );
            for t in ["spirv", "glsl"] {
                assert!(
                    !compiles(&g.source, &g.name, t),
                    "{intrinsic} must NOT compile for {t} (board 506: SPIR-V and GLSL stay declined):\n{}",
                    g.source
                );
            }
        }
        n += 1;
    }
    assert_eq!(n, 14, "no case was skipped");
}

/// A mutant of the emitter: the same text with the guard removed compiles for SPIR-V. This is what the test above would
/// not catch if `__target_switch` were dropped, so it proves the refusal comes from the guard and not from something else.
#[test]
fn without_the_guard_the_same_collective_compiles_for_spirv() {
    use CollectiveKind as C;
    let k = kernel(
        ElementKind::F32,
        C::Reduce(ReduceOp::Sum),
        Scope::Subgroup,
        ElementKind::F32,
    );
    let g = Slang.lower_kernel_hlsl(&k).expect("lowers");
    assert!(g.source.contains("__target_switch"), "{}", g.source);
    if !slangc_ok() {
        return;
    }
    let unguarded = g
        .source
        .replace("__target_switch\n", "")
        .replace("case hlsl:\n", "")
        .replace("break;\n", "");
    assert!(!unguarded.contains("__target_switch"), "{unguarded}");
    assert!(compiles(&unguarded, &g.name, "hlsl"), "{unguarded}");
    assert!(
        compiles(&unguarded, &g.name, "spirv"),
        "control: the unguarded text must compile for spirv, else the failing compile above proves nothing:\n{unguarded}"
    );
}

fn decline(r: Result<unpopped::backend::GeneratedKernel, LowerError>) -> String {
    match r {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!(
            "expected a typed decline, got {:?}",
            other.map(|g| g.source)
        ),
    }
}

/// The default entry point is unchanged: every collective, including each one the HLSL entry point lowers, still declines.
#[test]
fn the_default_lower_kernel_still_declines_what_the_hlsl_entry_lowers() {
    use CollectiveKind as C;
    let k = kernel(
        ElementKind::F32,
        C::Reduce(ReduceOp::Sum),
        Scope::Subgroup,
        ElementKind::F32,
    );
    assert!(
        Slang.lower_kernel_hlsl(&k).is_ok(),
        "control: the HLSL entry lowers it"
    );
    let d = decline(Slang.lower_kernel(&k));
    assert!(
        d.starts_with(GAP_PREFIX) && d.contains("collective statement"),
        "{d}"
    );
}

/// What stays declined, by name, even on the HLSL entry point.
#[test]
fn the_hlsl_entry_still_declines_scan_ballot_float_maxmin_block_scope_and_other_types() {
    use CollectiveKind as C;
    let f = ElementKind::F32;
    let u = ElementKind::U32;
    let sub = Scope::Subgroup;
    let cases: Vec<(KernelFn, &str)> = vec![
        (
            kernel(
                f,
                C::Scan {
                    op: ReduceOp::Sum,
                    exclusive: false,
                },
                sub,
                f,
            ),
            "Scan",
        ),
        (kernel(f, C::Vote(VoteKind::Ballot), sub, u), "Ballot"),
        (
            kernel(f, C::Reduce(ReduceOp::Max), sub, f),
            "float Reduce Max/Min",
        ),
        (
            kernel(f, C::Reduce(ReduceOp::Min), sub, f),
            "float Reduce Max/Min",
        ),
        (
            kernel(f, C::Reduce(ReduceOp::Sum), Scope::Block, f),
            "Block",
        ),
        (
            kernel(
                ElementKind::F64,
                C::Reduce(ReduceOp::Sum),
                sub,
                ElementKind::F64,
            ),
            "F64",
        ),
    ];
    for (k, want) in &cases {
        let d = decline(Slang.lower_kernel_hlsl(k));
        assert!(d.contains(want), "{want}: {d}");
    }
}
