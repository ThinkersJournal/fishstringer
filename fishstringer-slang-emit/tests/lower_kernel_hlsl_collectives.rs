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
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
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
    let mut dxc_n = 0;
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
        if have {
            if let Some(ok) = dxc_accepts(&g.source, &g.name) {
                assert!(
                    ok,
                    "{intrinsic}: dxc -T cs_6_6 refused the HLSL for:\n{}",
                    g.source
                );
                dxc_n += 1;
            }
        }
        n += 1;
    }
    assert_eq!(n, 14, "no case was skipped");
    // dxc ran on all of them or on none (it is not on PATH): never on some.
    assert!(dxc_n == 0 || dxc_n == 14, "dxc ran on {dxc_n} of 14");
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

/// Whether `dxc` (cs_6_6) accepts the HLSL slangc writes for `source`. `None` when slangc or dxc is not on PATH (UNMEASURED).
fn dxc_accepts(source: &str, entry: &str) -> Option<bool> {
    let dxc_ok = Command::new("dxc").arg("-help").output().is_ok();
    if !dxc_ok {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_DXC").is_none(),
            "FISHSTRINGER_REQUIRE_DXC is set but `dxc` could not be run"
        );
        return None;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    std::fs::write(&src, source).expect("write");
    let hlsl = dir.path().join("k.hlsl");
    let o = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", entry, "-stage", "compute", "-target", "hlsl", "-o",
        ])
        .arg(&hlsl)
        .output()
        .expect("run slangc");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = Command::new("dxc")
        .args(["-T", "cs_6_6", "-E", entry])
        .arg(&hlsl)
        .args(["-Fo"])
        .arg(dir.path().join("k.dxil"))
        .output()
        .expect("run dxc");
    if !o.status.success() {
        eprintln!("dxc:\n{}", String::from_utf8_lossy(&o.stderr));
    }
    Some(o.status.success())
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

/// The launch must state exactly one subgroup size, and the entry then carries `[WaveSize(n)]`: HLSL waves are 32 or 64 and
/// the oracle's `n` is the launch's, so anything else would silently disagree on a device.
#[test]
fn the_launch_must_state_one_subgroup_size_and_the_entry_carries_wavesize() {
    use CollectiveKind as C;
    let f = ElementKind::F32;
    let mk = |sg: Option<(u32, u32)>| {
        let mut k = kernel(f, C::Reduce(ReduceOp::Sum), Scope::Subgroup, f);
        let l = LaunchShape::new([64, 1, 1], [1, 1, 1]);
        k = k.with_launch(match sg {
            Some((a, b)) => l.with_subgroup(a, b),
            None => l,
        });
        k
    };
    let g = Slang
        .lower_kernel_hlsl(&mk(Some((32, 32))))
        .expect("lowers");
    assert!(
        g.source.contains("[WaveSize(32)]\n[numthreads(64, 1, 1)]"),
        "{}",
        g.source
    );
    let g = Slang
        .lower_kernel_hlsl(&mk(Some((64, 64))))
        .expect("lowers");
    assert!(g.source.contains("[WaveSize(64)]"), "{}", g.source);
    for (sg, want) in [
        (None, "states no subgroup size"),
        (Some((32, 64)), "not one power of two"),
        (Some((24, 24)), "not one power of two"),
        (Some((2, 2)), "not one power of two"),
        (Some((256, 256)), "not one power of two"),
    ] {
        let d = decline(Slang.lower_kernel_hlsl(&mk(sg)));
        assert!(d.contains(want), "{sg:?}: {d}");
    }
    // A kernel with no collective needs no size and gets no attribute.
    let mut plain = mk(None);
    plain.body.0.remove(1);
    plain.body.0[1] = Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![tid()],
        },
        value: ScalarExpr::Local(0),
    };
    let g = Slang.lower_kernel_hlsl(&plain).expect("lowers");
    assert!(!g.source.contains("WaveSize"), "{}", g.source);
}

/// Edge lanes of the four shuffles at n = 32: delta/mask/index 0, n-1, n, n+1 (and -1 for a signed operand). The expected
/// spelling of each is asserted in the emitted text, then compiled for HLSL; the oracle rules are Idx = `operand mod n`, and
/// Up/Down/Xor with a source outside `0..n` keep the lane's own value. The text is NOT executed (no device): UNMEASURED.
#[test]
fn shuffle_edge_operands_are_spelled_with_the_oracle_rules() {
    use CollectiveKind as C;
    let f = ElementKind::F32;
    let have = slangc_ok();
    let lower = |mode, op: ScalarExpr| {
        let k = kernel(f, C::Shuffle { mode, operand: op }, Scope::Subgroup, f);
        Slang.lower_kernel_hlsl(&k).expect("lowers")
    };
    let mut n = 0;
    for u in [0u128, 31, 32, 33] {
        let c = ScalarExpr::ConstInt(u as _, ElementKind::U32);
        let g = lower(ShuffleMode::Idx, c.clone());
        // mod n, unsigned: 32 -> lane 0, 33 -> lane 1 (the text takes `% 32u` at run time).
        assert!(g.source.contains("% 32u)"), "{}", g.source);
        let g2 = lower(ShuffleMode::Up, c.clone());
        // own value unless `delta <= lane`.
        assert!(
            g2.source
                .contains("<= WaveGetLaneIndex()) ? WaveReadLaneAt("),
            "{}",
            g2.source
        );
        assert!(g2.source.contains(": l0)"), "{}", g2.source);
        let g3 = lower(ShuffleMode::Down, c.clone());
        assert!(
            g3.source
                .contains("< 32u - WaveGetLaneIndex()) ? WaveReadLaneAt("),
            "{}",
            g3.source
        );
        let g4 = lower(ShuffleMode::Xor, c);
        assert!(
            g4.source.contains(") < 32u) ? WaveReadLaneAt("),
            "{}",
            g4.source
        );
        for g in [g, g2, g3, g4] {
            assert!(g.source.contains(&format!("{u}")), "{}", g.source);
            if have {
                assert!(compiles(&g.source, &g.name, "hlsl"), "{}", g.source);
                assert!(!compiles(&g.source, &g.name, "spirv"), "{}", g.source);
            }
            n += 1;
        }
    }
    // Signed operands: a negative one is `Up` by a negative delta, so the source is range-checked on both sides, and Idx
    // takes a non-negative remainder.
    for i in [-1i64, 0, 31, 32, 33] {
        let c = ScalarExpr::ConstInt(i as _, ElementKind::I32);
        let g = lower(ShuffleMode::Idx, c.clone());
        assert!(g.source.contains("% 32) + 32) % 32)"), "{}", g.source);
        let g2 = lower(ShuffleMode::Up, c.clone());
        assert!(g2.source.contains(") >= 0 && ("), "{}", g2.source);
        assert!(
            g2.source.contains(") < 32) ? WaveReadLaneAt("),
            "{}",
            g2.source
        );
        for g in [
            g,
            g2,
            lower(ShuffleMode::Down, c.clone()),
            lower(ShuffleMode::Xor, c),
        ] {
            if have {
                assert!(compiles(&g.source, &g.name, "hlsl"), "{}", g.source);
            }
            n += 1;
        }
    }
    assert_eq!(n, 4 * 4 + 5 * 4, "no case was skipped");
}

/// `dxc` (not slangc) on `hlsl` text with `args`; `None` when dxc is not on PATH.
fn dxc_direct(hlsl: &str, entry: &str, args: &[&str]) -> Option<bool> {
    if Command::new("dxc").arg("-help").output().is_err() {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_DXC").is_none(),
            "FISHSTRINGER_REQUIRE_DXC is set but `dxc` could not be run"
        );
        return None;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let f = dir.path().join("k.hlsl");
    std::fs::write(&f, hlsl).expect("write");
    let o = Command::new("dxc")
        .args(["-E", entry])
        .args(args)
        .arg(&f)
        .arg("-Fo")
        .arg(dir.path().join("k.out"))
        .output()
        .expect("run dxc");
    if !o.status.success() {
        eprintln!("dxc {args:?}:\n{}", String::from_utf8_lossy(&o.stderr));
    }
    Some(o.status.success())
}

/// `lower_kernel_hlsl_plain` is the same lowering with no `__target_switch`: `dxc` reads it directly, all 14 forms at
/// `cs_6_6`. It carries no guard, which the control shows: under `dxc -spirv` the unguarded text compiles to SPIR-V, and with
/// the guard a consumer must prepend (`#ifdef __spirv__ #error`) the same text fails there while plain `dxc` still passes.
#[test]
fn the_plain_hlsl_has_no_slang_guard_dxc_reads_it_and_a_consumer_guard_stops_spirv() {
    use CollectiveKind as C;
    let (f, i, u) = (ElementKind::F32, ElementKind::I32, ElementKind::U32);
    let sub = Scope::Subgroup;
    let c = |v: u128| ScalarExpr::ConstInt(v as _, u);
    let sh = |mode, op| C::Shuffle { mode, operand: op };
    let cases = vec![
        kernel(f, C::Reduce(ReduceOp::Sum), sub, f),
        kernel(f, C::Reduce(ReduceOp::Prod), sub, f),
        kernel(i, C::Reduce(ReduceOp::Max), sub, i),
        kernel(u, C::Reduce(ReduceOp::Min), sub, u),
        kernel(u, C::Reduce(ReduceOp::And), sub, u),
        kernel(i, C::Reduce(ReduceOp::Or), sub, i),
        kernel(i, C::Reduce(ReduceOp::Xor), sub, i),
        kernel(f, C::Vote(VoteKind::All), sub, u),
        kernel(i, C::Vote(VoteKind::Any), sub, u),
        kernel(f, C::Broadcast { lane: c(3) }, sub, f),
        kernel(f, sh(ShuffleMode::Idx, c(5)), sub, f),
        kernel(f, sh(ShuffleMode::Up, c(1)), sub, f),
        kernel(i, sh(ShuffleMode::Down, c(2)), sub, i),
        kernel(u, sh(ShuffleMode::Xor, c(1)), sub, u),
    ];
    let mut ran = 0;
    for k in &cases {
        let g = Slang.lower_kernel_hlsl_plain(k).expect("lowers");
        assert!(!g.source.contains("__target_switch"), "{}", g.source);
        assert!(g.source.contains("[WaveSize(32)]"), "{}", g.source);
        let spirv = ["-T", "cs_6_6", "-spirv", "-fspv-target-env=vulkan1.1"];
        match dxc_direct(&g.source, &g.name, &["-T", "cs_6_6"]) {
            None => continue,
            Some(ok) => assert!(ok, "plain HLSL must pass dxc cs_6_6:\n{}", g.source),
        }
        // Control: nothing in the plain text stops SPIR-V, so a consumer guard is required.
        assert_eq!(
            dxc_direct(&g.source, &g.name, &spirv),
            Some(true),
            "{}",
            g.source
        );
        let guarded = format!(
            "#ifdef __spirv__\n#error \"subgroup collectives are declined for SPIR-V (board 506)\"\n#endif\n{}",
            g.source
        );
        assert_eq!(
            dxc_direct(&guarded, &g.name, &spirv),
            Some(false),
            "{guarded}"
        );
        assert_eq!(
            dxc_direct(&guarded, &g.name, &["-T", "cs_6_6"]),
            Some(true),
            "{guarded}"
        );
        ran += 1;
    }
    assert!(ran == 0 || ran == 14, "dxc ran on {ran} of 14");
}
