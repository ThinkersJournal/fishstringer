// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 18 / E17 slice 3: `Copysign` and `Nextafter` lower at f32 (pm-kernels approval 2026-11-22; before it Slang
//! declined both and Baracuda's suite asserted the decline). Copysign is inline bit operations on `asuint`; Nextafter
//! calls `fs_nextafter`, prepended like the erf helpers. f64 declines both, typed.
//!
//! What is and is not shown. (1) The emitted kernels compile with `slangc`. (2) A Rust model of the SAME bit
//! operations is compared by bit pattern with std's `copysign` / `next_up` / `next_down` on an asymmetric edge set
//! (0, +-0, +-inf, NaN, sign crossing, the subnormal boundary, MAX) and a seeded sample of the whole f32 space.
//! (3) Each seeded mutant of the model must FAIL those checks, so the checks can catch what they claim to. (4) The
//! emitted source must contain the model's operations verbatim, so the model cannot drift from the shipped text.
//! NOT shown: a device run. The helper compares `a == 0.0`, which assumes denormals are preserved (the kernel header's
//! `-denorm-mode-fp32 preserve`); a device that flushes treats a subnormal operand as zero.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::LowerError;
use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn key(dt: ElementKind) -> unpopped_vocab::StructureKey {
    let a = OperandDesc::new(1, &[7], &[1], dt, 4);
    structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
}

fn gen_src(op: BinaryOp, dt: ElementKind) -> Result<String, LowerError> {
    let def = OpDef::elementwise("pb", 2, &[dt], input(0).binary(op, input(1)));
    try_generate(&def, &key(dt), &Slang).map(|k| k.source)
}

fn slangc_compiles(source: &str) {
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
            "-entry",
            "unpopped_slang_pb_f32",
            "-stage",
            "compute",
            "-target",
            "spirv",
            "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .args(["-denorm-mode-fp32", "preserve"])
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "slangc:\n{}\n{source}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn copysign_and_nextafter_lower_at_f32_and_compile() {
    for op in [BinaryOp::Copysign, BinaryOp::Nextafter] {
        let src = gen_src(op, ElementKind::F32).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        assert!(src.contains("output[i] ="), "{op:?}");
        slangc_compiles(&src);
    }
}

/// The helper block rides only with `Nextafter` (control: `Copysign` is inline and carries none), and the emitted text
/// holds the operations the model below mirrors.
#[test]
fn emitted_text_holds_the_modelled_operations() {
    let cs = gen_src(BinaryOp::Copysign, ElementKind::F32).unwrap();
    assert!(
        cs.contains(
            "asfloat((asuint(input0[i]) & 0x7fffffffu) | (asuint(input1[i]) & 0x80000000u))"
        ),
        "{cs}"
    );
    assert!(
        !cs.contains("fs_"),
        "Copysign must not carry the helper block:\n{cs}"
    );
    let na = gen_src(BinaryOp::Nextafter, ElementKind::F32).unwrap();
    for want in [
        "output[i] = fs_nextafter(input0[i], input1[i]);",
        "float fs_nextafter(float a, float b)",
        "if (a != a || b != b) { return a + b; }",
        "if (a == b) { return b; }",
        "if (a == 0.0) { return asfloat((asuint(b) & 0x80000000u) | 1u); }",
        "if ((a < b) == (a > 0.0)) { u += 1u; } else { u -= 1u; }",
    ] {
        assert!(na.contains(want), "missing `{want}` in:\n{na}");
    }
}

#[test]
fn f64_declines_both_typed() {
    for op in [BinaryOp::Copysign, BinaryOp::Nextafter] {
        match gen_src(op, ElementKind::F64) {
            Err(LowerError::UnsupportedDtype { dtype, detail }) => {
                assert_eq!(dtype, ElementKind::F64, "{op:?}");
                assert!(detail.contains("f32 only"), "{detail}");
            }
            other => panic!("{op:?} at f64: expected a typed decline, got {other:?}"),
        }
    }
}

// ---- model of the emitted bit operations, with seeded mutants ----
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mutant {
    None,
    CopysignKeepsASign,   // sign mask taken from `a`
    CopysignDropsSignBit, // magnitude mask keeps the sign bit too
    NextafterWrongDirection,
    NextafterZeroKeepsASign,
    NextafterNoEqualityShortcut,
    NextafterNoNanCheck,
    NextafterSignTestOnB, // `(a < b) == (b > 0)`
}

fn copysign_m(a: f32, b: f32, m: Mutant) -> f32 {
    let (ua, ub) = (a.to_bits(), b.to_bits());
    f32::from_bits(match m {
        Mutant::CopysignKeepsASign => (ua & 0x7fff_ffff) | (ua & 0x8000_0000),
        Mutant::CopysignDropsSignBit => ua | (ub & 0x8000_0000),
        _ => (ua & 0x7fff_ffff) | (ub & 0x8000_0000),
    })
}

fn nextafter_m(a: f32, b: f32, m: Mutant) -> f32 {
    if m != Mutant::NextafterNoNanCheck && (a.is_nan() || b.is_nan()) {
        return a + b;
    }
    if m != Mutant::NextafterNoEqualityShortcut && a == b {
        return b;
    }
    if a == 0.0 {
        let sign_from = if m == Mutant::NextafterZeroKeepsASign {
            a
        } else {
            b
        };
        return f32::from_bits((sign_from.to_bits() & 0x8000_0000) | 1);
    }
    let mut u = a.to_bits();
    let toward_larger_magnitude = if m == Mutant::NextafterSignTestOnB {
        (a < b) == (b > 0.0)
    } else {
        (a < b) == (a > 0.0)
    };
    let up = toward_larger_magnitude != (m == Mutant::NextafterWrongDirection);
    if up {
        u = u.wrapping_add(1);
    } else {
        u = u.wrapping_sub(1);
    }
    f32::from_bits(u)
}

/// Independent reference (C's `copysignf` / `nextafterf`), built from std rather than from the bit trick.
fn copysign_ref(a: f32, b: f32) -> f32 {
    a.copysign(b)
}
fn nextafter_ref(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        b
    } else if a == 0.0 {
        f32::from_bits(1).copysign(b)
    } else if b > a {
        a.next_up()
    } else {
        a.next_down()
    }
}

fn same(got: f32, want: f32) -> bool {
    (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
}

fn edges() -> Vec<f32> {
    let sub = f32::from_bits(1);
    let min_norm = f32::MIN_POSITIVE;
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        sub,
        -sub,
        f32::from_bits(0x007f_ffff), // largest subnormal
        min_norm,
        -min_norm,
        f32::EPSILON,
        1.0 - f32::EPSILON / 2.0,
    ];
    // Neighbours of each, so the sign-crossing and binade steps are in the set.
    let base = v.clone();
    for x in base {
        if x.is_finite() {
            v.push(x.next_up());
            v.push(x.next_down());
        }
    }
    v
}

/// xorshift, seeded: the sample is the same on every run.
fn sample(n: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f32::from_bits((s >> 16) as u32)
        })
        .collect()
}

/// Every (a, b) pair of the edge set and of a seeded sample; `None` if the model matches the reference everywhere,
/// else the first disagreement.
fn first_mismatch(m: Mutant) -> Option<String> {
    let e = edges();
    let r = sample(300_000);
    let pairs = e
        .iter()
        .flat_map(|&a| e.iter().map(move |&b| (a, b)))
        .chain(r.chunks(2).map(|c| (c[0], c[1])))
        // Close pairs, where nextafter is not just "far in one direction".
        .chain(
            r.iter()
                .flat_map(|&a| [(a, a.next_up()), (a, a.next_down()), (a, -a)]),
        );
    for (a, b) in pairs {
        let (c, c_want) = (copysign_m(a, b, m), copysign_ref(a, b));
        if !same(c, c_want) {
            return Some(format!(
                "copysign({a:e} {:#x}, {b:e} {:#x}) = {c:e}, want {c_want:e}",
                a.to_bits(),
                b.to_bits()
            ));
        }
        let (n, n_want) = (nextafter_m(a, b, m), nextafter_ref(a, b));
        if !same(n, n_want) {
            return Some(format!(
                "nextafter({a:e} {:#x}, {b:e} {:#x}) = {n:e}, want {n_want:e}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    None
}

#[test]
fn model_matches_the_reference_on_edges_and_a_seeded_sample() {
    assert_eq!(first_mismatch(Mutant::None), None);
}

/// Each seeded mutant is caught: the checks above are not vacuous.
#[test]
fn every_seeded_mutant_is_caught() {
    for m in [
        Mutant::CopysignKeepsASign,
        Mutant::CopysignDropsSignBit,
        Mutant::NextafterWrongDirection,
        Mutant::NextafterZeroKeepsASign,
        Mutant::NextafterNoEqualityShortcut,
        Mutant::NextafterNoNanCheck,
        Mutant::NextafterSignTestOnB,
    ] {
        assert!(first_mismatch(m).is_some(), "mutant {m:?} survived");
    }
}
