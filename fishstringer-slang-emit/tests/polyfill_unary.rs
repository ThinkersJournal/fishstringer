// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 18 / E17: `Asinh`/`Acosh`/`Atanh`/`Cbrt` lower on the elementwise path as inline polyfills.
//! (1) the generated kernels compile with `slangc`; (2) an f32 Rust model of the SAME formulas
//! is measured in ULP against an f64 reference over a sampled f32 grid, and the bound below is
//! the measured one. `Erf`/`Erfc`/`Gelu`/`Lgamma` stay typed declines (they need a helper function).
//! The model uses Rust's f32 `ln`/`sqrt`/`powf`, so the bound is "the formula's error given
//! accurate primitives"; a GPU's own `log`/`pow` ULP adds on top and is NOT measured here (no device run).

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::LowerError;
use unpopped::ir::{OpDef, UnaryOp, input};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn key() -> unpopped_vocab::StructureKey {
    let a = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
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
        .args(["-entry", entry, "-stage", "compute", "-target", "spirv", "-o"])
        .arg(dir.path().join("k.spv"))
        .output()
        .expect("run slangc");
    assert!(out.status.success(), "slangc:\n{}\n{source}", String::from_utf8_lossy(&out.stderr));
}

const POLYFILLED: [UnaryOp; 4] = [UnaryOp::Asinh, UnaryOp::Acosh, UnaryOp::Atanh, UnaryOp::Cbrt];
const STILL_DECLINED: [UnaryOp; 4] = [UnaryOp::Erf, UnaryOp::Erfc, UnaryOp::Gelu, UnaryOp::Lgamma];

#[test]
fn polyfilled_unaries_lower_and_compile() {
    for op in POLYFILLED {
        let def = OpDef::elementwise("pf", 1, &[ElementKind::F32], input(0).unary(op));
        let k = try_generate(&def, &key(), &Slang).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        assert!(k.source.contains("output[i] ="), "{op:?}");
        slangc_compiles(&k.source, &k.name);
    }
}

/// Control for the test above: the four helper-function ops are still typed declines, so a
/// generator that spelled everything would fail here.
#[test]
fn helper_function_unaries_stay_typed_declines() {
    for op in STILL_DECLINED {
        let def = OpDef::elementwise("pf", 1, &[ElementKind::F32], input(0).unary(op));
        match try_generate(&def, &key(), &Slang) {
            Err(LowerError::UnsupportedOp { detail }) => {
                assert!(detail.contains(&format!("{op:?}")), "{detail}")
            }
            other => panic!("{op:?}: expected a typed decline, got {:?}", other.map(|k| k.source)),
        }
    }
}

// ---- f32 model of the emitted formulas (same operation order) ----
fn l1p(t: f32) -> f32 {
    let u = 1.0 + t;
    if u == 1.0 { t } else if t.abs() > 0.5 { u.ln() } else { u.ln() * t / (u - 1.0) }
}
const LN2: f32 = 0.693_147_2;
fn asinh_m(x: f32) -> f32 {
    let a = x.abs();
    let t = a + a * a / (1.0 + (1.0 + a * a).sqrt());
    (if x < 0.0 { -1.0 } else { 1.0 }) * if a > 1.0e18 { a.ln() + LN2 } else { l1p(t) }
}
fn acosh_m(x: f32) -> f32 {
    if x > 1.0e18 { x.ln() + LN2 } else { l1p((x - 1.0) + ((x - 1.0) * (x + 1.0)).sqrt()) }
}
fn atanh_m(x: f32) -> f32 {
    if x.abs() < 0.5 { 0.5 * l1p(2.0 * x / (1.0 - x)) } else { 0.5 * ((1.0 + x) / (1.0 - x)).ln() }
}
fn cbrt_m(x: f32) -> f32 {
    let a = x.abs();
    let y0 = a.powf(0.333_333_34);
    let y = if a > 0.0 && a - a == 0.0 { y0 + (a / (y0 * y0) - y0) / 3.0 } else { y0 };
    (if x < 0.0 { -1.0 } else { 1.0 }) * y
}

/// ULP distance between an f32 result and the correctly-rounded f32 of the f64 reference.
fn ulp(got: f32, want64: f64) -> f64 {
    let want = want64 as f32;
    if got.is_nan() || want.is_nan() {
        return if got.is_nan() && want.is_nan() { 0.0 } else { f64::INFINITY };
    }
    if got == want {
        return 0.0;
    }
    let ord = |v: f32| {
        let b = v.to_bits() as i32;
        if b < 0 { i32::MIN.wrapping_sub(b) } else { b }
    };
    (ord(got) as i64 - ord(want) as i64).abs() as f64
}

/// Sample f32s: every 4097th bit pattern (~1M values) plus the edge cases.
fn grid() -> impl Iterator<Item = f32> {
    let edges = [0.0f32, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 1e-30, 1e30, 1e-38, f32::MAX, f32::MIN_POSITIVE];
    (0u32..=u32::MAX).step_by(4097).map(f32::from_bits).chain(edges)
}

fn worst(f: fn(f32) -> f32, r: fn(f64) -> f64, domain: fn(f32) -> bool) -> (f64, f32, u64) {
    let (mut w, mut at, mut n) = (0.0f64, 0.0f32, 0u64);
    for x in grid().filter(|x| x.is_finite() && domain(*x)) {
        let u = ulp(f(x), r(x as f64));
        n += 1;
        if u > w {
            (w, at) = (u, x);
        }
    }
    (w, at, n)
}

#[test]
fn polyfill_ulp_statement_against_f64_reference() {
    // (name, model, f64 libm reference, domain, measured bound in ULP)
    let rows: [(&str, fn(f32) -> f32, fn(f64) -> f64, fn(f32) -> bool, f64); 4] = [
        ("asinh", asinh_m, f64::asinh, |_| true, 4.0),
        ("acosh", acosh_m, f64::acosh, |x| x >= 1.0, 4.0),
        ("atanh", atanh_m, f64::atanh, |x| x.abs() < 1.0, 4.0),
        ("cbrt", cbrt_m, f64::cbrt, |_| true, 4.0),
    ];
    for (name, f, r, dom, bound) in rows {
        let (w, at, n) = worst(f, r, dom);
        println!("{name}: worst {w} ULP at x={at:e} over {n} samples (bound {bound})");
        assert!(w <= bound, "{name}: {w} ULP at {at:e} exceeds {bound}");
    }
}
