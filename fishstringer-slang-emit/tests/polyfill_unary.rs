// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 18 / E17: `Asinh`/`Acosh`/`Atanh`/`Cbrt` lower on the elementwise path as inline polyfills;
//! `Erf`/`Erfc`/`Gelu`/`Lgamma` call f32 helper functions the emitter prepends (f64 declines them).
//! (1) the generated kernels compile with `slangc`; (2) an f32 Rust model of the SAME formulas
//! is measured in ULP against an f64 reference over a sampled f32 grid, and the bound below is
//! the measured one. The erf-family f64 references are the same formulas as the model, so
//! `erf_family_matches_pinned_libm_values` pins independent libm values.
//! The model uses Rust's f32 `ln`/`sqrt`/`powf`/`exp`/`sin`, so the bound is "the formula's error given
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

const POLYFILLED: [UnaryOp; 8] = [
    UnaryOp::Asinh,
    UnaryOp::Acosh,
    UnaryOp::Atanh,
    UnaryOp::Cbrt,
    UnaryOp::Erf,
    UnaryOp::Erfc,
    UnaryOp::Gelu,
    UnaryOp::Lgamma,
];
const HELPER_FOUR: [UnaryOp; 4] = [UnaryOp::Erf, UnaryOp::Erfc, UnaryOp::Gelu, UnaryOp::Lgamma];

#[test]
fn polyfilled_unaries_lower_and_compile() {
    for op in POLYFILLED {
        let def = OpDef::elementwise("pf", 1, &[ElementKind::F32], input(0).unary(op));
        let k = try_generate(&def, &key(), &Slang).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        assert!(k.source.contains("output[i] ="), "{op:?}");
        slangc_compiles(&k.source, &k.name);
    }
}

/// The helper block is prepended exactly when a helper op is used (control: `Cbrt` beside it gets none),
/// and the helper four decline typed at f64, where the helpers do not exist.
#[test]
fn helper_block_only_when_used_and_f64_declines() {
    let gen_f32 = |op| {
        let def = OpDef::elementwise("pf", 1, &[ElementKind::F32], input(0).unary(op));
        try_generate(&def, &key(), &Slang)
            .expect("f32 lowers")
            .source
    };
    for (op, call) in [
        (UnaryOp::Erf, "fs_erf("),
        (UnaryOp::Erfc, "fs_erfc("),
        (UnaryOp::Gelu, "fs_gelu("),
        (UnaryOp::Lgamma, "fs_lgamma("),
    ] {
        let src = gen_f32(op);
        assert!(src.contains("float fs_erf("), "{op:?}: no helper block");
        // The kernel body (after the helper block) must call this op's own helper.
        let body = &src[src.find("output[i] =").expect("body")..];
        assert!(body.contains(call), "{op:?}: body does not call {call}:
{body}");
    }
    assert!(
        !gen_f32(UnaryOp::Cbrt).contains("fs_"),
        "Cbrt must not carry the helper block"
    );
    let a = OperandDesc::new(1, &[7], &[1], ElementKind::F64, 4);
    let k64 = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
    for op in HELPER_FOUR {
        let def = OpDef::elementwise("pf", 1, &[ElementKind::F64], input(0).unary(op));
        match try_generate(&def, &k64, &Slang) {
            Err(LowerError::UnsupportedDtype { dtype, detail }) => {
                assert_eq!(dtype, ElementKind::F64);
                assert!(detail.contains("f32 only"), "{detail}")
            }
            other => panic!(
                "{op:?} at f64: expected a typed decline, got {:?}",
                other.map(|k| k.source)
            ),
        }
    }
}

// ---- f32 model of the emitted formulas (same operation order) ----
fn l1p(t: f32) -> f32 {
    let u = 1.0 + t;
    if u == 1.0 {
        t
    } else if t.abs() > 0.5 {
        u.ln()
    } else {
        u.ln() * t / (u - 1.0)
    }
}
const LN2: f32 = std::f32::consts::LN_2;
fn asinh_m(x: f32) -> f32 {
    let a = x.abs();
    let t = a + a * a / (1.0 + (1.0 + a * a).sqrt());
    (if x < 0.0 { -1.0 } else { 1.0 }) * if a > 1.0e18 { a.ln() + LN2 } else { l1p(t) }
}
fn acosh_m(x: f32) -> f32 {
    if x > 1.0e18 {
        x.ln() + LN2
    } else {
        l1p((x - 1.0) + ((x - 1.0) * (x + 1.0)).sqrt())
    }
}
fn atanh_m(x: f32) -> f32 {
    if x.abs() < 0.5 {
        0.5 * l1p(2.0 * x / (1.0 - x))
    } else {
        0.5 * ((1.0 + x) / (1.0 - x)).ln()
    }
}
fn cbrt_m(x: f32) -> f32 {
    let a = x.abs();
    let y0 = a.powf(0.333_333_34);
    let y = if a > 0.0 && a.is_finite() {
        y0 + (a / (y0 * y0) - y0) / 3.0
    } else {
        y0
    };
    (if x < 0.0 { -1.0 } else { 1.0 }) * y
}

// ---- erf family / lgamma ----
fn expnsq(a: f32) -> f32 {
    let ah = (a * 128.0).floor() / 128.0;
    (-(ah * ah)).exp() * (-((a - ah) * (a + ah))).exp()
}
fn erf_series(a: f32) -> f32 {
    let (mut term, mut sum) = (a, a);
    for n in 1..32 {
        term = term * 2.0 * (a * a) / (2 * n + 1) as f32;
        sum += term;
    }
    1.128_379_2 * expnsq(a) * sum
}
fn erfc_cf(a: f32, e: f32) -> f32 {
    let mut t = a;
    for k in (1..=200).rev() {
        t = a + (k as f32 * 0.5) / t;
    }
    0.564_189_6 * e / t
}
fn erf_m(x: f32) -> f32 {
    let a = x.abs();
    let r = if a < 1.0 {
        erf_series(a)
    } else if a > 10.0 {
        1.0
    } else {
        1.0 - erfc_cf(a, expnsq(a))
    };
    if x < 0.0 { -r } else { r }
}
fn erfc_m(x: f32) -> f32 {
    let a = x.abs();
    let r = if a < 0.5 {
        1.0 - erf_series(a)
    } else if a > 10.0 {
        0.0
    } else {
        erfc_cf(a, expnsq(a))
    };
    if x < 0.0 { 2.0 - r } else { r }
}
fn gelu_m(x: f32) -> f32 {
    if x >= 0.0 {
        return 0.5 * x * (1.0 + erf_m(x * 0.707_106_8));
    }
    let m = -x;
    let a = m * 0.707_106_8;
    if a < 0.5 {
        return 0.5 * x * (1.0 - erf_series(a));
    }
    if a > 10.0 {
        return 0.0 * x;
    }
    let mh = (m * 128.0).floor() / 128.0;
    let e = (-0.5 * (mh * mh)).exp() * (-0.5 * ((m - mh) * (m + mh))).exp();
    0.5 * x * erfc_cf(a, e)
}
fn lgamma_pos(x: f32) -> f32 {
    let (mut prod, mut z) = (1.0f32, x);
    for _ in 0..8 {
        if z < 8.0 {
            prod *= z;
            z += 1.0;
        }
    }
    let zi = 1.0 / z;
    let zi2 = zi * zi;
    let series =
        zi * (1.0 / 12.0 - zi2 * (1.0 / 360.0 - zi2 * (1.0 / 1260.0 - zi2 * (1.0 / 1680.0))));
    (z - 0.5) * z.ln() - z + 0.918_938_5 + series - prod.ln()
}
fn lgamma_m(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    if x > 0.0 {
        return lgamma_pos(x);
    }
    let r = x - x.round_ties_even();
    if r == 0.0 {
        return f32::INFINITY;
    }
    let s = (std::f32::consts::PI * r).sin().abs();
    (std::f32::consts::PI / s).ln() - lgamma_pos(1.0 - x)
}
// f64 references (std has no erf/lgamma): series + continued fraction for erf, Lanczos g=7 for lgamma.
fn erf_series64(a: f64) -> f64 {
    let (mut t, mut s) = (a, a);
    for n in 1..200 {
        t = t * 2.0 * a * a / (2 * n + 1) as f64;
        s += t;
    }
    2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp() * s
}
fn erfc64(a: f64) -> f64 {
    if a < 3.0 {
        1.0 - erf_series64(a)
    } else {
        let mut t = a;
        for k in (1..=2000).rev() {
            t = a + (k as f64 * 0.5) / t;
        }
        (-a * a).exp() / std::f64::consts::PI.sqrt() / t
    }
}
fn erfc_ref(x: f64) -> f64 {
    if x >= 0.0 {
        erfc64(x)
    } else {
        2.0 - erfc64(-x)
    }
}
fn erf_ref(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 3.0 {
        erf_series64(a)
    } else {
        1.0 - erfc64(a)
    };
    if x < 0.0 { -r } else { r }
}
fn gelu_ref(x: f64) -> f64 {
    0.5 * x * erfc_ref(-x / std::f64::consts::SQRT_2)
}
fn lgamma_ref(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x < 0.5 {
        let r = x - x.round();
        return (PI / (PI * r).sin().abs()).ln() - lgamma_ref(1.0 - x);
    }
    let c = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let a = c[0] + (1..9).map(|i| c[i] / (x + i as f64)).sum::<f64>();
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// ULP distance between an f32 result and the correctly-rounded f32 of the f64 reference.
fn ulp(got: f32, want64: f64) -> f64 {
    let want = want64 as f32;
    if got.is_nan() || want.is_nan() {
        return if got.is_nan() && want.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
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
    let edges = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        1e-30,
        1e30,
        1e-38,
        f32::MAX,
        f32::MIN_POSITIVE,
    ];
    (0u32..=u32::MAX)
        .step_by(4097)
        .map(f32::from_bits)
        .chain(edges)
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
    let rows: [(&str, fn(f32) -> f32, fn(f64) -> f64, fn(f32) -> bool, f64); 8] = [
        ("asinh", asinh_m, f64::asinh, |_| true, 4.0),
        ("acosh", acosh_m, f64::acosh, |x| x >= 1.0, 4.0),
        ("atanh", atanh_m, f64::atanh, |x| x.abs() < 1.0, 4.0),
        ("cbrt", cbrt_m, f64::cbrt, |_| true, 4.0),
        ("erf", erf_m, erf_ref, |x| x.abs() < 12.0, 6.0),
        ("erfc", erfc_m, erfc_ref, |x| x.abs() < 12.0, 6.0),
        ("gelu", gelu_m, gelu_ref, |x| x.abs() < 12.0, 6.0),
        // lgamma: ULP only where the value is not near a root (the f32 Stirling shift has an
        // absolute error of a few 1e-6, so near x=1,2 and the negative-axis roots ULP is
        // unbounded); the absolute-error row below states that region.
        (
            "lgamma x >= 50",
            lgamma_m,
            lgamma_ref,
            |x| (50.0..1.0e30).contains(&x),
            4.0,
        ),
    ];
    for (name, f, r, dom, bound) in rows {
        let (w, at, n) = worst(f, r, dom);
        println!("{name}: worst {w} ULP at x={at:e} over {n} samples (bound {bound})");
        assert!(w <= bound, "{name}: {w} ULP at {at:e} exceeds {bound}");
    }
}

/// lgamma where ULP is not the right yardstick: absolute error on [-30, 50] (roots at x=1, 2 and the
/// negative-axis roots make a relative bound meaningless), plus the poles and special values.
#[test]
fn lgamma_absolute_error_and_special_values() {
    let mut worst = 0.0f64;
    let mut x = -30.0f32;
    while x < 50.0 {
        // Skip the immediate neighbourhood of a pole, where the value is large and f32 sin(pi*r) rounds.
        if (x - x.round()).abs() > 1.0e-3 || x > 0.0 {
            worst = worst.max((lgamma_m(x) as f64 - lgamma_ref(x as f64)).abs());
        }
        x += 0.00123;
    }
    println!("lgamma: max abs error {worst:e} on [-30, 50] (poles excluded)");
    assert!(worst <= 1.0e-4, "{worst:e}");
    assert_eq!(lgamma_m(0.0), f32::INFINITY);
    assert_eq!(lgamma_m(-3.0), f32::INFINITY);
    // Every f32 with |x| >= 2^24 is an integer, hence a pole: +inf, not inf - inf = NaN.
    assert_eq!(lgamma_m(-1.0e37), f32::INFINITY);
    assert_eq!(lgamma_m(-3.0e38), f32::INFINITY);
    assert_eq!(lgamma_m(-16_777_216.0), f32::INFINITY);
    assert_eq!(lgamma_m(f32::NEG_INFINITY), f32::INFINITY);
    assert!(lgamma_m(f32::NAN).is_nan());
}

/// Independent check for the erf family: values from libm (`math.erf`/`erfc`), not from the formulas
/// above, so a wrong coefficient or branch cannot pass by agreeing with itself.
#[test]
fn erf_family_matches_pinned_libm_values() {
    // (x, erf(x), erfc(x)) from libm, f64 digits.
    let rows: [(f32, f64, f64); 6] = [
        (0.1, 0.112_462_916_018_284_9, 0.887_537_083_981_715_1),
        (0.5, 0.520_499_877_813_046_5, 0.479_500_122_186_953_5),
        (1.0, 0.842_700_792_949_714_9, 0.157_299_207_050_285_1),
        (2.0, 0.995_322_265_018_952_7, 4.677_734_981_047_266e-3),
        (3.0, 0.999_977_909_503_001_4, 2.209_049_699_858_544e-5),
        (-2.0, -0.995_322_265_018_952_7, 1.995_322_265_018_952_7),
    ];
    let close = |got: f32, want: f64, what: &str| {
        let rel = ((got as f64 - want) / want).abs();
        assert!(rel < 1.0e-6, "{what}: got {got:e}, want {want:e}");
    };
    for (x, e, c) in rows {
        close(erf_m(x), e, &format!("erf({x})"));
        close(erfc_m(x), c, &format!("erfc({x})"));
    }
    // gelu(x) = 0.5*x*erfc(-x/sqrt2): gelu(1) = 0.5*(1+erf(1/sqrt2)); gelu(-3) = -1.5*erfc(3/sqrt2).
    close(gelu_m(1.0), 0.841_344_746_068_542_9, "gelu(1)");
    close(gelu_m(-3.0), -0.004_049_694_094_890_2, "gelu(-3)");
}
