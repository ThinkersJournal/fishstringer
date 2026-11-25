// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 1: `Backend::lower_with` and `CallerCaps`. A plan that reads a runtime scalar `Param` needs a `ConstantBuffer` the caller
//! fills and binds, so a legacy caller (which binds by declaration order and passes no block) is REFUSED with
//! `UnsupportedPlanShape` naming the capability, and a caller that states `with_interface()` gets the kernel plus the interface
//! (bindings, params block). A parameterless plan is byte-identical for both. NOT shown: a device run.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{CallerCaps, LowerError};
use unpopped::ir::{OpDef, input, param};
use unpopped::{try_generate, try_generate_with};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn key(dt: ElementKind) -> unpopped_vocab::StructureKey {
    let a = OperandDesc::new(1, &[7], &[1], dt, 4);
    structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89)
}

fn scale() -> OpDef {
    OpDef::elementwise("scale", 1, &[ElementKind::F32], input(0) * param(0))
}

fn plain() -> OpDef {
    OpDef::elementwise("plain", 1, &[ElementKind::F32], input(0) + input(0))
}

#[test]
fn a_legacy_caller_is_refused_a_plan_that_needs_the_interface() {
    let k = key(ElementKind::F32);
    for r in [
        try_generate(&scale(), &k, &Slang),
        try_generate_with(&scale(), &k, &Slang, &CallerCaps::LEGACY),
    ] {
        match r {
            Err(LowerError::UnsupportedPlanShape { detail }) => {
                assert!(detail.contains("CallerCaps::with_interface()"), "{detail}");
                assert!(detail.contains("params_block"), "{detail}");
            }
            other => panic!("expected UnsupportedPlanShape, got {other:?}"),
        }
    }
}

#[test]
fn an_interface_caller_gets_the_block_the_bindings_and_the_layout() {
    let g = try_generate_with(
        &scale(),
        &key(ElementKind::F32),
        &Slang,
        &CallerCaps::LEGACY.with_interface(),
    )
    .expect("lowers for an interface caller");
    let src = &g.source;
    assert!(
        src.contains("[[vk::binding(0, 0)]] StructuredBuffer<float> input0;"),
        "{src}"
    );
    assert!(
        src.contains("[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;"),
        "{src}"
    );
    assert!(
        src.contains("[[vk::binding(2, 0)]] ConstantBuffer<Params_> pb_;"),
        "{src}"
    );
    assert!(src.contains("float p0 = pb_.p0;"), "{src}");
    assert!(src.contains("uint n;"), "n is the first block field: {src}");
    let guard = src.find("if (i >= pb_.n)").expect("the n guard");
    assert!(
        guard < src.find("float p0 = pb_.p0;").unwrap() && guard < src.find("output[i] =").unwrap(),
        "the guard comes before any read or store: {src}"
    );
    let i = g.interface().expect("an interface is stated");
    let slots: Vec<_> = i.bindings.iter().map(|b| (b.param, b.slot)).collect();
    assert_eq!(
        slots,
        [(0, 0), (1, 1)],
        "buffers by slot; the block follows them at 2"
    );
    let pb = i.params_block.as_ref().expect("a params block");
    assert_eq!((pb.size, pb.align, pb.fields.len()), (8, 4, 2));
    assert_eq!((pb.fields[0].name.as_str(), pb.fields[0].offset), ("n", 0));
    assert_eq!((pb.fields[1].name.as_str(), pb.fields[1].offset), ("p0", 4));
    assert_eq!(pb.fields[0].ty, ElementKind::U32);
    assert_eq!(i.params.len(), 3, "input0, output, p0");
}

#[test]
fn a_parameterless_plan_is_the_same_for_both_callers_and_states_no_interface() {
    let k = key(ElementKind::F32);
    let a = try_generate(&plain(), &k, &Slang).unwrap();
    let b = try_generate_with(&plain(), &k, &Slang, &CallerCaps::LEGACY.with_interface()).unwrap();
    assert_eq!(a.source, b.source);
    assert!(!a.source.contains("vk::binding"), "historical text kept");
    assert!(a.interface().is_none());
    assert!(
        !a.source.contains("pb_"),
        "no block, no guard: exact dispatch kept"
    );
}

#[test]
fn the_interface_kernel_compiles_with_slangc() {
    if Command::new("slangc").arg("-v").output().is_err() {
        assert!(
            std::env::var_os("FISHSTRINGER_REQUIRE_SLANGC").is_none(),
            "FISHSTRINGER_REQUIRE_SLANGC is set but `slangc` could not be run"
        );
        eprintln!("skipping: `slangc` not on PATH");
        return;
    }
    let g = try_generate_with(
        &scale(),
        &key(ElementKind::F32),
        &Slang,
        &CallerCaps::LEGACY.with_interface(),
    )
    .unwrap();
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    std::fs::write(&src, &g.source).unwrap();
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", &g.name, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .args(["-denorm-mode-fp32", "preserve"])
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
