// SPDX-License-Identifier: MIT OR Apache-2.0
//! E17 slice 3: multi-output and hetero out-dtype plans on the `lower` path. They change the buffer list the caller binds, so
//! (like a `Param`) they are refused for a legacy caller and lowered, with explicit slots, the `n` block and guard, for a caller
//! that states `with_interface()`. A single uniform output keeps its historical text. Skips the slangc cases unless `slangc`
//! is on PATH (`FISHSTRINGER_REQUIRE_SLANGC` makes a missing one fatal). NOT shown: a device run.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::{CallerCaps, LowerError};
use unpopped::ir::{BinaryOp, OpDef, input, konst};
use unpopped::{try_generate, try_generate_with};
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

const VK8: &str = "vulkan:sg32.arith-f16-i8";

fn key(operands: usize, target: Option<&str>) -> StructureKey {
    let a = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    let ops = vec![a; operands];
    let mut k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    if let Some(t) = target {
        k.target = TargetId::parse(t).expect("valid target token");
    }
    k
}

/// out0 = a + b, out1 = a * b (two uniform f32 outputs sharing both loads).
fn sum_and_product() -> OpDef {
    OpDef::elementwise_multi(
        "sum_and_product",
        2,
        &[ElementKind::F32],
        vec![input(0) + input(1), input(0) * input(1)],
    )
}

/// out0 = a * select(b < 0.5, 2.0, 0.0) (f32), out1 = (b < 0.5) as a u8 keep-mask: the dropout shape.
fn dropout_like() -> OpDef {
    let keep = input(1).binary(BinaryOp::CmpLt, konst(0.5));
    OpDef::elementwise_multi_hetero(
        "dropout_like",
        2,
        &[ElementKind::F32],
        vec![
            (input(0) * keep.clone().select(konst(2.0), konst(0.0)), None),
            (keep, Some(ElementKind::U8)),
        ],
    )
}

fn iface() -> CallerCaps {
    CallerCaps::LEGACY.with_interface()
}

#[test]
fn a_legacy_caller_is_refused_a_multi_output_and_a_hetero_plan() {
    for (op, k) in [
        (sum_and_product(), key(4, None)),
        (dropout_like(), key(4, Some(VK8))),
    ] {
        for r in [
            try_generate(&op, &k, &Slang),
            try_generate_with(&op, &k, &Slang, &CallerCaps::LEGACY),
        ] {
            match r {
                Err(LowerError::UnsupportedPlanShape { detail }) => {
                    assert!(detail.contains("CallerCaps::with_interface()"), "{detail}");
                }
                other => panic!("expected UnsupportedPlanShape, got {other:?}"),
            }
        }
    }
}

#[test]
fn two_uniform_outputs_get_slots_names_the_block_and_the_guard() {
    let g = try_generate_with(&sum_and_product(), &key(4, None), &Slang, &iface()).expect("lowers");
    let s = &g.source;
    for want in [
        "[[vk::binding(0, 0)]] StructuredBuffer<float> input0;",
        "[[vk::binding(1, 0)]] StructuredBuffer<float> input1;",
        "[[vk::binding(2, 0)]] RWStructuredBuffer<float> output0;",
        "[[vk::binding(3, 0)]] RWStructuredBuffer<float> output1;",
        "[[vk::binding(4, 0)]] ConstantBuffer<Params_> pb_;",
        "output0[i] = ",
        "output1[i] = ",
    ] {
        assert!(s.contains(want), "missing `{want}`:\n{s}");
    }
    assert!(
        s.find("if (i >= pb_.n)").unwrap() < s.find("output0[i] =").unwrap(),
        "the guard precedes every store:\n{s}"
    );
    assert!(!s.contains("RWStructuredBuffer<float> output;"), "{s}");
    let i = g.interface().expect("interface");
    let slots: Vec<_> = i.bindings.iter().map(|b| (b.param, b.slot)).collect();
    assert_eq!(slots, [(0, 0), (1, 1), (2, 2), (3, 3)]);
    let pb = i.params_block.as_ref().expect("block");
    assert_eq!((pb.size, pb.fields.len()), (4, 1), "only `n`");
    assert_eq!(pb.fields[0].name, "n");
    let names: Vec<_> = i.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["input0", "input1", "output0", "output1"]);
}

#[test]
fn a_hetero_output_is_declared_u8_and_converted_at_its_own_store_only() {
    let g =
        try_generate_with(&dropout_like(), &key(4, Some(VK8)), &Slang, &iface()).expect("lowers");
    let s = &g.source;
    assert!(
        s.contains("[[vk::binding(2, 0)]] RWStructuredBuffer<float> output0;"),
        "{s}"
    );
    assert!(
        s.contains("[[vk::binding(3, 0)]] RWStructuredBuffer<uint8_t> output1;"),
        "{s}"
    );
    let line = |n: &str| {
        s.lines()
            .find(|l| l.contains(n))
            .unwrap_or_else(|| panic!("no {n}:\n{s}"))
    };
    assert!(
        line("output1[i] =").contains("(uint8_t)"),
        "u8 store converts: {s}"
    );
    assert!(
        !line("output0[i] =").contains("uint8_t"),
        "output 0 is not narrowed: {s}"
    );
    // The shared predicate is computed once, not twice.
    assert_eq!(
        s.matches("(float)0.5").count(),
        1,
        "shared compare hoisted once:\n{s}"
    );
    let i = g.interface().unwrap();
    assert_eq!(i.params[3].name, "output1");
    assert!(
        matches!(
            i.params[3].kind,
            unpopped::kernel::ParamKind::Buffer {
                elem: ElementKind::U8,
                ..
            }
        ),
        "{:?}",
        i.params[3]
    );
}

#[test]
fn a_hetero_u8_output_is_declined_where_the_target_has_no_8_bit_arithmetic() {
    // cuda:sm89 advertises no <arith> field, so the u8 store is refused with a typed error, not mis-spelled.
    match try_generate_with(&dropout_like(), &key(4, None), &Slang, &iface()) {
        Err(LowerError::UnsupportedPlanShape { detail }) => {
            assert!(detail.contains("u8 keep-mask"), "{detail}");
        }
        other => panic!("expected a typed decline, got {other:?}"),
    }
}

#[test]
fn a_single_uniform_output_keeps_its_historical_text() {
    let op = OpDef::elementwise("plain", 1, &[ElementKind::F32], input(0) + input(0));
    let a = try_generate(&op, &key(2, None), &Slang).unwrap();
    let b = try_generate_with(&op, &key(2, None), &Slang, &iface()).unwrap();
    assert_eq!(a.source, b.source);
    assert!(a.source.contains("RWStructuredBuffer<float> output;"));
    assert!(!a.source.contains("vk::binding") && a.interface().is_none());
}

fn slangc_available() -> bool {
    if Command::new("slangc").arg("-v").output().is_ok() {
        return true;
    }
    assert!(
        std::env::var_os("FISHSTRINGER_REQUIRE_SLANGC").is_none(),
        "FISHSTRINGER_REQUIRE_SLANGC is set but `slangc` could not be run"
    );
    eprintln!("skipping: `slangc` not on PATH");
    false
}

/// Compile with slangc and return its reflection JSON.
fn reflect(g: &unpopped::backend::GeneratedKernel) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    let refl = dir.path().join("k.json");
    std::fs::write(&src, &g.source).unwrap();
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", &g.name, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .args(["-denorm-mode-fp32", "preserve", "-reflection-json"])
        .arg(&refl)
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "slangc:\n{}\n{}",
        String::from_utf8_lossy(&out.stderr),
        g.source
    );
    serde_json::from_str(&std::fs::read_to_string(&refl).unwrap()).expect("json")
}

#[test]
fn the_stated_slots_and_block_offsets_are_what_slangc_reflects() {
    if !slangc_available() {
        return;
    }
    for (op, k) in [
        (sum_and_product(), key(4, None)),
        (dropout_like(), key(4, Some(VK8))),
    ] {
        let g = try_generate_with(&op, &k, &Slang, &iface()).unwrap();
        let j = reflect(&g);
        let i = g.interface().unwrap();
        let params = j["parameters"].as_array().expect("parameters");
        // Every buffer: slangc's binding index equals the stated slot, by name.
        for b in &i.bindings {
            let name = &i.params[b.param].name;
            let p = params
                .iter()
                .find(|p| p["name"] == name.as_str())
                .unwrap_or_else(|| panic!("no {name} in {j}"));
            assert_eq!(
                p["binding"]["index"].as_u64(),
                Some(u64::from(b.slot)),
                "slot of {name}\n{p}"
            );
        }
        // The block: its slot and every field offset.
        let pb = params.iter().find(|p| p["name"] == "pb_").expect("pb_");
        let block_slot = u64::try_from(i.bindings.len()).unwrap();
        assert_eq!(pb["binding"]["index"].as_u64(), Some(block_slot), "{pb}");
        let want: Vec<(String, u64)> = i
            .params_block
            .as_ref()
            .unwrap()
            .fields
            .iter()
            .map(|f| (f.name.clone(), u64::from(f.offset)))
            .collect();
        let got: Vec<(String, u64)> = pb["type"]["elementType"]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["name"].as_str().unwrap().to_owned(),
                    f["binding"]["offset"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(got, want, "{pb}");
    }
}
