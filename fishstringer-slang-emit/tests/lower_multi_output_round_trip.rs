// SPDX-License-Identifier: MIT OR Apache-2.0
//! E17 slice 3 round trip: the multi-output and hetero-u8 kernels the `lower` path emits for an interface caller are lifted back
//! through `unpopped_convert::lift_elementwise_multi` (the real spelling, `output0..`, `(uint8_t)`), and the lifted bodies are
//! compared with the originals. Vacuity mutants: a wrong declared output count and a swapped body must NOT pass. NOT shown: a
//! device run, nor that the lifted op re-emits byte-identically.

use fishstringer_slang_emit::Slang;
use fishstringer_slang_parse::SLANG;
use unpopped::backend::CallerCaps;
use unpopped::ir::{BinaryOp, OpDef, input, konst};
use unpopped::try_generate_with;
use unpopped_convert::lift_elementwise_multi;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

fn key(target: Option<&str>) -> StructureKey {
    let a = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    let mut k = structure_key(OpCategory::BinaryElementwise, &[a; 4], ArchSku::Sm89);
    if let Some(t) = target {
        k.target = TargetId::parse(t).expect("valid target token");
    }
    k
}

fn iface() -> CallerCaps {
    CallerCaps::LEGACY.with_interface()
}

fn sum_and_product() -> OpDef {
    OpDef::elementwise_multi(
        "sum_and_product",
        2,
        &[ElementKind::F32],
        vec![input(0) + input(1), input(0) * input(1)],
    )
}

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

#[test]
fn two_uniform_outputs_lift_back_to_the_same_bodies() {
    let g = try_generate_with(&sum_and_product(), &key(None), &Slang, &iface()).expect("lowers");
    let f = ElementKind::F32;
    let l = lift_elementwise_multi(&SLANG, &g.source, &g.name, &[f, f], &[None, None])
        .unwrap_or_else(|e| panic!("must re-lift: {e:?}\n{}", g.source));
    let want = sum_and_product();
    assert_eq!(
        format!("{:?}", l.op.body),
        format!("{:?}", want.body),
        "{}",
        g.source
    );
    assert_eq!(
        l.op.extra_out_bodies, want.extra_out_bodies,
        "output 1: {}",
        g.source
    );
}

#[test]
fn the_hetero_u8_kernel_lifts_back_with_its_u8_output() {
    let g = try_generate_with(
        &dropout_like(),
        &key(Some("vulkan:sg32.arith-f16-i8")),
        &Slang,
        &iface(),
    )
    .expect("lowers");
    let f = ElementKind::F32;
    let l = lift_elementwise_multi(
        &SLANG,
        &g.source,
        &g.name,
        &[f, f],
        &[None, Some(ElementKind::U8)],
    )
    .unwrap_or_else(|e| panic!("must re-lift: {e:?}\n{}", g.source));
    let want = dropout_like();
    // Output 1 (the u8 mask) and its dtype come back exactly. Output 0 does NOT come back structurally identical: the emitter
    // hoists the shared predicate into `float tmp0 = (cmp ? 1.0 : 0.0)` and the select re-tests `tmp0 != 0.0`; the lifter reads
    // that spelling literally. It is the same function of the inputs, but it is not the original tree, so the exact lifted form
    // is pinned rather than claimed equal.
    assert_eq!(l.op.extra_out_bodies, want.extra_out_bodies, "{}", g.source);
    assert_eq!(
        l.op.extra_out_dtypes,
        [Some(ElementKind::U8)],
        "{}",
        g.source
    );
    assert_eq!(
        format!("{:?}", l.op.body),
        "Mul(Input(0), Select(Binary(CmpNe, Select(Binary(CmpLt, Input(1), Const(0.5)), Const(1.0), Const(0.0)), Const(0.0)), Const(2.0), Const(0.0)))"
    );
    assert_ne!(
        format!("{:?}", l.op.body),
        format!("{:?}", want.body),
        "if equal, tighten this test"
    );
}

#[test]
fn vacuity_a_wrong_output_count_and_a_swapped_body_are_refused_or_differ() {
    let g = try_generate_with(&sum_and_product(), &key(None), &Slang, &iface()).expect("lowers");
    let f = ElementKind::F32;
    // Declaring three outputs when two are stored must refuse (a declared output with no store).
    assert!(
        lift_elementwise_multi(&SLANG, &g.source, &g.name, &[f, f], &[None, None, None]).is_err()
    );
    // Declaring one output when two are stored must refuse (a store past the declared count).
    assert!(lift_elementwise_multi(&SLANG, &g.source, &g.name, &[f, f], &[None]).is_err());
    // A mutated source (outputs swapped) lifts to a DIFFERENT body, so the equality above is not vacuous.
    let swapped = g
        .source
        .replace("output0[i] =", "outputX[i] =")
        .replace("output1[i] =", "output0[i] =")
        .replace("outputX[i] =", "output1[i] =");
    assert_ne!(swapped, g.source, "mutation applied");
    let l =
        lift_elementwise_multi(&SLANG, &swapped, &g.name, &[f, f], &[None, None]).expect("lifts");
    assert_ne!(
        format!("{:?}", l.op.body),
        format!("{:?}", sum_and_product().body)
    );
}
