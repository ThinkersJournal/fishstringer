// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E2: scalar parameters and strided views are one `ConstantBuffer`, and the `ParamsBlock` the emitter
//! states must equal the layout `slangc` itself reflects for it (not a hand count: the Unpopped lane's review of the
//! E1 shape, 2026-11-13). Skips unless `slangc` is on PATH; `FISHSTRINGER_REQUIRE_SLANGC` makes a missing one fatal.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn ex(a: u8) -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(a))
}

/// `out[y, x] = a[y, x] * scale + bias`, `a` and `out` rank-2 views, `scale: f32`, `bias: u32` scalars (the odd
/// u32 after an f32 makes the packing visible).
fn scaled_copy() -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ex(1), ex(0)],
    };
    KernelFn::new(
        "scaled_copy",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(2)),
            Param::scalar("scale", ElementKind::F32, 1),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(2)),
            Param::scalar("bias", ElementKind::U32, 2),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::U32),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(2),
                value: ScalarExpr::Mul(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(1)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 8, 1], [1, 1, 1]))
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

#[test]
fn scalars_and_strided_views_are_one_block_after_the_buffers() {
    let g = Slang.lower_kernel(&scaled_copy()).expect("lowers");
    let i = g.interface().expect("interface");
    // buffers get dense slots in order; the block follows them.
    let slots: Vec<(usize, u32)> = i.bindings.iter().map(|b| (b.param, b.slot)).collect();
    assert_eq!(slots, vec![(0, 0), (2, 1)]);
    assert!(
        g.source
            .contains("[[vk::binding(2, 0)]] ConstantBuffer<Params_> pb_;"),
        "{}",
        g.source
    );
    let pb = i.params_block.as_ref().expect("block");
    let names: Vec<&str> = pb.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "b0_stride0",
            "b0_stride1",
            "b0_base",
            "p1",
            "b2_stride0",
            "b2_stride1",
            "b2_base",
            "p3"
        ]
    );
}

#[test]
fn the_stated_block_layout_is_what_slangc_reflects() {
    if !slangc_available() {
        return;
    }
    let g = Slang.lower_kernel(&scaled_copy()).expect("lowers");
    let pb = g.interface().unwrap().params_block.clone().expect("block");
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("k.slang");
    let refl = dir.path().join("k.json");
    std::fs::write(&src, &g.source).expect("write");
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", &g.name, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(dir.path().join("k.spv"))
        .arg("-reflection-json")
        .arg(&refl)
        .output()
        .expect("run slangc");
    assert!(
        out.status.success(),
        "slangc:\n{}\n{}",
        String::from_utf8_lossy(&out.stderr),
        g.source
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&refl).expect("read")).expect("json");
    let params = j["parameters"].as_array().expect("parameters");
    let p = params
        .iter()
        .find(|p| p["name"] == "pb_")
        .unwrap_or_else(|| panic!("no pb_ in {j}"));
    let fields = p["type"]["elementType"]["fields"]
        .as_array()
        .unwrap_or_else(|| panic!("fields missing in {p}"));
    let got: Vec<(String, u64)> = fields
        .iter()
        .map(|f| {
            (
                f["name"].as_str().unwrap().to_owned(),
                f["binding"]["offset"].as_u64().unwrap(),
            )
        })
        .collect();
    let want: Vec<(String, u64)> = pb
        .fields
        .iter()
        .map(|f| (f.name.clone(), u64::from(f.offset)))
        .collect();
    assert_eq!(
        got, want,
        "slangc's reflected offsets vs the stated block\n{p}"
    );
}
