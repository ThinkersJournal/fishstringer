// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 17 / E12: multi-dimensional dispatch and execution ids. A 3-D copy over an odd grid, and every subgroup id,
//! lower to Slang that `slangc` compiles; the device ids are the single-device constants.

use std::process::Command;

use fishstringer_slang_emit::Slang;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn gid(axis: u8) -> ScalarExpr {
    // blockIdx * blockDim + threadIdx
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(axis))),
            Box::new(ScalarExpr::Exec(ExecId::BlockDim(axis))),
        )),
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(axis))),
    )
}

fn view(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![gid(2), gid(1), gid(0)],
    }
}

/// `out[z, y, x] = a[z, y, x]` over block 4x2x2 and the odd grid 3x5x7.
fn copy3d() -> KernelFn {
    KernelFn::new(
        "copy3d",
        ElementKind::F32,
        vec![
            Param::buffer("a", ElementKind::F32, BufferRole::Read, Some(3)),
            Param::buffer("out", ElementKind::F32, BufferRole::Write, Some(3)),
        ],
        vec![LocalDecl::scalar(ElementKind::F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: view(0),
            },
            Stmt::Store {
                place: view(1),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 2, 2], [3, 5, 7]))
}

/// `out[0] = lane + subgroup + size + rank + count` (one thread), to spell every id.
fn ids() -> KernelFn {
    let id = |e| ScalarExpr::Exec(e);
    let sum = [
        id(ExecId::LaneId),
        id(ExecId::SubgroupId),
        id(ExecId::SubgroupSize),
        id(ExecId::DeviceRank),
        id(ExecId::DeviceCount),
    ]
    .into_iter()
    .reduce(|a, b| ScalarExpr::Add(Box::new(a), Box::new(b)))
    .expect("five ids");
    KernelFn::new(
        "ids",
        ElementKind::U32,
        vec![Param::buffer(
            "out",
            ElementKind::U32,
            BufferRole::Write,
            Some(1),
        )],
        vec![],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, ElementKind::U32)],
            },
            value: sum,
        }]),
    )
    // the kernel declares the subgroup sizes it is valid for; the emitted source reads them from the device
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(16, 64))
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

#[test]
fn a_three_d_copy_over_an_odd_grid_compiles() {
    let g = Slang.lower_kernel(&copy3d()).expect("lowers");
    assert!(g.source.contains("[numthreads(4, 2, 2)]"), "{}", g.source);
    // grid sizes are baked from the launch shape, block ids and thread ids come from the builtins
    assert!(g.source.contains("bid_.z"), "{}", g.source);
    slangc_compiles(&g.source, &g.name);
}

#[test]
fn subgroup_and_device_ids_spell_and_compile() {
    let g = Slang.lower_kernel(&ids()).expect("lowers");
    for want in [
        "WaveGetLaneIndex()",
        "WaveGetLaneCount()",
        "(gi_ / WaveGetLaneCount())",
        "0u",
        "1u",
    ] {
        assert!(g.source.contains(want), "{want} in\n{}", g.source);
    }
    slangc_compiles(&g.source, &g.name);
}
