// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.62.0 tile statements and `MemorySpace::Tile`: each declines by name (a typed gap), none is `#[ignore]`d.
//! The 519 B text rides in the decline verbatim; it is CireSnave's risk call, not a clearance.

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, KernelFn, Place, Stmt};
use unpopped_vocab::ElementKind;

const TEXT: &str = "lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read";

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

/// Tile allocations declared: the allocation gap fires first (allocations are scanned before statements).
fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("k", ElementKind::F32, vec![], vec![], Block(body)).with_allocs(vec![
        AllocDecl::tile("ta", ElementKind::F16, 16, 16),
        AllocDecl::tile("tb", ElementKind::F16, 16, 16),
        AllocDecl::tile("tc", ElementKind::F32, 16, 16),
        AllocDecl::tile("td", ElementKind::F32, 16, 16),
    ])
}

/// No allocations declared (the gap scan runs before the checker), so the STATEMENT arm is what fires.
fn stmt_only(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("k", ElementKind::F32, vec![], vec![], Block(body))
}

fn zi() -> ScalarExpr {
    ScalarExpr::ConstInt(0, ElementKind::U32)
}

fn place() -> Place {
    Place::Alloc {
        alloc: 0,
        index: zi(),
    }
}

#[test]
fn each_tile_statement_declines_by_name() {
    let cases = vec![
        (
            Stmt::TileLoad {
                d: 0,
                src: place(),
                elements_per_row: zi(),
                origin: [zi(), zi()],
            },
            "tile statement",
        ),
        (
            Stmt::TileMma {
                d: 3,
                a: 0,
                b: 1,
                c: 2,
            },
            "tile statement",
        ),
        (
            Stmt::TileFill {
                d: 2,
                value: ScalarExpr::Const(0.0),
            },
            "tile statement",
        ),
        (
            Stmt::TileStore {
                a: 3,
                dst: place(),
                elements_per_row: zi(),
                origin: [zi(), zi()],
            },
            "tile statement",
        ),
    ];
    for (st, what) in cases {
        let name = format!("{st:?}");
        let d = decline(&stmt_only(vec![st]));
        assert!(d.starts_with(GAP_PREFIX), "{name}: {d}");
        assert!(d.contains(what), "{name}: {d}");
        assert!(d.contains(TEXT), "{name}: {d}");
        assert!(!d.contains("checker refuses"), "{name}: {d}");
    }
}

#[test]
fn a_tile_allocation_declines_by_name() {
    let d = decline(&kernel(vec![Stmt::Return]));
    assert!(d.starts_with(GAP_PREFIX), "{d}");
    assert!(d.contains("tile allocation"), "{d}");
    assert!(d.contains(TEXT), "{d}");
}

#[test]
fn control_a_kernel_without_tiles_does_not_decline_as_a_tile() {
    let k = KernelFn::new(
        "k",
        ElementKind::F32,
        vec![],
        vec![],
        Block(vec![Stmt::Return]),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(4),
    )]);
    let d = Slang
        .lower_kernel(&k)
        .map(|_| String::new())
        .unwrap_or_else(|e| format!("{e:?}"));
    assert!(!d.contains("tile"), "{d}");
}
