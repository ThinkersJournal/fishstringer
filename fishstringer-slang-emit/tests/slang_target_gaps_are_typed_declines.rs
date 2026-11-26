// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row E18: each recorded gap of the Slang target is a typed `UnsupportedKernel` decline whose reason starts with
//! `GAP_PREFIX` and names the construct. A kernel with no gap gets the generic "does not lower" decline instead,
//! so the prefix really discriminates (the control).

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, MemorySpace, Scope};
use unpopped::kernel::{Block, KernelFn, Stmt};
use unpopped_vocab::ElementKind;

fn empty() -> KernelFn {
    KernelFn::new("k", ElementKind::F32, vec![], vec![], Block(vec![]))
}

fn decline(k: &KernelFn) -> String {
    match Slang.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

fn gap(k: &KernelFn, names: &str) {
    let d = decline(k);
    assert!(d.starts_with(GAP_PREFIX), "{d}");
    assert!(d.contains(names), "{d}");
}

#[test]
fn a_kernel_without_a_gap_gets_the_generic_decline_not_the_gap_prefix() {
    assert!(!decline(&empty()).starts_with(GAP_PREFIX));
}

#[test]
fn dynamic_shared_extent_declines() {
    let mut k = empty();
    k.allocs.push(AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Param(0),
    ));
    gap(&k, "dynamic shared extent");
}

#[test]
fn a_static_shared_extent_is_not_a_gap() {
    let mut k = empty();
    k.allocs.push(AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(64),
    ));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}

#[test]
fn device_and_cluster_barriers_decline() {
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Device));
    gap(&k, "grid");
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Cluster));
    gap(&k, "cluster");
}

#[test]
fn a_block_barrier_is_not_a_gap() {
    let mut k = empty();
    k.body.0.push(Stmt::barrier(Scope::Block));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}

#[test]
fn the_patent_flagged_async_statements_decline_by_name() {
    for st in [Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }] {
        let mut k = empty();
        k.body.0.push(st);
        gap(&k, "async copy statement");
        gap(&k, "PATENT_SWEEP");
    }
}

#[test]
fn peer_cluster_and_multicast_places_decline() {
    use unpopped::ir::ScalarExpr;
    use unpopped::kernel::Place;
    let zero = || ScalarExpr::Const(0.0);
    let places = [
        (
            Place::Peer {
                buffer: 0,
                device: zero(),
                coords: vec![zero()],
            },
            "peer-device",
        ),
        (
            Place::ClusterPeer {
                alloc: 0,
                block: zero(),
                index: zero(),
            },
            "cluster-peer",
        ),
        (
            Place::Multicast {
                buffer: 0,
                coords: vec![zero()],
            },
            "multicast",
        ),
    ];
    for (place, what) in places {
        let mut k = empty();
        k.body.0.push(Stmt::Store {
            place,
            value: zero(),
        });
        gap(&k, what);
    }
}

#[test]
fn multi_device_and_cluster_launches_decline_and_one_device_does_not() {
    use unpopped::exec::LaunchShape;
    let mut l = LaunchShape::new([1, 1, 1], [1, 1, 1]);
    let one = empty().with_launch(l);
    // An empty launched kernel lowers now (E1); the control is only that it is not a gap decline.
    match Slang.lower_kernel(&one) {
        Ok(_) => {}
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(!detail.starts_with(GAP_PREFIX), "{detail}")
        }
        Err(other) => panic!("{other:?}"),
    }
    l.devices = 2;
    gap(&empty().with_launch(l), "multi-device");
    l.devices = 1;
    l.cluster = Some([2, 1, 1]);
    gap(&empty().with_launch(l), "cluster launch");
}

// --- E18 completion: the gaps the list names that had no test of their own ---

fn zero() -> unpopped::ir::ScalarExpr {
    unpopped::ir::ScalarExpr::Const(0.0)
}

#[test]
fn a_cluster_scope_fence_and_atomic_decline() {
    use unpopped::kernel::{AtomicOp, MemoryOrder, Place};
    let mut k = empty();
    k.body.0.push(Stmt::Fence {
        scope: Scope::Cluster,
        order: MemoryOrder::AcqRel,
    });
    gap(&k, "fence at cluster scope");
    let mut k = empty();
    k.body.0.push(Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: Place::Alloc {
            alloc: 0,
            index: zero(),
        },
        value: zero(),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Cluster,
    });
    gap(&k, "atomic at cluster scope");
}

#[test]
fn a_copy_async_into_a_peer_place_declines_as_the_async_gap() {
    use unpopped::kernel::Place;
    let mut k = empty();
    k.body.0.push(Stmt::CopyAsync {
        dst: Place::ClusterPeer {
            alloc: 0,
            block: zero(),
            index: zero(),
        },
        src: Place::Alloc {
            alloc: 1,
            index: zero(),
        },
        elems: 4,
        valid: None,
        cache: unpopped::kernel::CacheHint::Auto,
        l2_prefetch: None,
    });
    // Whichever gap the scan meets first, the decline is a named gap, never a silent lowering.
    let d = decline(&k);
    assert!(d.starts_with(GAP_PREFIX), "{d}");
}

#[test]
fn multicast_load_reduce_declines() {
    use unpopped::kernel::{AtomicOp, MemoryOrder, Place};
    let mut k = empty();
    k.body.0.push(Stmt::MulticastLoadReduce {
        local: 0,
        place: Place::Multicast {
            buffer: 0,
            coords: vec![zero()],
        },
        op: AtomicOp::Add,
        order: MemoryOrder::Relaxed,
    });
    gap(&k, "multicast");
}

#[test]
fn sparse_and_block_scaled_mma_atoms_decline_and_a_dense_one_does_not() {
    use unpopped::mma::MmaAtom;
    use unpopped_vocab::layout_algebra::Layout;
    let mma = |atom: MmaAtom| Stmt::Mma {
        atom,
        a: 0,
        b: 1,
        c: 2,
        d: 3,
        extra: vec![],
    };
    let leaf = |n: i64| Layout::leaf(n, 1).expect("a leaf layout");
    let dense = MmaAtom::nvidia_m16n8k16_f16_f32();

    let sparse = dense
        .clone()
        .with_sparse_24(&leaf(128), (ElementKind::U32, &leaf(64)))
        .expect("a sparse atom");
    let scaled = dense
        .clone()
        .with_block_scale(4, ElementKind::F32, (&leaf(64), &leaf(32)))
        .expect("a block-scaled atom");

    let mut k = empty();
    k.body.0.push(mma(sparse));
    gap(&k, "sparse (2:4) Mma atom");
    let mut k = empty();
    k.body.0.push(mma(scaled));
    gap(&k, "block-scaled Mma atom");

    // Control: the dense atom is not a gap (it is just not lowered yet).
    let mut k = empty();
    k.body.0.push(mma(dense));
    assert!(!decline(&k).starts_with(GAP_PREFIX));
}

#[test]
fn a_dense_mma_declines_by_name_but_is_not_a_target_gap() {
    use unpopped::mma::MmaAtom;
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    let (m, n, kk) = atom.mnk();
    let _ = (m, n, kk);
    let frag = |name: &str, which: usize, ek: ElementKind| {
        AllocDecl::new(
            name,
            MemorySpace::Private,
            ek,
            Extent::Static(atom.values(which)),
        )
    };
    let (ta, tb, tc) = atom.dtypes();
    let mut k = empty()
        .with_allocs(vec![
            frag("a", 0, ta),
            frag("b", 1, tb),
            frag("c", 2, tc),
            frag("d", 2, tc),
        ])
        .with_launch(unpopped::exec::LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.body.0.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: 3,
        extra: vec![],
    });
    let d = decline(&k);
    assert!(!d.starts_with(GAP_PREFIX), "{d}");
    assert!(
        d.contains("Mma: no cooperative-matrix shape for this atom"),
        "{d}"
    );
    assert!(d.contains(atom.name()), "{d}");
}

/// Board 434 C extended (CireSnave 2026-11-16): NVIDIA US11803380B2, US12204897B2, US12710961B2. Each statement is
/// refused by name; the control (an empty kernel) gets no gap prefix.
#[test]
fn the_patent_flagged_mbarrier_and_matrix_wait_statements_decline_by_name() {
    use unpopped::ir::ScalarExpr;
    use unpopped::kernel::Place;
    let bar = || Place::Alloc {
        alloc: 0,
        index: ScalarExpr::ConstInt(0, ElementKind::U32),
    };
    let one = || ScalarExpr::ConstInt(1, ElementKind::U32);
    let cases: Vec<(Stmt, &str)> = vec![
        (
            Stmt::MbarInit {
                bar: bar(),
                count: one(),
            },
            "mbarrier statement",
        ),
        (
            Stmt::MbarArrive {
                bar: bar(),
                tx: None,
            },
            "mbarrier statement",
        ),
        (
            Stmt::MbarCompleteTx {
                bar: bar(),
                bytes: one(),
            },
            "mbarrier statement",
        ),
        (
            Stmt::MbarWait {
                bar: bar(),
                parity: one(),
            },
            "mbarrier statement",
        ),
        (
            Stmt::CopyBulk {
                dst: bar(),
                src: bar(),
                elems: 4,
                bar: bar(),
            },
            "mbarrier statement",
        ),
        (
            Stmt::WgmmaWait { pending: 0 },
            "pending-matrix-multiply wait",
        ),
        (
            Stmt::Tcgen05Commit { bar: bar() },
            "pending-matrix-multiply wait",
        ),
    ];
    for (st, what) in cases {
        let mut k = empty();
        k.body.0.push(st);
        gap(&k, what);
        gap(&k, "board 434 C extended");
    }
}

/// Board 434 C: `Collective` and `CollectiveTuple` (every kind) are refused by name, at the top level and nested inside
/// `If`, `For` and `While` bodies at depth. The control (an empty kernel, and the same nesting around a harmless
/// statement) gets no gap prefix, so the nesting walk is what finds the collective.
#[test]
fn collective_and_collective_tuple_decline_by_name_including_nested() {
    use unpopped::ir::{ReduceOp, ScalarExpr};
    use unpopped::kernel::CollectiveKind;
    let one = || ScalarExpr::Const(1.0);
    let single = |kind: CollectiveKind| Stmt::Collective {
        kind,
        value: one(),
        result: 0,
        scope: Scope::Subgroup,
    };
    let tuple = |kind: CollectiveKind| Stmt::CollectiveTuple {
        kind,
        values: vec![one(), one()],
        results: vec![0, 1],
        scope: Scope::Subgroup,
    };
    let kinds = || {
        vec![
            CollectiveKind::Reduce(ReduceOp::Sum),
            CollectiveKind::Scan {
                op: ReduceOp::Sum,
                exclusive: false,
            },
            CollectiveKind::Broadcast { lane: zero() },
        ]
    };
    // Wrap `inner` as the only statement at depth: If { For { While { inner } } }, and one level of each alone.
    let wrap = |inner: Stmt| -> Vec<Stmt> {
        let w = |s: Stmt| Stmt::While {
            cond: one(),
            body: Block(vec![s]),
        };
        let f = |s: Stmt| Stmt::For {
            local: 0,
            start: zero(),
            end: one(),
            step: ScalarExpr::ConstInt(1, ElementKind::U32),
            body: Block(vec![s]),
            unroll: unpopped::kernel::Unroll::None,
        };
        let i = |s: Stmt| Stmt::If {
            cond: one(),
            then: Block(vec![s]),
            els: Block(vec![]),
        };
        vec![
            inner.clone(),
            i(inner.clone()),
            f(inner.clone()),
            w(inner.clone()),
            i(f(w(inner.clone()))),
            Stmt::If {
                cond: one(),
                then: Block(vec![]),
                els: Block(vec![w(i(inner))]),
            },
        ]
    };
    let mut n = 0;
    for kind in kinds() {
        for st in wrap(single(kind.clone()))
            .into_iter()
            .chain(wrap(tuple(kind)))
        {
            let mut k = empty();
            k.body.0.push(st);
            gap(&k, "collective statement");
            gap(&k, "PATENT_SWEEP");
            n += 1;
        }
    }
    // 3 kinds x 2 statements x 6 placements: none was skipped.
    assert_eq!(n, 36);
    // Control: the same nesting around a harmless statement is not a gap.
    for st in wrap(Stmt::Break) {
        let mut k = empty();
        k.body.0.push(st);
        assert!(!decline(&k).starts_with(GAP_PREFIX));
    }
}

/// unpopped 0.57.0 (U8c): `Stmt::ProxyFence` is a gap decline (`gaps.rs`) that names the statement and says why, not the
/// generic "this statement" wildcard. Control: the empty kernel declines generically and does not mention it.
#[test]
fn proxy_fence_declines_by_name() {
    let mut k = empty().with_launch(unpopped::exec::LaunchShape::new([1, 1, 1], [1, 1, 1]));
    k.body.0.push(Stmt::ProxyFence {
        space: MemorySpace::Shared,
    });
    let d = decline(&k);
    assert!(d.starts_with(GAP_PREFIX) && d.contains("ProxyFence"), "{d}");
    assert!(!decline(&empty()).contains("ProxyFence"));
}

/// unpopped 0.57.0 (U8c) parity: every NVIDIA async-proxy statement and byte-addressed place is a typed
/// `UnsupportedKernel` that names it, never the generic "this statement"/"this place". Control: a kernel that stores
/// through a `Place::Alloc` lowers past this point (its decline, if any, does not name any of these).
/// unpopped 0.57 U8c. Each kernel is checker-CONFORMING (so it is the Slang emitter, not unpopped's `check_kernel`,
/// that refuses it: a `[1,1,1]` launch is refused by the checker with `WgmmaUnsupported`, which also contains
/// "Wgmma"). The assertions pin WHICH site fires: the permanent NVIDIA-only constructs are gaps (`GAP_PREFIX`,
/// raised in `gaps.rs`), the byte places are "not lowered yet" (`kernel_lower.rs`), and no decline is the checker's.
#[test]
fn u8c_statements_and_byte_places_decline_by_name() {
    use unpopped::exec::{LaunchShape, SubgroupSize};
    use unpopped::ir::ScalarExpr;
    use unpopped::kernel::{BufferRole, LocalDecl, OperandLayout, Param, Place};
    let z = || ScalarExpr::Const(0.0);
    let zi = || ScalarExpr::ConstInt(0, ElementKind::I32);
    let alloc = |i: u32| Place::Alloc {
        alloc: i,
        index: zi(),
    };
    let launch = || {
        let mut l = LaunchShape::new([128, 1, 1], [1, 1, 1]);
        l.subgroup = Some(SubgroupSize::new(32, 32));
        l
    };
    let shared = |n: &str, e| AllocDecl::new(n, MemorySpace::Shared, e, Extent::Static(4096));
    // allocs: 0,1 Shared f16 tiles; 2 Private f32 fragment (n/2 = 4); 3 Private f16 (2 = 2 * count); 4 Tensor f32 (128 lanes x 1 column)
    let allocs = |tensor: bool| {
        let mut v = vec![
            shared("a", ElementKind::F16),
            shared("b", ElementKind::F16),
            AllocDecl::new(
                "d",
                MemorySpace::Private,
                ElementKind::F32,
                Extent::Static(4),
            ),
            AllocDecl::new(
                "h",
                MemorySpace::Private,
                ElementKind::F16,
                Extent::Static(2),
            ),
        ];
        if tensor {
            v.push(AllocDecl::new(
                "t",
                MemorySpace::Tensor,
                ElementKind::F32,
                Extent::Static(128),
            ));
        }
        v
    };
    let cases: Vec<(Stmt, &str, bool)> = vec![
        (
            Stmt::Wgmma {
                n: 8,
                ab: ElementKind::F16,
                acc: ElementKind::F32,
                a: alloc(0),
                b: alloc(1),
                d: 2,
                scale_d: zi(),
                a_layout: OperandLayout::RowMajor,
                b_layout: OperandLayout::RowMajor,
                a_regs: None,
            },
            "Wgmma statement",
            true,
        ),
        (Stmt::WgmmaCommit, "Wgmma statement", true),
        (
            Stmt::Tcgen05Mma {
                n: 8,
                ab: ElementKind::F16,
                acc: ElementKind::F32,
                a: alloc(0),
                b: alloc(1),
                d: 4,
                d_col: zi(),
                scale_d: zi(),
                a_layout: OperandLayout::RowMajor,
                b_layout: OperandLayout::RowMajor,
                m: 128,
                d_lane: 0,
                b_ty: None,
                negate_a: false,
                negate_b: false,
            },
            "tensor-memory allocation",
            true,
        ),
        (
            Stmt::Tcgen05Ld {
                src: 4,
                col: zi(),
                count: 1,
                dst: 2,
            },
            "tensor-memory allocation",
            true,
        ),
        // unpopped 0.58.0 statements, by name; the gap fires before the checker.
        (
            Stmt::Tcgen05Alloc { d: 4, ncols: 32 },
            "Tcgen05 statement",
            true,
        ),
        (
            Stmt::Tcgen05Dealloc { d: 4, ncols: 32 },
            "Tcgen05 statement",
            true,
        ),
        (Stmt::Tcgen05RelinquishPermit, "Tcgen05 statement", true),
        (
            Stmt::CopyUnpack {
                dst: alloc(0),
                src: alloc(1),
                groups: 1,
                bar: alloc(0),
            },
            "mbarrier statement",
            true,
        ),
        (Stmt::GridDepWait, "grid-dependency", false),
        (Stmt::GridDepLaunchDependents, "grid-dependency", false),
        (
            Stmt::MatrixLoad {
                count: 1,
                trans: false,
                src: alloc(0),
                dst: 3,
            },
            "matrix load/store",
            false,
        ),
        (
            Stmt::MatrixStore {
                count: 1,
                trans: false,
                dst: alloc(0),
                src: 3,
            },
            "matrix load/store",
            false,
        ),
        (
            Stmt::Load {
                local: 0,
                place: Place::Raw {
                    buffer: 0,
                    byte_offset: zi(),
                    elem: ElementKind::F32,
                },
            },
            "Place::Raw",
            false,
        ),
        (
            Stmt::Load {
                local: 0,
                place: Place::AllocRaw {
                    alloc: 0,
                    byte_offset: zi(),
                    elem: ElementKind::F32,
                },
            },
            "Place::AllocRaw",
            false,
        ),
    ];
    let _ = z;
    for (st, what, is_gap) in cases {
        let mut k = KernelFn::new(
            "k",
            ElementKind::F32,
            vec![Param::buffer(
                "b",
                ElementKind::F32,
                BufferRole::ReadWrite,
                Some(1),
            )],
            vec![LocalDecl::scalar(ElementKind::F32)],
            Block(vec![]),
        )
        .with_allocs(allocs(what.contains("tensor")))
        .with_launch(launch());
        k.body.0.push(st);
        let d = decline(&k);
        assert!(d.contains(what), "{what}: {d}");
        assert!(
            !d.contains("checker refuses"),
            "{what} reached the unpopped checker, not the emitter: {d}"
        );
        assert_eq!(d.starts_with(GAP_PREFIX), is_gap, "{what}: {d}");
    }
    // Control: the same launch and allocations with a statement that lowers do not decline by any of these names.
    let mut ok = empty().with_allocs(allocs(false)).with_launch(launch());
    ok.body.0.push(Stmt::Return);
    let d = Slang
        .lower_kernel(&ok)
        .map(|_| String::new())
        .unwrap_or_else(|e| format!("{e:?}"));
    assert!(
        !d.contains("Wgmma statement") && !d.contains("Place::"),
        "{d}"
    );
}
