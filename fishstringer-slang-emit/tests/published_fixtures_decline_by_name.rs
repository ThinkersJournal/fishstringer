// SPDX-License-Identifier: MIT OR Apache-2.0
//! The published `unpopped::fixtures` (feature `fixtures`, 0.58.5: one checker-valid kernel or module per `Stmt`
//! variant) drive the by-name decline tests: because the checker passes each fixture, the refusal seen here is the
//! Slang emitter's OWN text, never the checker's. The expectation table below is explicit per variant, so a variant
//! that moves between "lowers", "named gap" and "not lowered yet" fails by name (the mutants are the table rows:
//! flip any row and the test fails).

use fishstringer_slang_emit::{GAP_PREFIX, Slang};
use unpopped::backend::{Backend, LowerError};
use unpopped::fixtures::{Fixture, fixture, fixtures};
use unpopped::ir::ReduceOp;
use unpopped::kernel::{CollectiveKind, KernelFn, Stmt, check_kernel};

/// What the Slang emitter does with a fixture.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    /// Lowers to a kernel.
    Lowers,
    /// A typed decline starting `GAP_PREFIX` (declined by name).
    Gap,
    /// A typed `slang: ... is not lowered yet` decline (a coverage hole, not a recorded gap).
    NotYet,
}

use Outcome::{Gap, Lowers, NotYet};

/// Every `Stmt` variant the fixtures cover, with the emitter's outcome at this commit.
const EXPECTED: &[(&str, Outcome)] = &[
    ("Let", NotYet),
    ("Load", NotYet),
    ("Assign", NotYet),
    ("Store", NotYet),
    ("MaskedLoad", NotYet),
    ("MaskedStore", NotYet),
    ("If", NotYet),
    ("For", NotYet),
    ("While", NotYet),
    ("Break", NotYet),
    ("Continue", NotYet),
    ("Return", NotYet),
    ("Barrier", Lowers),
    ("AtomicRmw", NotYet),
    ("AtomicCas", Lowers),
    ("MulticastLoadReduce", Gap),
    ("GridDepWait", NotYet),
    ("GridDepLaunchDependents", NotYet),
    ("Fence", Lowers),
    ("Assume", Lowers),
    ("Collective", Gap),
    ("CollectiveTuple", Gap),
    ("Mma", NotYet),
    ("MatrixLoad", NotYet),
    ("MatrixStore", NotYet),
    ("MbarInit", Gap),
    ("MbarArrive", Gap),
    ("MbarCompleteTx", Gap),
    ("MbarWait", Gap),
    ("ProxyFence", Gap),
    ("Wgmma", Gap),
    ("WgmmaCommit", Gap),
    ("WgmmaWait", Gap),
    ("Tcgen05Mma", Gap),
    ("Tcgen05Commit", Gap),
    ("Tcgen05Alloc", Gap),
    ("Tcgen05Dealloc", Gap),
    ("Tcgen05RelinquishPermit", Gap),
    ("Tcgen05Ld", Gap),
    ("CopyAsync", Gap),
    ("CopyBulk", Gap),
    ("CopyUnpack", Gap),
    ("AsyncCommit", Gap),
    ("AsyncWait", Gap),
    ("TileLoad", Gap),
    ("TileMma", Gap),
    ("TileFill", Gap),
    ("TileStore", Gap),
    ("Call", NotYet),
    ("CallTuple", NotYet),
    ("ExternCall", NotYet),
];

fn lower(f: &Fixture) -> Result<String, LowerError> {
    match f {
        Fixture::Kernel(k) => Slang.lower_kernel(k).map(|g| g.name),
        Fixture::Module { module, entry } => Slang.lower_module(module, *entry).map(|g| g.name),
        _ => panic!("a Fixture shape this test does not know"),
    }
}

fn classify(name: &str, f: &Fixture) -> Outcome {
    match lower(f) {
        Ok(_) => Lowers,
        Err(LowerError::UnsupportedKernel { detail }) if detail.starts_with(GAP_PREFIX) => Gap,
        Err(LowerError::UnsupportedKernel { detail }) if detail.starts_with("slang: ") => NotYet,
        Err(e) => panic!("{name}: neither a lowering nor a typed slang decline: {e:?}"),
    }
}

#[test]
fn every_fixture_passes_the_checker() {
    for (name, f) in fixtures() {
        f.check()
            .unwrap_or_else(|e| panic!("{name}: the fixture is not checker-valid: {e:?}"));
    }
}

#[test]
fn the_table_covers_exactly_the_published_fixtures() {
    let published: Vec<&str> = fixtures().iter().map(|(n, _)| *n).collect();
    let mut table: Vec<&str> = EXPECTED.iter().map(|(n, _)| *n).collect();
    let mut sorted = published.clone();
    sorted.sort_unstable();
    table.sort_unstable();
    assert_eq!(sorted, table, "a new Stmt variant needs a row in EXPECTED");
}

#[test]
fn each_variant_has_the_recorded_outcome() {
    let mut wrong = Vec::new();
    for (name, want) in EXPECTED {
        let f = fixture(name).unwrap_or_else(|| panic!("no fixture `{name}`"));
        let got = classify(name, &f);
        if got != *want {
            wrong.push(format!("{name}: want {want:?}, got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn the_gap_declines_name_their_construct() {
    // Spot rows: the reason text names the statement, so a gap cannot pass by carrying only the prefix.
    for (name, needle) in [
        ("Collective", "collective statement"),
        ("CollectiveTuple", "collective statement"),
        ("CopyAsync", "async copy statement"),
        ("CopyUnpack", "CopyUnpack"),
        ("MbarInit", "mbarrier statement"),
        ("Wgmma", "Wgmma"),
    ] {
        let Err(LowerError::UnsupportedKernel { detail }) = lower(&fixture(name).unwrap()) else {
            panic!("{name} did not decline");
        };
        assert!(detail.contains(needle), "{name}: {detail}");
    }
}

/// The Collective fixture with its kind swapped; it must stay checker-valid.
fn collective_with(kind: CollectiveKind) -> KernelFn {
    let Some(Fixture::Kernel(mut k)) = fixture("Collective") else {
        panic!("the Collective fixture is a kernel");
    };
    let mut swapped = 0;
    for st in &mut k.body.0 {
        if let Stmt::Collective { kind: slot, .. } = st {
            *slot = kind.clone();
            swapped += 1;
        }
    }
    assert_eq!(swapped, 1, "the fixture holds one Collective statement");
    k
}

#[test]
fn exclusive_and_inclusive_scans_decline_by_name() {
    // unpopped 0.58.4 added exclusive Max/Min/Prod scans; the whole family is the collective gap, never a lowering.
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
        for exclusive in [false, true] {
            let k = collective_with(CollectiveKind::Scan { op, exclusive });
            check_kernel(&k)
                .unwrap_or_else(|e| panic!("{op:?} exclusive={exclusive}: not valid: {e:?}"));
            let Err(LowerError::UnsupportedKernel { detail }) = Slang.lower_kernel(&k) else {
                panic!("{op:?} exclusive={exclusive}: not a typed decline");
            };
            assert!(
                detail.starts_with(GAP_PREFIX),
                "{op:?} exclusive={exclusive}: {detail}"
            );
            assert!(detail.contains("collective statement"), "{detail}");
        }
    }
}
