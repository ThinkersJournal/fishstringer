// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Slang front end for [Unpopped](https://crates.io/crates/unpopped).
//!
//! FishStringer owns Slang in the CireSnave ecosystem (it moved here from Vulkane's `vulkane-slang-parse`), so the language's parser lives here
//! (the emitter is `fishstringer-slang-emit`). Unpopped keeps the language-neutral half:
//! [`Frontend`] and the `lift_*` walkers in `unpopped::convert`, which this crate feeds with
//! a grammar binding, exactly as `baracuda-cuda-parse` does for CUDA.
//!
//! * [`parse_slang`] - Slang source to an error-tolerant tree-sitter CST.
//! * [`SLANG_RESIDUE`] - constructs refused as inexpressible (shared memory, barriers, atomics).
//! * [`SLANG`] - the [`Frontend`]: `numthreads` kernels with `output` / `input{K}` buffers.
//!   The buffer names are not a style choice: `out` and `in` are HLSL keywords.
//!
//! ```
//! use unpopped_convert::lift_elementwise;
//! use unpopped_vocab::ElementKind;
//! use fishstringer_slang_parse::SLANG;
//!
//! let src = "StructuredBuffer<float> input0;\n\
//!     RWStructuredBuffer<float> output;\n\
//!     [numthreads(256, 1, 1)]\n\
//!     void copy(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
//! let lifted = lift_elementwise(&SLANG, src, "copy", &[ElementKind::F32]).unwrap();
//! assert_eq!(lifted.n_inputs, 1);
//! ```
//!
//! The grammar crate (`tree-sitter-slang`) is named in exactly one place, [`parse_slang`].
//! Moved here from `unpopped::convert` (Unpopped `213df4f`, v0.34.11); see
//! `THIRD-PARTY-NOTICES.md`.

use tree_sitter::{Node, Parser, Tree};
use unpopped_convert::{EntryModel, EntryModelError, Frontend, Resource, ResourceKind, Semantic};

/// Parse Slang source into a tree-sitter CST.
///
/// `None` only if the grammar fails to load or the parser is cancelled; syntax errors
/// do not return `None`, they appear as `ERROR` nodes in the tree.
#[must_use]
pub fn parse_slang(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    parser.parse(src, None)
}

/// Slang constructs that aren't IR-expressible - group-shared memory, barriers, atomics.
pub const SLANG_RESIDUE: &[&str] = &[
    "groupshared",
    "GroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "AllMemoryBarrier",
    "InterlockedAdd",
    "InterlockedCompareExchange",
    "RWByteAddressBuffer",
];

/// Slang: `numthreads` compute kernels, `output`/`input{K}` buffers.
///
/// The buffer names differ from CUDA's because `out` and `in` are **HLSL keywords** - a
/// language constraint, not a style choice.
pub const SLANG: Frontend = Frontend::new(
    "slang",
    "numthreads",
    SLANG_RESIDUE,
    "output",
    "input",
    parse_slang,
)
.with_entry_model(read_entry_model);

fn text<'a>(n: Node<'_>, src: &'a str) -> &'a str {
    n.utf8_text(src.as_bytes()).unwrap_or("")
}

/// Every node of `kind` under `n`, in source order.
fn collect<'t>(n: Node<'t>, kind: &str, out: &mut Vec<Node<'t>>) {
    if n.kind() == kind {
        out.push(n);
    }
    let mut c = n.walk();
    for ch in n.children(&mut c) {
        collect(ch, kind, out);
    }
}

fn numbers(args: Node<'_>, src: &str) -> Vec<u32> {
    let mut c = args.walk();
    args.named_children(&mut c)
        .filter(|a| a.kind() == "number_literal")
        .filter_map(|a| text(a, src).trim_end_matches(['u', 'U']).parse().ok())
        .collect()
}

/// `[[vk::binding(slot, set)]]` on a file-scope declaration: the slot, if it gives one.
fn binding_slot(decl: Node<'_>, src: &str) -> Option<u32> {
    let mut c = decl.walk();
    for a in decl.children(&mut c) {
        if a.kind() != "attribute_declaration" {
            continue;
        }
        let mut c2 = a.walk();
        for attr in a.named_children(&mut c2) {
            let name = attr.child_by_field_name("name").map(|n| text(n, src));
            if attr.kind() == "attribute" && name == Some("binding") {
                let mut c3 = attr.walk();
                let args = attr
                    .named_children(&mut c3)
                    .find(|n| n.kind() == "argument_list")?;
                return numbers(args, src).first().copied();
            }
        }
    }
    None
}

/// Reads a Slang entry point off the CST for [`Frontend::with_entry_model`]: the `[numthreads]` block, the `SV_*` semantics on the
/// entry's parameters, and the file-scope `StructuredBuffer` / `RWStructuredBuffer` / `ConstantBuffer` globals the entry body
/// references, in slot order (the walker's convention, which fishstringer's own emission follows).
///
/// # Errors
///
/// [`EntryModelError`] when the entry is missing, a semantic is not an execution id, or a referenced global is a resource type
/// this reader does not know.
pub fn read_entry_model(
    tree: &Tree,
    src: &str,
    entry: &str,
) -> Result<EntryModel, EntryModelError> {
    let root = tree.root_node();
    let mut fns = Vec::new();
    collect(root, "function_definition", &mut fns);
    let func = fns
        .into_iter()
        .find(|f| {
            f.child_by_field_name("declarator")
                .and_then(|d| d.child_by_field_name("declarator"))
                .is_some_and(|n| text(n, src) == entry)
        })
        .ok_or_else(|| EntryModelError::new(format!("no entry point named `{entry}`")))?;

    // [numthreads(x, y, z)]
    let mut block = None;
    let mut c = func.walk();
    for a in func
        .children(&mut c)
        .filter(|a| a.kind() == "hlsl_attribute")
    {
        let mut calls = Vec::new();
        collect(a, "call_expression", &mut calls);
        for call in calls {
            let is_nt = call
                .child_by_field_name("function")
                .is_some_and(|f| text(f, src) == "numthreads");
            if let (true, Some(args)) = (is_nt, call.child_by_field_name("arguments")) {
                if let [x, y, z] = numbers(args, src)[..] {
                    block = Some([x, y, z]);
                } else {
                    return Err(EntryModelError::new(
                        "numthreads without three constant extents",
                    ));
                }
            }
        }
    }

    // SV_* semantics on the parameters; a parameter with none is left for the walker to name.
    let mut semantics = Vec::new();
    let mut params = Vec::new();
    collect(func, "parameter_declaration", &mut params);
    for p in params {
        let Some(name) = p.child_by_field_name("declarator") else {
            continue;
        };
        let mut sems = Vec::new();
        collect(p, "semantics", &mut sems);
        let Some(sem) = sems.first() else { continue };
        let sem_text = text(*sem, src).trim_start_matches(':').trim();
        let s = match sem_text {
            "SV_DispatchThreadID" => Semantic::DispatchThreadId,
            "SV_GroupID" => Semantic::GroupId,
            "SV_GroupThreadID" => Semantic::GroupThreadId,
            "SV_GroupIndex" => Semantic::GroupIndex,
            other => return Err(EntryModelError::new(format!("the semantic `{other}`"))),
        };
        semantics.push((text(name, src).to_string(), s));
    }

    // Identifiers the entry body mentions.
    let mut idents = Vec::new();
    if let Some(body) = func.child_by_field_name("body") {
        collect(body, "identifier", &mut idents);
    }
    let used: Vec<&str> = idents.iter().map(|n| text(*n, src)).collect();

    // File-scope resources: `[[vk::binding(N, 0)]] KIND<T> name;`, referenced ones only.
    let mut resources = Vec::new();
    let mut c = root.walk();
    for (order, d) in root
        .children(&mut c)
        .filter(|d| d.kind() == "declaration")
        .enumerate()
    {
        let (Some(ty), Some(decl)) = (
            d.child_by_field_name("type"),
            d.child_by_field_name("declarator"),
        ) else {
            continue;
        };
        if ty.kind() != "template_type" {
            continue;
        }
        let name = text(decl, src);
        if !used.contains(&name) {
            continue;
        }
        let kind_name = ty.child_by_field_name("name").map_or("", |n| text(n, src));
        let kind = match kind_name {
            "StructuredBuffer" => ResourceKind::StructuredRO,
            "RWStructuredBuffer" => ResourceKind::StructuredRW,
            "ConstantBuffer" => ResourceKind::ConstantBuffer,
            other => return Err(EntryModelError::new(format!("the resource type `{other}`"))),
        };
        let mut targs = Vec::new();
        collect(ty, "type_descriptor", &mut targs);
        let elem = targs
            .first()
            .and_then(|t| t.child_by_field_name("type"))
            .map(|t| text(t, src).to_string())
            .ok_or_else(|| EntryModelError::new("a resource without an element type"))?;
        let slot = binding_slot(d, src).unwrap_or(u32::try_from(order).unwrap_or(u32::MAX));
        resources.push(Resource::new(name, kind, elem, slot));
    }
    resources.sort_by_key(|r| (r.kind == ResourceKind::ConstantBuffer, r.slot));
    Ok(EntryModel::new(block, semantics, resources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::LiftError;
    use unpopped::ir::{Access, ReduceOp, ScalarExpr};
    use unpopped_convert::{lift, lift_elementwise, lift_reduction, lift_scan};
    use unpopped_vocab::ElementKind;

    const F32: &[ElementKind] = &[ElementKind::F32];

    const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
        StructuredBuffer<float> input1;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void mul(uint3 tid : SV_DispatchThreadID) {\n\
            uint i = tid.x;\n\
            output[i] = input0[i] * input1[i];\n\
        }";
    const SLANG_COPY: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
    const SLANG_SUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += input0[i]; output[0] = acc; }";
    const SLANG_CSUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += input0[i]; output[i] = acc; } }";

    #[test]
    fn grammar_loads_and_parses() {
        let tree = parse_slang("float4 f(float x) { return x; }").expect("grammar loads");
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn syntax_errors_are_error_nodes_not_none() {
        let tree = parse_slang("void f( {").expect("a parse tree even for broken source");
        assert!(
            tree.root_node().has_error(),
            "broken source must show an ERROR node"
        );
    }

    #[test]
    fn slang_lifts_multiply() {
        let lifted = lift_elementwise(&SLANG, SLANG_MUL, "slang_mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            )
        );
    }

    #[test]
    fn every_residue_marker_is_refused_as_inexpressible() {
        for marker in SLANG_RESIDUE {
            let src = format!(
                "[numthreads(256,1,1)]\nvoid k(uint3 tid : SV_DispatchThreadID) {{ {marker}; output[tid.x] = input0[tid.x]; }}"
            );
            assert!(
                matches!(
                    lift_elementwise(&SLANG, &src, "x", F32),
                    Err(LiftError::Inexpressible(ref w)) if w == marker
                ),
                "{marker}: {:?}",
                lift_elementwise(&SLANG, &src, "x", F32)
            );
        }
    }

    #[test]
    fn slang_lifts_sum_reduction() {
        let lifted = lift_reduction(&SLANG, SLANG_SUM, "rsum", F32).unwrap();
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert!(matches!(
            lifted.op.access,
            Access::Reduction {
                op: ReduceOp::Sum,
                ..
            }
        ));
    }

    #[test]
    fn slang_lifts_cumsum() {
        let lifted = lift_scan(&SLANG, SLANG_CSUM, "csum", F32).unwrap();
        assert!(matches!(
            lifted.op.access,
            Access::Scan {
                op: ReduceOp::Sum,
                ..
            }
        ));
    }

    #[test]
    fn the_three_lifters_do_not_cross() {
        assert!(lift_elementwise(&SLANG, SLANG_SUM, "x", F32).is_err());
        assert!(lift_reduction(&SLANG, SLANG_COPY, "x", F32).is_err());
        assert!(lift_scan(&SLANG, SLANG_CSUM, "x", F32).is_ok());
        assert!(lift_reduction(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(lift_elementwise(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(lift(&SLANG, SLANG_SUM, "x", F32).is_ok());
        assert!(lift(&SLANG, SLANG_COPY, "x", F32).is_ok());
    }

    #[test]
    fn source_without_numthreads_is_not_a_kernel() {
        assert!(matches!(
            lift_elementwise(&SLANG, "float f(float x) { return x; }", "x", F32),
            Err(LiftError::NotAKernel)
        ));
    }
}
