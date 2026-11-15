// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for Slang (ROADMAP row 17, doc rows E1, E2, E4, E5 and E7 in part).
//!
//! **Covered:** a *launched* kernel whose parameters are buffers of `F32`/`I32`/`U32` (any known rank) and scalars of
//! those types, with the statements `Let`, `Load`, `Assign`, `Store` and the expressions `Local`, `Const`, `ConstInt`,
//! `Exec` (`ThreadIdx`, `BlockIdx`, `BlockDim`, `GridDim`), `Add`, `Sub`, `Mul`.
//!
//! **Control flow and masks (E4).** `If`, `While`, `For`, `Break`, `Continue`, `Return`, `MaskedLoad`, `MaskedStore`, with
//! the comparisons `CmpEq/Ne/Lt/Le/Gt/Ge` as conditions. A condition or mask that is not a comparison is tested `!= 0`. A
//! masked access sits inside the branch, so memory is untouched when the mask is false. A `For` is emitted only when its
//! step is a nonzero constant (the oracle traps on a zero step and a GPU cannot): otherwise it declines
//! `For: step not a nonzero constant`. Its bounds are evaluated once. No loop hint is emitted (the IR has none). Locals are
//! declared once at the top of the entry, because the IR scopes a local by its declaring statement.
//!
//! **Views (E2).** `Place::Logical` means "through the view": the element offset is
//! `base + sum(coord_d * stride_d)`, computed in `int64_t`. Each buffer's strides and base, and every scalar parameter,
//! live in one `ConstantBuffer` bound after the buffers, and the interface's `ParamsBlock` gives the byte offset of each
//! field. Field order is parameter order: a scalar is `p<i>`; a buffer is `b<i>_stride0 .. b<i>_stride<rank-1>`, then
//! `b<i>_base`. There is no shape field: **no bounds check is emitted** (the oracle's out-of-range access is an error and a
//! GPU cannot trap, so the launcher must pass views that keep every access in range). The block holds `int64_t` fields,
//! so the device needs `shaderInt64` (the capability vocabulary for that is row E19; the interface carries it as a
//! fidelity note until then).
//!
//! **Allocations (E5).** A static `Shared` allocation is a module-scope `groupshared T a<k>[N]`, a static `Private` one a
//! function-local array; `Place::Alloc` indexes it by its flat element index. `Constant`/`Global` allocations and
//! `Extent::Param` decline. Without barriers (E7) a shared array can only be used per-thread.
//!
//! **Barriers and fences (E7).** `Barrier{Block}` is `GroupMemoryBarrierWithGroupSync()`, `Barrier{Subgroup}`
//! `AllMemoryBarrierWithWaveSync()`; a `Fence` is `GroupMemoryBarrier()` (block), `AllMemoryBarrier()` (subgroup) or
//! `DeviceMemoryBarrier()` (device). Slang's barriers take no memory order, so every `MemoryOrder` maps to the same full
//! barrier: stronger and so correct, the order distinction is lost. Device and cluster barriers are target gaps (`gaps.rs`).
//!
//! **Everything else is refused** with `UnsupportedKernel` and a reason saying what is not lowered yet; nothing is
//! approximated. The gap scan in `gaps.rs` runs first, so a construct outside the target gets its own reason.

use std::collections::BTreeSet;
use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, ParamKind, Place, Stmt, check_kernel};
use unpopped::kernel_interface::{Binding, KernelInterface, ParamsBlock, ParamsBlockField};
use unpopped_vocab::ElementKind;

fn no(kernel: &KernelFn, what: &str) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!(
            "slang: {what} is not lowered yet (kernel `{}`)",
            kernel.name
        ),
    }
}

fn ctype(kernel: &KernelFn, dt: ElementKind) -> Result<&'static str, LowerError> {
    match dt {
        ElementKind::F32 => Ok("float"),
        ElementKind::I32 => Ok("int"),
        ElementKind::U32 => Ok("uint"),
        _ => Err(no(kernel, &format!("element type {dt:?}"))),
    }
}

fn axis(a: u8) -> Option<&'static str> {
    ["x", "y", "z"].get(usize::from(a)).copied()
}

fn spell(kernel: &KernelFn, e: &ScalarExpr) -> Result<String, LowerError> {
    let launch = kernel.launch.as_ref().expect("checked by the caller");
    let bin = |a: &ScalarExpr, op: &str, b: &ScalarExpr| -> Result<String, LowerError> {
        Ok(format!(
            "({} {op} {})",
            spell(kernel, a)?,
            spell(kernel, b)?
        ))
    };
    match e {
        ScalarExpr::Local(i) => Ok(format!("l{i}")),
        ScalarExpr::Const(v) if kernel.compute == ElementKind::F32 => Ok(crate::slang_const(*v)),
        ScalarExpr::ConstInt(v, ek) => Ok(format!("(({}){v})", ctype(kernel, *ek)?)),
        ScalarExpr::Exec(ExecId::ThreadIdx(a)) => axis(*a)
            .map(|c| format!("tid_.{c}"))
            .ok_or_else(|| no(kernel, "ThreadIdx axis")),
        ScalarExpr::Exec(ExecId::BlockIdx(a)) => axis(*a)
            .map(|c| format!("bid_.{c}"))
            .ok_or_else(|| no(kernel, "BlockIdx axis")),
        ScalarExpr::Exec(ExecId::BlockDim(a)) => launch
            .block
            .get(usize::from(*a))
            .map(|n| format!("{n}u"))
            .ok_or_else(|| no(kernel, "BlockDim axis")),
        ScalarExpr::Exec(ExecId::GridDim(a)) => launch
            .grid
            .get(usize::from(*a))
            .map(|n| format!("{n}u"))
            .ok_or_else(|| no(kernel, "GridDim axis")),
        ScalarExpr::Add(a, b) => bin(a, "+", b),
        ScalarExpr::Sub(a, b) => bin(a, "-", b),
        ScalarExpr::Mul(a, b) => bin(a, "*", b),
        ScalarExpr::Binary(op, a, b) => match cmp_op(*op) {
            Some(c) => bin(a, c, b),
            None => Err(no(kernel, "this expression")),
        },
        _ => Err(no(kernel, "this expression")),
    }
}

fn cmp_op(op: BinaryOp) -> Option<&'static str> {
    Some(match op {
        BinaryOp::CmpEq => "==",
        BinaryOp::CmpNe => "!=",
        BinaryOp::CmpLt => "<",
        BinaryOp::CmpLe => "<=",
        BinaryOp::CmpGt => ">",
        BinaryOp::CmpGe => ">=",
        _ => return None,
    })
}

/// What an expression is, as far as the conditions need to know: a comparison is a `bool`, everything else a number.
fn is_bool(e: &ScalarExpr) -> bool {
    matches!(e, ScalarExpr::Binary(op, ..) if cmp_op(*op).is_some())
}

/// `e != 0` spelled so that a comparison (already a `bool`) is not compared again.
fn truth(kernel: &KernelFn, e: &ScalarExpr) -> Result<String, LowerError> {
    let s = spell(kernel, e)?;
    Ok(if is_bool(e) { s } else { format!("({s} != 0)") })
}

#[derive(Default)]
struct Cx {
    /// Locals a `Let`, `Load`, `MaskedLoad` or `For` declares, in first-seen order.
    declared: BTreeSet<u32>,
    /// Counter for the temporaries a loop's bound needs.
    tmp: u32,
}

fn pad(ind: usize) -> String {
    "    ".repeat(ind)
}

fn stmts(
    kernel: &KernelFn,
    b: &Block,
    ind: usize,
    cx: &mut Cx,
    out: &mut String,
) -> Result<(), LowerError> {
    let p = pad(ind);
    for st in &b.0 {
        match st {
            Stmt::Let { local, init } => {
                cx.declared.insert(*local);
                out.push_str(&format!("{p}l{local} = {};\n", spell(kernel, init)?));
            }
            Stmt::Load { local, place: pl } => {
                cx.declared.insert(*local);
                out.push_str(&format!("{p}l{local} = {};\n", place(kernel, pl)?));
            }
            Stmt::Assign { local, value } => {
                out.push_str(&format!("{p}l{local} = {};\n", spell(kernel, value)?));
            }
            Stmt::Store { place: pl, value } => {
                out.push_str(&format!(
                    "{p}{} = {};\n",
                    place(kernel, pl)?,
                    spell(kernel, value)?
                ));
            }
            // Memory is untouched when the mask is false, so the access sits inside the branch.
            Stmt::MaskedLoad {
                local,
                place: pl,
                mask,
                fill,
            } => {
                cx.declared.insert(*local);
                out.push_str(&format!(
                    "{p}if ({}) {{ l{local} = {}; }} else {{ l{local} = {}; }}\n",
                    truth(kernel, mask)?,
                    place(kernel, pl)?,
                    spell(kernel, fill)?
                ));
            }
            Stmt::MaskedStore {
                place: pl,
                value,
                mask,
            } => {
                out.push_str(&format!(
                    "{p}if ({}) {{ {} = {}; }}\n",
                    truth(kernel, mask)?,
                    place(kernel, pl)?,
                    spell(kernel, value)?
                ));
            }
            Stmt::If { cond, then, els } => {
                out.push_str(&format!("{p}if ({}) {{\n", truth(kernel, cond)?));
                stmts(kernel, then, ind + 1, cx, out)?;
                if els.0.is_empty() {
                    out.push_str(&format!("{p}}}\n"));
                } else {
                    out.push_str(&format!("{p}}} else {{\n"));
                    stmts(kernel, els, ind + 1, cx, out)?;
                    out.push_str(&format!("{p}}}\n"));
                }
            }
            Stmt::While { cond, body } => {
                out.push_str(&format!("{p}while ({}) {{\n", truth(kernel, cond)?));
                stmts(kernel, body, ind + 1, cx, out)?;
                out.push_str(&format!("{p}}}\n"));
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
            } => {
                // The oracle traps on a zero step and a GPU cannot, so only a constant, nonzero step is emitted.
                let ScalarExpr::ConstInt(k, _) = step else {
                    return Err(no(kernel, "For: step not a nonzero constant"));
                };
                if *k == 0 {
                    return Err(no(kernel, "For: step not a nonzero constant"));
                }
                let ty = local_ty(kernel, *local)?;
                cx.declared.insert(*local);
                let t = cx.tmp;
                cx.tmp += 1;
                let cmp = if *k > 0 { "<" } else { ">" };
                // The bounds are evaluated once, before the first pass.
                out.push_str(&format!(
                    "{p}{{ {ty} e{t}_ = {}; for (l{local} = {}; l{local} {cmp} e{t}_; l{local} += (({ty}){k})) {{\n",
                    spell(kernel, end)?,
                    spell(kernel, start)?
                ));
                stmts(kernel, body, ind + 1, cx, out)?;
                out.push_str(&format!("{p}}} }}\n"));
            }
            // Slang's barriers take no memory order, so every `MemoryOrder` of a fence maps to the same full barrier of its
            // scope: stronger, hence correct, and the order distinction is lost (recorded in the module docs).
            Stmt::Barrier { scope, .. } => match scope {
                Scope::Block => out.push_str(&format!(
                    "{p}GroupMemoryBarrierWithGroupSync();
"
                )),
                Scope::Subgroup => out.push_str(&format!(
                    "{p}AllMemoryBarrierWithWaveSync();
"
                )),
                _ => return Err(no(kernel, "a barrier of this scope")),
            },
            Stmt::Fence { scope, .. } => match scope {
                Scope::Block => out.push_str(&format!(
                    "{p}GroupMemoryBarrier();
"
                )),
                Scope::Subgroup => out.push_str(&format!(
                    "{p}AllMemoryBarrier();
"
                )),
                Scope::Device => out.push_str(&format!(
                    "{p}DeviceMemoryBarrier();
"
                )),
                _ => return Err(no(kernel, "a fence of this scope")),
            },
            Stmt::Break => out.push_str(&format!("{p}break;\n")),
            Stmt::Continue => out.push_str(&format!("{p}continue;\n")),
            Stmt::Return => out.push_str(&format!("{p}return;\n")),
            _ => return Err(no(kernel, "this statement")),
        }
    }
    Ok(())
}

fn place(kernel: &KernelFn, p: &Place) -> Result<String, LowerError> {
    if let Place::Alloc { alloc, index } = p {
        return Ok(format!("a{alloc}[{}]", spell(kernel, index)?));
    }
    let Place::Logical { buffer, coords } = p else {
        return Err(no(kernel, "this place"));
    };
    let mut off = format!("pb_.b{buffer}_base");
    for (d, c) in coords.iter().enumerate() {
        off.push_str(&format!(
            " + (int64_t)({}) * pb_.b{buffer}_stride{d}",
            spell(kernel, c)?
        ));
    }
    Ok(format!("b{buffer}[{off}]"))
}

/// Lower `kernel`, or say why not. Callers run the gap scan first.
pub(crate) fn lower(kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    check_kernel(kernel).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!(
            "slang: the kernel-layer checker refuses kernel `{}`: {e:?}",
            kernel.name
        ),
    })?;
    let Some(launch) = kernel.launch.as_ref() else {
        return Err(no(kernel, "a kernel with no launch"));
    };
    ctype(kernel, kernel.compute)?;

    let entry: String = kernel
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mut s = String::new();
    s.push_str("// Generated by fishstringer-slang-emit (lower_kernel) — do not edit.\n");
    let mut iface = KernelInterface::of(kernel).with_entry_symbol(entry.clone());
    // Pass 1: bindings (buffers get dense slots in order) and the params block fields.
    let mut decls = String::new();
    let mut members = String::new();
    let mut prologue = String::new();
    let mut block = Vec::<(String, ElementKind)>::new();
    let mut slot = 0u32;
    for (i, p) in kernel.params.iter().enumerate() {
        match &p.kind {
            ParamKind::Buffer { elem, role, rank } => {
                let Some(rank) = rank else {
                    return Err(no(kernel, "a buffer whose rank is not declared"));
                };
                let ty = ctype(kernel, *elem)?;
                let qual = match role {
                    BufferRole::Read => "StructuredBuffer",
                    _ => "RWStructuredBuffer",
                };
                decls.push_str(&format!(
                    "[[vk::binding({slot}, 0)]] {qual}<{ty}> b{i}; // {}\n",
                    p.name
                ));
                iface = iface.with_binding(Binding::new(i, 0, slot));
                slot += 1;
                for d in 0..*rank {
                    members.push_str(&format!("    int64_t b{i}_stride{d};\n"));
                    block.push((format!("b{i}_stride{d}"), ElementKind::I64));
                }
                members.push_str(&format!("    int64_t b{i}_base;\n"));
                block.push((format!("b{i}_base"), ElementKind::I64));
            }
            ParamKind::Scalar { ty, local } => {
                let t = ctype(kernel, *ty)?;
                members.push_str(&format!("    {t} p{i};\n"));
                block.push((format!("p{i}"), *ty));
                prologue.push_str(&format!("    {t} l{local} = pb_.p{i};\n"));
            }
            _ => return Err(no(kernel, "this kind of parameter")),
        }
    }
    if !block.is_empty() {
        s.push_str(&decls);
        s.push_str(&format!(
            "struct Params_\n{{\n{members}}};\n[[vk::binding({slot}, 0)]] ConstantBuffer<Params_> pb_;\n"
        ));
        let mut pb = ParamsBlock::new(0, 1);
        let (mut off, mut align) = (0u32, 1u32);
        for (name, ty) in &block {
            let sz = if *ty == ElementKind::I64 { 8 } else { 4 };
            off = off.next_multiple_of(sz);
            pb = pb.with_field(ParamsBlockField::new(name.clone(), off, *ty));
            off += sz;
            align = align.max(sz);
        }
        pb.size = off.next_multiple_of(align);
        pb.align = align;
        iface = iface.with_params_block(pb).with_fidelity_note(
            "params block holds int64_t strides/bases: needs shaderInt64; no bounds check is emitted",
        );
    } else {
        s.push_str(&decls);
    }
    // Static allocations (E5). `Shared` is a module-scope `groupshared` (an entry-local one is not allowed), `Private` a
    // function-local array. `Constant` has no initializer in the IR, so there is nothing to put in it: declined.
    let mut private = String::new();
    for (k, a) in kernel.allocs.iter().enumerate() {
        let Extent::Static(n) = a.extent else {
            return Err(no(
                kernel,
                "an allocation whose extent is a launch parameter",
            ));
        };
        let ty = ctype(kernel, a.elem)?;
        match a.space {
            MemorySpace::Shared => {
                s.push_str(&format!(
                    "groupshared {ty} a{k}[{n}]; // {}
",
                    a.name
                ));
            }
            MemorySpace::Private => {
                private.push_str(&format!(
                    "    {ty} a{k}[{n}]; // {}
",
                    a.name
                ));
            }
            _ => return Err(no(kernel, "a Constant or Global allocation")),
        }
    }
    let [bx, by, bz] = launch.block;
    s.push_str(&format!(
        "[shader(\"compute\")]\n[numthreads({bx}, {by}, {bz})]\nvoid {entry}(uint3 tid_ : SV_GroupThreadID, uint3 bid_ : SV_GroupID)\n{{\n"
    ));
    s.push_str(&prologue);
    s.push_str(&private);
    // Statements first (collecting every local they declare), then the declarations, hoisted to the top of the entry:
    // the IR scopes a local by its declaring statement, the source declares it once where every use can see it.
    let mut cx = Cx::default();
    let mut body = String::new();
    stmts(kernel, &kernel.body, 1, &mut cx, &mut body)?;
    for l in &cx.declared {
        let ty = local_ty(kernel, *l)?;
        s.push_str(&format!("    {ty} l{l} = ({ty})0;\n"));
    }
    s.push_str(&body);
    s.push_str("}\n");
    for a in crate::required_slangc_args(kernel.compute) {
        iface = iface.with_compiler_arg(*a);
    }
    Ok(GeneratedKernel::new(entry, s).with_interface(iface))
}

fn local_ty(kernel: &KernelFn, local: u32) -> Result<&'static str, LowerError> {
    let d = kernel
        .locals
        .get(local as usize)
        .ok_or_else(|| no(kernel, "an undeclared local"))?;
    if d.lanes != 1 {
        return Err(no(kernel, "a packed (multi-lane) local"));
    }
    ctype(kernel, d.elem)
}
