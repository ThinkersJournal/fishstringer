// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for Slang, first slice (ROADMAP row 17, doc rows E1/E2 in part).
//!
//! **Covered:** a *launched* kernel whose parameters are buffers of `F32`/`I32`/`U32` (any known rank) and scalars of
//! those types, with the statements `Let`, `Load`, `Assign`, `Store` and the expressions `Local`, `Const`, `ConstInt`,
//! `Exec` (`ThreadIdx`, `BlockIdx`, `BlockDim`, `GridDim`), `Add`, `Sub`, `Mul`.
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
//! **Everything else is refused** with `UnsupportedKernel` and a reason saying what is not lowered yet; nothing is
//! approximated. The gap scan in `gaps.rs` runs first, so a construct outside the target gets its own reason.

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::ExecId;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BufferRole, KernelFn, ParamKind, Place, Stmt, check_kernel};
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
        _ => Err(no(kernel, "this expression")),
    }
}

fn place(kernel: &KernelFn, p: &Place) -> Result<String, LowerError> {
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
    if !kernel.allocs.is_empty() {
        return Err(no(kernel, "an allocation (shared or private array)"));
    }
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
    let [bx, by, bz] = launch.block;
    s.push_str(&format!(
        "[shader(\"compute\")]\n[numthreads({bx}, {by}, {bz})]\nvoid {entry}(uint3 tid_ : SV_GroupThreadID, uint3 bid_ : SV_GroupID)\n{{\n"
    ));
    s.push_str(&prologue);
    for st in &kernel.body.0 {
        match st {
            Stmt::Let { local, init } => {
                let ty = local_ty(kernel, *local)?;
                s.push_str(&format!("    {ty} l{local} = {};\n", spell(kernel, init)?));
            }
            Stmt::Load { local, place: p } => {
                let ty = local_ty(kernel, *local)?;
                s.push_str(&format!("    {ty} l{local} = {};\n", place(kernel, p)?));
            }
            Stmt::Assign { local, value } => {
                s.push_str(&format!("    l{local} = {};\n", spell(kernel, value)?));
            }
            Stmt::Store { place: p, value } => {
                s.push_str(&format!(
                    "    {} = {};\n",
                    place(kernel, p)?,
                    spell(kernel, value)?
                ));
            }
            _ => return Err(no(kernel, "this statement")),
        }
    }
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
