// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel` for Slang (ROADMAP row 17, doc rows E1, E2, E4, E5, E7 and E12 in part).
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
//! **Execution ids (E12).** `LaneId` is `WaveGetLaneIndex()`, `SubgroupSize` `WaveGetLaneCount()`, `SubgroupId` the derived
//! `SV_GroupIndex / WaveGetLaneCount()`, `DeviceRank`/`DeviceCount` the constants `0`/`1` (a multi-device launch is a gap). The subgroup ids read the device's own subgroup size; the sizes the kernel is valid for are
//! `LaunchShape::subgroup`, which the core-derived half of the interface already carries for the launcher to enforce.
//! Vulkan's workgroup-size and dispatch limits are device properties this emitter does not see: a launch over them is
//! the launcher's error until the capability vocabulary of row E19 gives the emitter the limits to decline on.
//!
//! **Atomics (E10).** `AtomicRmw` is `InterlockedAdd/Max/Min/And/Or/Xor/Exchange(dest, value[, old])` and `AtomicCas`
//! `InterlockedCompareExchange(dest, expected, desired[, old])`, on an `I32`/`U32` buffer or shared element. Slang takes
//! neither scope nor memory order (Vulkan fixes the scope by the storage), so both are dropped: the result is at least as
//! strong as asked. Float and 64-bit atomics decline by name until the capability vocabulary (row E19) can say the device
//! has them; a cluster scope or a peer place is a target gap (`gaps.rs`).
//!
//! **16-bit floats (E11).** `F16` is `half`. `Bf16` has no Slang scalar: a buffer element is `uint16_t` (widened by
//! `bf16_to_f32_`, narrowed by `f32_to_bf16_`, round-to-nearest-even), a local or allocation is a `float` holding a
//! bf16-representable value, and each `Add`/`Sub`/`Mul` is rounded back (`bf16r_`). Needs `shaderFloat16` and 16-bit
//! storage (a fidelity note until row E19). 16-bit scalar parameters, packed locals and fp8 decline. Not yet compared
//! against the oracle exhaustively: that needs a device.
//!
//! **Paged gather (E13).** `I64` is `int64_t` (needs `shaderInt64`) and `Cast` lowers for int to int (`Wrap`) and to
//! `F32`/`F16` (`NearestEven`, `Wrap`); float to int, other modes and `Bf16` decline. A buffer index is 32-bit in Slang, so
//! a cache past 2^32 elements is the launcher's to keep out. Integer `Div`/`Rem` are not in the IR: loop over pages.
//!
//! **Required capabilities (E19).** The interface's `requires` lists the `vulkan:` `<arith>` tokens the source needs (`i64`,
//! `f16`, `st16`, `i16`); the consumer gates on them against its target. The emitter sees no target.
//!
//! **Everything else is refused** with `UnsupportedKernel` and a reason saying what is not lowered yet; nothing is
//! approximated. The gap scan in `gaps.rs` runs first, so a construct outside the target gets its own reason.

use std::collections::BTreeSet;
use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};

use unpopped::ir::{BinaryOp, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, CallArg, KernelFn, Module, ParamKind, Place, Stmt, check_kernel,
    check_module,
};
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
        // `half` is native. `Bf16` has no Slang scalar: a value is a `float` that always holds a bf16-representable number,
        // rounded after every operation, and is only 16 bits wide in a buffer (see `storage_ty`).
        ElementKind::F16 => Ok("half"),
        ElementKind::Bf16 => Ok("float"),
        ElementKind::I32 => Ok("int"),
        ElementKind::U32 => Ok("uint"),
        // 64-bit indices and offsets (E13); needs shaderInt64, as the params block already does.
        ElementKind::I64 => Ok("int64_t"),
        _ => Err(no(kernel, &format!("element type {dt:?}"))),
    }
}

/// What a buffer element is stored as: `Bf16` is its 16 raw bits, everything else is its compute type.
fn storage_ty(kernel: &KernelFn, dt: ElementKind) -> Result<&'static str, LowerError> {
    if dt == ElementKind::Bf16 {
        return Ok("uint16_t");
    }
    ctype(kernel, dt)
}

/// The element type of a place.
fn place_elem(kernel: &KernelFn, p: &Place) -> Option<ElementKind> {
    match p {
        Place::Alloc { alloc, .. } => kernel.allocs.get(*alloc as usize).map(|a| a.elem),
        Place::Logical { buffer, .. } => match kernel.params.get(*buffer as usize).map(|q| &q.kind)
        {
            Some(ParamKind::Buffer { elem, .. }) => Some(*elem),
            _ => None,
        },
        _ => None,
    }
}

/// A `Bf16` buffer element (not an allocation: those hold the compute `float` directly) needs a conversion on every access.
fn is_bf16_buffer(kernel: &KernelFn, p: &Place) -> bool {
    matches!(p, Place::Logical { .. }) && place_elem(kernel, p) == Some(ElementKind::Bf16)
}

/// The element type of an expression, where the lowering needs it (a `Cast` is checked against its source).
fn expr_ty(kernel: &KernelFn, e: &ScalarExpr) -> Option<ElementKind> {
    match e {
        ScalarExpr::Local(i) => kernel.locals.get(*i as usize).map(|d| d.elem),
        ScalarExpr::Const(_) => Some(kernel.compute),
        ScalarExpr::ConstInt(_, ek) => Some(*ek),
        ScalarExpr::Exec(_) => Some(ElementKind::U32),
        ScalarExpr::Cast(_, spec) => Some(spec.to),
        ScalarExpr::Add(a, _) | ScalarExpr::Sub(a, _) | ScalarExpr::Mul(a, _) => expr_ty(kernel, a),
        _ => None,
    }
}

fn is_int_ty(t: ElementKind) -> bool {
    matches!(t, ElementKind::I32 | ElementKind::U32 | ElementKind::I64)
}

fn is_float_ty(t: ElementKind) -> bool {
    matches!(t, ElementKind::F32 | ElementKind::F16)
}

/// Whether an expression is a `Bf16` value (so an arithmetic result must be rounded back to bf16).
fn is_bf16(kernel: &KernelFn, e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Local(i) => kernel
            .locals
            .get(*i as usize)
            .is_some_and(|d| d.elem == ElementKind::Bf16),
        ScalarExpr::Const(_) => kernel.compute == ElementKind::Bf16,
        ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
            is_bf16(kernel, a) || is_bf16(kernel, b)
        }
        _ => false,
    }
}

/// Round `v` to the nearest bf16 (ties to even), as the `f32_to_bf16_` helper does at run time.
fn round_bf16(v: f64) -> f64 {
    let f = v as f32;
    if f.is_nan() || f.is_infinite() {
        return f64::from(f);
    }
    let b = f.to_bits();
    let r = (b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16;
    f64::from(f32::from_bits(r << 16))
}

const BF16_HELPERS: &str = "float bf16_to_f32_(uint16_t v) { return asfloat(((uint)v) << 16); }
uint16_t f32_to_bf16_(float f)
{
    uint b = asuint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return (uint16_t)((b >> 16) | 0x40u); } // NaN stays a (quiet) NaN
    return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
float bf16r_(float f) { return bf16_to_f32_(f32_to_bf16_(f)); }
";

fn axis(a: u8) -> Option<&'static str> {
    ["x", "y", "z"].get(usize::from(a)).copied()
}

fn spell(kernel: &KernelFn, e: &ScalarExpr) -> Result<String, LowerError> {
    // A callee has no launch: BlockDim/GridDim in it are declined by name; the entry always has one.
    let launch = kernel.launch.as_ref();
    let bin = |a: &ScalarExpr, op: &str, b: &ScalarExpr| -> Result<String, LowerError> {
        Ok(format!(
            "({} {op} {})",
            spell(kernel, a)?,
            spell(kernel, b)?
        ))
    };
    // A bf16 result is rounded after each operation, as the oracle does.
    let arith = |a: &ScalarExpr, op: &str, b: &ScalarExpr| -> Result<String, LowerError> {
        let t = bin(a, op, b)?;
        Ok(if is_bf16(kernel, a) || is_bf16(kernel, b) {
            format!("bf16r_{t}")
        } else {
            t
        })
    };
    match e {
        ScalarExpr::Local(i) => Ok(format!("l{i}")),
        ScalarExpr::Const(v) if kernel.compute == ElementKind::F32 => Ok(crate::slang_const(*v)),
        ScalarExpr::Const(v) if kernel.compute == ElementKind::F16 => {
            Ok(format!("((half){})", crate::slang_const(*v)))
        }
        ScalarExpr::Const(v) if kernel.compute == ElementKind::Bf16 => {
            Ok(crate::slang_const(round_bf16(*v)))
        }
        ScalarExpr::ConstInt(v, ek) => Ok(format!("(({}){v})", ctype(kernel, *ek)?)),
        ScalarExpr::Exec(ExecId::ThreadIdx(a)) => axis(*a)
            .map(|c| format!("tid_.{c}"))
            .ok_or_else(|| no(kernel, "ThreadIdx axis")),
        ScalarExpr::Exec(ExecId::BlockIdx(a)) => axis(*a)
            .map(|c| format!("bid_.{c}"))
            .ok_or_else(|| no(kernel, "BlockIdx axis")),
        ScalarExpr::Exec(ExecId::BlockDim(a)) => launch
            .ok_or_else(|| no(kernel, "a launch id in a callee"))?
            .block
            .get(usize::from(*a))
            .map(|n| format!("{n}u"))
            .ok_or_else(|| no(kernel, "BlockDim axis")),
        ScalarExpr::Exec(ExecId::GridDim(a)) => launch
            .ok_or_else(|| no(kernel, "a launch id in a callee"))?
            .grid
            .get(usize::from(*a))
            .map(|n| format!("{n}u"))
            .ok_or_else(|| no(kernel, "GridDim axis")),
        // Subgroup ids are WaveGet* intrinsics; the subgroup id is derived (linear thread index / subgroup size). One device
        // only: a multi-device launch is a target gap before this runs.
        ScalarExpr::Exec(ExecId::LaneId) => Ok("WaveGetLaneIndex()".into()),
        ScalarExpr::Exec(ExecId::SubgroupSize) => Ok("WaveGetLaneCount()".into()),
        ScalarExpr::Exec(ExecId::SubgroupId) => Ok("(gi_ / WaveGetLaneCount())".into()),
        ScalarExpr::Exec(ExecId::DeviceRank) => Ok("0u".into()),
        ScalarExpr::Exec(ExecId::DeviceCount) => Ok("1u".into()),
        // Conversions the C cast does exactly: integer to integer (wrapping) and integer or float to a float
        // (round to nearest even). Float to integer, other rounding or saturation, and `Bf16` decline.
        ScalarExpr::Cast(x, spec) => {
            let to = spec.to;
            let from = expr_ty(kernel, x).ok_or_else(|| no(kernel, "a Cast of this operand"))?;
            let ok = (is_int_ty(from) && is_int_ty(to) && spec.sat == Saturation::Wrap)
                || ((is_int_ty(from) || is_float_ty(from))
                    && is_float_ty(to)
                    && spec.round == RoundMode::NearestEven
                    && spec.sat == Saturation::Wrap);
            if !ok {
                return Err(no(
                    kernel,
                    "this Cast (only int-to-int Wrap and to-float NearestEven lower)",
                ));
            }
            Ok(format!("(({})({}))", ctype(kernel, to)?, spell(kernel, x)?))
        }
        ScalarExpr::Add(a, b) => arith(a, "+", b),
        ScalarExpr::Sub(a, b) => arith(a, "-", b),
        ScalarExpr::Mul(a, b) => arith(a, "*", b),
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
    /// The module a `Call` or `ExternCall` resolves against (E16); `None` for a lone kernel.
    module: Option<std::rc::Rc<ModInfo>>,
    /// What a `Return` returns in a callee with a result (`l<r>`); `None` in an entry and a void callee.
    ret: Option<String>,
}

/// What a call site needs to know about the module: each function's Slang name and whether it returns a value, and each
/// extern's name and reference function.
struct ModInfo {
    fns: Vec<(String, bool)>,
    /// Per function, how many values a `CallTuple` receives after the first (the callee's `results`).
    extra: Vec<usize>,
    externs: Vec<(String, Option<u32>)>,
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
                out.push_str(&format!("{p}l{local} = {};\n", read(kernel, pl)?));
            }
            Stmt::Assign { local, value } => {
                out.push_str(&format!("{p}l{local} = {};\n", spell(kernel, value)?));
            }
            Stmt::Store { place: pl, value } => {
                out.push_str(&format!(
                    "{p}{};\n",
                    write(kernel, pl, &spell(kernel, value)?)?
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
                    read(kernel, pl)?,
                    spell(kernel, fill)?
                ));
            }
            Stmt::MaskedStore {
                place: pl,
                value,
                mask,
            } => {
                out.push_str(&format!(
                    "{p}if ({}) {{ {}; }}\n",
                    truth(kernel, mask)?,
                    write(kernel, pl, &spell(kernel, value)?)?
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
                ..
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
            // Atomics (E10). `InterlockedX(dest, value[, out original])` on a buffer or `groupshared` element. Vulkan fixes the
            // scope by the storage (device for a buffer, workgroup for shared) and Slang takes no memory order, so `scope` and
            // `order` are not spelled; the atomic is at least as strong as asked. Float atomics need a device feature
            // (`VK_EXT_shader_atomic_float`) the emitter cannot see yet (row E19), so they decline by name.
            Stmt::AtomicRmw {
                op,
                place: pl,
                value,
                old,
                ..
            } => {
                atomic_elem(kernel, pl)?;
                let f = match op {
                    AtomicOp::Add => "InterlockedAdd",
                    AtomicOp::Max => "InterlockedMax",
                    AtomicOp::Min => "InterlockedMin",
                    AtomicOp::And => "InterlockedAnd",
                    AtomicOp::Or => "InterlockedOr",
                    AtomicOp::Xor => "InterlockedXor",
                    AtomicOp::Exchange => "InterlockedExchange",
                    _ => return Err(no(kernel, "this atomic operation")),
                };
                let tail = match old {
                    Some(l) => {
                        cx.declared.insert(*l);
                        format!(", l{l}")
                    }
                    None => String::new(),
                };
                out.push_str(&format!(
                    "{p}{f}({}, {}{tail});\n",
                    place(kernel, pl)?,
                    spell(kernel, value)?
                ));
            }
            Stmt::AtomicCas {
                place: pl,
                expected,
                desired,
                old,
                ..
            } => {
                atomic_elem(kernel, pl)?;
                let tail = match old {
                    Some(l) => {
                        cx.declared.insert(*l);
                        format!(", l{l}")
                    }
                    None => String::new(),
                };
                out.push_str(&format!(
                    "{p}InterlockedCompareExchange({}, {}, {}{tail});\n",
                    place(kernel, pl)?,
                    spell(kernel, expected)?,
                    spell(kernel, desired)?
                ));
            }
            // Route (b) of doc row E15: no cooperative-matrix shape is mapped yet, so a dense `Mma` declines by name. Sparse and
            // block-scaled atoms are target gaps and never get here. Mapping an atom to a `CoopMat` shape needs the device's
            // shape list (row E19) and a device measurement.
            Stmt::Mma { atom, .. } => {
                return Err(LowerError::UnsupportedKernel {
                    detail: format!(
                        "slang: Mma: no cooperative-matrix shape for this atom `{}` (kernel `{}`)",
                        atom.name(),
                        kernel.name
                    ),
                });
            }
            Stmt::Break => out.push_str(&format!("{p}break;\n")),
            Stmt::Continue => out.push_str(&format!("{p}continue;\n")),
            Stmt::Return => match &cx.ret {
                Some(r) => out.push_str(&format!("{p}return {r};\n")),
                None => out.push_str(&format!("{p}return;\n")),
            },
            // Calls (E16). Scalar arguments by value only; a callee is a Slang function `f<index>`. An `ExternCall` with a
            // `reference` kernel emits a call of the reference body (its semantics are the declaration's by construction).
            Stmt::Call { .. } | Stmt::CallTuple { .. } => {
                let (callee, args, results): (&u32, &Vec<CallArg>, Vec<u32>) = match st {
                    Stmt::Call {
                        callee,
                        args,
                        result,
                    } => (callee, args, result.iter().copied().collect()),
                    Stmt::CallTuple {
                        callee,
                        args,
                        results,
                    } => (callee, args, results.clone()),
                    _ => unreachable!(),
                };
                let mi = cx
                    .module
                    .clone()
                    .ok_or_else(|| no(kernel, "Call outside a module"))?;
                let mut a = Vec::new();
                for x in args.iter() {
                    match x {
                        CallArg::Scalar(e) => a.push(spell(kernel, e)?),
                        // A buffer is passed by alias: the binding, then its strides and base (the callee's parameters).
                        CallArg::Buffer(b) => {
                            let Some(ParamKind::Buffer {
                                rank: Some(rank), ..
                            }) = kernel.params.get(*b as usize).map(|p| &p.kind)
                            else {
                                return Err(no(
                                    kernel,
                                    "Call: a buffer argument whose rank is not declared",
                                ));
                            };
                            let pre = if kernel.launch.is_some() { "pb_." } else { "" };
                            a.push(format!("b{b}"));
                            for d in 0..*rank {
                                a.push(format!("{pre}b{b}_stride{d}"));
                            }
                            a.push(format!("{pre}b{b}_base"));
                        }
                        _ => return Err(no(kernel, "Call: an allocation argument")),
                    }
                }
                let (name, _) = mi
                    .fns
                    .get(*callee as usize)
                    .ok_or_else(|| no(kernel, "Call: an unknown callee"))?;
                let call = format!("{name}({})", a.join(", "));
                let extra = mi.extra[*callee as usize];
                if results.len() > 1 && results.len() != extra + 1 {
                    return Err(no(kernel, "CallTuple: results do not match the callee"));
                }
                match results.as_slice() {
                    [] => out.push_str(&format!("{p}{call};\n")),
                    [r] if extra == 0 => {
                        cx.declared.insert(*r);
                        out.push_str(&format!("{p}l{r} = {call};\n"));
                    }
                    [r] => {
                        cx.declared.insert(*r);
                        out.push_str(&format!("{p}l{r} = {call}.v0;\n"));
                    }
                    rs => {
                        let t = format!("t{}", out.len());
                        out.push_str(&format!("{p}R{callee} {t} = {call};\n"));
                        for (i, r) in rs.iter().enumerate() {
                            cx.declared.insert(*r);
                            out.push_str(&format!("{p}l{r} = {t}.v{i};\n"));
                        }
                    }
                }
            }
            Stmt::ExternCall {
                decl,
                args,
                results,
            } => {
                let mi = cx
                    .module
                    .clone()
                    .ok_or_else(|| no(kernel, "ExternCall outside a module"))?;
                let (ename, reference) = mi
                    .externs
                    .get(*decl as usize)
                    .ok_or_else(|| no(kernel, "ExternCall: an unknown declaration"))?;
                let Some(fi) = reference else {
                    return Err(LowerError::UnsupportedKernel {
                        detail: format!(
                            "slang: ExternCall: no Slang body for extern `{ename}` (kernel `{}`)",
                            kernel.name
                        ),
                    });
                };
                if results.len() > 1 {
                    return Err(no(kernel, "ExternCall with several results"));
                }
                let (name, _) = &mi.fns[*fi as usize];
                let mut a = Vec::new();
                for x in args {
                    a.push(spell(kernel, x)?);
                }
                let call = format!("{name}({})", a.join(", "));
                match results.first() {
                    Some(r) => {
                        cx.declared.insert(*r);
                        out.push_str(&format!("{p}l{r} = {call};\n"));
                    }
                    None => out.push_str(&format!("{p}{call};\n")),
                }
            }
            _ => return Err(no(kernel, "this statement")),
        }
    }
    Ok(())
}

/// The element type an atomic acts on; only the 32-bit integers have an atomic in core Vulkan compute.
fn atomic_elem(kernel: &KernelFn, p: &Place) -> Result<(), LowerError> {
    let elem = match p {
        Place::Alloc { alloc, .. } => kernel.allocs.get(*alloc as usize).map(|a| a.elem),
        Place::Logical { buffer, .. } => match kernel.params.get(*buffer as usize).map(|q| &q.kind)
        {
            Some(ParamKind::Buffer { elem, .. }) => Some(*elem),
            _ => None,
        },
        _ => None,
    };
    match elem {
        Some(ElementKind::I32 | ElementKind::U32) => Ok(()),
        Some(ElementKind::F32) => Err(no(
            kernel,
            "a float atomic (needs VK_EXT_shader_atomic_float, a capability the emitter does not model yet)",
        )),
        _ => Err(no(kernel, "an atomic on this element type")),
    }
}

/// A place read as a value (a `Bf16` buffer element is widened to its `float`).
fn read(kernel: &KernelFn, p: &Place) -> Result<String, LowerError> {
    let pl = place(kernel, p)?;
    Ok(if is_bf16_buffer(kernel, p) {
        format!("bf16_to_f32_({pl})")
    } else {
        pl
    })
}

/// The statement text storing `value` to a place (a `Bf16` buffer element is rounded and narrowed).
fn write(kernel: &KernelFn, p: &Place, value: &str) -> Result<String, LowerError> {
    let pl = place(kernel, p)?;
    Ok(if is_bf16_buffer(kernel, p) {
        format!("{pl} = f32_to_bf16_({value})")
    } else {
        format!("{pl} = {value}")
    })
}

fn place(kernel: &KernelFn, p: &Place) -> Result<String, LowerError> {
    if let Place::Alloc { alloc, index } = p {
        return Ok(format!("a{alloc}[{}]", spell(kernel, index)?));
    }
    let Place::Logical { buffer, coords } = p else {
        return Err(no(kernel, "this place"));
    };
    // The entry reads strides/base from the params block; a callee (no launch) takes them as function parameters.
    let pre = if kernel.launch.is_some() { "pb_." } else { "" };
    let mut off = format!("{pre}b{buffer}_base");
    for (d, c) in coords.iter().enumerate() {
        off.push_str(&format!(
            " + (int64_t)({}) * {pre}b{buffer}_stride{d}",
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
    lower_entry(kernel, None, "")
}

/// The entry kernel, with `prelude` (the callee functions) ahead of the entry point and `mcx` to resolve calls.
fn lower_entry(
    kernel: &KernelFn,
    mcx: Option<std::rc::Rc<ModInfo>>,
    prelude: &str,
) -> Result<GeneratedKernel, LowerError> {
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
    // 16-bit floats: `half` needs the 16-bit arithmetic and storage features; `Bf16` is `uint16_t` storage plus helpers.
    let mut elems = vec![kernel.compute];
    elems.extend(kernel.locals.iter().map(|d| d.elem));
    elems.extend(kernel.allocs.iter().map(|a| a.elem));
    for q in &kernel.params {
        if let ParamKind::Buffer { elem, .. } = &q.kind {
            elems.push(*elem);
        }
    }
    let uses_bf16 = elems.contains(&ElementKind::Bf16);
    let uses_f16 = elems.contains(&ElementKind::F16);
    if uses_bf16 {
        s.push_str(BF16_HELPERS);
    }
    if uses_bf16 || uses_f16 {
        iface = iface.with_fidelity_note(
            "16-bit floats: needs shaderFloat16 and 16-bit storage (bf16 is uint16_t storage with float arithmetic rounded to bf16 after each operation, round-to-nearest-even)",
        );
    }
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
                let ty = storage_ty(kernel, *elem)?;
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
                if matches!(ty, ElementKind::F16 | ElementKind::Bf16) {
                    return Err(no(kernel, "a 16-bit float scalar parameter"));
                }
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
    s.push_str(prelude);
    s.push_str(&format!(
        "[shader(\"compute\")]\n[numthreads({bx}, {by}, {bz})]\nvoid {entry}(uint3 tid_ : SV_GroupThreadID, uint3 bid_ : SV_GroupID, uint gi_ : SV_GroupIndex)\n{{\n"
    ));
    s.push_str(&prologue);
    s.push_str(&private);
    // Statements first (collecting every local they declare), then the declarations, hoisted to the top of the entry:
    // the IR scopes a local by its declaring statement, the source declares it once where every use can see it.
    let mut cx = Cx {
        module: mcx,
        ..Cx::default()
    };
    let mut body = String::new();
    stmts(kernel, &kernel.body, 1, &mut cx, &mut body)?;
    for l in &cx.declared {
        let ty = local_ty(kernel, *l)?;
        s.push_str(&format!("    {ty} l{l} = ({ty})0;\n"));
    }
    s.push_str(&body);
    s.push_str("}\n");
    // bf16 arithmetic runs in `float`, so it takes the float denorm argument.
    let arg_dt = if uses_bf16 {
        ElementKind::F32
    } else {
        kernel.compute
    };
    for a in crate::required_slangc_args(arg_dt) {
        iface = iface.with_compiler_arg(*a);
    }
    // Capabilities the source needs, as `vulkan:` `<arith>` tokens (doc row E19). `with_requires` only records the name; the
    // consumer that holds the target gates on it (`crate::unmet_requirements`). Sorted, so the list is stable.
    let mut needs = BTreeSet::new();
    if s.contains("int64_t") {
        needs.insert("i64");
    }
    if uses_f16 {
        needs.insert("f16");
        needs.insert("st16");
    }
    if uses_bf16 {
        // `uint16_t` buffer elements: 16-bit integer type plus 16-bit storage access.
        needs.insert("i16");
        needs.insert("st16");
    }
    for n in needs {
        iface = iface.with_requires(n);
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

/// Every function `b` calls, directly: a `Call`'s callee and an `ExternCall`'s reference kernel.
fn callees(m: &Module, b: &Block, out: &mut Vec<u32>) {
    for st in &b.0 {
        match st {
            Stmt::Call { callee, .. } | Stmt::CallTuple { callee, .. } => out.push(*callee),
            Stmt::ExternCall { decl, .. } => {
                if let Some(r) = m.externs.get(*decl as usize).and_then(|d| d.reference) {
                    out.push(r);
                }
            }
            Stmt::If { then, els, .. } => {
                callees(m, then, out);
                callees(m, els, out);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => callees(m, body, out),
            _ => {}
        }
    }
}

/// Lower the module's `entry` kernel with every function it reaches as a Slang function ahead of it (E16). The checker
/// resolves the calls (no cycle, valid callees) first; the callees come out callee-first.
pub(crate) fn lower_module(m: &Module, entry: u32) -> Result<GeneratedKernel, LowerError> {
    let Some(ek) = m.fns.get(entry as usize) else {
        return Err(LowerError::UnsupportedKernel {
            detail: format!("slang: the module has no entry function {entry}"),
        });
    };
    check_module(m, entry).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!(
            "slang: the kernel-layer checker refuses module entry `{}`: {e:?}",
            ek.name
        ),
    })?;
    // Reachable functions, callee-first (the call graph is acyclic: the checker said so).
    fn visit(m: &Module, f: u32, seen: &mut Vec<u32>, order: &mut Vec<u32>) {
        if seen.contains(&f) {
            return;
        }
        seen.push(f);
        let mut cs = Vec::new();
        callees(m, &m.fns[f as usize].body, &mut cs);
        for c in cs {
            visit(m, c, seen, order);
        }
        order.push(f);
    }
    let (mut seen, mut order) = (Vec::new(), Vec::new());
    visit(m, entry, &mut seen, &mut order);
    let mi = std::rc::Rc::new(ModInfo {
        fns: m
            .fns
            .iter()
            .enumerate()
            .map(|(i, f)| (format!("f{i}"), f.result.is_some()))
            .collect(),
        extra: m.fns.iter().map(|f| f.results.len()).collect(),
        externs: m
            .externs
            .iter()
            .map(|d| (d.name.clone(), d.reference))
            .collect(),
    });
    let mut prelude = String::new();
    for f in order {
        let k = &m.fns[f as usize];
        if let Some(gap) = crate::gaps::first_gap(k) {
            return Err(LowerError::UnsupportedKernel {
                detail: format!("{}{gap} (kernel `{}`)", crate::gaps::GAP_PREFIX, k.name),
            });
        }
        if f == entry {
            continue;
        }
        prelude.push_str(&lower_callee(k, f, &mi)?);
    }
    if let Some(gap) = crate::gaps::first_gap(ek) {
        return Err(LowerError::UnsupportedKernel {
            detail: format!("{}{gap} (kernel `{}`)", crate::gaps::GAP_PREFIX, ek.name),
        });
    }
    lower_entry(ek, Some(mi), &prelude)
}

/// One callee as a Slang function `f<index>`: scalar parameters by value, the result local returned, no launch, no
/// allocations, no 16-bit floats, no launch ids (a callee takes no hidden state).
fn lower_callee(k: &KernelFn, f: u32, mi: &std::rc::Rc<ModInfo>) -> Result<String, LowerError> {
    if k.launch.is_some() || !k.allocs.is_empty() {
        return Err(no(k, "a callee with a launch or an allocation"));
    }
    if !k.results.is_empty() && k.result.is_none() {
        return Err(no(k, "a callee with tuple results but no first result"));
    }
    let mut params = Vec::new();
    let mut prologue = String::new();
    for (i, p) in k.params.iter().enumerate() {
        match &p.kind {
            ParamKind::Scalar { ty, local } => {
                if matches!(ty, ElementKind::F16 | ElementKind::Bf16) {
                    return Err(no(k, "a callee with a 16-bit float parameter"));
                }
                let t = ctype(k, *ty)?;
                params.push(format!("{t} p{i}"));
                prologue.push_str(&format!("    {t} l{local} = p{i};\n"));
            }
            ParamKind::Buffer { elem, role, rank } => {
                let Some(rank) = rank else {
                    return Err(no(k, "a callee with a buffer whose rank is not declared"));
                };
                if matches!(elem, ElementKind::F16 | ElementKind::Bf16) {
                    return Err(no(k, "a callee with a 16-bit float buffer"));
                }
                let ty = storage_ty(k, *elem)?;
                let qual = match role {
                    BufferRole::Read => "StructuredBuffer",
                    _ => "RWStructuredBuffer",
                };
                params.push(format!("{qual}<{ty}> b{i}"));
                for d in 0..*rank {
                    params.push(format!("int64_t b{i}_stride{d}"));
                }
                params.push(format!("int64_t b{i}_base"));
            }
            _ => return Err(no(k, "a callee with this kind of parameter")),
        }
    }
    if k.locals
        .iter()
        .any(|d| matches!(d.elem, ElementKind::F16 | ElementKind::Bf16))
    {
        return Err(no(k, "a callee with a 16-bit float local"));
    }
    // A function returning several values returns a struct `R<f>` of fields `v0, v1, ...` (the first is `result`).
    let mut tuple_def = String::new();
    let mut ret_expr = k.result.map(|r| format!("l{r}"));
    let ret_ty = match k.result {
        Some(r) if !k.results.is_empty() => {
            let all: Vec<u32> = std::iter::once(r)
                .chain(k.results.iter().copied())
                .collect();
            tuple_def.push_str(&format!("struct R{f}\n{{\n"));
            for (i, l) in all.iter().enumerate() {
                tuple_def.push_str(&format!("    {} v{i};\n", local_ty(k, *l)?));
            }
            tuple_def.push_str("};\n");
            let vals: Vec<String> = all.iter().map(|l| format!("l{l}")).collect();
            ret_expr = Some(format!("R{f}({})", vals.join(", ")));
            format!("R{f}")
        }
        Some(r) => local_ty(k, r)?.to_string(),
        None => "void".to_string(),
    };
    let mut cx = Cx {
        module: Some(mi.clone()),
        ret: ret_expr,
        ..Cx::default()
    };
    let mut body = String::new();
    stmts(k, &k.body, 1, &mut cx, &mut body)?;
    if ["tid_", "bid_", "gi_"].iter().any(|n| body.contains(n)) {
        return Err(no(k, "a callee that reads a launch id"));
    }
    let mut s = tuple_def;
    s.push_str(&format!("{ret_ty} f{f}({})\n{{\n", params.join(", ")));
    s.push_str(&prologue);
    for l in &cx.declared {
        let ty = local_ty(k, *l)?;
        s.push_str(&format!("    {ty} l{l} = ({ty})0;\n"));
    }
    s.push_str(&body);
    if let Some(r) = &cx.ret {
        s.push_str(&format!("    return {r};\n"));
    }
    s.push_str("}\n");
    Ok(s)
}
