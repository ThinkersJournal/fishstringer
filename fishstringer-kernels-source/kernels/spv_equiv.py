#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Decide whether two SPIR-V modules compute the same thing, ignoring what a recompile may legitimately change.

Why: a byte-compare of regenerated SPIR-V against a committed copy cannot pass across `slangc` builds
(fishstringer 0.0.12 vs Fuel's committed files: 119 of 153 differ, 111 in debug records only, 8 in id numbering and
instruction order). This tool is the gate that can: it passes when the modules differ only in

  - debug records (OpName, OpMemberName, OpSource*, OpString, OpLine, OpNoLine, OpModuleProcessed, header comments),
  - the numbering and names of ids,
  - the order of instructions that cannot interact (pure value instructions, declarations, decorations),
  - repeated identical pure instructions (common subexpressions).

It keeps, and so reports as different: any change of opcode, type, constant, decoration, binding, execution mode,
entry point or capability; any change of an instruction's operands; the control-flow graph; the ORDER of stores,
atomics and barriers within a block; and which store a load can observe (a load is tied to the number of
stores/atomics/barriers before it in its block, except loads of Input / UniformConstant / PushConstant variables,
which nothing in the module can write).

How: every value id gets a hash of its defining instruction and, recursively, its operands (a Merkle DAG, so the id
numbers never enter). A block's label hash is then refined over the control-flow graph from the block's ordered
store/barrier sequence. The module is the sorted multiset of those hashes.

What it does NOT do: it is a conservative structural check, not a proof of semantic equality. "different" means "not
shown equivalent" (for example two commutative operands swapped, or a load moved past an unrelated store, are
reported different). It refuses (exit 2) a module with OpPhi, OpFunctionCall or decoration groups, which `slangc`
does not emit for these kernels; a refusal is never a pass. It does not run the kernels: numeric agreement is the
consumer's live test.

Usage:
    python spv_equiv.py A.spv B.spv            # needs spirv-dis (Vulkan SDK) on PATH
    python spv_equiv.py A.spvasm B.spvasm      # disassembly text, no tools needed
    python spv_equiv.py --explain A.spv B.spv  # on a difference, say which part differs

Exit status: 0 equivalent, 1 different (not shown equivalent), 2 cannot decide or bad input.
"""

import hashlib
import re
import shutil
import subprocess  # nosec B404 - runs one fixed Vulkan SDK tool (spirv-dis) by argv list, no shell
import sys
from pathlib import Path

DEBUG_OPS = {"OpName", "OpMemberName", "OpSource", "OpSourceContinued", "OpSourceExtension", "OpString",
             "OpLine", "OpNoLine", "OpModuleProcessed"}
UNSUPPORTED_OPS = {"OpPhi", "OpFunctionCall", "OpGroupDecorate", "OpGroupMemberDecorate", "OpDecorationGroup"}
READ_ONLY_CLASSES = {"Input", "UniformConstant", "PushConstant"}
CHAIN_OPS = {"OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain", "OpInBoundsPtrAccessChain", "OpCopyObject"}
MODULE_OPS = {"OpCapability", "OpExtension", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode", "OpExecutionModeId"}
EFFECT_OPS = {"OpStore", "OpControlBarrier", "OpMemoryBarrier", "OpCopyMemory", "OpCopyMemorySized", "OpImageWrite"}

INSTR = re.compile(r"^\s*(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')


class Undecidable(Exception):
    """The module uses something this checker does not model; the answer is neither yes nor no."""


class Inst:
    __slots__ = ("result", "op", "args", "block", "ver")

    def __init__(self, result, op, args):
        self.result, self.op, self.args = result, op, args
        self.block = None
        self.ver = 0


def _h(*parts) -> str:
    return hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=12).hexdigest()


def parse(text: str) -> list[Inst]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        m = INSTR.match(line)
        if not m:
            raise Undecidable(f"cannot parse line: {line!r}")
        result, op, rest = m.groups()
        if op in DEBUG_OPS:
            continue
        if op in UNSUPPORTED_OPS:
            raise Undecidable(f"{op} is not modelled")
        out.append(Inst(result, op, TOKEN.findall(rest)))
    return out


def _is_effect(op: str) -> bool:
    return op in EFFECT_OPS or op.startswith("OpAtomic")


class Module:
    def __init__(self, text: str):
        self.insts = parse(text)
        self.defs = {i.result: i for i in self.insts if i.result}
        self.labels = {i.result for i in self.insts if i.op == "OpLabel"}
        self.deco = {}
        for i in self.insts:
            if i.op == "OpDecorate":
                self.deco.setdefault(i.args[0], []).append(("D", tuple(i.args[1:])))
            elif i.op == "OpMemberDecorate":
                self.deco.setdefault(i.args[0], []).append(("M", tuple(i.args[1:])))
        self._memo = {}
        self._cls = {}
        self._assign_blocks()

    def _assign_blocks(self):
        block, ver = None, 0
        for i in self.insts:
            if i.op == "OpFunction":
                block = None
            elif i.op == "OpLabel":
                block, ver = i.result, 0
            elif block is not None:
                i.block, i.ver = block, ver
                if _is_effect(i.op):
                    ver += 1

    def storage_class(self, ptr: str):
        if ptr in self._cls:
            return self._cls[ptr]
        d = self.defs.get(ptr)
        cls = None
        if d is not None and d.op == "OpVariable":
            cls = d.args[1]  # %v = OpVariable %ptr_type <StorageClass> [initialiser]
        elif d is not None and d.op in CHAIN_OPS:
            cls = self.storage_class(d.args[2] if d.op != "OpCopyObject" else d.args[1])
        self._cls[ptr] = cls
        return cls

    def vhash(self, ident: str) -> str:
        """Hash of a value id: its defining instruction and, recursively, its operands. Id numbers never enter."""
        if ident in self.labels:
            return "<label>"
        if ident in self._memo:
            return self._memo[ident]
        d = self.defs.get(ident)
        if d is None:
            raise Undecidable(f"id {ident} is used but never defined")
        self._memo[ident] = "<cycle>"  # a value cycle would need OpPhi, which is refused; keeps recursion finite
        extra = ()
        if d.op == "OpLoad":
            cls = self.storage_class(d.args[1])
            extra = (None,) if cls in READ_ONLY_CLASSES else (d.ver,)
        deco = tuple(sorted(self.deco.get(ident, ())))
        val = _h("v", d.op, self._args(d), extra, deco)
        self._memo[ident] = val
        return val

    def _args(self, inst: Inst) -> tuple:
        return tuple(self.vhash(a) if a.startswith("%") else a for a in inst.args)

    def ihash(self, inst: Inst) -> str:
        return self.vhash(inst.result) if inst.result else _h("i", inst.op, self._args(inst))

    def signature(self) -> dict:
        module_sigs = []
        for i in self.insts:
            if i.op in MODULE_OPS or (i.op == "OpVariable" and self.storage_class(i.result) != "Function"):
                module_sigs.append(self._module_hash(i))
        fn_sigs = []
        cur = None
        for i in self.insts:
            if i.op == "OpFunction":
                cur = [i, []]
            elif i.op == "OpFunctionEnd":
                fn_sigs.append(self._function_sig(cur[0], cur[1]))
                cur = None
            elif cur is not None:
                cur[1].append(i)
        return {"module": sorted(module_sigs), "functions": sorted(fn_sigs)}

    def _module_hash(self, i: Inst) -> str:
        if i.op == "OpEntryPoint":
            model, fn, name, *iface = i.args
            return _h("entry", model, self.vhash(fn), name, tuple(sorted(self.vhash(x) for x in iface)))
        if i.op in ("OpExecutionMode", "OpExecutionModeId"):
            return _h("mode", self.vhash(i.args[0]), i.op, tuple(i.args[1:]))
        return self.ihash(i)

    def _function_sig(self, head: Inst, body: list[Inst]) -> str:
        blocks = {}
        order = []
        for i in body:
            if i.op == "OpLabel":
                order.append(i.result)
                blocks[i.result] = {"effects": [], "tail": []}
            elif i.block is not None:
                b = blocks[i.block]
                if _is_effect(i.op):
                    b["effects"].append(self.ihash(i))
                elif i.op in ("OpSelectionMerge", "OpLoopMerge", "OpBranch", "OpBranchConditional", "OpSwitch",
                              "OpReturn", "OpReturnValue", "OpKill", "OpUnreachable"):
                    b["tail"].append(i)
        if not order:
            return _h("fn", self.ihash(head), ())
        base = {}
        for lab, b in blocks.items():
            tail = tuple(self._tail_hash(t) for t in b["tail"])
            base[lab] = _h("blk", tuple(b["effects"]), tail)
        cur = dict(base)
        for _ in range(len(order) + 1):
            nxt = {}
            for lab, b in blocks.items():
                succ = tuple(cur[a] for t in b["tail"] for a in t.args if a in blocks)
                nxt[lab] = _h("lab", base[lab], succ)
            if len(set(nxt.values())) == len(set(cur.values())) and _same_partition(cur, nxt):
                cur = nxt
                break
            cur = nxt
        return _h("fn", self.ihash(head), cur[order[0]], tuple(sorted(cur.values())))

    def _tail_hash(self, t: Inst) -> str:
        return _h("t", t.op, tuple("<L>" if a in self.labels else (self.vhash(a) if a.startswith("%") else a)
                                   for a in t.args))


def _same_partition(a: dict, b: dict) -> bool:
    ka = {}
    for k in a:
        if ka.setdefault(a[k], b[k]) != b[k]:
            return False
    kb = {}
    for k in a:
        if kb.setdefault(b[k], a[k]) != a[k]:
            return False
    return True


def signature(text: str) -> dict:
    return Module(text).signature()


def equivalent(text_a: str, text_b: str) -> bool:
    return signature(text_a) == signature(text_b)


def explain(text_a: str, text_b: str) -> str:
    a, b = signature(text_a), signature(text_b)
    if a == b:
        return "equivalent"
    parts = []
    for key in ("module", "functions"):
        if a[key] != b[key]:
            only_a = [x for x in a[key] if x not in b[key]]
            only_b = [x for x in b[key] if x not in a[key]]
            parts.append(f"{key}: {len(only_a)} only in A, {len(only_b)} only in B (of {len(a[key])} / {len(b[key])})")
    return "different: " + "; ".join(parts)


def _load(arg: str) -> str:
    path = Path(arg)
    if not path.is_file():
        raise Undecidable(f"{arg} is not a file")
    if path.suffix == ".spv":
        dis = shutil.which("spirv-dis")
        if dis is None:
            raise Undecidable("spirv-dis not found on PATH (install the Vulkan SDK), or pass .spvasm text")
        run = subprocess.run([dis, "--no-color", str(path)], capture_output=True, text=True)  # nosec B603
        if run.returncode != 0:
            raise Undecidable(f"spirv-dis failed on {arg}: {run.stderr.strip()}")
        return run.stdout
    return path.read_text(encoding="utf-8")


def main(argv: list[str]) -> int:
    want_explain = "--explain" in argv
    args = [a for a in argv if a != "--explain"]
    if len(args) != 2:
        print("usage: spv_equiv.py [--explain] A.spv|A.spvasm B.spv|B.spvasm", file=sys.stderr)
        return 2
    try:
        ta, tb = _load(args[0]), _load(args[1])
        msg = explain(ta, tb)
    except Undecidable as e:
        print(f"cannot decide: {e}", file=sys.stderr)
        return 2
    if msg == "equivalent":
        print("equivalent")
        return 0
    print(msg if want_explain else "different (not shown equivalent)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
