#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regeneration gate: regenerate SPIR-V with a pinned slangc and compare it to a consumer's committed .spv.

A consumer (Fuel, Vulkane) commits the .spv it built from `fishstringer-kernels-source`. This script answers
"is what I committed still what the pinned toolchain makes from these sources?" without a byte-compare (slangc output
is not byte-stable: debug records, id numbering and the order of independent instructions move between builds).

    python3 regen_gate.py --committed DIR --slangc-version 2026.21 [--src-dir DIR] [--report FILE]

Steps: (1) the toolchain is pinned: `compile.sh --print-toolchain` must report exactly `--slangc-version` (slangc is
taken from `$SLANGC` or the Vulkan SDK, as compile.sh does), else exit 2; (2) `compile.sh` regenerates every output
into a temporary directory; (3) each regenerated file is compared with the committed one of the same name:

    identical     byte-for-byte the same
    equivalent    spv_equiv.py says equivalent (needs spirv-dis on PATH)
    different     spv_equiv.py does not show them equivalent
    undecidable   spv_equiv.py refused (OpPhi, OpFunctionCall, bad input, spirv-dis missing)
    missing       the toolchain makes it, nothing committed under that name
    extra         committed, but the toolchain no longer makes it (only checked when regenerating all outputs)

Exit 0: every kernel identical or equivalent. Exit 1: at least one different, missing or extra. Exit 2: the gate could
not decide (pin mismatch, compile failure, undecidable). A refusal is never a pass.

`--regen-dir DIR` compares an already-regenerated directory instead of running the toolchain (no pin check). Files may
be `.spv` or, for tests and offline use, `spirv-dis` text named `.spvasm`; names are matched by stem.

This is a structural check, not a proof: see spv_equiv.py for exactly what it ignores and what it keeps. It runs
nothing on a device; a consumer's live test still decides numeric agreement.
"""

import argparse
import subprocess  # nosec B404 - runs this crate's own compile.sh by argv list, no shell
import sys
import tempfile
from pathlib import Path

import spv_equiv

HERE = Path(__file__).resolve().parent
SUFFIXES = (".spv", ".spvasm")


def _index(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in SUFFIXES}


def classify(committed: Path, regenerated: Path) -> str:
    """One of identical / equivalent / different / undecidable for a pair of files."""
    if committed.read_bytes() == regenerated.read_bytes():
        return "identical"
    try:
        text_a, text_b = spv_equiv._load(str(committed)), spv_equiv._load(str(regenerated))
        return "equivalent" if spv_equiv.equivalent(text_a, text_b) else "different"
    except spv_equiv.Undecidable:
        return "undecidable"


def compare_dirs(committed_dir: Path, regen_dir: Path, check_extra: bool = True) -> dict[str, list[str]]:
    committed, regen = _index(committed_dir), _index(regen_dir)
    result: dict[str, list[str]] = {k: [] for k in ("identical", "equivalent", "different", "undecidable", "missing", "extra")}
    for name, path in regen.items():
        if name not in committed:
            result["missing"].append(name)
        else:
            result[classify(committed[name], path)].append(name)
    if check_extra:
        result["extra"] = sorted(set(committed) - set(regen))
    return result


def exit_code(result: dict[str, list[str]]) -> int:
    if result["undecidable"]:
        return 2
    if result["different"] or result["missing"] or result["extra"]:
        return 1
    return 0


def render(result: dict[str, list[str]]) -> str:
    total = sum(len(v) for k, v in result.items() if k != "extra")
    lines = [f"{k}: {len(v)}" for k, v in result.items()]
    lines.append(f"compared: {total}")
    for key in ("different", "undecidable", "missing", "extra"):
        for name in result[key]:
            lines.append(f"  {key}: {name}")
    return "\n".join(lines)


def check_pin(version: str, env=None) -> str | None:
    """None if `compile.sh --print-toolchain` reports slangc == version, else a message."""
    run = subprocess.run(["bash", str(HERE / "compile.sh"), "--print-toolchain"], capture_output=True, text=True, env=env)  # nosec B603 B607
    if run.returncode != 0:
        return f"compile.sh --print-toolchain failed: {run.stderr.strip() or run.stdout.strip()}"
    for line in run.stdout.splitlines():
        if line.startswith("slangc:"):
            got = line.split(":", 1)[1].strip().split()[0] if line.split(":", 1)[1].strip() else ""
            return None if got == version else f"slangc is {got!r}, pinned {version!r}"
    return "compile.sh --print-toolchain printed no slangc line"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Regenerate SPIR-V with a pinned slangc and compare to committed .spv.")
    ap.add_argument("--committed", required=True, type=Path, help="directory of the consumer's committed .spv")
    ap.add_argument("--slangc-version", help="exact slangc version to require, e.g. 2026.21 (required unless --regen-dir)")
    ap.add_argument("--src-dir", type=Path, help="sources to compile instead of the ones next to compile.sh")
    ap.add_argument("--regen-dir", type=Path, help="compare this already-regenerated directory; do not run the toolchain")
    ap.add_argument("--report", type=Path, help="also write the summary to this file")
    args = ap.parse_args(argv)

    if not args.committed.is_dir():
        print(f"error: --committed {args.committed} is not a directory", file=sys.stderr)
        return 2

    tmp = None
    if args.regen_dir is not None:
        regen_dir = args.regen_dir
        if not regen_dir.is_dir():
            print(f"error: --regen-dir {regen_dir} is not a directory", file=sys.stderr)
            return 2
    else:
        if not args.slangc_version:
            print("error: --slangc-version is required (the gate pins the toolchain)", file=sys.stderr)
            return 2
        problem = check_pin(args.slangc_version)
        if problem:
            print(f"toolchain not pinned: {problem}", file=sys.stderr)
            return 2
        tmp = tempfile.TemporaryDirectory(prefix="regen-gate-")
        regen_dir = Path(tmp.name)
        cmd = ["bash", str(HERE / "compile.sh"), "--out-dir", str(regen_dir)]
        if args.src_dir:
            cmd += ["--src-dir", str(args.src_dir)]
        run = subprocess.run(cmd)  # nosec B603 B607
        if run.returncode != 0:
            print(f"compile.sh failed (exit {run.returncode})", file=sys.stderr)
            return 2

    try:
        result = compare_dirs(args.committed, regen_dir)
        summary = render(result)
        print(summary)
        if args.report:
            args.report.write_text(summary + "\n", encoding="utf-8")
        return exit_code(result)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
