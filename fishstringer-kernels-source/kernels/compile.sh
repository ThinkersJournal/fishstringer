#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Compile every shader source (*.wgsl, *.glsl, *.slang) in this
# directory to SPIR-V, writing the output to --out-dir DIR (default ../spv/).
#
# Toolchain: requires the Vulkan SDK (for slangc + glslangValidator)
# and naga-cli (cargo install naga-cli) on PATH.
#
# Compiled SPIR-V is not kept in this crate; each consumer builds (and, if it
# wants, commits) its own from these sources.
#
# Usage:
#   ./compile.sh                         # compile all into ../spv/
#   ./compile.sh --out-dir out/spv       # compile all into out/spv/
#   ./compile.sh rope.slang              # compile a single file
#   ./compile.sh --out-dir out rope.slang
#   ./compile.sh --src-dir DIR ...       # compile the sources in DIR instead of next to this script
#   ./compile.sh --print-manifest        # print "<source><TAB><kind><TAB><output.spv>" per output; needs no tools
#   ./compile.sh --print-toolchain       # print the versions of the tools that would be used (for provenance)
#
# Options go before the file names. Exit status is non-zero if a needed tool is missing or a compile fails.
#
# Conventions:
#   .wgsl   -> naga (Vulkan 1.1 / SPIR-V 1.3)
#   .glsl   -> glslangValidator, compute stage, Vulkan 1.1
#   .slang  -> slangc, SPIR-V, profile glsl_450
#             Default entry point is 'main', output is <basename>.spv.
#             To compile multiple entry points from one source, add a
#             directive line near the top:
#                 // entries: silu_forward, silu_backward
#             Each listed entry becomes its own <entry>.spv.
#
# GLSL files that need a specific entry or non-default SPIR-V version
# can override via a sidecar comment line `// compile: <flags>` at
# the top of the file.

set -euo pipefail

# SCRIPT_DIR holds this script and its post-processing helpers (*.py); SRC_DIR holds the sources to compile
# (--src-dir, default: SCRIPT_DIR).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
# Output directory: --out-dir DIR, else spv/ next to kernels/ (git-ignored).
# A relative DIR is resolved against the caller's working directory, before
# this script changes into the source directory.
OUT_DIR="$SCRIPT_DIR/../spv"
MODE=compile
while (( $# > 0 )); do
  case "$1" in
    --out-dir)  [[ -n "${2:-}" ]] || { echo "error: --out-dir needs a directory" >&2; exit 2; }; OUT_DIR="$2"; shift 2 ;;
    --src-dir)  [[ -n "${2:-}" ]] || { echo "error: --src-dir needs a directory" >&2; exit 2; }
                SRC_DIR="$(cd "$2" && pwd)" || exit 2; shift 2 ;;
    --print-manifest)  MODE=manifest; shift ;;
    --print-toolchain) MODE=toolchain; shift ;;
    --) shift; break ;;
    -*) echo "error: unknown option $1" >&2; exit 2 ;;
    *)  break ;;
  esac
done
if [[ "$MODE" == compile ]]; then
  mkdir -p "$OUT_DIR"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

# Resolve tools. slangc and glslangValidator come from the Vulkan SDK.
SLANGC="${SLANGC:-${VULKAN_SDK:-}/Bin/slangc}"
GLSLANG="${GLSLANG:-${VULKAN_SDK:-}/Bin/glslangValidator}"
NAGA="${NAGA:-naga}"

# On non-Windows the SDK usually puts binaries in bin/ (lowercase).
if [[ ! -x "$SLANGC" && -n "${VULKAN_SDK:-}" ]]; then
  [[ -x "$VULKAN_SDK/bin/slangc" ]] && SLANGC="$VULKAN_SDK/bin/slangc"
fi
if [[ ! -x "$GLSLANG" && -n "${VULKAN_SDK:-}" ]]; then
  [[ -x "$VULKAN_SDK/bin/glslangValidator" ]] && GLSLANG="$VULKAN_SDK/bin/glslangValidator"
fi

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1 && [[ ! -x "$1" ]]; then
    echo "error: $2 not found (looked for '$1')" >&2
    echo "       Install the Vulkan SDK and/or run 'cargo install naga-cli'." >&2
    return 1
  fi
}

compile_wgsl() {
  local src="$1"
  local base; base="$(basename "$src" .wgsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  wgsl  $base.wgsl -> $OUT_DIR/$base.spv"
  "$NAGA" "$src" "$out"
}

compile_glsl() {
  local src="$1"
  local base; base="$(basename "$src" .glsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  glsl  $base.glsl -> $OUT_DIR/$base.spv"
  "$GLSLANG" -V -S comp --target-env vulkan1.1 "$src" -o "$out" >/dev/null
}

compile_slang() {
  local src="$1"
  local base; base="$(basename "$src" .slang)"

  # Look for a `// entries: a, b, c` directive anywhere in the first
  # 40 lines. If present, compile each listed entry to <entry>.spv.
  # Otherwise compile the default `main` entry to <basename>.spv.
  local entries_line
  entries_line="$(head -n 40 "$src" | grep -E '^\s*//\s*entries\s*:' | head -n 1 || true)"

  if [[ -n "$entries_line" ]]; then
    # Strip the comment prefix, the 'entries:' keyword, and split on commas.
    local list
    list="$(echo "$entries_line" | sed -E 's|^\s*//\s*entries\s*:\s*||; s|,| |g')"
    for entry in $list; do
      # Trim whitespace.
      entry="$(echo "$entry" | tr -d '[:space:]')"
      [[ -z "$entry" ]] && continue
      local out="$OUT_DIR/$entry.spv"
      echo "  slang $base.slang[$entry] -> $OUT_DIR/$entry.spv"
      "$SLANGC" "$src" -target spirv -profile glsl_450 -entry "$entry" -o "$out"
      hoist_imports_if_needed "$out"
      no_contraction_if_requested "$src" "$out"
    done
  else
    local out="$OUT_DIR/$base.spv"
    echo "  slang $base.slang -> $OUT_DIR/$base.spv"
    "$SLANGC" "$src" -target spirv -profile glsl_450 -entry main -o "$out"
    hoist_imports_if_needed "$out"
    no_contraction_if_requested "$src" "$out"
  fi
}

# Post-process: if the SPV contains OpExtInstImport instructions
# inside function bodies (slang's `spirv_asm` lowers them there
# instead of module scope, which spirv-val rejects), run the hoist
# script to move them up + dedupe. The script is a no-op when there
# are no in-function imports.
hoist_imports_if_needed() {
  local spv="$1"
  if command -v python >/dev/null 2>&1; then
    python "$SCRIPT_DIR/hoist_extinst_imports.py" "$spv"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/hoist_extinst_imports.py" "$spv"
  else
    echo "    error: python not found; $spv needs hoist_extinst_imports.py" >&2
    return 1
  fi
}

# Post-process: a kernel whose first 40 lines carry a `// no-contraction` comment gets NoContraction on
# every OpFAdd/OpFSub/OpFMul (see no_contraction.py: slangc cannot emit it itself, and the
# double-double kernels need `a * b` and `a + b` to round separately).
no_contraction_if_requested() {
  local src="$1" spv="$2"
  if head -n 40 "$src" | grep -qE '^\s*//\s*no-contraction\b'; then
    if command -v python >/dev/null 2>&1; then
      python "$SCRIPT_DIR/no_contraction.py" "$spv"
    elif command -v python3 >/dev/null 2>&1; then
      python3 "$SCRIPT_DIR/no_contraction.py" "$spv"
    else
      echo "    error: python not found; $spv needs no_contraction.py" >&2
      return 1
    fi
  fi
}

compile_one() {
  local f="$1"
  case "$f" in
    *.wgsl)  compile_wgsl  "$f" ;;
    *.glsl)  compile_glsl  "$f" ;;
    *.slang) compile_slang "$f" ;;
    *) echo "warning: unknown shader type: $f" >&2 ;;
  esac
}

# "<source><TAB><kind><TAB><output.spv>" for every output of one source, using the same entry rules as the
# compile functions (a `// entries:` directive gives one output per entry; otherwise <basename>.spv).
manifest_one() {
  local f="$1" base kind line list entry
  local -a head40=()
  case "$f" in
    *.wgsl)  kind=wgsl;  base="${f%.wgsl}" ;;
    *.glsl)  kind=glsl;  base="${f%.glsl}" ;;
    *.slang) kind=slang; base="${f%.slang}" ;;
    *) return 0 ;;
  esac
  base="${base##*/}"
  list=""
  if [[ "$kind" == slang ]]; then
    mapfile -n 40 -t head40 < "$f"
    for line in "${head40[@]}"; do
      if [[ "$line" =~ ^[[:space:]]*//[[:space:]]*entries[[:space:]]*:[[:space:]]*(.*)$ ]]; then
        list="${BASH_REMATCH[1]//,/ }"
        break
      fi
    done
  fi
  if [[ -n "$list" ]]; then
    for entry in $list; do
      entry="${entry//[[:space:]]/}"
      [[ -z "$entry" ]] && continue
      printf '%s\t%s\t%s.spv\n' "$f" "$kind" "$entry"
    done
  else
    printf '%s\t%s\t%s.spv\n' "$f" "$kind" "$base"
  fi
}

print_toolchain() {
  local rc=0
  echo "slangc: $("$SLANGC" -v 2>&1 | head -n 1)" || rc=1
  echo "glslangValidator: $("$GLSLANG" --version 2>&1 | head -n 1)" || rc=1
  echo "python: $( (python --version || python3 --version) 2>&1 | head -n 1)" || rc=1
  return $rc
}

list_sources() {
  # One source per line, sorted, from SRC_DIR (the ones named on the command line, else all).
  if (( $# > 0 )); then printf '%s
' "$@"; else
    ( cd "$SRC_DIR" && shopt -s nullglob && printf '%s
' *.wgsl *.glsl *.slang | LC_ALL=C sort ); fi
}

case "$MODE" in
  manifest)
    cd "$SRC_DIR"
    while IFS= read -r f; do [[ -n "$f" ]] && manifest_one "$f"; done < <(list_sources "$@")
    exit 0 ;;
  toolchain)
    check_tool "$SLANGC" "slangc"
    check_tool "$GLSLANG" "glslangValidator"
    print_toolchain
    exit $? ;;
esac

cd "$SRC_DIR"
if (( $# > 0 )); then
  # Check only the tools we actually need.
  for arg in "$@"; do
    case "$arg" in
      *.wgsl)  check_tool "$NAGA" "naga-cli" ;;
      *.glsl)  check_tool "$GLSLANG" "glslangValidator" ;;
      *.slang) check_tool "$SLANGC" "slangc" ;;
    esac
  done
  for f in "$@"; do compile_one "$f"; done
else
  # Compile everything: every tool that a present source type needs must exist (a missing one is an error, not a skip).
  shopt -s nullglob
  files=( *.wgsl *.glsl *.slang )
  (( ${#files[@]} > 0 )) || { echo "error: no shader sources in $SRC_DIR" >&2; exit 1; }
  for kind in wgsl glsl slang; do
    present=( *."$kind" )
    (( ${#present[@]} > 0 )) || continue
    case "$kind" in
      wgsl)  check_tool "$NAGA" "naga-cli" ;;
      glsl)  check_tool "$GLSLANG" "glslangValidator" ;;
      slang) check_tool "$SLANGC" "slangc" ;;
    esac
  done
  for f in *.wgsl; do compile_one "$f"; done
  for f in *.glsl; do compile_one "$f"; done
  for f in *.slang; do compile_one "$f"; done
fi

echo "Done. Output: $OUT_DIR"
