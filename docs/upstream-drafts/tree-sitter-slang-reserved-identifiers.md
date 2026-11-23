# DRAFT (not posted): tree-sitter-slang 0.3.1 rejects `var`, `is` and `out` as identifiers

Status: draft only. Nothing is posted on any third-party repo without CireSnave's go.
Measured 2026-11-22 with `tree-sitter-slang =0.3.1` (latest on crates.io), `tree-sitter 0.27.0`,
through `fishstringer_slang_parse::parse_slang`. Not yet checked against upstream `main`.

## Title
Parse errors when `var`, `is` or `out` is used as an ordinary identifier

## Repro (each parses without error when the identifier is renamed)
Prefix for every case:
```slang
StructuredBuffer<float> input0;
RWStructuredBuffer<float> output;
[numthreads(256, 1, 1)]
void copy(uint3 tid : SV_DispatchThreadID) {
```
| body | `root.has_error()` |
|---|---|
| `uint i = tid.x; output[i] = input0[i];` (positive control) | false |
| `float var = 1.0; output[0] = var;` | **true** |
| `float v = 1.0; output[0] = v;` (renamed control) | false |
| `uint is = 2u * tid.x; output[is] = 1.0;` | **true** |
| `uint k = 2u * tid.x; output[k] = 1.0;` (renamed control) | false |

Buffer named `out`: `RWStructuredBuffer<float> out; [numthreads(256,1,1)] void copy(uint3 tid : SV_DispatchThreadID) { out[tid.x] = 1.0; }` -> has_error **true**; same text with `out` renamed `dst` -> false.

## Why it matters
Real kernels use these names: unpopped measured 13 entry points (8 `var`, 1 `is`, 4 `out`).
`var` and `is` are contextual in Slang (not reserved in declarations of locals), and `out` is a
parameter modifier only in parameter position, so they should parse as identifiers elsewhere.
(Slang compiler acceptance of these sources is the unpopped measurement; not re-run by this lane.)

## To do before any post
Recheck upstream `main`; confirm slangc accepts the repros; get CireSnave's go.

— lane: fishstringer
