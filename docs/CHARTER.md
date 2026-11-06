# fishstringer charter

**Ruling (CireSnave 2026-11-06, relayed by the PM, verbatim):** *"I believe FishStringer is a better home for Slang.
Lets move all Slang code that isn't specific to one of the other projects into FishStringer.  Lets get all Slang
kernels moved into FishStringer and removed from Fuel and Vulkane.  FishStringer is the owner of Slang."*

## Scope

FishStringer owns the **high-level GPU kernel languages that have no backend of their own** (Triton, Mojo and others
are parked; **Slang is first**), as three kinds of artefact:

1. **Kernel sources** (`.slang`, `.slangh` and the scripts that compile them).
2. **Parsers** (language source to Unpopped IR) and **emitters** (Unpopped IR to language source).
3. Anything else about the language that is not specific to one consuming project.

Not ours: GLSL, HLSL, WGSL and SPIR-V (Vulkane's); CUDA (Baracuda's); the `.glsl` kernel files that sit beside Slang
ones in Fuel and Vulkane (Vulkane's: listed in `SLANG_INVENTORY.md`, not moved); Unpopped's IR; Fuel's and Vulkane's
dispatch and runtime code, which consume our kernels.

## Crates (names to be claimed at 0.0.0 by the PM, who publishes)

| Crate | Holds | Source today |
|---|---|---|
| `fishstringer` | reserved (0.0.0, `publish = false`); to become the umbrella or stay a name | this repo |
| `fishstringer-sys` | reserved | PM's reservation |
| kernel-source crate (name to be chosen, see the plan) | the Slang kernels and compile scripts | `fuel-kernels-source`, `vulkane-kernels-source` |
| Slang emitter | IR to Slang | `vulkane-slang-emit` |
| Slang parser | Slang to IR | `vulkane-slang-parse` |

Naming follows Baracuda's `<owner>-<lang>-emit` / `-parse` rule (Vulkane's design note, D1). The final names are an
`[ASK]` to the PM before anything is claimed.

## Rules this repo follows

- Every pushed change changes the version; the PM allocates the number at gate time. One version series across all
  crates here.
- Third-party provenance, licences and notices travel with the files (CireSnave 2026-09-17: *"I am not a plageurist."*).
  A moved file keeps its history where it can (path-limited history import, with a blob-identity check against the
  source commit).
- Dependencies are crates.io versions, latest at all times; no path or git dependencies across repos.
- The lane writes and opens PRs; the PM merges, allocates versions and publishes.
- A duplicate or conflicting file is never resolved silently: it is listed and asked (`[ASK]`).

## Order of work

See `SLANG_INVENTORY.md` section 6 (move plan). Nothing is deleted from Fuel or Vulkane by this lane; their lanes
switch to our published crates and delete their own copies.
