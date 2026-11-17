# fishstringer

The Slang home of the portfolio. The root crate `fishstringer` is a reserved name (0.0.0, `publish = false`) with no API; the work is in the members:

- `fishstringer-slang-emit`: the Slang emitter, including `Slang::lower_kernel` (IR kernel to Slang source; compiled with `slangc` in tests, not yet device-run).
- `fishstringer-slang-parse`: the Slang parser and IR lifter.
- `fishstringer-kernels-source`: the hand-written `.slang` kernels, `compile.sh`, `spv_equiv.py`, `regen_gate.py`.

What is verified, and how, is in `ROADMAP.md` and each crate's README.

Licensed under either of the Apache License, Version 2.0 (`LICENSE-APACHE`) or the MIT licence (`LICENSE-MIT`), at your
option.
