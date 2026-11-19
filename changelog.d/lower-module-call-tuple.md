kind: `Slang::lower_module` lowers `CallTuple` and callees with several results (E16, third slice)
---
A callee with `KernelFn::results` becomes a Slang function returning a struct `R<index>` (fields `v0, v1, …`, the first being `result`), and `Stmt::CallTuple` unpacks it into the named locals. A plain `Call` of such a callee takes `.v0`. A results list of the wrong length is a typed decline. slangc-compiled in tests, not device-run. Allocation arguments and `Asm`/`Symbol` extern bodies stay named declines.
