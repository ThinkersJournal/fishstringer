kind: CI moves the CodeRipper portfolio workflow pin to dc45e363
---
The `portfolio` job now calls the shared workflow at `dc45e3632ab9f471292458e5ccba5f375084a70a` (was `c72a6e03`). That commit installs CodeRipper 0.47.15 (its `CODERIPPER_VERSION`); coderipper main is 0.47.21, so the pin is not the newest release. CI configuration only; no code or version change. The job stays advisory, and this repo's lock carries no cookie.
