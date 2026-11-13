kind: Slang declines peer, cluster and multicast places and multi-device launches
---
Completes ROADMAP row 19 / doc E18: `Slang::lower_kernel` declines `Place::ClusterPeer`, `Place::Peer`, `Place::Multicast`, `MulticastLoadReduce`, `devices > 1` and cluster launches with the `slang target gap: ` prefix. Tests: 3 new in `slang_target_gaps_are_typed_declines.rs` (8 pass), including the one-device control. Not covered: `Mma` sparse/scaled arms (no test builds an atom). Stacked on #23 (unpopped 0.45.12).
