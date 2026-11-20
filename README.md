# KV cache group study: notes and fix branches

This branch holds notes only. They come from a study of vLLM's KV cache group machinery at upstream `d33bfeef40` (2026-11-19) and from the fixes that followed, based on upstream `e2bab26e51` (2026-11-20). Each fix is its own branch in this fork. No PR is open yet.

- `kv-group-pain-points.md`: the study. It covers the root causes, the bugs found, numbers from the real planner, the GPU checks and the fix status table.
- `fixes/<branch>/PR.md`: the draft PR description for branch `aarnphm/<branch>`. `STATUS.md`, where present, has the test commands and their result lines.
- `upstream/<name>/`: bugs that already have an open upstream PR, with the analysis and draft review comments.

| Branch | Bug | Diff |
| --- | --- | --- |
| `aarnphm/sink-block-reservation` | 2: static-sink blocks reused as request KV; the sink writer's view layout | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/sink-block-reservation) |
| `aarnphm/nixl-conv-spec-mismatch` | 3: P/D conv state offsets with different draft lengths | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/nixl-conv-spec-mismatch) |
| `aarnphm/mooncake-mamba-slots` | 4: Mooncake Mamba state slot pairing | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/mooncake-mamba-slots) |
| `aarnphm/worker-null-masking` | 5, 22: freed R-SWA and PCP window blocks read by attention | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/worker-null-masking) |
| `aarnphm/nixl-dcp-group-index` | 6: NIXL pull DCP prefix hits indexed by transfer group | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/nixl-dcp-group-index) |
| `aarnphm/encdec-block-table-width` | 12: block-table width computed in two places | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/encdec-block-table-width) |
| `aarnphm/cross-attn-prefix-cache` | 13: prefix caching on encoder-decoder models | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/cross-attn-prefix-cache) |
| `aarnphm/mla-stale-page-padding` | 16: stale MLA page padding after a spec rewrite | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/mla-stale-page-padding) |
| `aarnphm/backend-per-kind-keys` | 18: `backend_per_kind` keys that never take effect | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/backend-per-kind-keys) |
| `aarnphm/packed-grouping-zip` | 19: packed grouping truncated by `zip` | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/packed-grouping-zip) |
| `aarnphm/sched-mamba-all-stages` | 20: scheduler Mamba flag from pipeline stage 0 only | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/sched-mamba-all-stages) |
| `aarnphm/sentinel-pool-ids` | 23: fault sentinel evicts host ids from the device pool | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/sentinel-pool-ids) |
| `aarnphm/chunked-local-vbatch` | 24: chunked-local virtual batches overflow FlashInfer and FA3 buffers | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/chunked-local-vbatch) |
| `aarnphm/k3-fp8-prefill-check` | Kimi-K3 fp8 KV cache without an fp8 prefill query, refused at startup | [compare](https://github.com/aarnphm/vllm/compare/e2bab26e51...aarnphm/k3-fp8-prefill-check) |

These open upstream PRs already fix other bugs from the study and need a rebase or a review: #57358 (bugs 1, 7), #49965 (8), #60045 and #66725 (9), #65690 (10), #63691 (11), #54599 (17), #67855 (NUMA binding race), #58851 (Mamba2 aligned state indices).

Some notes stay private: internal cluster details, and items under responsible disclosure.
