# No new PR: bug 17 is already fixed by the third hunk of vllm-project/vllm#54599

The fix Claude would have sent is the same 12-line `unify_kv_cache_spec_page_size` hunk that #54599 carries. vLLM's duplicate rule says don't open a second PR for it. Below are the fix-choice notes and a comment I could post on #54599.

## Overview

`--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers <full-attention index>` on a sliding-window hybrid fails `get_kv_cache_configs` with a bare `AssertionError` (`kv_cache_interface.py:512`). Gemma 3 with layer 5 and gpt-oss with layer 1 both hit it.
- The bf16 skip layer's page is 2x the fp8 shared page, so `unify` doubles the block size of every fp8 sliding-window spec but keeps their absolute `page_size_padded` (`attention.py:722`, `kv_cache_planning.py:381`).

## Fix

Scale `page_size_padded` by the same ratio as `block_size` in unify's block-scaling branch. This is #54599's third hunk, and `_ensure_min_page_size` already does the same thing.
- Claude also weighed refusing the config at startup. That seems worse: main already plans the sibling case (a sliding-window skip index) into the same plan shape, and a refusal would also block TurboQuant's boundary skips when they land on full-attention layers.
- **IMPORTANT:** the plan that results has one KV cache group per layer: 62 for Gemma-3-27B (7 without the skip) and 36 for gpt-oss-120b (2 without). Main already does this for a sliding-window skip index. #58207 looks like the place to fix it.

## Validation

Planner-level only, CPU, base `e2bab26e51`. `repro_planner.py` builds a real `ModelConfig` from a Gemma-3-27B or gpt-oss-120b config.json and runs the real platform alignment, `Attention.get_kv_cache_config` and `get_kv_cache_configs`. It then admits, frees and re-admits one 4000-token prompt through `KVCacheManager`.

```
VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 PYTHONPATH=<worktree> python repro_planner.py gemma3 5
PLANNING FAILED: AssertionError: ... kv_cache_interface.py:512

PORT_54599=1 ... python repro_planner.py gemma3 5
PLANNED: 62 groups
attention bytes/token/layer (dtype, bytes): {('torch.uint8', 4096.0): 61, ('torch.bfloat16', 8192.0): 1}
MANAGER OK: scheduler_block=64 hash_block=32 ... second_hit=3968

PORT_54599=1 ... python repro_planner.py gptoss 1
PLANNED: 36 groups ... MANAGER OK ... second_hit=3968
```

- I haven't run it on a GPU. On FA3 the skipped bf16 group probably also needs #65891.

## Duplicate check

- `gh pr list --state open --search` for "kv-cache-dtype-skip-layers", "skip layers sliding window fp8", "unify_kv_cache_spec_page_size page_size_padded", "kv_cache_dtype_skip_layers gemma", "page_size_padded". The search found #54599 (same hunk), #59178 (phase-3 pricing, different), #65891 (FA3 runtime, different) and #58207 (group sizing).
- `gh issue list --state all` for "kv-cache-dtype-skip-layers", "skip layers AssertionError page_size_padded" and "fp8 skip layers sliding window AssertionError" found no issue for this bug.

## Draft comment for #54599

> Claude and I hit your third hunk's assert from a different direction, without TurboQuant: `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers <full-attention index>` on a sliding-window hybrid (Gemma 3 layer 5, gpt-oss layer 1) dies in `get_kv_cache_configs` with the same bare `AssertionError`.
> - fp8 is exactly half of bf16, so the skipped full-attention layer's page is 2x the shared page. unify then takes the block-scaling branch for every fp8 sliding-window layer, and those layers carry `page_size_padded=skip_page_size_padded` from `Attention.get_kv_cache_config`.
> - Ported to current main (the function now lives in `vllm/v1/core/kv_cache_planning.py`), your hunk alone fixes it. Gemma-3-27B geometry plans, the fp8 layers keep 4096 B/token, and `KVCacheManager` re-hits a 4000-token prompt for 3968 tokens, the same as `--kv-cache-dtype-skip-layers sliding_window`.
>
> Would you be up for splitting that hunk into its own small bugfix PR, or rebasing this one? It doesn't depend on the TurboQuant pricing changes, so it seems easy to land by itself. I'm happy to contribute a planner-level regression test for it.
>
> _Note that_ the resulting plan puts each layer in its own KV cache group (62 for Gemma-3-27B). Main already does that for a sliding-window skip index, and #58207 looks like the place to fix it.

AI assistance (Claude) was used for the investigation and these drafts; I reviewed them.
