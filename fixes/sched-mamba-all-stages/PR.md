[Bugfix][Core] Report Mamba layers held by any pipeline stage

## Overview

The scheduler gets worker 0's KV cache config (`generate_scheduler_kv_cache_config`, `vllm/v1/core/kv_cache_utils.py:1079-1093`), and `KVCacheConfig.has_mamba_layers` (`vllm/v1/kv_cache_interface.py:943-949`) reads the projected `kv_cache_specs`, which are empty on a stage that holds none of a group's layers.
So when PP stage 0 has no Mamba layer and a later stage does, the scheduler thinks the model has none. The planner already ORs `needs_kv_cache_zeroing` across workers for this reason (`kv_cache_planning.py:1599-1602`); this flag was left out.

Claude and I hit this while going through the KV cache group code. What reads the flag on the scheduler side:
- `Scheduler.need_mamba_block_aligned_split` and everything derived from it (`scheduler.py:424-455`): with prefix caching, prefills stop being split at Mamba block boundaries, so the reusable Mamba state is never written.
- the sync-scheduling publish of finalized decode states (`scheduler.py:2503-2517`).
- `KVCacheManager.get_computed_blocks_for_connector` (`kv_cache_manager.py:364-370`), which falls back to the reconciled lookup. Conservative, no harm.
- the NIXL and Mooncake scheduler connectors' Mamba prefill backoff (`nixl/base_scheduler.py:104, 395-410`; `mooncake/mooncake_connector.py:1054, 1112, 1154`).

Measured on CPU with the real planner and the real `Scheduler` (block size 16): a repeated 100-token prompt gets 0 cached tokens with Mamba only on stage 1, against 96 with PP=1 or with Mamba on stage 0.
For P/D I only read the code: a prefiller in this layout computes the whole prompt instead of stopping one token short, and the decoder then recomputes the last prompt token on a Mamba state that already includes it. That should be wrong output with no error.

**IMPORTANT:** I couldn't find a public checkpoint that reaches this today.
- All 43 hybrid configs I pulled from HF (22 architectures: Nemotron-H, Granite-4, Jamba, Qwen3-Next, Qwen3.5, Kimi-Linear, Kimi-K3, GLM-5.3-Flash, MiniMax-M1, LFM2/3, Falcon-H1, Zamba2, ZAYA, Bamba, OLMo-Hybrid, AliceAI, Ling-3, Ring-linear) have a Mamba-state layer at layer 0, so stage 0 always holds one.
- No in-tree drafter has Mamba layers (Nemotron-H MTP allows `*` and `E` only; K3, GLM-5 Next and Qwen3-Next DFlash drafters are attention).
- `VLLM_PP_LAYER_PARTITION` accepts a 0 for stage 0, but the planner then dies with `ZeroDivisionError` at `kv_cache_planning.py:674` before any scheduler exists.

So it takes an attention-first layout (a custom `layer_types` / `hybrid_override_pattern`, or an out-of-tree hybrid) or a drafter with Mamba layers on an attention-only target. It's cheap to close, and the P/D failure would be silent.

## Fix

`has_mamba_layers` reads each group's `distinct_specs`. A group projected onto a stage keeps the whole group's specs as `unprojected_specs` (`kv_cache_planning.py:1444-1451`), and `distinct_specs` is documented as the same on every pipeline stage; `KVCacheGroup.uses_slot_mapping` already reads it.

Workers on a stage without Mamba layers now report `True` too:
- Mooncake store: the worker's retention rewrite (`mooncake_store/worker.py:1758-1763`) now agrees with its scheduler side, which already counts Mamba groups by manager (`mooncake_store/scheduler.py:154-166`).
- Mooncake P2P worker: an attention-only stage takes the hybrid path in the head-count checks (`mooncake_connector.py:1862, 3354, 3419`), the same path a hybrid without PP already uses for those layers. `TransferTopology.is_mamba` (`:1584`) is not read anywhere.
- FULL cudagraph capacity check (`config/compilation.py:1717-1731`): stage 0 now raises too, but every stage has the same `num_blocks`, so the Mamba stage raised anyway.

_Note that I kept it a property._ A resolved field ORed in `get_kv_cache_configs` like `needs_kv_cache_zeroing` gives the same values with more code.

## Validation

CPU only (macOS). The prefix of each pytest command is `VLLM_TARGET_DEVICE=cpu PYTHONPATH=<tree> python -m pytest`.

New test `test_generate_scheduler_kv_cache_config_sees_mamba_on_later_stage` in `tests/v1/core/test_kv_cache_utils.py`: two stages through `get_kv_cache_configs`, stage 0 attention only, stage 1 attention + Mamba.

```
pytest "tests/v1/core/test_kv_cache_utils.py::test_generate_scheduler_kv_cache_config_sees_mamba_on_later_stage" -q
```

Before, on e2bab26e51:
```
>       assert scheduler_config.has_mamba_layers
E       AssertionError: assert False
E        +  where False = KVCacheConfig(num_blocks=64, ..., needs_kv_cache_zeroing=True).has_mamba_layers
1 failed, 1 warning in 1.57s
```
After, on 05c1b61e6e:
```
1 passed, 1 warning in 1.79s
```

Scheduler behavior (real planner + real `Scheduler`, request A then the same 100-token prompt as B):

| budget | layout | before: A chunks / B hit | after: A chunks / B hit |
| --- | --- | --- | --- |
| 8192 | PP=1 | [96, 4] / 96 | [96, 4] / 96 |
| 8192 | PP=2, Mamba on stage 0 | [96, 4] / 96 | [96, 4] / 96 |
| 8192 | PP=2, Mamba on stage 1 only | [100] / 0 | [96, 4] / 96 |
| 40 | PP=1 | [32, 32, 32, 4] / 96 | [32, 32, 32, 4] / 96 |
| 40 | PP=2, Mamba on stage 0 | [32, 32, 32, 4] / 96 | [32, 32, 32, 4] / 96 |
| 40 | PP=2, Mamba on stage 1 only | [40, 40, 20] / 0 | [32, 32, 32, 4] / 96 |

Nearby suites, same failures before and after:
- `tests/v1/core/test_kv_cache_utils.py`: `10 failed, 105 passed` before (9 env + the new test), `9 failed, 106 passed` after. The 9 are `test_num_hash_lookahead_tokens[*]`: "The CPU model runner without Triton does not yet support: speculative decoding".
- `tests/v1/core/test_kv_cache_planning.py`: `75 passed` before and after
- `tests/v1/core/test_prefix_lm_scheduler.py`: `12 passed` before and after
- `tests/v1/core/test_mamba_align_chunk_split.py`: `49 passed` before and after
- `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`: `1 failed, 103 passed` before and after (gated `google/gemma-3-1b-it` config, no HF token)

I haven't run a GPU e2e. I have one ready that mirrors #67994's `test_pipeline_parallel_mamba_only_stage_prefix_caching` the other way round: granite-4.0-tiny-preview with dummy weights, `layer_types` moved so layer 0 is attention, `VLLM_PP_LAYER_PARTITION=1,39`, comparing `num_cached_tokens` between PP=1 and PP=2. I'll add it here once it fails on main and passes with the fix on 2 GPUs.

## Duplicate check

- `gh pr list --repo vllm-project/vllm --state open --search` with "has_mamba_layers", "has_mamba_layers pipeline", "mamba pipeline parallel stage scheduler", "unprojected_specs OR distinct_specs" and "generate_scheduler_kv_cache_config": no open PR changes `has_mamba_layers` or the scheduler's projection for this. #68559 (UMBP PP prefill producers) reads `distinct_specs` in its own connector for the same reason. #65335 touches `generate_scheduler_kv_cache_config` on an older API, for offloading page sizes. #67994 (merged) fixed the Mamba block size for the opposite layout, a Mamba-only stage 0.
- `gh issue list --repo vllm-project/vllm --search` with "has_mamba_layers" and "mamba pipeline parallel prefix caching stage": nothing on this.

AI assistance was used (Claude Code). I reviewed every changed line and ran the commands above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
