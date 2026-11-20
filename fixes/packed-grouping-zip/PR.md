# [Bugfix][Core] Keep every layer when packed grouping splits a state bucket

## Overview

Claude and I found this while auditing how vLLM forms KV cache groups. No in-tree model is known to hit it yet.

`_get_packed_kv_cache_groups` splits a state bucket into groups that fit the block, and it built that split from `zip(*page_size_layers.values())`, which stops at the shortest page-size list.
Sliding-window and circular-buffer buckets may mix page sizes (only `MambaManager.can_share_block_table` requires one, `single_type_kv_cache_manager.py:1873`), so on a block-outermost layout:
- With unequal counts, the layers past the shortest list are dropped. `get_kv_cache_groups` returns without them and they get no KV tensor. `get_global_kv_cache_groups` catches it only through the `assert` in `_check_exact_partition`, which is gone under `python -O`.
- When the state cap asks for more groups than `zip` made repeats, a group comes out empty and `KVCacheGroup.from_layers({})` raises a bare `StopIteration` (`kv_cache_interface.py:713`). A balanced mixed-page circular-buffer bucket hits this as well.

**IMPORTANT** Users can reach this from their own config: `validate_kv_cache_layout` tells a model with mixed page sizes to `set VLLM_KV_CACHE_LAYOUT=BLHNC` (`kv_cache_planning.py:632`), which routes it into the packed path.

## Fix

Claude proposes that we split these buckets instead of refusing them:
- The bucketing admits them, and the anchor already budgets for them: `widest_group_bytes` with one group per state sizes the block for a split down to one layer of each page size per group. So the only broken step is the `zip`, and a refusal would reject a bucket that the rest of the planner accepts.
- The split now builds repeats with `zip_longest`. Each repeat holds at most one layer of each page size, and the group count is capped at the number of repeats. No group can be empty and no layer can be dropped.
- For balanced buckets `zip_longest` gives the same repeats as `zip`, and any split that works today already has no more groups than repeats. So every bucket that groups correctly today gets the same groups.

_Note that the state cap still counts every layer at the first layer's page size_. A mixed-page bucket that is split into fewer groups than repeats can therefore come out wider than the anchor. With six 8 KiB and one 4 KiB sliding-window layer next to a 16 KiB full-attention page, the block is 20480 B where 16384 B would do (on main this shape raises `StopIteration`). That costs padding and loses no layers. #60996, #66503 and #67171 are reworking this sizing, so I left it alone.

## Validation

CPU only, on macOS. I set `VLLM_TARGET_DEVICE=cpu` because platform detection fails on a source checkout that has no installed package. Without it, 43 tests in `test_kv_cache_planning.py` fail with `RuntimeError: Failed to infer device type`, both on main and with the fix.

The new test is `TestSlidingWindowBucketCap::test_unbalanced_bucket_split_places_every_layer`. It builds one full-attention layer and five sliding-window layers that share a window (three 8 KiB pages, two 4 KiB pages) and checks that every layer gets exactly one KV tensor. The full-attention page controls how far the cap splits the bucket: a 4 KiB page gives `[empty-group]` and a 24 KiB page gives `[dropped-layer]`.

```bash
VLLM_TARGET_DEVICE=cpu python -m pytest "tests/v1/core/test_contiguous_kv_packing.py::TestSlidingWindowBucketCap" -q
```

Before (main `e2bab26e51` plus the new test):

```
vllm/v1/core/kv_cache_planning.py:1012: in _get_packed_kv_cache_groups
vllm/v1/kv_cache_interface.py:713: in from_layers
E   StopIteration
tests/v1/core/test_contiguous_kv_packing.py:644: AssertionError: assert ['full', 'swa...a.3', 'swa.4'] == ['full', 'swa...a.3', 'swa.4']
      At index 3 diff: 'swa.3' != 'swa.2'
FAILED tests/v1/core/test_contiguous_kv_packing.py::TestSlidingWindowBucketCap::test_unbalanced_bucket_split_places_every_layer[empty-group]
FAILED tests/v1/core/test_contiguous_kv_packing.py::TestSlidingWindowBucketCap::test_unbalanced_bucket_split_places_every_layer[dropped-layer]
2 failed, 1 passed in 0.39s
```

After:

```
3 passed in 0.31s
```

Nearest test files, before and after:

```bash
VLLM_TARGET_DEVICE=cpu python -m pytest tests/v1/core/test_contiguous_kv_packing.py tests/v1/core/test_kv_cache_planning.py -q
VLLM_TARGET_DEVICE=cpu python -m pytest tests/v1/core/test_kv_cache_config_builder.py -q
```

| | packing + planning | config builder |
|---|---|---|
| main + new test | `2 failed, 109 passed in 8.09s` (the two new cases) | `15 passed in 1.38s` |
| this PR | `111 passed in 7.72s` | `15 passed in 1.41s` |

A standalone repro (`get_global_kv_cache_groups`, then `get_kv_cache_groups` and tensor placement):

| shape | main | this PR |
|---|---|---|
| full 4 KiB; SWA 3 x 8 KiB + 1 x 4 KiB | `StopIteration` | `[full.0] [swa.0 swa.3] [swa.1] [swa.2]`, block 12288 B |
| full 24 KiB; SWA 3 x 8 KiB + 2 x 4 KiB | `AssertionError: KV cache groups miss layers: ['swa.2']`; under `-O` the groups come back silently and `swa.2` has no KV tensor | `[full.0] [swa.0 swa.3 swa.2] [swa.1 swa.4]`, block 24576 B |
| full 4 KiB; ring 2 x 8 KiB + 2 x 4 KiB (balanced) | `StopIteration` | `[full.0] [ring.0 ring.1] [ring.2 ring.3]`, block 12288 B |

In each fixed row the block stays at the size the anchor planned.

## Duplicate check

- `gh pr list --repo vllm-project/vllm --state open --search "packed kv cache groups zip"`, the same with `"_get_packed_kv_cache_groups"`, and `gh issue list --search "StopIteration kv cache group packed"`: none of them fixes this.
- #60996 and #66503 refactor the same split, but both keep `zip(*page_size_layers.values())` as it is. #67171 (packed state anchor) and #60967 (packed Mamba sizing) change other lines of the same function. Whichever lands first, the others need a small rebase.

AI assistance (Claude) was used to find, reproduce and fix this. I reviewed every changed line and ran the tests above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
