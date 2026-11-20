## [Bugfix][Core] Keep static-sink KV blocks reserved, written and valid

### Overview

Claude and I found that static-sink models have been reading the wrong KV as their sink, in several ways at once.

- `StaticSinkAttention` (OpenPangu with `param_sink_number > 0`, e.g. `openpangu/openPangu-R-7B-2512` and `FreedomIntelligence/openPangu-R-72B-2512`, 128 sink tokens each) keeps each layer's learned sink KV in blocks `1..sink_len // block_size`. `StaticSinkAttentionBuilder` prepends those ids to every block table row.
- **Nothing reserves those blocks.** #53941 removed the free-queue pop in `SinkFullAttentionManager.__init__` as dead code, but the pop was the reservation.
  - The pool hands blocks 1..k to requests, and the V2 warmup hands them to its synthetic requests too.
  - Any request that holds them overwrites the sink every later request reads.
- **The sink was never written into those blocks.**
  - `populate_sink_kv` passed the layer's `[blocks, kv_heads, block_tokens, K | V]` view straight to `triton_reshape_and_cache_flash_diffkv`, which expects `[blocks, block_tokens, kv_heads, C]`. The backend's own writer passes `kv_cache.transpose(1, 2)` (`flash_attn_diffkv.py:124`).
  - The kernel took `kv_heads` as the block size, so the sink landed at `slot // kv_heads`: inside request pages, or out of bounds.
- **It was written once.** The extensible KV cache, on by default on CUDA, backs only block 0 at allocation and remaps blocks as zeroed pages before graph capture and on wake. `sink_populated` stayed True through all of it.
- **The worker used another block size.** The builder and the writer used `CacheConfig.block_size`, but the planner grows a group's blocks when it equalizes page sizes with a larger-page group (`unify_kv_cache_spec_page_size`).

The output is wrong in every case, and nothing errors.

### Fix

- `BlockPool.reserve_blocks(block_ids)` takes fixed-id blocks out of the free queue for the pool's lifetime.
  - `reset_prefix_cache` and `get_usage` count them like the null block.
  - Restoring only the old pop would leave `reset_prefix_cache` returning False forever, and `sleep(level>=1)` would then raise.
- `KVCacheManager` reserves the sink's blocks in the device pool right after building it. Host pools that CPU offload builds from the same config don't reserve, since they only hold copies of request blocks.
- **One geometry.** `SinkFullAttentionManager.sink_block_ids` defines the ids, always in the group's resolved block size:
  - the scheduler reserves in it, from the group;
  - the builder prepends in it, from the `kv_cache_spec` it is built with;
  - the writer writes in it, from the cache view it writes.
  - Kernel-block splitting can't occur for FlashAttention DiffKV. `select_common_block_size` keeps the manager size whenever the backend accepts it, and DiffKV accepts any multiple of 16. The one fixed-size FA case is FP8 on SM90, which DiffKV refuses.
- Startup refuses two setups with a `ValueError`, both before warmup:
  - Sink layers split into several KV cache groups. Groups alias the same bytes, so their sinks would overwrite each other. `first_request_block_id` raises, and the planner calls it before allocation. Stock OpenPangu, MTP included, is one group; a drafter with plain attention can split it.
  - Decode context parallelism. Each rank writes `sink_len // block_size` blocks, while DCP-scaled scheduler blocks would reserve fewer. `StaticSinkAttention.__init__` raises when the model is built.
- `first_request_block_id(kv_cache_groups)` also feeds the capacity checks, warmup, the extensible cache floor and `vllm bench forward`.
  - Capacity: `check_enough_kv_cache_memory`, the auto-fit and admission check in `get_kv_cache_configs`, the post-warmup `max_model_len` fit, and `shrink_kv_cache_configs` hold back the sink blocks with the null block.
- **The sink write rides on the KV update.** The static-sink backend subclasses its impl. `do_kv_cache_update` writes the sink through the backend's own writer (`write_sink` → `super().do_kv_cache_update`, so the same transpose and scales), then the request's KV.
  - Ordering: `unified_kv_cache_update` returns the dummy that `unified_attention_with_output` consumes, so the sink is written before this step's attention, also under torch.compile.
  - Lifetime: it runs every step and CUDA graphs replay it. A remap, a wake, or a connector that fills the cache is healed by the next step.
- `ExtensibleKVCache.min_committed_blocks` covers the sink blocks, so the write never targets unbacked pages.

Why every step rather than only after a remap: a remap-only write needs hooks at the end of startup, `extend_kv_cache`, wake/restore and every weight-update path. It also can't run inside decode replays of graphs captured without it. The every-step write costs one `reshape_and_cache` of `sink_len` tokens per sink layer per step: 27 layers (28 with MTP) × 128 tokens on R-7B, about 16 MB of writes. I haven't measured it yet.

_Note that_ a weight reload that replaces the sink K tensor, rather than updating it in place, leaves captured graphs writing the old tensor (`post_weight_load` builds `k_layernorm(param_sink_key)` anew). That needs an in-place refresh or a recapture; it is a follow-up.

### Validation

CPU-only Mac, so nothing below ran a model or the Triton writer.

```
pytest -q tests/v1/core/test_prefix_caching.py -k static_sink
pytest -q tests/v1/core/test_kv_cache_utils.py::test_capacity_check_holds_back_static_sink_blocks
pytest -q tests/model_executor/layers/test_static_sink_attention.py
```

At e2bab26e51:
- `test_static_sink_blocks_are_never_allocated`: `E assert [1, 2, 3, 4, 5, 6, ...] == [3, 4, 5, 6, 7, 8, ...]`
- `test_static_sink_layers_in_several_groups_are_refused`: `E Failed: DID NOT RAISE ValueError`
- `test_capacity_check_holds_back_static_sink_blocks`: `E Failed: DID NOT RAISE ValueError`

At 82966074ba (this PR's second commit):
- `test_static_sink_blocks_stay_out_of_offload_host_pools`: `E IndexError: list index out of range` in `reserve_blocks`.
- `test_sink_read_ids_are_the_blocks_the_device_pool_keeps`: `E assert [1, 2] == [1]`. The group resolved to 32-token blocks while `CacheConfig.block_size` is 16.
- `test_decode_context_parallelism_is_refused_when_the_model_is_built`: the layer didn't refuse, so construction went on into backend selection.

This PR: all pass. The layer test file gives `2 passed, 2 skipped`, the skips being the GPU test off CUDA.

I also ran the allocation test against a variant that only restores the old pop. It hands out the right blocks but fails on accounting: `assert 0.19999999999999996 == 0.0` on `manager.usage`.

```
pytest -q tests/v1/worker/test_gpu_warmup_blocks.py   # CUDA-gated; I ran it with a local plugin
# e2bab26e51:  E assert not ({1, 2, 3, 4, 5, 6, ...} & {1, 2})   1 failed, 50 passed
# this PR:     -k "static_sink or rank_agreed or too_few_blocks or capped_by_the_committable
#                  or reserves_lookahead_blocks or sizing_warmup"   29 passed, 22 deselected
```

The plugin passes the file's `is_cuda_alike()` gate and turns `torch.accelerator.synchronize()` into a no-op; the tests drive warmup through a stub runner.

Existing tests on the reset, usage, warmup-capacity and bench paths, with the new ones: `3 failed, 23 passed`. The 3 failures are `test_scheduler.py` reset tests with `RuntimeError: Failed to infer device type` on macOS; they fail the same way on e2bab26e51.

`pre-commit run --files` on every changed file passes, mypy and the CI-tethering check included.

**IMPORTANT**: the byte layout and graph replay need a GPU, and I haven't run them. `test_sink_write_lands_in_sink_blocks_and_replays`, in `tests/model_executor/layers/test_static_sink_attention.py`, needs SM90+ with FA3/FA4.
- With 1 and 4 KV heads it writes a request beside the sink into a canary-filled cache and checks every byte.
- Then it clears the sink, replays a captured `do_kv_cache_update`, and checks again.
- An engine-level check on openPangu-R-7B-2512 is pending too.

### Duplicate check

- Open PRs, searched for `static sink`, `SinkFullAttentionManager`, `sink blocks reserved`, `openpangu sink`, `sink block reservation`, `reserve_blocks block pool`. Issues (all states), searched for `static sink`, `openpangu sink`, `SinkFullAttentionManager`, `sink kv overwritten`. Nothing fixes this.
- #59512 (open) changes the same constructor to accept `needs_kv_cache_zeroing`. Main already takes `**kwargs` there.
- #47271 (open, idle since 2026-08-19) passes the sink KV to the kernel instead of keeping it in KV blocks, which would delete this path.
- #53696 (merged) described a sink reservation and a reset fix, but that KV-manager part was dropped in review.
- Upstream main at 673618e0a4 (2026-11-20) still has these problems.

AI assistance (Claude Code) was used for the investigation, the fix and the tests. I reviewed every changed line and ran the tests above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
