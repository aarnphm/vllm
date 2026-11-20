## [Bugfix][Core] Evict fault-contaminated blocks from the pool that owns them

### Overview

Claude and I hit this while mapping which code assumes a single block pool. The fault-tolerance sentinel is one such place, and HiSparse breaks that assumption.

- `EngineCoreSentinel._clear_contaminated_blocks` walks every manager's `req_to_blocks` for the failed step's token range. It puts all the `block_id`s into one `set[int]` and calls `kv_cache_manager.evict_blocks(...)`, which evicts from the device pool only (`vllm/v1/fault_tolerance/engine_core_sentinel.py:231-255`, `vllm/v1/core/kv_cache_manager.py:686-693` on `e2bab26e51`).
- HiSparse's host group allocates from its own `BlockPool` (`HiSparseSourceManager.bind_host_pool`, `vllm/v1/core/single_type_kv_cache_manager.py:2842-2860`). Its IDs start at 1, the same as the device pool's, so every host ID is also the ID of some unrelated device block.

That gives two failures on a fault, both reproduced on CPU through the sentinel's own entry points:

- The failed step's host IDs evict device pages that the step never wrote, and the engine publishes `BlockRemoved(medium=GPU)` for them while their owner still holds them. In the repro, request B loses 2 of the 3 prompt pages that the completed step wrote. One goes through B's own decode-page host ID, and the other through request C's host ID.
- **IMPORTANT:** once a failed-step host ID is at least the device pool size, `BlockPool.evict_blocks` fails its assertion inside `on_fault`:
  `AssertionError: Invalid block_id 13 >= 12. This indicates a bug in the KV connector - workers should only report block IDs that were allocated by the scheduler.`
  `fault_tolerant_wrapper` calls `on_fault` inside its `except` block, so the assertion escapes. The engine core exits with status still HEALTHY and with no status push to the client, so fault tolerance never engages.

The planner sizes the two pools independently (`host_pool_gib` for the host pool, GPU memory for the device pool), and on a server that has been up for a while the host IDs spread over the whole host pool. So I'd expect the second failure on the first fault whenever the host pool has more blocks than the device pool. I haven't checked the block counts of a real HiSparse deployment, though.
Nothing rejects HiSparse together with `--enable-fault-tolerance`. `rg -i hisparse vllm/v1/fault_tolerance vllm/config/fault_tolerance.py vllm/config/parallel.py` finds nothing.

_Note that the host pages that the failed step wrote are usually not in the host prefix cache yet:_ HiSparse publishes a host page only after its spill is acked, and that ack comes back with the step's output. So I could not show stale host KV being served. The fix sends those IDs to the host pool anyway, which is where they belong.

### Fix

The sentinel now groups dirty IDs by `mgr.block_pool` and evicts each group from its own pool. `KVCacheCoordinator` already uses this rule for prefix lookups (`manager.block_pool`).

- With one pool, which is every non-HiSparse config including hybrid models, all managers share the coordinator's `BlockPool`. The same IDs are then evicted from the same pool as before.
- Host evictions go through the host pool's `evict_blocks`, as the HiSparse tests already do. Their `BlockRemoved` events carry `medium=CPU` and go to the shared event queue.

### Validation

New test `tests/v1/fault_tolerance/test_engine_core_sentinel.py::test_fault_evicts_only_failed_pages_with_hisparse_host_pool`. It builds a real `Scheduler` with a HiSparse `KVCacheManager` (32 device blocks, 16 host blocks) and runs two steps:

- Requests A (1.5 pages) and B (3.5 pages) prefill, and that step completes.
- In the next step, A and B decode inside their partial last pages and C prefills 4 pages. This step fails, and the test calls `on_fault` with `request_failure_policy="recompute"`.

The test then checks two things. First, B's 3 full pages are still in the device prefix cache, because the completed step wrote them. Second, all 4 of C's pages are evicted. The test also asserts its own precondition: the failed step's host IDs must overlap the IDs of B's cached device pages.

Fail before (the new test, run against `e2bab26e51`'s `engine_core_sentinel.py`):

```
VLLM_TARGET_DEVICE=cpu pytest -p noclean tests/v1/fault_tolerance/test_engine_core_sentinel.py -q
>       assert all(pool.get_cached_block(h, [1]) for h in b.block_hashes[:3])
E       assert False
FAILED tests/v1/fault_tolerance/test_engine_core_sentinel.py::test_fault_evicts_only_failed_pages_with_hisparse_host_pool
1 failed, 3 passed, 1 warning in 0.21s
```

Pass after:

```
VLLM_TARGET_DEVICE=cpu pytest -p noclean tests/v1/fault_tolerance/test_engine_core_sentinel.py -q
4 passed, 1 warning in 0.17s
```

The 3 existing tests in that file pass both before and after.
I need `VLLM_TARGET_DEVICE=cpu` on my Mac because there's no installed vLLM for platform detection. CI's fault-tolerance step already runs this file.

For the crash, a CPU repro script drives `fault_tolerant_wrapper` around a busy loop that raises. It uses 12 device blocks and 64 host blocks, and three earlier requests finish with their host pages published, so that the failed request's host IDs come out at 13-16:

```
base: wrapper raised AssertionError: Invalid block_id 13 >= 12. ...
      sentinel status HEALTHY; status pushes to the client: 0
fix:  wrapper raised RuntimeError: step failed
      sentinel status UNHEALTHY; status pushes to the client: 2
```

With the fix, the wrapper only re-raises the original error because the repro sets `engine_recovery_timeout_sec=0` and sends no recovery instruction.

I haven't run this on a GPU. The change is scheduler-side Python. The FT E2E step (`v1/fault_tolerance/test_fault_tolerance_e2e.py`, 4xH100) covers the single-pool path. There is no HiSparse + FT E2E test in-tree.

### Duplicate check

- `gh pr list --state open --search` with "sentinel evict_blocks", "fault tolerance contaminated blocks", "hisparse fault tolerance", "_clear_contaminated_blocks", "EngineCoreSentinel", "[FT]" and "fault tolerance in:title". None of these PRs fixes this bug.
- I grepped the diffs of the open FT PRs that touch the sentinel or nearby code for `evict_blocks|_clear_contaminated_blocks|dirty_block_ids|single_type_managers`: #65200, #68727, #64254, #60290, #41438, #62235, #68132. None of them touch this code.
- `gh issue list --search` with "fault tolerance evict host pool", "engine_core_sentinel contaminated" and "hisparse fault tolerance" found nothing related.
- The eviction comes from #50265 and the separate host pool from #53781.

One open doubt that I haven't checked: a spill of a page that the failed step wrote might still be acked later, through the batch-queue salvage in `_drain_batch_queue`, and that would publish the page to the host prefix cache. That is a separate HiSparse + FT question, outside the scope of this PR.

AI assistance (Claude) was used for this change; I reviewed every changed line and ran the tests above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
