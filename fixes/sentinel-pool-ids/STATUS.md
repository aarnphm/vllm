# sentinel-pool-ids (bug 23): STATUS

status: fixed
branch: `aarnphm/sentinel-pool-ids` in ~/workspace/worktrees/vllm/sentinel-pool-ids (base `e2bab26e51`). Not pushed.
commit: `9d7dffc2639de8c3e058358a04547a229bc8914c` "[Bugfix][Core] Evict fault-contaminated blocks from the pool that owns them" (one commit, Signed-off-by aarnphm, no co-author trailer; pre-commit incl. mypy passed)

## Bug, confirmed at base (CPU)
- `EngineCoreSentinel._clear_contaminated_blocks` (`vllm/v1/fault_tolerance/engine_core_sentinel.py:231-255` at e2bab26e51) puts the failed step's block IDs from every manager into one `set[int]`, then calls `kv_cache_manager.evict_blocks`, which evicts from the device pool only (`kv_cache_manager.py:686-693` -> `block_pool.py:841-859`).
- `HiSparseSourceManager.bind_host_pool` (`single_type_kv_cache_manager.py:2842-2860`, called by `HiSparseCoordinator.__init__`, `vllm/v1/hisparse/coordinator.py:142`) gives the host group its own BlockPool. Its IDs start at 1, the same as the device pool's.
- No config guard rejects HiSparse with `--enable-fault-tolerance`.
- Consequence 1 (repro scenario 1 and the new test). Layout: device pool 32 blocks, host pool 16 blocks. A (1.5 pages) and B (3.5 pages) complete a step. The next step (A and B decode, C prefills 4 pages) fails. On base, B's cached device pages 1 and 2 (device ids 6 and 7) are evicted. The completed step wrote them, and B still holds them. Page 1 goes through B's own host id 6, page 2 through C's host id 7. `BlockRemoved(GPU)` events are published for B1 and B2. With the fix, only C0-C3 are removed.
- Consequence 2 (repro scenario 2). Layout: device pool 12 blocks, host pool 64 blocks. Three earlier requests finish with their host pages published, so the failed request gets host ids 13-16. On base, `fault_tolerant_wrapper` raises `AssertionError: Invalid block_id 13 >= 12 ...` out of `on_fault`. The sentinel status stays HEALTHY with 0 status pushes, so FT never engages and the engine core exits. With the fix, the status goes to UNHEALTHY with 2 pushes, and the wrapper waits for recovery. It re-raises only because the repro sets the recovery timeout to 0.
- Not shown: stale host KV being served. The failed step's host pages are normally unpublished at fault time, because the spill ack comes with that step's output.

## Fix
- Group the dirty IDs by `mgr.block_pool` (a `defaultdict[BlockPool, set[int]]`), then call `block_pool.evict_blocks(ids)` on each pool.
- With one pool (every non-HiSparse config, hybrid models included), the same IDs are evicted from the same pool as before.

## Files changed (+82 / -4)
- `vllm/v1/fault_tolerance/engine_core_sentinel.py` (+8/-4)
- `tests/v1/fault_tolerance/test_engine_core_sentinel.py`: new `test_fault_evicts_only_failed_pages_with_hisparse_host_pool` (real Scheduler + HiSparse KVCacheManager, drives `on_fault`)

## Tests run (CPU, macOS, one process at a time)
Prefix: `VLLM_TARGET_DEVICE=cpu PYTHONPATH=~/workspace/worktrees/vllm/sentinel-pool-ids:<scratch>/plug <scratch>/venv/bin/python -m pytest -p noclean tests/v1/fault_tolerance/test_engine_core_sentinel.py -q -p no:cacheprovider`

| tree | result line | log |
|---|---|---|
| base source (fix stashed) + new test | `1 failed, 3 passed, 1 warning in 0.21s`; the new test fails at `assert all(pool.get_cached_block(h, [1]) for h in b.block_hashes[:3])` | `pytest_base.log` |
| fix 9d7dffc263 | `4 passed, 1 warning in 0.17s` | `pytest_fix.log` |

- The 3 existing tests in the file pass both before and after. No other test file exercises the sentinel (`rg on_fault|EngineCoreSentinel|fault_tolerant_wrapper tests/`).
- Environment: without `VLLM_TARGET_DEVICE=cpu`, the new test errors with "Failed to infer device type". No vLLM package is installed for platform detection. This is environmental only; Linux CI detects the platform.
- Repro: `VLLM_TARGET_DEVICE=cpu PYTHONPATH=<worktree>:<plug> <venv>/bin/python repro_sentinel_pools.py`. Output is in `repro_base.log` / `repro_fix.log`, with noise-filtered copies in `*.trimmed.log`. Summary lines:
  - base: `SUMMARY clean_pages_lost=2 wrapper_raised=AssertionError`
  - fix: `SUMMARY clean_pages_lost=0 wrapper_raised=RuntimeError`
- Lint: `uvx ruff check` and `uvx ruff format --check` on both files passed.

## Duplicate check
- Open PRs: searched "sentinel evict_blocks", "fault tolerance contaminated blocks", "hisparse fault tolerance", "_clear_contaminated_blocks", "EngineCoreSentinel", "[FT]" and "fault tolerance in:title". None fixes this bug.
- I grepped the diffs of #65200, #68727, #64254, #60290, #41438, #62235 and #68132 for `evict_blocks|_clear_contaminated_blocks|dirty_block_ids|single_type_managers`. None touch this code.
- Issues: searched "fault tolerance evict host pool", "engine_core_sentinel contaminated" and "hisparse fault tolerance". Nothing related.
- Origin: the eviction came in #50265 (0ef2a95cc8) and the host pool in #53781 (d43bb2f37f).

## GPU checks still needed
- None strictly. The change is scheduler-side Python.
- Optional regression check of the single-pool recovery path: `pytest -v -s v1/fault_tolerance/test_fault_tolerance_e2e.py` (4xH100, from `tests/`, as in `.buildkite/test_areas/fault_tolerance.yaml`).
- There is no HiSparse + FT E2E test in-tree, so I could not run one. To see consequence 2 on a real model, run HiSparseConnector with `--enable-fault-tolerance`, a host pool larger than the device pool in blocks (both counts are in the "HiSparse HMA: ... host source (N blocks) ... pool (M blocks)" log line), and an injected step fault.

## Open (not verified)
- A spill of a page written by the failed step might still be acked later through `_drain_batch_queue`'s salvaged kv_connector_output, and that would publish the page to the host prefix cache. This is a separate HiSparse + FT question, outside the scope of this fix.
