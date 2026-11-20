## Overview

Claude and I found this while reading how the KV connectors index KV cache groups.

The NIXL pull worker's DCP branch reads `ReqMeta.local_num_computed_blocks` with a transfer-group index.
That tuple is per KV cache group: the scheduler builds it over every group in `blocks.blocks` (`pull_scheduler.py:163-172`), and the comment on the `ReqMeta` field says so.
`local_block_ids`, `_group_spec_types` and `_dcp_replicated_groups` are all per transfer group, so `g` in that loop is a transfer-group index.
The region branch of the same function already maps the index through `transfer_group_ids` (`pull_worker.py:316-319`). The DCP branch did not (`pull_worker.py:416-418`).

The flow, for a request that is partly cached on D:
1. D's scheduler allocates blocks and finds a local prefix hit. `update_state_after_alloc` sends the uncached block ids per transfer group (`local_block_ids`) and the hit per KV cache group (`local_num_computed_blocks`).
2. P prefilled with DCP=4, so page p of a group is block p // 4 on P rank p % 4.
3. D's worker (`_read_blocks_for_req`, DCP branch) walks the transfer groups and, for each P rank, calls `_apply_dcp_prefix_caching(..., local_num_computed_blocks=hit)`, which starts the pairing at page `hit`.
4. If `hit` comes from another group, every uncached page of that group pairs with the wrong P page, and decode attends to that KV.

The two index spaces only differ when a group that does not transfer comes before one that does.
Today only HiSparse makes such groups: its layout is `[source, indexer, resident*, hot*, regular*]`, and resident/hot do not transfer.
When the indices differ, a DCP pull starts a group's slice at another group's prefix hit, so D's uncached pages read P's pages at the wrong positions, including pages D already has cached.
That is silent wrong KV, with no error.

How reachable this is today, as far as I can tell from the code (I have not run it):
- HiSparse refuses local DCP > 1 at config time (`vllm/config/vllm.py:1962`), so the decode side must be DCP=1.
- A HiSparse decode can still pull from a DCP prefill: `dcp_active` is also true when only the remote has DCP > 1, the compat hash includes neither HiSparse nor DCP, and none of the handshake's DCP checks (sizes divide, equal logical block sizes) look at HiSparse.
- HiSparse decode with a NIXL pull is a real deployment: #63811 runs GLM-5.3 decode with `MultiConnector[NixlConnector, HiSparseConnector]` and prefix caching on. That issue does not say whether the prefill used DCP.
- The source and indexer groups keep the same index in both spaces. Only the regular groups (dense attention layers next to the sparse ones) shift, so the model needs one of those.
- So I would call it latent: DSA model with a dense attention group, HiSparse decode, DCP prefill, and a local prefix hit.

## Fix

Claude proposes to map the index the same way the region branch does: `meta.local_num_computed_blocks[transfer_groups[g]]`.
It is two lines in `pull_worker.py`. The scheduler, the metadata and the region branch do not change.

## Validation

New test: `test_dcp_pull_skips_each_groups_own_prefix_hit` in `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`.
It uses the KV cache config of the `region_pull_worker` fixture, whose groups are `[resident (no transfer), host, indexer]`. The resident group has no cached pages, host has 1 and indexer has 2.
A real `NixlPullConnectorScheduler` runs `update_state_after_alloc` and `build_connector_meta` for that request, and the worker runs `_read_blocks_for_req` on the result against a DCP=4 prefill.
It checks that every uncached local page reads the same page on P.
Because the counts come from the scheduler and not from the test, the test also fails if the scheduler and the worker stop agreeing on which index the counts use.

All runs are CPU-only on macOS (`cpu_test`), with the transport mocked as in the other tests that use this fixture.
The new test ran on commit `f85a7d5397` (this PR) and on its base `e2bab26e51`.

```bash
PYTHONPATH=$PWD python -m pytest \
  "tests/v1/kv_connector/unit/test_nixl_connector_hma.py::test_dcp_pull_skips_each_groups_own_prefix_hit" -q
```

**Before** (base `e2bab26e51` with only the test added, i.e. `git checkout e2bab26e51 -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`):

```
E           assert [(np.int64(31...p.int64(121))] == [(34, 11), (144, 121)]
E             At index 0 diff: (np.int64(31), np.int64(10)) != (34, 11)
E             Left contains one more item: (np.int64(145), np.int64(121))
1 failed, 1 warning in 33.52s
```

On remote rank 0, D's block 31 (page 1, not cached) reads P's block 10, which holds page 0. D already has page 0 cached.
(That run also logged a failed Hub metadata fetch for `facebook/opt-125m`, which `create_vllm_config` makes. I think that is where the extra 30 s went, but I did not time it.)

**After:**

```
1 passed, 1 warning in 3.17s
```

Existing NIXL unit tests, base vs. this branch, same machine. These ran on `8086ed8eee`, which has the same source change and an earlier version of the new test:

```bash
python -m pytest -q tests/v1/kv_connector/unit/test_nixl_connector_hma.py \
  tests/v1/kv_connector/unit/test_nixl_desc_geometry.py \
  tests/v1/kv_connector/unit/test_nixl_push_connector.py \
  tests/v1/kv_connector/unit/test_nixl_heartbeat.py \
  tests/v1/kv_connector/unit/test_tp_mapping.py
# base: 262 failed, 340 passed, 1 warning in 340.63s (0:05:40)
# PR:   262 failed, 341 passed, 1 warning in 360.80s (0:06:00)

python -m pytest -q tests/v1/kv_connector/unit/test_nixl_connector.py \
  tests/v1/kv_connector/unit/test_nixl_remote_read_expiry.py
# base: 2 failed, 89 passed, 1 xfailed, 1 warning, 162 errors in 224.49s (0:03:44)
# PR:   2 failed, 89 passed, 1 xfailed, 1 warning, 162 errors in 224.09s (0:03:44)
```

The failure sets are identical (426 failed or errored tests on each side, compared by test id). The one extra pass is the new test.
On the final commit `f85a7d5397`, `test_nixl_connector_hma.py` alone gives `2 failed, 103 passed, 1 warning in 12.18s` (base: `2 failed, 102 passed, 1 warning in 21.64s`), with the same two failures.
All of them come from my Mac, not from this change:
- 238 hit `assert self._is_hma_required` in the worker's init, because vLLM turns the hybrid KV cache manager off on non-GPU platforms.
- 162 errors are `Failed to infer device type`, and 3 more are `Device string must not be empty`.
- 23 in `test_nixl_desc_geometry.py` are `RuntimeError:  is not supported.` or `Failed to promote local KV cache specs to one unified type.`

CI: the `:computer: (CPU) V1 Others` step runs `-m cpu_test v1/kv_connector/unit`, so it picks up the new test, and it triggers on this PR because `tests/v1` changed.
That step excludes `vllm/distributed/kv_transfer/` from its triggers, so a later change to `pull_worker.py` alone would not run it. This is not new, and #63933 adds `pull_scheduler.py` back to the same list.

I have not run this end to end. That needs a HiSparse decode pulling from a DCP prefill on GPUs, with a model that has a dense attention group next to its sparse layers.

## Duplicate check

I searched open PRs for `nixl dcp local_num_computed_blocks`, `nixl dcp transfer group`, `hisparse nixl dcp`, `transfer_group_ids nixl`, `nixl dcp prefix caching`, and issues for the same terms. No PR or issue reports this bug.

#55374 (piecewise prefix loading with NIXL) changes the same field in a different way: it builds `local_num_computed_blocks` per transfer group in the scheduler and drops the mapping in the region branch.
That fixes this bug as a side effect, but the PR is a large feature, does not mention this bug, has merge conflicts (`needs-rebase`), and has not been updated since 2026-10-31.
If #55374 lands after this PR, it must also drop the mapping that this PR adds, or the DCP branch would map the index twice. The new test then fails, because it takes the counts from the scheduler.
I checked this by making the scheduler count per transfer group on top of this PR: the test fails with `IndexError: tuple index out of range` at `pull_worker.py:418`.

#63933 changes how the scheduler counts the cached blocks, but it keeps them per KV cache group, so it does not interact with this fix.

AI assistance (Claude) was used to find the bug, write the fix and the test, and run the checks. I reviewed every changed line.
