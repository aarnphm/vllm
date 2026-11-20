# nixl-dcp-group-index (bug 6) status

status: fixed
branch: aarnphm/nixl-dcp-group-index (worktree ~/workspace/worktrees/vllm/nixl-dcp-group-index), not pushed
base: e2bab26e51
commit: f85a7d53972553951e3c4b9bbaaf241caaef303a `[Bugfix][NIXL] Index DCP prefix hits by KV cache group` (author and Signed-off-by: Aaron Pham, no Co-Authored-By)

## Files changed
- vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+2/-1): the DCP branch reads `meta.local_num_computed_blocks[transfer_groups[g]]`, with `transfer_groups = self.kv_cache_config.transfer_group_ids`, the same mapping as the region branch.
- tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+75): `test_dcp_pull_skips_each_groups_own_prefix_hit`. A real `NixlPullConnectorScheduler` builds the request metadata (`update_state_after_alloc` + `build_connector_meta`) on the `region_pull_worker` fixture's `[resident (no transfer), host, indexer]` config, and the worker pulls from a DCP=4 prefill. The test asserts the (local, remote) descriptor pairs per remote rank.

## Bug re-verified at base
- pull_worker.py:416-418 at e2bab26e51 still reads `meta.local_num_computed_blocks[g]` with a transfer-group index; region branch at :316-319 maps it. Scheduler builds the tuple over all KV cache groups (pull_scheduler.py:163-172); `ReqMeta` field comment says per KV cache group.
- Repro script part B at base prints the same as the report (`untranslated (DCP path) : cached=0 -> D [13, 14, 15] <- P [100, 101, 102]`). Part B calls `_apply_dcp_prefix_caching` directly, so it does not change with the fix; the new test is the before/after evidence.
- Report correction (connectors.md B3): it says "No guard rejects HiSparse with DCP". `vllm/config/vllm.py:1962` (d33bfeef40:1991) rejects local DCP > 1 when HiSparse is on. The bug is reachable only as a HiSparse decode (DCP=1) pulling from a DCP > 1 prefill (`dcp_active` covers remote DCP; compat hash has no HiSparse or DCP factor), and only for regular (dense attention) transfer groups, which sit after the resident/hot groups. Source and indexer groups keep identical indices in both spaces. Also "Sparse-MLA DCP is still a draft upstream" is stale: deepseek_v32 has `supports_pcp_dcp = True` and flashmla_sparse / flashinfer_mla_sparse have `supports_dcp = True` at this base.

## Tests
New test, fail-before / pass-after:
- before (pristine e2bab26e51 source + new test, <scratch>/fixes/nixl-dcp-group-index/base2): `1 failed, 1 warning in 33.52s`, `At index 0 diff: (np.int64(31), np.int64(10)) != (34, 11)` (log: new-test-before2.log)
- after (f85a7d5397): `1 passed, 1 warning in 3.17s` (log: new-test-after2.log)
- contract check: the same test on f85a7d5397 with the scheduler switched to per-transfer-group counts (what open PR #55374 does; one-line `sd` edit in <scratch>/fixes/nixl-dcp-group-index/hazard): `1 failed, 1 warning in 7.03s`, `IndexError: tuple index out of range` at pull_worker.py:418 (log: hazard.log)

Existing suites, base e2bab26e51 vs 8086ed8eee (same fix, earlier test version), CPU macOS:
- test_nixl_connector_hma + test_nixl_desc_geometry + test_nixl_push_connector + test_nixl_heartbeat + test_tp_mapping: base `262 failed, 340 passed, 1 warning in 340.63s`; fix `262 failed, 341 passed, 1 warning in 360.80s`
- test_nixl_connector + test_nixl_remote_read_expiry (needs `ray`; added via `uv pip install --target overlay --no-deps ray==2.56.1 msgpack`, shared venv untouched): base and fix both `2 failed, 89 passed, 1 xfailed, 1 warning, 162 errors` (224.49s / 224.09s)
- failure sets identical by test id (compare_junit.py, compare.out): 426 bad each side, none new, none fixed; the only extra id is the new test (passes).
- all 426 are environment: 238 `assert self._is_hma_required` (HMA auto-disabled on non-GPU platforms, vllm/config/vllm.py:2314-2316), 162 `Failed to infer device type`, 3 `Device string must not be empty`, 23 desc_geometry `is not supported` / `Failed to promote local KV cache specs`.
- test_nixl_connector_hma.py on final f85a7d5397: `2 failed, 103 passed, 1 warning in 12.18s`; base `2 failed, 102 passed, 1 warning in 21.64s`; same 2 failures by id (compare-final-hma.out), both environmental (`Device string must not be empty`, `assert self._is_hma_required`)
- lint: `uvx ruff check` and `uvx ruff format --check` clean on both files; vLLM pre-commit (ruff, typos, mypy, check-test-tethering, signoff) passed on commit.

## GPU checks still needed
No GPU-only code changed; the CPU test covers the pairing. An end-to-end run is still missing, and it needs GPUs:
- Topology: HiSparse decode with DCP=1, NIXL pull, from a prefill with DCP=2 or 4; prefix caching on; send a prompt, then the same prompt plus a suffix, so that D has a local hit and pulls the rest.
- Model: it needs a HiSparse layout with a regular attention transfer group. Check at D startup that `kv_cache_config.transfer_group_ids != tuple(range(len(kv_cache_config.kv_cache_groups)))`. If they are equal, the bug cannot occur for that model. Candidate: GLM-5.x (`Glm5NextForCausalLM`), whose `Glm5NextTailCache` layers are dense `AttentionSpec`s with no `cache_role`, so they go to a regular group. I have not confirmed that NIXL accepts a GLM-5 DCP prefill with a HiSparse decode. Kimi K3 is out, because NIXL needs equal DCP for Mamba hybrids (base_worker.py:2943). DeepSeek V3.2 without dense layers is not affected (no regular group).
- Sketch (unverified flags, adapt to the slurm-lease harness):
  - P: `vllm serve nvidia/GLM-5.2-NVFP4 -tp 4 -dcp 4 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`
  - D: `VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/GLM-5.2-NVFP4 -tp 4 --attention-config '{"hisparse_config":{}}' --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":32}}]}}'`
  - Pass: greedy output of the second request matches aggregated serving (or a DCP=1 prefill) on main+fix; on main it should diverge.

## Decisions for aarnphm
- #55374 (open, needs-rebase, idle since 2026-10-31) moves `local_num_computed_blocks` to transfer space in the scheduler. That fixes this bug as a side effect, but it is a different contract. If it lands after this fix, it must drop the mapping this commit adds; the new test fails until it does.
- CI: the CPU step that runs `cpu_test` NIXL unit tests (`:computer: (CPU) V1 Others`) excludes `vllm/distributed/kv_transfer/` from its triggers. It runs for this PR because `tests/v1` changed, but a later `pull_worker.py`-only change would not run it. Not new; #63933 adds `pull_scheduler.py` back to that list, and `pull_worker.py` could join it.
- progress.md: kept in this scratch dir (STATUS.md) and not at the workspace root, because the brief limits writes to the worktree and scratch.
