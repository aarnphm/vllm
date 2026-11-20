# packed-grouping-zip (study bug 19)

status: fixed
branch: aarnphm/packed-grouping-zip (worktree ~/workspace/worktrees/vllm/packed-grouping-zip), base e2bab26e51
commit: e257d4b9e734bc976f42e2bc7d24b88c91e6359b (not pushed; all pre-commit hooks passed, mypy included)

## Files changed
- vllm/v1/core/kv_cache_planning.py (+9 -1): the state-bucket split in `_get_packed_kv_cache_groups` builds repeats with `zip_longest` (at most one layer of each page size per repeat) and caps `num_groups` at the number of repeats.
- tests/v1/core/test_contiguous_kv_packing.py (+27): `TestSlidingWindowBucketCap::test_unbalanced_bucket_split_places_every_layer[empty-group|dropped-layer]`.

## Bug at base (re-verified, `repro.py`, outputs `repro.*.out`)
- Sliding-window and circular-buffer buckets may mix page sizes; only Mamba requires one (`single_type_kv_cache_manager.py:1873`).
- Empty group: full 4 KiB + SWA 3x8 KiB + 1x4 KiB gives a bare `StopIteration` at `kv_cache_interface.py:713`, reached from `kv_cache_planning.py:1012`.
- Lost layer: full 24 KiB + SWA 3x8 KiB + 2x4 KiB. `get_kv_cache_groups` returns `[full.0] [swa.0 swa.3] [swa.1 swa.4]`, and `swa.2` gets no KV tensor. `get_global_kv_cache_groups` raises `AssertionError: KV cache groups miss layers: ['swa.2']`. Under `python -O` it is silent (`repro.base.O.out`).
- Balanced ring 2x8 KiB + 2x4 KiB: `StopIteration` too (`repro.ring.base.out`).

## Tests (CPU, macOS, venv per BRIEF; VLLM_TARGET_DEVICE=cpu)
- New test, base + test: `2 failed, 1 passed in 0.39s` ([empty-group] StopIteration, [dropped-layer] swa.2 missing)
- New test, fix: `3 passed in 0.31s`
- test_contiguous_kv_packing.py + test_kv_cache_planning.py: base `2 failed, 109 passed in 8.09s` (only the new cases), fix `111 passed in 7.72s`
- test_kv_cache_config_builder.py: base `15 passed in 1.38s`, fix `15 passed in 1.41s`
- Without VLLM_TARGET_DEVICE=cpu: 43 environment failures in test_kv_cache_planning.py (`RuntimeError: Failed to infer device type`), the same set at base and fix.
- ruff check / ruff format --check: clean. Peak RSS of the pytest runs: 567 MB.

## GPU checks still needed
- None. The change is pure-Python planning, and the test covers grouping plus tensor placement at the planner boundary.

## For aarnphm to decide
- I handled the mismatch rather than refusing it. The bucketing admits these buckets, and the anchor already budgets for one layer of each page size per group. Justified in PR.md.
- Follow-up, not fixed: the state cap counts every layer at the first layer's page size, so a mixed-page bucket can still widen the block (6x8 KiB + 1x4 KiB SWA next to a 16 KiB full page gives a 20480 B block where 16384 B would do; `widen.fix.out`). That is padding, not a lost layer. #60996, #66503 and #67171 rework this sizing.
- Not audited: five in-tree models build circular-buffer rings (deepseek_v41, glm5next, qwen4_exp, step5, hy_v4_flash/nvidia). I did not check each one's ring page mix. The fix leaves every bucket that groups correctly today unchanged, so the audit would only show who crashes on main today.
- PR.md's last line says aarnphm reviewed every changed line. That is true only after your review.
