I found this bug on my own while reading the KV cache manager, and this PR has the fix I would have sent.

It still reproduces on main at `e2bab26e51`. With FA block 16, DCP 2, 64 computed tokens and 8 lookahead tokens, `get_block_ids_for_computed_tokens` returns 3 blocks. Only 2 cover the computed tokens.

The conflict comes from #61452. That PR rewrote the attention check (`group.distinct_specs`, `group.manager_cls`) and changed the divisor to `group.block_size`, which is still the size before DCP.

The resolution I tested keeps the new check and divides by `manager.block_size`:

```diff
-        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
+        for group, manager, ids in zip(
+            self.kv_cache_config.kv_cache_groups,
+            self.coordinator.single_type_managers,
+            block_ids,
+            strict=True,
+        ):
             is_attention = all(
                 isinstance(spec, AttentionSpec) for spec in group.distinct_specs
             )
@@
-            num_valid_blocks = cdiv(num_computed_tokens, group.block_size)
+            num_valid_blocks = cdiv(num_computed_tokens, manager.block_size)
```

Your test needs no change, only a new position in `tests/v1/core/test_prefix_caching.py`.

`pytest tests/v1/core/test_prefix_caching.py::test_computed_block_ids_use_dcp_effective_block_size -q` (CPU, macOS):
- main: `assert ([1, 2],) == ([1],)`, `1 failed, 4 warnings in 4.96s`
- with the resolution: `1 passed, 4 warnings in 1.83s`

I also ran `test_prefix_caching.py`, `test_single_type_kv_cache_manager.py`, `prefix_cache/test_partial_prefix_cache_hits.py`, `test_remote_decode_lifecycle.py`, `test_remote_prefill_lifecycle.py` and `test_multi_connector.py`:
- main: `13 failed, 564 passed, 1 skipped in 23.10s`
- with the resolution: `12 failed, 565 passed, 1 skipped in 19.57s`
- The two failure sets differ only in your test. The other 12 are macOS environment failures on both sides. In seven, no device is detected. In five, `VllmConfig` rejects the hybrid KV cache manager on this platform.

With NIXL pull, the only effect is over-export: P sends and pins one extra block, and D's DCP pairing (`_apply_dcp_prefix_caching`) never reads it.
With NIXL push, the result is wrong KV. `_xfer_blocks` end-trims P's list to D's count (`_apply_prefix_caching`) and has no DCP case.
I gave NIXL's two pairing functions the real `KVCacheManager` outputs on CPU (P: DCP 2, block 16, 64 computed + 8 lookahead tokens; D: async load, cold and with a 1-block hit):
- push on main, cold D: P's tokens [32, 64) go to D's slot for [0, 32), and P's uncomputed lookahead page goes to D's slot for [32, 64)
- push on main, warm D: P's lookahead page goes to D's slot for [32, 64)
- pull on main: every pair is aligned
- with this PR: every push and pull pair is aligned
I have not run push with DCP and MTP on GPUs.

@jongukc, could you rebase? The diff above resolves the conflict.

I used AI assistance (Claude) to investigate and to write this comment, and I reviewed it.
