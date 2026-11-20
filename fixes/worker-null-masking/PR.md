# [Bugfix][Core] Keep freed R-SWA gap and PCP window blocks out of attention

## Overview

Claude and I found this while auditing how KV cache groups hand freed blocks back to the worker. R-SWA frees blocks in the scheduler that the worker still hands to attention.

- `RSWAManager.remove_skipped_blocks` frees the blocks between the prompt and the decode window and swaps them for the null block in `req_to_blocks` (`single_type_kv_cache_manager.py:1035-1065`). The worker never sees that swap: `CachedRequestData.new_block_ids` only appends.
- The worker's gather nulls freed entries only by `sliding_window_reach` (#60767), and R-SWA inherited 0, so its forward-pass table keeps the freed ids. Once the pool hands a freed id to another request, the row points at that request's page.
- FA4 gets the R-SWA `mask_mod` and no block-sparse tensors, so it loads every KV tile from block 0. `use_fast_sampling` only matters to `compute_block_sparsity`, which we never call ([`block_info.py:43-50`](https://github.com/vllm-project/flash-attention/blob/1b32c5caf71f5183248bdeda0855d4669b1d8744/flash_attn/cute/block_info.py#L43-L50), [`flash_fwd_sm100.py:1647-1700`](https://github.com/vllm-project/flash-attention/blob/1b32c5caf71f5183248bdeda0855d4669b1d8744/flash_attn/cute/flash_fwd_sm100.py#L1647-L1700)). Triton unified attention turns tile pruning off under R-SWA and zeroes V only for an ordinary sliding window.
- The mask is exact, so finite bytes from the other request get zero weight. A reallocated page whose bytes decode as NaN or Inf still poisons the row through `0 * NaN` in `P @ V`. That is the same failure #60767 fixed for ordinary sliding windows.
- The `RSWAManager` docstring claimed the table FA4 sees is valid because the null block is all-zero. That only holds for the scheduler's copy.

PCP had the same gap for ordinary sliding windows. `PCPManager.prepare_attn` and the DFlash and target-dependent AR draft gathers called `gather_block_tables` without the request cursor. #60767 left CP out because "a block-table column is not a contiguous token range there". That holds for DCP, but with DCP = 1 every PCP rank keeps the full KV, so the columns are plain token ranges. DCP > 1 still turns masking off inside `BlockTables`.

_Note that this is a correctness fix: I haven't seen NaN from this in a serving run, and the default Unlimited-OCR v1 setup (BF16 KV, one attention spec) needs bytes from another dtype or a NaN encoding before the output goes wrong._

## Fix

The worker now nulls exactly what the scheduler can have freed, in the same gather launch.

- `SingleTypeKVCacheManager.sliding_window_keeps_prompt()` is a new manager fact, exposed as `KVCacheGroup.sliding_window_keeps_prompt`. `RSWAManager` reports `True`, with reach `rswa_window`, which is the boundary its own `remove_skipped_blocks` frees below. `W - 1` would be exact for the mask, but it collides with the `reach == 0` "no limit" sentinel at `W = 1`.
- `_gather_block_tables_kernel` nulls `[cdiv(prompt_len, bs), (first_pos - reach) // bs)` for those groups and keeps the leading cutoff for the others. A new constexpr keeps the prompt-length loads out of the common path.
- The runner passes `req_states.prompt_len.gpu`. Prompt-keeping groups assert it, so a caller can't silently null a prompt.
- The CPU fork does the same, because R-SWA now reports a nonzero reach.
- Under PCP, the runner gives `PCPManager` the request cursor at KV init. `prepare_attn` and the two draft gathers pass it.
- Slot mappings still read the persistent tables, so no write moves.

**IMPORTANT:** FA4 still walks the gap tiles. They now read the zero null page instead of someone else's. Skipping those tiles needs block-sparse tensors, which is a bigger change than this bug fix.

## Validation

Head `4a97a7e42c`, base `e2bab26e51`. Everything below ran CPU-only on my Mac. The GPU runs go through a separate harness, see the end of this section.

The behavioral check is a scratch script, `rswa_gap_check.py` (I'll attach it as a gist). It drives the real `RSWAManager` and the real worker table. Row A decodes past its window, row B, a fresh prefill, takes A's reclaimed pages from a pool with no spares, and the table is built the way each tree's runner builds it:

```
PYTHONPATH=<tree> python rswa_gap_check.py --device cpu
```

<details><summary>base vs this PR, block size 16 (64 and 128 behave the same)</summary>

```
# base e2bab26e51
block_size=16 tree=base reach=0 prompt_a=40 window=128 decode_pos_a=213
  scheduler freed row-A columns [3, 4]
  row B (new prefill) took reclaimed ids [4, 5] of [5, 4]
  row A forward table: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  row A stale reclaimed entries {3: 4, 4: 5}; ids shared with B [4, 5]
RESULT: BUG REPRODUCED   (exit 1)

# this PR
block_size=16 tree=fix reach=128 prompt_a=40 window=128 decode_pos_a=213
  scheduler freed row-A columns [3, 4]
  row B (new prefill) took reclaimed ids [4, 5] of [5, 4]
  row A forward table: [1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  row A stale reclaimed entries {}; ids shared with B []
RESULT: clean   (exit 0)
```
</details>

The CPU block-table tests can't run on macOS as-is: the file skips off the CPU platform, and `gather_block_table_cpu` is a compiled op. So I ran them with a scratch plugin that swaps in a torch copy of `csrc/cpu/block_table.cpp` and lifts the skip:

```
pytest -p cpu_gather_stub_plugin tests/v1/worker/cpu/test_cpu_block_table.py -k "gather or shared_block_tables or external_block_table"
base vllm + new test:  1 failed, 2 warnings in 0.10s   (AttributeError: 'KVCacheGroup' object has no attribute 'sliding_window_keeps_prompt')
base:                  4 passed, 5 deselected, 2 warnings in 0.10s
this PR:               5 passed, 5 deselected, 1 warning in 0.09s
```

Existing CPU tests near the change give the same results on both trees:

```
pytest tests/v1/worker/test_gpu_pcp_manager.py -k "rebind or capture_uses or dummy_context or prepare_attn_nulls"
base:     3 passed, 92 deselected, 2 warnings in 4.09s
this PR:  3 passed, 1 skipped, 92 deselected, 1 warning in 2.81s   (new GPU test skipped)
pytest tests/v1/core/test_single_type_kv_cache_manager.py -k "rswa or sliding_window"
base:     5 passed, 65 deselected, 3 warnings in 0.86s
this PR:  5 passed, 65 deselected, 1 warning in 0.23s
pytest tests/v1/core/test_single_type_kv_cache_manager.py   (this PR)
70 passed, 1 warning in 3.29s
```

Lint: `ruff check`, `ruff format --check`, the docstring hook and the mypy hook (`Success: no issues found in 9 source files`) are clean.

Still to run on a GPU, which I'm queuing on GB300:
- the new `test_gather_block_tables_nulls_rswa_gap_the_manager_freed` and `test_prepare_attn_nulls_blocks_below_sliding_window`, fail-before and pass-after;
- `rswa_gap_check.py --device cuda`, which fills row B's pages with NaN and then with finite values and runs row A's decode through FA4 (the `mask_mod` path) and Triton against a dense reference. Gap nulling is the control.

I'll paste those logs here before this leaves draft.

## Duplicate check

- `gh pr list --state open` for "RSWA", "R-SWA gap block table", "unlimited ocr", "gather_block_tables sliding window", "PCP sliding window block table", "pcp_manager num_computed_tokens" and "stale block ids worker block table freed": nothing fixes this. #64253 (DSv4.1 SWA decode metadata) and #65437 (stale row tails on shrinking `add_row`) touch nearby code for other reasons.
- `gh issue list` for "RSWA" and "unlimited ocr NaN": nothing related.
- #60767 (merged) is the precedent this extends to R-SWA and PCP. #42276 (open, stalled) tries the scheduler-side alternative for ordinary sliding windows, resending the rewritten table.

---

AI assistance (Claude) was used to trace the bug, write the change, the tests and this description. I reviewed every changed line and the results above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
