## [Bugfix][Core] Size kernel block tables from the runner's block-table width

### Overview

Claude and I hit this while mapping how the worker sizes per-group block tables: the row width is computed twice, and the two formulas disagree for encoder-decoder models.

- The runner sizes `BlockTables` with `KVCacheGroup.block_table_width(vllm_config, max_len)`, where `max_len` is widened for encoder-decoder models because cross-attention tables index encoder tokens (`vllm/v1/worker/gpu/model_runner.py:710` on `e2bab26e51`).
- `AttentionGroup.create_metadata_builders` sizes `kernel_block_table` with `max_num_blocks_per_req(vllm_config)`, which uses the decoder `max_model_len` and a fixed 128-token alignment instead of the manager's (`vllm/v1/worker/utils.py:311` on `e2bab26e51`).
- With a kernel-block split (manager block larger than the backend's kernel block), the first metadata build then fails in `map_to_kernel_block_table`. Whisper-shaped: 24 manager blocks per row (1500 encoder positions, block 64), mapped into a 32-column kernel table where 96 are needed:
  `RuntimeError: unflatten: Provided sizes [24, -1] don't multiply up to the size of dim 1 (32) in the input tensor`

**IMPORTANT:** this is latent today. The backends that support `ENCODER_DECODER` (FA, Triton, AITER unified, CPU) take `MultipleOf(16)` or `MultipleOf(32)`, so they don't split at default block sizes. FA's SM90 FP8 path for head size 512 already takes only 64-token kernel blocks (`flash_attn.py:409-435`), but no encoder-decoder model has that head size. It fires once an encoder-decoder model lands on a backend with fixed kernel blocks.

### Fix

Claude proposes that `KVCacheGroup.block_table_width()` owns the width and every path asks it:

- `get_block_table_max_len(vllm_config)` (`vllm/v1/kv_cache_interface.py:286`) holds the encoder-decoder widening that used to live in both runners: `max(max_model_len, max_num_encoder_input_tokens, max_source_positions)`.
- `max_num_blocks_per_req` and `block_table_width` default `max_len` to it (`:792`, `:803`). The explicit `max_len` stays for warmup's cap and tests.
- The GPU runner (`gpu/model_runner.py:712`), the standalone CPU runner (`cpu/standalone_model_runner.py:374`) and `create_metadata_builders` (`worker/utils.py:310`) all call `block_table_width(vllm_config)`.

Side effect: builder widths now use the manager's block-table alignment, so kernel tables for managers with `block_table_token_alignment = None` (circular buffer, Mamba, HiSparse hot) shrink to the row they map instead of a 128-token-aligned superset. The one builder that takes `block_table_width` (`DeepseekV32IndexerMetadataBuilder`) sits on `FullAttentionManager` groups, so its width doesn't change.

_Note that two paths can still hand the owner a different `vllm_config` than the runner used:_ the standalone AR drafter with `--speculative-config.kv_cache_dtype` builds its builders from the draft model's config (its `max_model_len` can be smaller), and DeepSeek-V4.1's lazy replay-group copies run after an auto-fit `update_max_model_len`. Both were already exposed on main and need a kernel split to fire. Passing the runner's widths into `init_attn_backend` would close them too, but it touches the speculator, a model file and #59279's lines, so I kept this PR to the encoder-decoder bug.

### Validation

New test `tests/v1/worker/test_gpu_model_runner_v2.py::test_encoder_decoder_kernel_block_tables_cover_block_table_rows`: a real `openai/whisper-tiny` `VllmConfig` (448-token decoder, 1500 encoder positions), `GPUModelRunner.initialize_kv_cache` with a backend that only takes 16-token kernel blocks, then `build_attn_metadata` over the runner's full rows for the self-attention and cross-attention groups. It checks that every manager block `b` reaches the backend as kernel blocks `4b..4b+3`.

Four existing tests get `is_encoder_decoder=False` on their `SimpleNamespace` model configs, since the owner now reads it (`test_attn_utils.py`: `_VLLM_CONFIG`, `test_no_attn_metadata_layers_keep_their_layout_without_builders`; `test_gpu_model_runner_v2.py`: `test_qsa_circular_group_uses_custom_slot_mapping`, `test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table`).

Fail before, on `e2bab26e51` with only the test files from this PR (CPU, macOS):

```
pytest -p noclean tests/v1/worker/test_attn_utils.py tests/v1/worker/test_gpu_model_runner_v2.py -v
FAILED tests/v1/worker/test_gpu_model_runner_v2.py::test_encoder_decoder_kernel_block_tables_cover_block_table_rows
E   RuntimeError: unflatten: Provided sizes [24, -1] don't multiply up to the size of dim 1 (32) in the input tensor
============== 4 failed, 110 passed, 11 skipped, 1 error in 4.18s ==============
```

Pass after:

```
pytest -p noclean tests/v1/worker/test_attn_utils.py tests/v1/worker/test_gpu_model_runner_v2.py -v
tests/v1/worker/test_gpu_model_runner_v2.py::test_encoder_decoder_kernel_block_tables_cover_block_table_rows PASSED
============= 3 failed, 111 passed, 11 skipped, 1 error in 20.10s ==============
```

The other 3 failures and 1 error show up on unmodified `e2bab26e51` too (`3 failed, 110 passed, 11 skipped, 1 error in 6.71s`). They come from my Mac: `VllmConfig` without a device config can't infer a device (`test_resolve_hisparse_specs_before_kv_cache_layout`, `test_dummy_inputs_embeds_holds_random_input_id_embeddings[True|False]`), and `test_speculator_request_hooks` needs CUDA.
The width tests in `tests/v1/core/test_kv_cache_utils.py` and `tests/v1/worker/test_gpu_model_runner.py` (`-k "block_table_width or dcp_target_allocates or merge_preserves_replicated or late_interleave"`) give `13 failed, 2 passed` both before and after, all the same device-inference error. CI's "V1 Executor + Worker" step runs them for real.
I haven't run this on a GPU. The change is Python-only sizing, so I'm counting on that CI step.

### Duplicate check

- `gh pr list --state open --search` with "block table width encoder decoder", "map_to_kernel_block_table", "kernel_block_table cross attention", "block_table_width"; `gh issue list --search` with "block table width encoder decoder kernel block" and "unflatten don't multiply up whisper". No PR or issue covers this.
- #59279 ([Model][DCP] Support Qwen4Exp QSA, open) touches the same code another way: it threads the runner's widths into `AttentionGroup.block_table_width` and keeps the old formula as a fallback, for QSA's replicated layers. It doesn't target encoder-decoder, and it predates #61452 (it still calls `KVCacheSpec.max_num_blocks_per_req`), so the two will conflict in `create_metadata_builders`.

AI assistance (Claude) was used for this change; I reviewed every changed line and ran the tests above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
