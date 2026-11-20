# encdec-block-table-width (bug 12): STATUS

status: fixed
branch: `aarnphm/encdec-block-table-width` in ~/workspace/worktrees/vllm/encdec-block-table-width (base `e2bab26e51`). Not pushed.
commit: `8e71f1f2fd7f28aab056177ffb873a7771cb8deb` "[Bugfix][Core] Size kernel block tables from the runner's block-table width" (one commit, Signed-off-by aarnphm, no co-author trailer)

## Bug, confirmed at base
- Runner: `BlockTables` rows = `g.block_table_width(vllm_config, block_table_max_model_len)`. For encoder-decoder models it widens to `max(max_model_len, max_num_encoder_input_tokens, max_source_positions)` (`vllm/v1/worker/gpu/model_runner.py:710-724` at e2bab26e51; the same code is in `cpu/standalone_model_runner.py:365-384`).
- Builders: `kernel_block_table` width = `get_block_table_width(kv_cache_group.max_num_blocks_per_req(vllm_config), spec.block_size)`. That uses the decoder `max_model_len` and a fixed 128-token alignment (`vllm/v1/worker/utils.py:311-312` at e2bab26e51).
- Study §2 repro at base (`scripts/repro_section2.py`, output in `repro_section2.base.out`): BlockTables width 24, kernel table `(1, 4, 32)`, 96 columns needed, `RuntimeError: unflatten: Provided sizes [24, -1] don't multiply up to the size of dim 1 (32)`.
- Latent: Triton and AITER unified take `MultipleOf(16)`, CPU takes `MultipleOf(32)`, and FA takes `MultipleOf(16)` except its SM90 FP8 path for head size 512 (fixed 64), which no encoder-decoder model hits.

## Fix
`KVCacheGroup.block_table_width()` is the one owner.
- `get_block_table_max_len(vllm_config)` (`kv_cache_interface.py:286`) holds the encoder-decoder widening that used to live in both runners.
- `max_num_blocks_per_req` and `block_table_width` default `max_len` to it. An explicit `max_len` still works, for warmup's cap and the GPU kpool test.
- The GPU runner, the standalone CPU runner and `AttentionGroup.create_metadata_builders` all call `block_table_width(vllm_config)`.
- Side effect: builder widths now use the manager's alignment. For managers with `block_table_token_alignment = None`, kernel tables shrink to exactly r x the row width. The only builder that takes `block_table_width` (`DeepseekV32IndexerMetadataBuilder`) sits on `FullAttentionManager` groups (alignment 128), so its width is unchanged. The ring-buffer tail builders (`KpoolTailMetadataBuilder`, HY-V4 `IndexPoolMetadataBuilder`) take no width.

## Files changed (+159 / -32)
- `vllm/v1/kv_cache_interface.py`: new `get_block_table_max_len`; default `max_len` on `max_num_blocks_per_req` and `block_table_width`
- `vllm/v1/worker/utils.py`: `create_metadata_builders` uses `kv_cache_group.block_table_width(vllm_config)`; drops the unused import
- `vllm/v1/worker/gpu/model_runner.py`: local widening removed
- `vllm/v1/worker/cpu/standalone_model_runner.py`: local widening removed
- `tests/v1/worker/test_gpu_model_runner_v2.py`: new `test_encoder_decoder_kernel_block_tables_cover_block_table_rows`; `model_config` with `is_encoder_decoder=False` on the stub configs of `test_qsa_circular_group_uses_custom_slot_mapping` and `test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table`
- `tests/v1/worker/test_attn_utils.py`: `is_encoder_decoder=False` on `_VLLM_CONFIG` and on `test_no_attn_metadata_layers_keep_their_layout_without_builders`

## Tests run (CPU, macOS, one process at a time)
Prefix: `HF_HUB_OFFLINE=1 PYTHONPATH=<tree>:<scratch>/plug <scratch>/venv/bin/python -m pytest -p noclean <files> -v --tb=line -W ignore`
Files: `tests/v1/worker/test_attn_utils.py tests/v1/worker/test_gpu_model_runner_v2.py`

| tree | result line | log |
|---|---|---|
| base e2bab26e51 (`git archive`), base tests | `3 failed, 110 passed, 11 skipped, 1 error in 6.71s` | `base.near.out` |
| base source + this branch's test files | `4 failed, 110 passed, 11 skipped, 1 error in 4.18s`; the extra failure is the new test: `RuntimeError: unflatten: Provided sizes [24, -1] don't multiply up to the size of dim 1 (32)` | `base_with_new_tests.near.out`, `test.base.out` |
| fix 8e71f1f2fd | `3 failed, 111 passed, 11 skipped, 1 error in 20.10s`; new test PASSED | `fix.near.out` |

- The same 4 fail before and after, for environment reasons:
  - `test_resolve_hisparse_specs_before_kv_cache_layout` and `test_dummy_inputs_embeds_holds_random_input_id_embeddings[True|False]`: "Failed to infer device type". `VllmConfig` without a device config cannot detect a platform here.
  - `test_speculator_request_hooks`: ERROR, because its fixture needs CUDA.
- The 4 fixture edits are only needed by the change. With the fix and the old fixtures, the run had 9 `AttributeError`s (stub configs without `model_config.is_encoder_decoder`; `fix_prefixture.out`). With the new fixtures, those tests pass on base and on the fix.
- `tests/v1/worker/test_gpu_model_runner.py tests/v1/core/test_kv_cache_utils.py -k "block_table_width or dcp_target_allocates or merge_preserves_replicated or late_interleave"`: base `13 failed, 2 passed`, fix `13 failed, 2 passed`. The failure sets are identical, all "Failed to infer device type" (`base.more.out`, `fix.more.out`).
- `scripts/repro_section2_fixed_api.py` at the fix: kernel table `(1, 4, 96)`; manager block 95 maps to kernel blocks 380-383 (`repro_section2.fix.out`).
- Lint: `uvx ruff check` and `uvx ruff format --check` pass on all 6 files. The pre-commit hooks at commit (ruff, mypy, typos, check-test-tethering, signoff) passed.
- CI: the "V1 Executor + Worker" step (`.buildkite/test_areas/misc.yaml`, `pytest -v -s v1/worker`, depends on `vllm/v1/`) selects the new test.

## GPU checks still needed
None are needed to prove the fix: it changes Python sizing only, and the regression test runs on CPU. Recommended on a GPU node (slurm-lease), where platform detection works:
1. The 4 tests that only fail from the environment, plus the width tests:
   `pytest -v -s tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_attn_utils.py`
   `pytest -v -s tests/v1/core/test_kv_cache_utils.py -k "block_table_width or dcp_target_allocates or merge_preserves_replicated"`
2. Whisper end-to-end regression. The Whisper block-table widths do not change, and FA does not split blocks, so no kernel tables are built:
   `pytest -v -s tests/models/multimodal/generation/test_whisper.py -k "test_models and not distributed"`

## For aarnphm to decide
- Design: this branch makes the group the owner (`block_table_width(vllm_config)`). Two paths can still pass that owner a different `vllm_config` than the runner used, and both already behave this way on main:
  - The standalone AR drafter with `speculative_config.kv_cache_dtype` builds with the draft model's config, whose `max_model_len` can be smaller.
  - DeepSeek-V4.1's lazy `_copy_attn_groups` can run after an auto-fit `update_max_model_len` shrink.
  Both need a kernel split to fail. Passing the runner's widths into `init_attn_backend` closes them, but it touches the speculator and a model file and collides with #59279.
- Open PR #59279 (QSA DCP, last updated 2026-11-11) threads runner widths into `AttentionGroup.block_table_width` and keeps the old formula as a fallback. It conflicts with this branch in `create_metadata_builders` and predates #61452.
- `progress.md` was not updated, because the brief limits writes to the worktree and the scratch dir.
