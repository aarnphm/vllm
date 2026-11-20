# sink-sentinel-followups: status (2026-11-20)

Worktree: ~/workspace/worktrees/vllm/sink-sentinel-followups
Branch: aarnphm/sink-sentinel-followups, base e82b7fca7e (upstream main). Not pushed.

| item | verdict | evidence |
| --- | --- | --- |
| 1. static-sink builder changes the shared seq_lens | REAL (fixed, 54206277ce) | CPU repro through `build_attn_metadata`: the batch seq_lens goes from [37, 100, 0] to [53, 116, 0]. The sampler's chunked-prefill predicate flips for a chunk ending 10 tokens short of its prompt. Group mixing and MTP are latent, because no shipped model mixes a sink group with another group. |
| 2. zero-layer PP stage divides by zero in the planner | REAL (fixed, 7ce91b3818) | `VLLM_PP_LAYER_PARTITION=4,0` is accepted. `get_kv_cache_configs` with an empty worker raises ZeroDivisionError at kv_cache_planning.py:674 (LBNHC) and AssertionError at :553 (BLHNC). No config check refuses it. |
| 3. host copy of a page the failed step wrote is recorded as stored | DESIGN | CPU repro (real Scheduler + OffloadingConnector + sentinel methods): `_clear_contaminated_blocks` evicts the GPU prefix-cache entries of both prompt blocks, then `_drain_batch_queue` delivers `complete_store(keys of those 2 chunks, success=True)`. The fix needs a new KV connector hook. |
| 4. weight reload vs the OpenPangu sink K under CUDA graphs | REAL, only with aarnphm/sink-block-reservation and FULL graphs (verdict only) | `load_weights` calls `post_weight_load`, which makes a new tensor (`k_layernorm(param_sink_key)`) and rebinds `attn.sink_key`. Nothing is copied in place. |

Status: fixed (items 1 and 2), design (item 3), verdict only (item 4).

## Item 1 detail

- Path: `default.py prepare_attn` passes `input_batch.seq_lens`, a view of `input_buffers.seq_lens`, to `build_attn_metadata`. That function puts a slice of it into each group's CommonAttentionMetadata, and the sink builder wrote `seq_lens[:] = seq_lens + sink_len`.
- Readers later in the same step: `sampler.py` (`get_num_sampled_and_rejected`: `seq_len < prefill_len`), `rejection_sampler.py:470`, the drafters (`target_dependent_ar/speculator.py:85, 557`, `standalone_ar`, `multi_module_mtp`) and later KV cache groups.
- Effect without spec decode: when a chunk ends within `sink_len` tokens of the prompt end, the chunk counts as finished. The runner then returns a token sampled mid-prompt, and the scheduler appends it as output.
- Why the rest is latent: OpenPangu picks MLA, sink or plain attention per model. All sink layers form one SINK_FULL_ATTENTION group. MTP is mapped only for `pangu_ultra_moe`, which uses MLA.
- Fix: a persistent `seq_lens_with_sink` buffer in the builder, `common_attn_metadata.replace(...)`, and `supports_draft_decode_metadata_update = False` (build() restages lengths, as Triton does for R-SWA).
- Merge note: aarnphm/sink-block-reservation (252e88eb30) also edits `StaticSinkAttentionBuilder.__init__` and `build`, so the two will conflict textually. Keep that branch's block size and sink ids, plus this commit's private buffer.

## Item 3 detail (neutral; no fix)

- Store candidates for a step include that step's scheduled tokens (`store_triggers/candidates.py:50-58`). The worker queues the jobs in `wait_for_save` and submits them at the start of the next step (`offloading/worker.py`).
- `on_fault` evicts only GPU prefix-cache entries. `_drain_batch_queue` and later steps still deliver store completions, and `_update_connector_output` calls `complete_store(success=True)` (`offloading/scheduler.py:1513-1555`).
- Under the recompute policy, preempting the requests also flushes their pending stores (`jobs_to_flush`).
- Fix options, both through the KV connector interface:
  - (a) Add a fault hook that receives the dirty block ids. The offloading scheduler marks pending store jobs sourced from those ids as failed, and their completions still count, so fences release.
  - (b) Reset the connector cache on fault. This drops the whole host tier, and `reset_cache` asserts that it is not called mid-schedule.
- #56430 (merged) added the drain and does not cover this. Open #50283 (NaN fault tolerance) would add a fault mode where kernels succeed but the KV values are bad.
- Security: none of my items was marked security-relevant. If item 3 is treated as one, I rate it LOW: it needs fault tolerance, KV offloading and an internal step failure, and I found no direct input path.

## Item 4 detail

- `weight_loader` updates `param_sink_key` and `param_sink_value` in place (`param_data.copy_`). The normalized K is a new RMSNorm output on every `post_weight_load`.
- The sink branch writes the sink inside `do_kv_cache_update`. PIECEWISE runs that op eagerly, because `unified_kv_cache_update` is a splitting op. FULL graphs, the decode default with FA3/FA4, capture it, so replays after a reload read the old tensor's memory.
- Kernel-format reload (`is_checkpoint_format=False`) never calls `post_weight_load`, so the normalized K keeps its pre-reload values. This also holds on base.
- Suggested change for the sink branch: have `update_sink_kv` copy into persistent sink tensors once they exist.

## Files changed

- 54206277ce: vllm/model_executor/layers/attention/static_sink_attention.py, tests/v1/worker/test_attn_utils.py
- 7ce91b3818: vllm/distributed/utils.py, tests/distributed/test_pipeline_partition.py

## Tests run (CPU, macOS venv; `-p noclean -p cpuplat`)

- `tests/v1/worker/test_attn_utils.py -k static_sink_builder_keeps_batch_seq_lens`: base `1 failed, 81 deselected, 1 warning in 0.28s`, fix `1 passed, 81 deselected, 1 warning in 0.31s`
- `tests/v1/worker/test_attn_utils.py`: base `70 passed, 11 skipped, 1 warning in 0.75s`, fix `71 passed, 11 skipped, 1 warning in 0.68s`
- `tests/distributed/test_pipeline_partition.py`: base `2 failed, 13 passed, 1 warning in 0.04s`, fix `15 passed, 1 warning in 0.02s`
- `tests/test_config.py -k "layer_range_with_prefill_context_parallel or hybrid_layer_counts"`: fix `5 passed`
- ruff check, ruff format and the pre-commit hooks (including mypy) passed on both commits.
- Repro outputs: item1_repro_{base,fix}.txt, item2_repro_{base,fix}.txt, item3_repro_base.txt. Scripts are in scripts/.

## GPU checks still needed

- Item 1, on SM90/SM100 with an OpenPangu checkpoint that sets `param_sink_number > 0`, default cudagraph mode, greedy, a 70-token prompt, `max_tokens=8`:
  - `vllm serve <ckpt> --trust-remote-code --max-num-batched-tokens 64` against `--max-num-batched-tokens 4096`.
  - Base: the outputs differ (the first token comes from mid-prompt).
  - Fix: the outputs are identical, and FULL decode graphs still replay correctly.
- Item 2 (optional, 2 GPUs): `VLLM_PP_LAYER_PARTITION=4,0 vllm serve <4-layer model> -pp 2` should fail at model construction with the ValueError, not after profiling.
