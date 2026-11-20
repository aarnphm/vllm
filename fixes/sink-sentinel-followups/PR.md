# PR draft 1: [Bugfix][Attention] Keep static-sink lengths out of the batch seq_lens

Commit: 54206277ce on aarnphm/sink-sentinel-followups (base e82b7fca7e).

## Overview

The static-sink metadata builder added the sink length to `common_attn_metadata.seq_lens` in place.
In the GPU model runner that tensor is a view of the step's `input_batch.seq_lens`.
Everything that reads `input_batch.seq_lens` later in the same step therefore saw `seq_len + sink_len`.
The sampler's chunked-prefill check is one of those readers: `is_chunked_prefilling = seq_len < prefill_len`.
So a prefill chunk that ends fewer than `sink_len` tokens before the end of its prompt counts as finished, and the request returns a token sampled from the middle of its prompt.
I think this affects any OpenPangu checkpoint with `param_sink_number > 0` that runs with chunked prefill, with no draft model and no second KV cache group.
I could not test a real checkpoint: the registry lists `PanguProMoEV2ForCausalLM` without a public repo.
The rejection sampler and the drafters read the same tensor.
A KV cache group built after the sink group got the inflated lengths together with the original `max_seq_len`.
No shipped model mixes a sink group with another group, so that last effect is latent today.

## Fix

The builder writes the sink lengths into a persistent buffer of its own, and padded rows stay zero.
It passes a replaced `CommonAttentionMetadata` to the underlying builder, so the caller's object and tensor are left as they were.
The buffer is persistent, so FULL CUDA graphs read the same address on every replay, as they did before.
The builder reports `supports_draft_decode_metadata_update = False`, because `build()` restages the lengths and the in-place update does not.
`TritonAttentionMetadataBuilder` does the same for R-SWA.

## Validation

New test: `tests/v1/worker/test_attn_utils.py::test_static_sink_builder_keeps_batch_seq_lens`.
It builds a sink group and a full-attention group through `build_attn_metadata`.

```
PYTHONPATH=<worktree>:<plug> <venv>/bin/python -m pytest -p noclean -p cpuplat \
  tests/v1/worker/test_attn_utils.py -k static_sink_builder_keeps_batch_seq_lens -q
```

- Base (fix stashed): `E assert [53, 116, 0] == [37, 100, 0]`, `1 failed, 81 deselected, 1 warning in 0.28s`
- Fix: `1 passed, 81 deselected, 1 warning in 0.31s`
- Whole file, base: `70 passed, 11 skipped, 1 warning in 0.75s`
- Whole file, fix: `71 passed, 11 skipped, 1 warning in 0.68s`

Repro script: `scripts/repro_sink_seq_lens.py` in this folder. With a decode at 37, a chunk ending at 100 of a 110-token prompt and a padded row, the sampler predicate flips from `[False, True, False]` to `[False, False, False]` on the base. It stays `[False, True, False]` with the fix.

I could not run this on a GPU. It still needs a GPU run (see STATUS.md): an OpenPangu sink checkpoint under the default FULL_AND_PIECEWISE graphs, chunked against unchunked.

## Duplicate check

I searched open PRs for "static sink seq_lens", "StaticSinkAttention", "openpangu sink" and "sink attention builder".
None of them fixes this.
#47271 (Add OpenPangu V2, open, last updated 2026-08-19) removes `StaticSinkAttentionBuilder` as part of a larger redesign. That would also remove the in-place write, but it is a model-addition PR, not this fix.
#59512 and #47535 touch the sink manager and the DiffKV cache kernel, not the builder.

AI assistance (Claude) was used to find, fix and test this. I reviewed every changed line.

---

# PR draft 2: [Bugfix][Distributed] Refuse pipeline stages without layers

Commit: 7ce91b3818 on aarnphm/sink-sentinel-followups (base e82b7fca7e).

## Overview

`get_pp_indices` accepted a `VLLM_PP_LAYER_PARTITION` entry of 0.
It refused negative entries only.
Its default split also leaves stages empty when a model has fewer layers than stages: 2 layers over 4 stages gives `[(0, 0), (0, 1), (1, 2), (2, 2)]`.
A stage like that owns no KV cache layers.
After model loading and profiling, the KV cache planner then failed.
With a layer-outer layout (LBNHC) it raised `ZeroDivisionError` at `kv_cache_planning.py:674`.
With a block-outer layout (BLHNC) it raised a bare `AssertionError` at `kv_cache_planning.py:553`.
No config validation refuses such a partition today.

## Fix

`get_pp_indices` now raises a `ValueError` for a zero entry in the partition and for a default split with fewer layers than stages.
It runs when each worker builds its model, before weights load.
Draft models run as a single stage (`pp_size == 1`), so this does not affect them.
I chose a refusal over teaching the planner about empty stages.
An empty stage computes nothing.
Supporting it would also need the attention backend setup, kernel block sizes and KV connectors to handle a worker with no layers, and I cannot exercise that on CPU.

## Validation

```
PYTHONPATH=<worktree>:<plug> <venv>/bin/python -m pytest -p noclean -p cpuplat \
  tests/distributed/test_pipeline_partition.py -q
```

- Base (fix stashed): `2 failed, 13 passed, 1 warning in 0.04s`, both failures `DID NOT RAISE ValueError`
- Fix: `15 passed, 1 warning in 0.02s`
- `tests/test_config.py -k "layer_range_with_prefill_context_parallel or hybrid_layer_counts"`: `5 passed` with the fix

Repro script: `scripts/repro_zero_layer_stage.py` in this folder. It shows the accepted partitions and both planner failures on the base. With the fix it shows `ValueError: Every pipeline stage needs at least one layer: partition_list_str='4,0'.`

## Duplicate check

I searched open PRs for "VLLM_PP_LAYER_PARTITION", "get_pp_indices", "pipeline stage layers partition" and "ZeroDivisionError kv cache pipeline".
I searched issues for "VLLM_PP_LAYER_PARTITION 0" and "pipeline parallel ZeroDivisionError".
Nothing covers a zero-layer stage.
#66013 (a stage with no MoE layers) and #63898 (one pipeline stage per instance) touch PP but not this check.

AI assistance (Claude) was used to find, fix and test this. I reviewed every changed line.
