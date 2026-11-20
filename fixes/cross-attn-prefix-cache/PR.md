# [Bugfix][Core] Keep prefix caching off for encoder-decoder models

## Overview

Claude and I found this while going through the KV cache group code.

Whisper, or any encoder-decoder model, started with `--enable-prefix-caching` fails its first prefix-cache lookup:

```
NotImplementedError: CrossAttentionManager does not support caching
```

I reproduced the raise with the real `KVCacheManager` and `Scheduler` on CPU; the GPU end-to-end run is still to do (below).

Whisper with `--runner pooling` gets the same configuration without any flag.

## Claims

- An explicit `--enable-prefix-caching` on an encoder-decoder model is now turned off with a warning, where the first request used to raise.
- Whisper with `--runner pooling` no longer turns prefix caching on by default.
- `CrossAttentionManager` now reports itself as not prefix-cacheable, so cache-hit lookups skip the cross-attention group instead of raising.
- Decoder-only models, and encoder-decoder models started without the flag, are unchanged.

## Fix

Root cause:

- `CrossAttentionManager` inherits `prefix_cacheable() == True` from the base manager. `prefix_lookup_groups` therefore puts the cross-attention group into the lookups of `HybridKVCacheCoordinator`, and its `find_longest_cache_hit` raises. Its `cache_blocks` raised as well.
- Only a default kept prefix caching off: `ModelConfig.is_prefix_caching_supported` returns False for `encoder_decoder`, and `EngineArgs` uses it only when the flag is unset.
- With `--runner pooling`, Whisper's `attn_type` is `decoder`, so that default is True.
- `SchedulerConfig` logs "Encoder-decoder models do not support chunked prefill nor prefix caching; disabling both." but it can only turn off chunked prefill. Before #25075, this check was in `VllmConfig.__post_init__` and turned off both.

Change:

- `VllmConfig.__post_init__` turns prefix caching off when `model_config.is_encoder_decoder`, with a warning. It sits next to the existing check for encoder-only instances, before the model config hooks run.
- `CrossAttentionManager.prefix_cacheable()` returns False, and its `cache_blocks` override, which raised, is removed. The base `cache_blocks` already returns early for a group that is not prefix-cacheable.

Why the decoder groups are not cached either:

- The manager change alone would let the decoder self-attention group take prefix hits.
- That is not safe today. The scheduler takes any computed decoder token as proof that the encoder already ran (`_try_schedule_encoder_inputs`), so a decoder prefix hit skips the encoder and the request gets no cross-attention blocks. The run below shows it.
- Caching the decoder groups would need the scheduler to schedule the encoder after a hit, and the block hashes to cover the encoder input. I left that out of this fix.

Why a warning and not an error:

- The existing log line, the chunked-prefill handling for the same models, and the Unlimited-OCR (R-SWA) config hooks all turn the setting off. I did the same.
- If reviewers prefer an error for the explicit flag, it belongs in `EngineArgs`. `CacheConfig` defaults `enable_prefix_caching=True`, so `VllmConfig` cannot tell a request from a default.

`docs/features/README.md` already marks APC as unsupported for encoder-decoder models, so the docs do not change.

## Validation

All runs are CPU-only on a MacBook, without a vLLM build, at base `e2bab26e51`. Environment for every command:

```bash
export VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1
export PYTHONPATH=<tree>:<plugin dir that skips the conftest cleanup, which segfaults on macOS>
```

**New test, fails before and passes after:**

```bash
python -m pytest -p noclean "tests/test_config.py::test_encoder_decoder_disables_prefix_caching" -q --tb=short
```

```
# base e2bab26e51 (base tree + the new test)
E   AssertionError: assert not True   (both parametrizations)
2 failed, 1 warning in 3.17s
# this branch (783eb99e95)
2 passed, 1 warning in 1.33s
```

The two cases are `[generate-True]` (explicit flag) and `[pooling-None]` (Whisper with `--runner pooling`, no flag).

**Same config path, by hand** (`EngineArgs(...).create_engine_config()` on `openai/whisper-tiny`, condensed from the probe output):

```
# base
default flags                                     enable_prefix_caching=False enable_chunked_prefill=False
--enable-prefix-caching                           enable_prefix_caching=True  enable_chunked_prefill=False
--enable-prefix-caching --enable-chunked-prefill  enable_prefix_caching=True  enable_chunked_prefill=False
--runner pooling                                  enable_prefix_caching=True
# this branch
WARNING [vllm.py:1619] Encoder-decoder models do not support prefix caching; disabling it.
--enable-prefix-caching                           enable_prefix_caching=False enable_chunked_prefill=False
--enable-prefix-caching --enable-chunked-prefill  enable_prefix_caching=False enable_chunked_prefill=False
--runner pooling                                  enable_prefix_caching=False
```

On base, every one of these configs logs "disabling both", but only chunked prefill is turned off.

**KV cache manager**, Whisper-like groups (4 self-attention + 4 cross-attention layers), real `KVCacheManager`, prefix caching on:

```
# base
cross group prefix_cacheable: True | lookups: [('FullAttentionManager', [0]), ('CrossAttentionManager', [1])]
get_computed_blocks raised: NotImplementedError - CrossAttentionManager does not support caching
# this branch
cross group prefix_cacheable: False | lookups: [('FullAttentionManager', [0])]
no error
```

**Why the decoder groups stay uncached.** Real `Scheduler` with a self-attention group and a cross-attention group, prefix caching on, the config guard bypassed (opt-125m config, patched to encoder-decoder as in `_create_encoder_decoder_scheduler` in `tests/v1/core/test_scheduler.py`). Requests A and B have the same 48-token decoder prompt and the same encoder input (100 tokens, 7 blocks):

```
# base
schedule(A) raised: NotImplementedError - CrossAttentionManager does not support caching
# manager change only (this branch, config guard bypassed)
A: encoder scheduled = True | cross-attn blocks = 7
B: prefix-hit tokens = 32 | scheduled tokens = 16 | encoder scheduled = False | cross-attn blocks = 0 (needs 7)
```

B's decoder would read cross-attention KV that was never written. With the config change, this scheduler state is not reachable through `EngineArgs` or `VllmConfig`.

**Existing tests**, same environment, base against this branch:

| tests | base | this branch |
|---|---|---|
| `tests/v1/core/test_scheduler.py` | 136 failed, 171 passed in 12.75s | 136 failed, 171 passed in 12.67s |
| `tests/v1/core/test_single_type_kv_cache_manager.py` | 70 passed in 0.47s | 70 passed in 0.47s |
| `tests/v1/core/test_prefix_caching.py` | 1 failed, 235 passed in 4.75s | 1 failed, 235 passed in 4.80s |
| `tests/test_config.py -k encoder_decoder` | 3 failed, 4 passed in 3.55s | 1 failed, 6 passed in 2.54s |
| `tests/engine/test_arg_utils.py -k prefix_cache` | 1 passed in 0.15s | 1 passed in 0.61s |

Apart from the two new test cases, the failure sets are identical. They come from the environment: the `test_scheduler.py` failures I sampled and the `test_prefix_caching.py` one are "The CPU model runner without Triton does not yet support: speculative decoding", and `test_is_encoder_decoder[meta-llama/Llama-3.2-1B-Instruct-False]` needs a gated checkpoint. The encoder-decoder scheduler tests (cross-attention block allocation, chunked prefill disabled for encoder-decoder) pass on both.

**Not yet run: GPU end-to-end.** With the built branch and base on one GPU:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python e2e_whisper_prefix_caching.py   # LLM(openai/whisper-tiny, enable_prefix_caching=True), 3 transcriptions
```

On base I expect the engine to die on its first request (or its JIT warmup request) with the `NotImplementedError` above. On this branch I expect `prefix caching on: False` and three transcripts, with the first and third identical.

## Duplicate check

- `gh pr list --state open --search "CrossAttentionManager prefix caching"`: #66903 only. It skips idle groups in per-step allocation and touches the coordinator's cross-attention `isinstance` checks, not `prefix_cacheable`.
- `--search "encoder-decoder enable-prefix-caching"`: #55159 rejects `--enable-prefix-caching` for encoder-only pooling models in `arg_utils.py`. That is a different model class with the same kind of gap; it is open with merge conflicts.
- `--search "whisper prefix caching"`: nothing.
- `gh issue list --state all` for "CrossAttentionManager does not support caching", "encoder decoder prefix caching NotImplementedError" and "whisper enable-prefix-caching": nothing about this bug.

AI assistance was used for this PR (Claude). I reviewed every changed line and ran the commands above.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI. (GPU end-to-end still to run.)
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
