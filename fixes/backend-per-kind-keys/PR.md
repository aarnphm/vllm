[Bugfix][Attention] Reject backend_per_kind keys that backend selection never reads

## Overview

`AttentionConfig.backend_per_kind` accepts every `KVCacheSpecKind` value as a key, but the selector (`get_attn_spec_kind`, `vllm/v1/attention/selector.py:70-108`) only ever assigns 6 of the 10 kinds to a layer. Keys for `chunked_local_attention`, `mamba`, `sink_full_attention` and `unknown` parse fine and are then never looked up.

Claude and I hit this while going through the KV cache group code. Llama 4 is where it shows:
- `ChunkedLocalAttention` picks its backend with `get_attn_backend(head_size, dtype, kv_cache_dtype)` (`vllm/model_executor/layers/attention/chunked_local_attention.py:103`), so the selector sees a plain `full_attention` layer.
- With `--attention-config '{"backend_per_kind": {"full_attention": "TRITON_ATTN", "chunked_local_attention": "FLEX_ATTENTION"}}'` the chunked-local layers run TRITON_ATTN, and the FLEX_ATTENTION entry is dropped without a word.
- Mamba layers never reach `get_attn_backend` (`vllm/model_executor/layers/mamba/abstract.py:110-112` goes through `get_mamba_attn_backend`), and the only in-tree `StaticSinkAttention` (openPangu) pins `FlashAttentionDiffKVBackend`, so `mamba` and `sink_full_attention` entries can't do anything either.

#48012 listed the chunked-local and sink cases under Limitations, but the validator kept accepting those keys.

## Fix

The validator now accepts only the 6 kinds the selector assigns: `full_attention`, `mla_attention`, `sliding_window`, `sliding_window_mla`, `encoder_only_attention`, `cross_attention`.
Any other `KVCacheSpecKind` raises with the list of valid kinds; unknown strings keep the existing "Unknown KV cache group kind" error.
The field docstring now says that chunked-local layers select as `full_attention` and that Mamba layers don't use attention backends, so that mapping is written down instead of implied.

**IMPORTANT:** a config that sets one of the four keys used to start (and ignore the entry); it now fails at startup. I think that's the right trade, since the entry never did anything.

_Note that this keeps `full_attention` steering Llama 4's chunked-local layers._ Making `chunked_local_attention` a real key looks doable (a `use_chunked_local` flag from `ChunkedLocalAttention` through `get_attn_backend` into `get_attn_spec_kind`), but it changes what an existing `full_attention` entry does on Llama 4, so I left it out of this bugfix. Happy to follow up if reviewers want it.

## Validation

New test `test_backend_per_kind_accepts_only_selectable_kinds` in `tests/v1/attention/test_backend_per_kind.py`: it runs `get_attn_spec_kind` over its whole input domain and checks that the accepted keys are exactly those kinds and that every rejection lists them.

CPU only (macOS); the change is config validation, so no GPU is involved.

```
pytest tests/v1/attention/test_backend_per_kind.py::test_backend_per_kind_accepts_only_selectable_kinds -q
```

Before, on e2bab26e51:
```
E         Extra items in the left set:
E         <KVCacheSpecKind.SINK_FULL_ATTENTION: 'sink_full_attention'>
E         <KVCacheSpecKind.MAMBA: 'mamba'>
E         <KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION: 'chunked_local_attention'>
E         <KVCacheSpecKind.UNKNOWN: 'unknown'>
1 failed, 1 warning in 0.06s
```
After, on 73d595add1:
```
1 passed, 1 warning in 0.23s
```

Nearby suites:
- `pytest tests/v1/attention/test_backend_per_kind.py -q`: `26 passed, 1 warning in 0.06s` before, `27 passed, 1 warning in 0.06s` after (the new test)
- `pytest tests/engine/test_arg_utils.py -k attention -q`: `3 passed, 117 deselected` before and after

Through the `vllm serve` argument parser (`EngineArgs.add_cli_args` + `EngineArgs.from_cli_args`) with `--attention-config '{"backend_per_kind": {"chunked_local_attention": "FLEX_ATTENTION"}}'`:
- before: accepted, `{'chunked_local_attention': FLEX_ATTENTION}`
- after:
```
error: argument --attention-config/-ac: 1 validation error for AttentionConfig
  Value error, backend_per_kind cannot target 'chunked_local_attention': backend selection never assigns that kind. Chunked-local layers select as 'full_attention', and Mamba layers do not use attention backends. Valid kinds are: cross_attention, encoder_only_attention, full_attention, mla_attention, sliding_window, sliding_window_mla.
```

I haven't rerun the GPU e2e `tests/v1/e2e/general/test_attention_backend_per_kind.py`; it only uses `full_attention` and `sliding_window`, which stay valid.

## Duplicate check

- `gh pr list --repo vllm-project/vllm --state open --search` with "backend_per_kind", "backend_per_kind chunked", "per kind attention backend", "get_attn_spec_kind" and "chunked local attention backend selector": no open PR changes the validator or `get_attn_spec_kind`. #68629 (named backends in spawned workers) and #67798 (encoder-only hooks) read `backend_per_kind`, and #67548 edits a different test in the same test file; none overlap.
- `gh issue list --repo vllm-project/vllm --state all --search` with "backend_per_kind", "backend per kind chunked local" and "backend_per_kind ignored": nothing on this.

AI assistance was used (Claude Code). I reviewed every changed line and ran the commands above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
