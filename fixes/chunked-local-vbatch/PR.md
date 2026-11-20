[Bugfix][Attention] Size FlashInfer and FA3 request buffers for chunked local virtual batches

## Overview

Claude and I hit this while mapping KV cache groups: with chunked local attention (Llama 4), FlashInfer can get more requests in one `build()` than `max_num_seqs`, and on the native prefill path that stops the engine. Closes #49980.

- `ChunkedLocalAttentionBuilder.build()` rewrites the batch into virtual batches, one per chunk that a request's new tokens touch, and hands their count to the underlying builder as `num_reqs` (`make_local_attention_virtual_batches`).
- `FlashInferMetadataBuilder` sizes `paged_kv_indptr`, `paged_kv_last_page_len`, `qo_indptr_prefill` and the XQA draft-mask cache by `max_num_seqs`.
- So a batch that fits the scheduler limits can overflow them. Example: `max_num_seqs=256`, `max_num_batched_tokens=16384`, 255 decodes plus a 16,129-token prefill chunk starting at position 4,096 gives 258 virtual batches, and `paged_kv_indptr` has 257 entries where 259 are needed.
- Native FlashInfer prefill (FlashInfer selected on SM90/SM12x, or SM100 with `use_trtllm_attention=False`) raises `ValueError: provided out is the wrong size for the accumulation` in `_compute_flashinfer_kv_metadata`, and the engine core shuts down with every request in flight.
- The TRTLLM prefill path (SM100 default) does not raise: `torch.cumsum(..., out=...)` resizes a too-short view of `paged_kv_indptr.gpu`, which moves the persistent buffer's storage. It only gives the right offsets because the resized storage happens to hold them.

## Fix

- New `AttentionMetadataBuilder.get_max_num_reqs(vllm_config)`: the largest `num_reqs` that `build()` can receive. It defaults to `max_num_seqs`, so every other builder is unchanged.
- The chunked local builder overrides it with `2 * max_num_seqs + cdiv(max_num_batched_tokens, attention_chunk_size)`. A request's new tokens span their full chunks plus at most two partial ones. The wrapper already closes over the chunk size, which #61452 moved off `KVCacheSpec`.
- FlashInfer sizes its per-request buffers by it in `__init__`. They are never regrown, so the decode wrappers that alias them stay valid.
- `paged_kv_indices` keeps the `max_num_seqs` bound, because virtual batches only split each request's pages.
- The cost is a few KiB per chunked FlashInfer builder: at 256/16384, `paged_kv_indptr` grows from 257 to 515 entries.
- FA3 (second commit): a FlashAttention builder decides at construction whether it stages scheduler metadata in a persistent buffer of `max(max_num_seqs, max capture size)` rows. When graph-memory profiling is skipped (e.g. `kv_cache_memory_bytes`), the serving builders are built before the cudagraph mode is downgraded for the chunked local builder, so its FA3 builder keeps staging and a large virtual batch fails the staging copy. That buffer now uses `get_max_num_reqs` too.

_Note that FlexAttention has the same overflow in its persistent buffers. #68142 fixes it: its diff applies cleanly on this branch and sizes all of them by the same bound, so I left Flex alone. It could call `get_max_num_reqs` instead of scanning the layers. b12x's plan table also keys on `max_num_seqs` (explicit backend only); I left it out._

## Validation

Base `e2bab26e51`; this branch at `161e567dc4` (FlashInfer in `45a0d8f63a`, FA3 in `161e567dc4`).

New CPU test: the capacity the chunked local builder reports vs vLLM's real virtual batching, on the 258-row batch above and on the worst case (every request starts one token before a chunk boundary, 513 rows):

```bash
pytest tests/v1/attention/test_chunked_local_attention.py -k "builder_holds_virtual_batches or flashinfer_chunked_local"
```

- main: `2 failed` with `AttributeError: type object 'ChunkedLocalAttentionBuilder' has no attribute 'get_max_num_reqs'`. Both batches do exceed `max_num_seqs`: that assert passes first.
- this PR: `2 passed`.

```
# main (e2bab26e51) with only the new test file added, macOS CPU
2 failed, 1 skipped, 6 deselected, 2 warnings in 0.56s
# this PR
2 passed, 1 skipped, 6 deselected, 1 warning in 1.57s
```

The skip is the GPU test below. The rest of `test_attention_backends.py` and this file show the same failure sets with and without the change on my Mac; those failures are macOS environment errors (no device type).

I also replayed FlashInfer's buffer writes on CPU (real virtual batching and `split_decodes_and_prefills`, FlashInfer's exact `cumsum`/`sub` statements) at the old and new capacity:

| batch | native prefill | TRTLLM prefill KV offsets | XQA mask rows |
|---|---|---|---|
| 255 decodes + 16,129 at 4,096 (258 rows) | `ValueError` → ok | 2 of 4 entries, storage moved → 4 of 4, no move | |
| 255 fresh 2-token prompts + fresh 8,193 (257 rows) | `ValueError` → ok | 257 of 258, both buffers moved → exact | |
| k=3 verify windows split 2+2 by a chunk boundary (512 rows) | | | 256 of 512 → 512 of 512 |

**IMPORTANT**: I haven't run the GPU test yet. `test_flashinfer_chunked_local_prefill_beyond_max_num_seqs` serves a tiny Llama 4 (dummy weights) through FlashInfer's native prefill at `max_num_seqs=1` with a 200-token prompt, which is 4 virtual batches. It should fail on main with the `ValueError` above and pass here on H100 and B200:

```bash
pytest -v -s tests/v1/attention/test_chunked_local_attention.py -k flashinfer_chunked_local_prefill_beyond_max_num_seqs
```

The FA3 commit has no test of its own, and I haven't run it on a GPU either. The repro is the same tiny Llama 4 on H100 with `FLASH_ATTN`, `max_num_seqs=1`, capture sizes `[1]`, `kv_cache_memory_bytes` set and a 400-token prompt (7 virtual batches). FA3 then stages 33 metadata entries; main's buffer has 17, and this PR's has 49.

`ruff check`, `ruff format --check`, the docstring/torch.cuda/test-tethering hooks and the mypy hook pass on the changed files.

## Duplicate check

- #50022 (open, `needs-rebase`, conflicting) targets this same bug. It keys on `getattr(kv_cache_spec, "attention_chunk_size", None)`; #61452 removed that field from every `KVCacheSpec`, so at current main the check is always `None` and nothing would be resized even after a rebase. It also predates `qo_indptr_prefill` (#65598). This PR moves the bound to the chunked local wrapper and covers that buffer. Happy to coordinate with the #50022 author on credit or on which PR lands.
- #68142 (open) is the FlexAttention fix; it does not touch FlashInfer or FA3.
- Searched open PRs for "chunked local attention flashinfer", "virtual batches", "chunked local max_num_seqs", "make_local_attention_virtual_batches", "local attention llama4 flashinfer", and issues for the same plus "paged_kv_indptr" and the error text. Only #49980 (open) and #28993 (closed as stale) came up; #28993 is the same `ValueError` from FlashInfer's `np.cumsum` on Llama 4 Scout, 2x B200, `--max-num-seqs 128`.

AI assistance (Claude) was used for the investigation, the patch, the tests and this description; I reviewed every changed line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
