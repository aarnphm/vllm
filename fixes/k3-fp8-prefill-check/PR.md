# [Bugfix][Kimi-K3] Refuse an fp8 KV cache without an fp8 prefill query at startup

<!-- markdownlint-disable -->

## Overview

Claude and I hit this on GB300 while mapping KV cache groups with Kimi-K3 (8 layers, dummy weights, TP4).

With `--kv-cache-dtype fp8`, the engine starts, reports ready, and then dies on the first request:

```
AssertionError: Kimi-K3 fp8 KV cache requires an fp8 prefill query; enable --attention-config '{"use_prefill_query_quantization": true}'.
```

Setting that flag alone gives the same error (#50056). On Hopper no setting avoids it (#51313).

## Claims

- Kimi-K3 with a plain fp8 KV cache that would get a bf16 prefill query now fails while the model is built, before any weights load. The error names both settings and a backend that works. On `e2bab26e51` the engine starts and fails the first request.
- Configurations that work today should be unchanged. The check calls the same `determine_prefill_query_data_type` the MLA metadata builder calls, so it refuses only when the first prefill would assert.
- Only Kimi-K3's NVIDIA MLA layer changes. As far as I can tell no other MLA model has this requirement (see Details).

## Validation

**Before, on GPU.** vLLM nightly `9c1d45c2` (arm64 image), one GB300 tray, TP4, the Kimi-K3 config with 8 layers and dummy weights, `--attention-backend TOKENSPEED_MLA --kv-cache-dtype fp8`:

- job 5643751, fp8 only: the engine finishes init, then every worker raises the AssertionError in `execute_model`.
- job 5643930, plus `use_prefill_query_quantization`: same AssertionError.
- job 5644413, plus `"mla_prefill_backend": "TRTLLM_RAGGED"`: runs. The log shows `Using TRTLLM_RAGGED MLA prefill backend.` and `FP8 prefill attention enabled: query data type is FP8`.

<details>
<summary>job 5643930 log excerpt</summary>

```
INFO 11-19 13:25:32 [selector.py:131] Using FLASH_ATTN MLA prefill backend.
INFO 11-19 13:25:37 [mla_attention.py:2510] Unable to perform FP8 prefill attention when use_prefill_query_quantization is enabled. Please ensure that --kv-cache-dtype is set to fp8 and your prefill backend is compatible with FP8 attention.
INFO 11-19 13:25:54 [core.py:449] init engine (profile, create kv cache, warmup model) took 21.17 s
ERROR 11-19 13:25:56 [multiproc_executor.py:1224]   File ".../vllm/models/kimi_k3/nvidia/mla.py", line 1056, in _forward_prefill_fused
ERROR 11-19 13:25:56 [multiproc_executor.py:1224] AssertionError: Kimi-K3 fp8 KV cache requires an fp8 prefill query; enable --attention-config '{"use_prefill_query_quantization": true}'.
```
</details>

**After, on GPU.** Not run yet. The same three engine configurations go through `gpu_e2e.py` (`--case fp8_only | flag_only | recipe`). I expect `fp8_only` and `flag_only` to raise the new `ValueError` inside `LLM()` and `recipe` to generate tokens.

**Unit test, CPU (macOS, no vLLM build), commit `ace4a7aae9`.**

```bash
VLLM_TARGET_DEVICE=cpu python -m pytest tests/models/kimi_k3/test_mla_fp8_kv_cache.py -v
```

- With this change: `2 passed, 1 warning in 9.14s`
- With `mla.py` from `e2bab26e51`: `2 failed, 1 warning in 35.28s`. On CPU the base fails because the CPU MLA backend rejects fp8 first (`--kv-cache-dtype fp8 is not supported by MLA attention on CPU.`). On a GPU the base builds the layer without that refusal, which is the bug. I have not run the test on a GPU; it runs in `:nvidia: (B200) Kimi K3`.
- `tests/models/test_dspark_mla.py`: `31 passed, 1 warning in 5.77s`. `tests/models/kimi_k3/test_mla_prefill_context.py`: `11 skipped in 2.16s` (CUDA only).

The new error for the two test configurations:

```
ValueError: Kimi-K3 with --kv-cache-dtype fp8 needs an fp8 prefill query: set use_prefill_query_quantization and an MLA prefill backend that takes an fp8 query, such as TRTLLM_RAGGED, on an SM100-family GPU. Got use_prefill_query_quantization=False, mla_prefill_backend=auto. Pass --attention-config '{"use_prefill_query_quantization": true, "mla_prefill_backend": "TRTLLM_RAGGED"}', or use --kv-cache-dtype bfloat16.
ValueError: Kimi-K3 with --kv-cache-dtype fp8 needs an fp8 prefill query: set use_prefill_query_quantization and an MLA prefill backend that takes an fp8 query, such as TRTLLM_RAGGED, on an SM100-family GPU. Got use_prefill_query_quantization=True, mla_prefill_backend=FLASH_ATTN. Pass --attention-config '{"use_prefill_query_quantization": true, "mla_prefill_backend": "TRTLLM_RAGGED"}', or use --kv-cache-dtype bfloat16.
```

## Details

### Fix

- Root cause: K3's `_forward_prefill_fused` implements a plain fp8 cache only with an fp8 query. The query is fp8 only with `use_prefill_query_quantization` and an MLA prefill backend that takes an fp8 query (FLASHINFER, TRTLLM_RAGGED, TOKENSPEED_MLA) on an SM100-family GPU. For K3's head dims, SM100 auto-selection puts FLASH_ATTN first, and `determine_prefill_query_data_type` drops the flag with an info line. Nothing checked the three settings together before the first prefill.
- Claude proposes to check this in `MultiHeadLatentAttention.__init__`, right after the cache dtype is resolved and before the decode backend is built. The check calls `MLACommonMetadataBuilder.determine_prefill_query_data_type`, so startup and the first prefill reach the same answer.
- The error suggests `--kv-cache-dtype bfloat16` as the fallback. It does not suggest `fp8_ds_mla`, because per #51313 K3's dense decode backends reject that layout.
- Scope: the generic MLA path (DeepSeek-V3.x, and K3 on ROCm) and `dots3_note` dequantize an fp8 cache for a bf16 query. The DeepSeek-V3.2/V4.x sparse paths do not use these prefill backends. So the check lives in the K3 layer, and the shared selector is untouched.
- I kept the assert in `_forward_prefill_fused` as a backstop.
- The test clears the process-wide cache of `backend_supports_prefill_query_quantization` around each case, as `tests/v1/attention/test_mla_backends.py` does, so results do not depend on test order. It uses the DeepSeek-V2-Lite config because the prefill selector reads MLA head dims from the model config, and DeepSeek-V2-Lite has K3's dims (128/64/128).
- Docs: the K3 recipe's FP8 KV note already gives the flag pairing. It does not say that plain fp8 needs an SM100-family GPU for K3; I left the recipe alone.

### Duplicate check

I searched open PRs for "fp8 prefill query", "use_prefill_query_quantization", "Kimi-K3 fp8 KV cache prefill" and "K3 mla.py fp8 prefill", all PRs for "mla_prefill_backend fp8 query startup", and issues for "requires an fp8 prefill query" and "Kimi-K3 fp8 kv cache prefill".

- #50181 (open) makes auto-selection prefer prefill backends that take an fp8 query when the flag and an fp8 cache are both set. That fixes the flag-only case on GB200/GB300. With it, the fp8-only case, an explicit FLASH_ATTN, and Hopper still fail at the first request. The two compose: with #50181 the flag-only case passes this check, because the check reads the same predicate.
- #51394 validated in the same layer. Its author closed it on 2026-08-17 before any review. Its tests patched `backend_supports_prefill_query_quantization`; this test builds the real layer.
- #51313 (open) asks for this fail-fast check. #50056 (closed) is the flag-only report, answered with the TRTLLM_RAGGED pairing.

AI assistance (Claude Code) was used for this change and this description. I reviewed every changed line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
