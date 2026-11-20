# STATUS: cross-attn-prefix-cache (bug 13)

status: fixed
branch: aarnphm/cross-attn-prefix-cache (worktree ~/workspace/worktrees/vllm/cross-attn-prefix-cache), base e2bab26e51
commit: 783eb99e9568732a557676f93690f6ecbe4b4100 (one commit, signed off by Aaron Pham, not pushed)

## Files changed

- vllm/config/vllm.py (+13): `VllmConfig.__post_init__` turns prefix caching off, with a warning, when `model_config.is_encoder_decoder`. It sits next to the encoder-only check, before the model hooks.
- vllm/v1/core/single_type_kv_cache_manager.py (+5 -12): `CrossAttentionManager.prefix_cacheable()` returns False. Its `cache_blocks` override, which raised, is removed; the base one returns early for groups that are not prefix-cacheable.
- tests/test_config.py (+18): `test_encoder_decoder_disables_prefix_caching[generate-True]` (explicit flag) and `[pooling-None]` (Whisper with `--runner pooling`, no flag).

## What I found

- Bug confirmed at e2bab26e51 with the bends repro (NotImplementedError from the cross-attention lookup).
- `SchedulerConfig.__post_init__` logs "Encoder-decoder models do not support chunked prefill nor prefix caching; disabling both." but turns off only chunked prefill. #21088 turned off both in `VllmConfig`; #25075 (e23cacda35) moved the check into `SchedulerConfig` and lost the prefix-caching half. The fix restores that intent.
- Wider than the explicit flag: Whisper with `--runner pooling` resolves `attn_type=decoder`, so the pooling default turns prefix caching on with no flag at all (evidence/probe_pooling.base.out).
- Caching only the decoder groups is unsafe: the scheduler treats any computed decoder token as "encoder done" (`_try_schedule_encoder_inputs`). With the manager change alone, request B got a 32-token decoder hit, its encoder was not scheduled, and it got 0 of the 7 cross-attention blocks it needs (evidence/hazard_decoder_hit.fix.out).
- R-SWA left unchanged: both R-SWA users (Unlimited-OCR 1 via `models/config.py:388-395`, Unlimited-OCR-2 via `_configure_v2`, `enable_prefix_caching = False`) turn prefix caching off whatever the flag says, so an explicit flag never reaches `RSWAManager`.

## Tests run

Environment: CPU, macOS, no vLLM build. `VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 PYTHONPATH=<tree>:<scratch>/plug`, `pytest -p noclean`. Base tree = `git archive e2bab26e51 vllm tests` in base/ plus the new test.

New test:
- base: `2 failed, 1 warning in 3.17s` (`AssertionError: assert not True` in both cases), log evidence/new_test.base.log
- fix: `2 passed, 1 warning in 1.33s`, log evidence/new_test.fix.log

Existing tests, base vs fix, failure sets compared:
- tests/v1/core/test_scheduler.py (whole file, run before the memory limit): `136 failed, 171 passed` on both, identical failure ids. The enc-dec tests (cross-attention block allocation, chunked prefill disabled for enc-dec) pass on both. Failures are environment: 22 sampled are all "The CPU model runner without Triton does not yet support: speculative decoding" (VllmConfig validation on the CPU platform).
- tests/v1/core/test_single_type_kv_cache_manager.py (whole file, before the limit): `70 passed` on both.
- tests/v1/core/test_prefix_caching.py (whole file, before the limit): `1 failed, 235 passed` on both, the same baseline failure `test_hisparse_admitted_async_loads_can_finish_with_nothing_running[7-3-47-4]`, also the CPU-without-Triton spec-decode validation error (suites/hisparse_base.log).
- tests/test_config.py `-k encoder_decoder` (after the memory limit, one process): base `3 failed, 4 passed, 659 deselected, 1 warning in 3.55s`; fix `1 failed, 6 passed, 659 deselected, 1 warning in 2.54s`. The failure left on fix, `test_is_encoder_decoder[meta-llama/Llama-3.2-1B-Instruct-False]`, fails on base too: the checkpoint is gated.
- tests/engine/test_arg_utils.py `-k prefix_cache`: base `1 passed, 119 deselected, 1 warning in 0.15s`; fix `1 passed, 119 deselected, 1 warning in 0.61s`.
- Logs: suites/before, suites/after (whole files), suites/narrow (narrow runs).

Probes (rerun: `sh run_evidence.sh`; outputs in evidence/*.base.out, evidence/*.fix.out):
- repro_crossattn_prefix.py (bends repro, real KVCacheManager): base raises NotImplementedError; fix `prefix_cacheable: False | lookups: [('FullAttentionManager', [0])]`, no error.
- probe_paths.py (EngineArgs on whisper-tiny): base `--enable-prefix-caching` -> True; fix -> False plus the warning.
- probe_pooling.py: base `--runner pooling` -> True with no flag; fix -> False plus the warning.
- hazard_decoder_hit.py: base raises at the first schedule; manager change alone -> decoder hit with the encoder skipped (above).

## Not run here

- tests/v1/core/test_kv_cache_utils.py: started, stopped by the memory limit. No test in it builds a cross-attention group (`rg CrossAttention tests` finds only test_scheduler.py and test_gpu_model_runner.py).
- tests/test_config.py beyond `-k encoder_decoder`, under the memory limit.

## GPU checks still needed

1. End to end, base vs fix, one GPU, the compiled build of each:
   `VLLM_WORKER_MULTIPROC_METHOD=spawn python <scratch>/fixes/cross-attn-prefix-cache/e2e_whisper_prefix_caching.py`
   Expected on base: the engine dies on its first request (or on its JIT warmup request) with `NotImplementedError: CrossAttentionManager does not support caching`. Expected on fix: `prefix caching on: False`, then three transcripts with the first and third identical.
2. Optional, same check from the server: `vllm serve openai/whisper-tiny --enable-prefix-caching --enforce-eager`, then one `/v1/audio/transcriptions` request.

## For aarnphm to decide

1. Warning versus error. I turned the explicit flag off with a warning, as the existing log line, chunked prefill for the same models, and the Unlimited-OCR hooks do. An error for the explicit flag would go in `EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`, because `CacheConfig` defaults `enable_prefix_caching=True` and `VllmConfig` cannot tell a request from a default. Open PR #55159 takes the error route for encoder-only pooling models.
2. vLLM's tests/AGENTS.md asks for an `LLM`/`vllm_runner` regression test for user-visible fixes. My test goes through `EngineArgs.create_engine_config()`, which the CLI and `LLM` use, and it runs on CPU. If you want the LLM-level test as well, after the GPU run, this would go in tests/models/multimodal/generation/test_whisper.py:
   ```python
   @pytest.mark.core_model
   def test_enable_prefix_caching_is_turned_off(vllm_runner, input_audios) -> None:
       with vllm_runner("openai/whisper-tiny", dtype="half", max_model_len=448,
                        enforce_eager=True, enable_prefix_caching=True) as vllm_model:
           config = vllm_model.llm.llm_engine.vllm_config
           assert not config.cache_config.enable_prefix_caching
           for prompts, _, audios in input_audios:
               assert vllm_model.generate_greedy(prompts, max_tokens=16, audios=audios)[0][1]
   ```
3. Whisper with `--runner pooling` now logs the warning on every start, because `ModelConfig.is_prefix_caching_supported` still says True for it. Making that property return False for `is_encoder_decoder` would silence it. I left the property alone to keep the diff small.
4. Dead code after this change: the `issubclass(..., CrossAttentionManager)` skip in `KVCacheManager.estimate_cached_tokens` and the cross-attention branch of MooncakeStore's `_validate_kv_cache_config` no longer trigger, because cross groups are no longer prefix-cacheable. Encoder-decoder models already reject KV connectors in the scheduler. I left both in place.

## Cleanup

- Background processes started by this agent: none left (checked with `ps`). The parallel whole-file runs were stopped when the memory limit came in; later runs used one process at a time.
- base/ (the scratch git-archive copy) is deleted. To rerun `run_evidence.sh` or `run_narrow.sh`, recreate it first: `mkdir base && git -C ~/workspace/worktrees/vllm/cross-attn-prefix-cache archive e2bab26e51 vllm tests | tar -x -C base && git -C ~/workspace/worktrees/vllm/cross-attn-prefix-cache show 783eb99e95:tests/test_config.py > base/tests/test_config.py`.
