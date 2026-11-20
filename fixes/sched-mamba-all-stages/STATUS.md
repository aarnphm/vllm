status: fixed (latent: no public checkpoint reaches it, see "Reach")
branch: aarnphm/sched-mamba-all-stages (worktree ~/workspace/worktrees/vllm/sched-mamba-all-stages), base e2bab26e51, not pushed
commit: 05c1b61e6eb9432778a65f9b0dead55c4ee8317b

## Bug 20 at the base (reproduced by running)
- generate_scheduler_kv_cache_config deep-copies worker 0's config (kv_cache_utils.py:1079-1093). KVCacheConfig.has_mamba_layers (kv_cache_interface.py:943-949) read projected kv_cache_specs, which are empty on a stage holding none of a group's layers. needs_kv_cache_zeroing is ORed across workers (kv_cache_planning.py:1599-1602); has_mamba_layers was not.
- repro_planner.py (real get_kv_cache_configs): stage 0 attention only, stage 1 attention + Mamba -> worker 0 False, worker 1 True, scheduler copy False, zeroing True everywhere. Control (Mamba on stage 0) -> scheduler True. Outputs: repro_planner.base.out, repro_planner.fix.out.

## Everything that reads the flag
Scheduler side (gets the stage-0 copy):
- scheduler.py:424-455: need_mamba_block_aligned_split, mamba_has_prefill_checkpoint_blocks, mamba_partial_cache_hit, mamba_shared_prefix_checkpoint -> _mamba_block_aligned_split call sites :845, :1392. This is the Mamba prefix-caching alignment, chunked-prefill alignment, partial-tail and Marconi junction stops.
- scheduler.py:2503-2517: publish finalized decode states (sync scheduling + prefix caching). Spec decode ORs in num_spec_tokens > 0 here, so spec decode itself does not depend on the flag.
- kv_cache_manager.py:364-370 get_computed_blocks_for_connector: False -> reconciled get_computed_blocks (conservative, no harm).
- NIXL base_scheduler.py:104 -> _prefill_backoff :395-410; Mooncake mooncake_connector.py:1054 -> :1112 (D load count), :1154 (P truncation).
Not dependent on the flag: block-size rounding (resolve_kv_cache_block_sizes, partial_hash_hits_enabled use manager configs), resolve_mamba_block_size (#67994, merges all workers), _spec_decode_step_blocks (manager class).
Worker side (per-stage config): config/compilation.py:1717-1731 (FULL cudagraph capacity check), mooncake_store/worker.py:1758-1763 (retention rewrite), mooncake_connector.py:1584 (TransferTopology.is_mamba, never read), :1862, :3354, :3419 (head-count checks).

## What goes wrong (stage 0 without Mamba, later stage with Mamba)
- Measured (repro_scheduler.py, real planner + real Scheduler, block 16, repeated 100-token prompt): B's prefix hit 0 instead of 96; A's prefill chunks [100] or [40, 40, 20] instead of [96, 4] or [32, 32, 32, 4]. Prefix caching is off for the hybrid; outputs stay correct because MambaManager keeps mid-block states out of full-block hashes (_record_mid_block_states / _refine_block_mask), by reading. Outputs: repro_scheduler.base.out, repro_scheduler.fix.out.
- P/D (by reading, not run): the prefiller does not truncate the last prompt token, the decoder recomputes it on a state that already includes it -> silent wrong decode. Also when both sides have the bug.

## Reach (pp_stage0_survey.py + pp_stage0_survey.out, configs in hf/)
- 43 public hybrid configs, 22 architectures (Nemotron-H x10, Granite-4 x6, Jamba x4, Qwen3-Next x3, Qwen3.5 x3, LFM2/3 x4, Kimi-Linear, Kimi-K3, GLM-5.3-Flash, MiniMax-M1, Falcon-H1 x2, Zamba2, ZAYA, Bamba, OLMo-Hybrid, AliceAI, Ling-3.0, Ring-flash-linear-2.0): every one has a Mamba-state layer at layer 0 -> stage 0 holds one under any partition with >= 1 layer on stage 0, at PP 2/3/4/6/8/16. Jamba-1.5 Mini/Large are gated (not checked; same arch as v0.1).
- Drafters: none in-tree has Mamba layers (nemotron_h_mtp.py accepts only '*' and 'E'; K3 GQA/dense/DSpark drafts are attention; GLM-5 Next MTP is DSA; z-lab/Qwen3-Coder-Next-DFlash is 8 full-attention layers). The planner report's "hybrid drafter" case was synthetic.
- VLLM_PP_LAYER_PARTITION allows 0 (distributed/utils.py:241 rejects only negatives), but a 0-layer stage 0 crashes the planner first: ZeroDivisionError at kv_cache_planning.py:674 via the probe at :306 (zero_stage_tb.py). Separate pre-existing bug, not fixed here.
- So reaching bug 20 needs an attention-first layer layout (custom layer_types / hybrid_override_pattern / out-of-tree hybrid) or a drafter with Mamba layers on an attention-only target.

## Fix
- has_mamba_layers reads group.distinct_specs (the projection keeps unprojected_specs, kv_cache_planning.py:1444-1451). One line plus a docstring.
- Worker-side consequence: a stage without Mamba layers now reports True. Mooncake store worker retention now matches its scheduler side (mooncake_store/scheduler.py:154-166); Mooncake P2P head-count checks on an attention-only stage take the hybrid path (same as non-PP hybrids); cudagraph capacity check raises on stage 0 as well (same num_blocks, the Mamba stage raised anyway).

## Files changed
- vllm/v1/kv_cache_interface.py (+2 -1)
- tests/v1/core/test_kv_cache_utils.py (+23): test_generate_scheduler_kv_cache_config_sees_mamba_on_later_stage

## Tests (CPU, macOS; one process at a time)
Prefix: cd <worktree> && VLLM_TARGET_DEVICE=cpu PYTHONPATH=<worktree>:<scratch>/plug <scratch>/venv/bin/python -m pytest -p noclean
(VLLM_TARGET_DEVICE=cpu: without vllm._version, platform detection raises before its macOS check.)
- new test, base: 1 failed, 1 warning in 1.57s (scheduler has_mamba_layers False, needs_kv_cache_zeroing True) -> test_fail_before.out
- new test, fix: 1 passed, 1 warning in 1.79s -> test_pass_after.out
- tests/v1/core/test_kv_cache_utils.py: base 10 failed, 105 passed; fix 9 failed, 106 passed. The 9 are test_num_hash_lookahead_tokens[*], env: "The CPU model runner without Triton does not yet support: speculative decoding".
- tests/v1/core/test_kv_cache_planning.py: 75 passed before and after
- tests/v1/core/test_prefix_lm_scheduler.py: 12 passed before and after
- tests/v1/core/test_mamba_align_chunk_split.py: 49 passed before and after
- tests/v1/kv_connector/unit/test_nixl_connector_hma.py: 1 failed, 103 passed before and after (env: gated google/gemma-3-1b-it config)
- ruff check / ruff format --check: clean; pre-commit (incl. mypy, docstring style, test tethering) passed on commit.
- CI: test_kv_cache_utils.py is cpu_test, run by `pytest -v -s -m 'cpu_test' v1/core` (misc.yaml:413), which depends on vllm/v1/.

## GPU checks still needed
- gpu_e2e/test_hybrid_attention_only_stage.patch adds test_pipeline_parallel_attention_only_stage_prefix_caching to tests/models/language/generation/test_hybrid.py (granite-4.0-tiny-preview, load_format=dummy, layer_types with attention at 0/15/25/35, VLLM_PP_LAYER_PARTITION=1,39; asserts num_cached_tokens PP=2 == PP=1 and nonzero). Not committed because I could not run it. On a 2-GPU node:
    git apply <scratch>/fixes/sched-mamba-all-stages/gpu_e2e/test_hybrid_attention_only_stage.patch
    pytest -v -s tests/models/language/generation/test_hybrid.py -k attention_only_stage   # on HEAD~1: expect PP=2 counts all 0 -> fail; on HEAD: pass
  If it passes on HEAD and fails on HEAD~1, fold it into the commit.

## For aarnphm to decide
- Latent for every public checkpoint checked. Upstream alone (1-line fix + test) or bundle with other PP-projection fixes; vLLM AGENTS.md frowns on tiny standalone PRs.
- Worker-side semantics change (above). The alternative, a field ORed only into the scheduler copy, keeps workers per-stage but leaves the Mooncake store worker/scheduler retention mismatch.

## Scratch artifacts
repro_planner.py, repro_scheduler.py, zero_stage_tb.py, pp_stage0_survey.py, *.out, hf/ (downloaded config.json files), COMMIT_MSG.txt, PR.md, gpu_e2e/.
