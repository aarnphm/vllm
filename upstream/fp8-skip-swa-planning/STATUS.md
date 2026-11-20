# STATUS: bug 17, fp8 + full-attention skip layer on a sliding-window hybrid

- status: duplicate-open-PR. The fix is the third hunk of vllm-project/vllm#54599 ("[Bugfix][TurboQuant] Let a TurboQuant KV cache coexist with sliding-window layers", yeasah). That PR is open, labeled `needs-rebase` since 2026-10-15, has no human review, and bundles the hunk with TurboQuant platform and producer changes.
- commit sha: none. Branch `aarnphm/fp8-skip-swa-planning` is left clean at base `e2bab26e51`.
- files changed: none.

## Bug, re-verified at e2bab26e51

Command (one process; `repro_planner.py` is in this dir):

```
cd <scratch>/fixes/fp8-skip-swa-planning
VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
PYTHONPATH=~/workspace/worktrees/vllm/fp8-skip-swa-planning \
<scratch>/venv/bin/python repro_planner.py gemma3 5
```

The script uses real code where it matters: a `ModelConfig` built from a local Gemma-3-27B text config.json (62 layers, 16 KV heads, head 128, window 1024, 5:1 pattern), the real `FlashAttentionBackend` class, `Platform._align_heterogeneous_kv_block_size`, the producer `Attention.get_kv_cache_config`, then `get_global_kv_cache_groups` + `get_kv_cache_configs`. Only the layer objects are stand-ins that carry what `Attention.__init__` resolves (dtype after the skip rule, window). `VLLM_TARGET_DEVICE=cpu` is needed on this Mac because platform detection fails without `vllm._version`.

Output at base (`runs/base-gemma3-5.txt`):

```
platform: block_size 32 skip_page_size_padded 131072
    ('AttentionSpec', 'FullAttentionManager', 'torch.bfloat16', 32, 262144, None) 1
    ('AttentionSpec', 'FullAttentionManager', 'torch.uint8', 32, 131072, None) 9
    ('AttentionSpec', 'SlidingWindowManager', 'torch.uint8', 32, 131072, 131072) 52
PLANNING FAILED: AssertionError:
    vllm/v1/core/kv_cache_planning.py:1119 filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    vllm/v1/core/kv_cache_planning.py:392 assert new_spec.page_size_bytes == max_page_size
    vllm/v1/kv_cache_interface.py:512 assert self.page_size_padded >= self.unpadded_page_size_bytes
```

Mechanism (line numbers at e2bab26e51):

1. `vllm/platforms/interface.py:855-966` raises the block size from 16 to 32 and sets `skip_page_size_padded` to the fp8 page at 32 tokens (131072 B).
2. `vllm/model_executor/layers/attention/attention.py:689-722` gives every sliding-window layer `page_size_padded=skip_page_size_padded`, quantized or not. For an fp8 SW layer the hint equals its real page, so it pads nothing, but the hint is still an absolute byte count. The full-attention branch (`:741-750`) builds the bf16 skip layer at the primary block size with no padding, so its page is 2x the shared page (262144 B).
3. `vllm/v1/core/kv_cache_planning.py:378-381` (`unify_kv_cache_spec_page_size`) sees 262144 % 131072 == 0 and doubles the fp8 SW specs' `block_size`, keeping `page_size_padded=131072`. The unpadded page is now 262144, so `page_size_bytes` asserts. It is an `AssertionError`, so the `NotImplementedError` fallback at `:1120-1124` does not catch it.

The planner can already unify two page sizes. The defect is that the block-scaling branch forgets that `page_size_padded` scales with the block. `_ensure_min_page_size` (`kv_cache_planning.py:1053-1058`) already scales both. Only the sliding-window branch sets the skip padding (no other producer reads `skip_page_size_padded`), so the crash needs a sliding-window layer plus an integer bf16:primary page ratio (fp8: 2). For a non-integer ratio, unify takes the padding branch instead and does not crash.

## Fix choice: scale the padding (= #54599 hunk 3), not refuse

- Scaling is exact: `padded * r == max_page`, and `unpadded <= padded` still holds after both scale by `r`.
- With the #54599 hunk ported into base `unify` (the script swaps in the base function's source with only that hunk replaced; `PORT_54599=1`), Gemma 3 plans, and so does gpt-oss. The scheduler's `KVCacheManager` (prefix caching on) admits a 4000-token prompt, frees it, and re-admits it with a 3968-token hit. This is the same hit as the known-good `--kv-cache-dtype-skip-layers sliding_window` config.
- fp8 memory is kept: 4096 B/token on the 61 fp8 layers and 8192 on the bf16 layer (Gemma-3-27B).
- A refusal at config time would be inconsistent. Base already plans the sibling case (a sliding-window skip index, e.g. Gemma 3 `0`) into the same one-layer-per-group shape, and the refusal would also block TurboQuant's automatic boundary skips when they land on full-attention layers (gpt-oss alternates SW/full). A refusal also needs per-model layer-type knowledge at config time, so it is the larger change.

## Results (all fp8 KV, FlashAttention kernel blocks `MultipleOf(16)`, layout LBNHC; files in `runs/`)

| model geometry | skip | base | with #54599 hunk | groups (no skip -> skip) |
|---|---|---|---|---|
| Gemma-3-27B | `5` (full) | AssertionError | plans; manager OK, re-hit 3968/4000 | 7 -> 62 |
| Gemma-3-27B | `0` (SW) | plans; manager OK, re-hit 3968 | unchanged | 7 -> 62 |
| Gemma-3-27B | `sliding_window` | plans; manager OK, re-hit 3968 | unchanged | 7 -> 7 |
| gpt-oss-120b | `1` (full) | AssertionError | plans; manager OK, re-hit 3968/4000 | 2 -> 36 |
| gpt-oss-120b | `0` (SW) | plans, 36 groups | unchanged | 2 -> 36 |
| Qwen3-Next-80B | `3` (attention) | plans; GDN pages padded 2x | unchanged | 4 -> 48 |

Item 4, other hybrids:

- gpt-oss (alternating SW/full) fails the same way with a full-attention index and is fixed by the same hunk.
- Qwen3-Next and other GDN hybrids do not crash: unify pads Mamba pages instead of rescaling them. But the bf16 attention page (2,195,456 B) is 2x the aligned page, so all 36 GDN state pages are padded from 1,097,728 to 2,195,456 B. That doubles GDN state memory per block, and the plan splits into 48 one-layer groups (12x). By page arithmetic (not measured), an 8k-token request costs about 27% more KV memory (188 vs 149 MB). The #54599 hunk does not change this. It is a separate pessimization, not a crash.
- The manager check for Qwen3-Next is not valid: my harness never advances `num_computed_tokens`, and it fails the same way on plain Qwen3-Next fp8 with no skip layers (`runs/base-qwen3next-noskip.txt`). For Qwen3-Next only planning is verified.

## Tests run

- Sanity, existing tests at base: `VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 PYTHONPATH=<worktree>:<scratch>/plug .../venv/bin/python -m pytest -p noclean "tests/v1/core/test_kv_cache_planning.py::test_unify_kv_cache_spec_page_size_mamba" "tests/v1/core/test_kv_cache_planning.py::test_get_kv_cache_config_one_worker" -q` gives `2 passed, 1 warning in 0.58s`. Without `VLLM_TARGET_DEVICE=cpu` the second test fails with "Failed to infer device type" (environment).
- Repro matrix: 12 single-process runs, recorded in `runs/*.txt`. No new test was added and no code was changed.

## GPU checks still needed (after the hunk lands upstream)

- `vllm serve google/gemma-3-27b-it --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers 5` and `vllm serve openai/gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers 1`, each with one greedy request compared against `--kv-cache-dtype fp8` without the skip list (`check_logprobs_close`). On Hopper with FA3, the skipped bf16 group can hit #65891 (FA3 scheduler metadata built with the engine-wide fp8 dtype; issue #65821), so either take #65891 too or force `--attention-backend FLASHINFER` / `TRITON_ATTN`.

## Related open work (noted, not duplicates of bug 17)

- #58207 (LucasWilkinson, "Pick hybrid KV cache group size by worst-case padding bytes") targets the one-group-per-layer plans in the table.
- #59178 reprices phase 3 per actual skip layer (Gemma 4 per-layer head sizes). It touches `interface.py` only and does not fix this bug.
- #65891 fixes FA3 runtime metadata for skip-layer groups (see GPU checks).
- Sibling study bug B2 (MLA stale `page_size_padded`) is the same defect class. Its worktree branch `aarnphm/mla-stale-page-padding` had no commits past the base when I looked. If that fix makes `copy_with_new_block_size` scale the padding, unify could call it instead of `replace`.

## What aarnphm must decide

1. Comment on #54599 asking yeasah to split hunk 3 (or rebase). A draft is in PR.md. Or:
2. Open a split-out bugfix PR with `Co-authored-by: yeasah`, plus a planner-boundary regression test (#54599 has no tests). vLLM's duplicate rule says not to without the author's or a maintainer's agreement. Or:
3. Leave it to #54599.
