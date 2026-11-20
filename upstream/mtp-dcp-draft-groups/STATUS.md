# mtp-dcp-draft-groups (study bug 14): STATUS

- status: blocked. The right fix is a design change, too large for a short-term branch. No code change.
- commit: none. Branch `aarnphm/mtp-dcp-draft-groups` stays at base `e2bab26e51`, clean.
- files changed: none in the worktree. Scratch evidence is in this dir: `repro_groups.py`, `bench_plans.py`, `probe_scaled_planner.py`, `bench_*.log`.
- PR.md: not written, because there is no change to describe.

## Reproduced at e2bab26e51

`repro_groups.py` runs the real `get_kv_cache_groups` on 61 bf16 MLA target layers plus 1 MTP layer:

```
draft_dcp_sharded=False layout=LBNHC: groups=62 layers_per_group=[1] replicated_groups=1 bytes_per_block=73728
draft_dcp_sharded=False layout=BLHNC: groups=62 layers_per_group=[1] replicated_groups=1 bytes_per_block=73728
draft_dcp_sharded=True  layout=LBNHC: groups=1 layers_per_group=[62] replicated_groups=0 bytes_per_block=4571136
```

## Why the specs differ (question 1)

Neither candidate is the cause. The DCP size is not stored in the spec, and the draft does inherit the target's DCP:
- `draft_decode_context_parallel_size=None` resolves to the target's DCP (`config/speculative.py:1685-1695`). MTP and EAGLE do not override `default_draft_decode_context_parallel_size`; only DFlash/DSpark default to 1 (`spec_decode/dflash/config.py:35-48`). `tests/test_config.py::test_draft_decode_context_parallel_size` pins this default.
- The specs differ only when someone sets `draft_decode_context_parallel_size=1` explicitly. The V2 runner then rewrites the draft specs to `dcp_sharded=False` (`v1/worker/gpu/model_runner.py:670-683`). The spec holds only that boolean. The DCP size comes in later, as `block_size * dcp` for sharded groups (`kv_cache_utils.py:730-732`).
- None of aarnphm's pd-bench recipes sets the knob, so the main P/D DCP 4-8 MTP runs use the default (1 group) and do not hit this bug.

## Why 62 groups instead of 2 (question 2)

- `can_share_block_table` puts `(block_size, dcp_sharded)` in the bucket key (`kv_cache_planning.py:239`), so the draft forms its own 1-layer bucket.
- The pages are equal, so `_get_packed_kv_cache_groups` returns None (`:913-914`). `_get_kv_cache_groups_uniform_page_size` then sets `group_size = min bucket = 1`, because 61 >= 1.5 x 1 (`:496-507`), and splits the target into 61 groups (`:509-530`).
- Upstream chose this on purpose. #68533 (merged 2026-11-18) added `test_full_attention_draft_only_bucket_still_sets_the_group_size`, which asserts 63 one-layer groups for the same shape, because an unbounded draft "still sizes the groups instead of being padded per token".

## Measurements (DCP 8, one process at a time)

Method: the study's `bench_groups.py`/`bench_pool.py` (64 requests x 8192 unique prompt tokens, prefix caching on, the study's budget of 70,000 x 62 pages = 298 GiB). Changes: real `KVCacheManager(dcp_world_size=8)` with the scheduler and hash block sizes that `resolve_kv_cache_block_sizes` would give, the pool build timed without tracemalloc, and the heap measured in a second pass.
Command: `PYTHONPATH=<worktree> <venv>/bin/python bench_plans.py --plans <plan>`. Peak RSS was 1.94 GB for `base` and under 0.6 GB for the others.

| plan | groups | block ids/req | KV/req (8K) | admit ms/req | free ms/req | pool build | heap |
|---|---:|---:|---:|---:|---:|---:|---:|
| base: today, draft DCP=1 | 62 | 1,104 | 77.6 MiB | 0.522 | 0.081 | 1.95 s | 562.9 MiB |
| default: draft DCP = target DCP | 1 | 16 | 69.8 MiB | 0.010 | 0.001 | 0.01 s | 9.1 MiB |
| pad: one draft group, padded to 61 layers | 2 | 144 | 617.6 MiB | 0.046 | 0.008 | 0.02 s | 9.3 MiB |
| scaled: draft block = 64 x DCP, 8-layer target groups, BLHNC (hand-built) | 9 | 144 | 81.0 MiB | 0.064 | 0.011 | 0.38 s | 70.7 MiB |

- The study's 68.9x and 6.91 s did not scale the target block by DCP (7,936 block ids/req), and tracemalloc was on during that build. With DCP 8 the costs are 52x admission against the default, 69x block ids, a 1.95 s build and 62x heap.
- The "one drafter group" fix inside today's single pool is `pad`: 11x faster admission, but 8.0x KV per request, so 8x fewer tokens fit. #68533 rejected this trade.
- Today's planner cannot produce `scaled` (`probe_scaled_planner.py`). With a 512-token draft block, LBNHC gives 62 groups and raises every target block to 512 tokens (a 4096-token DCP span). BLHNC gives 2 groups `[61, 1]` with a 61-page block, and the draft uses 8 of those 61 pages.
- `default` wins on both axes: it needs 10% less KV per request than `base` and admission is 52x faster. Draft DCP=1 can only win on draft-step latency (no DCP collective in the draft layer). That needs a GPU A/B; I did not measure it.

## What the structural fix needs (the `scaled` plan: 9 groups, +4.4% KV against base)

1. Runner: give replicated draft specs `block_size * dcp` so their block covers the target's token span (`model_runner.py:670-683`).
2. Planner: split a single-page target bucket into groups whose bytes match the draft page, in `_get_packed_kv_cache_groups`. Open PR #66503 ("Automatically fit packed groups to draft pages", for issue #66451, K3 + replicated DSpark at DCP 8) rewrites this function now, so the two changes must be coordinated.
3. Layout: DeepSeek + MTP moves to a block-outer layout. The resolver already narrows to block-outer when sharded and replicated pages differ (`v1/attention/backends/utils.py:369-394`). Dense FlashMLA declares no layout preference and 64-token kernel blocks, so a 512-to-64 kernel split looks possible. I did not run it.
4. P/D: NIXL must move a replicated draft group whose block size differs between P (DCP 1, block 64) and D (DCP 8, block 512). This is the riskiest part for aarnphm's main configuration.

## Decisions for aarnphm

- (a) Refuse `draft_decode_context_parallel_size=1` for full-attention (unbounded) drafts when DCP > 1. This is a small startup check. It contradicts #68533's test and removes a supported configuration. The numbers favour it, except for the draft-latency question.
- (b) Build the `scaled` design together with #66503's author.
- (c) Leave it as it is. The default MTP setting does not hit the bug.

## Tests run

- None in pytest, because there is no code change. Repro and bench commands are above. Output is in `bench_base.log`, `bench_default.log` and `bench_pad_scaled.log`.

## GPU checks still needed (only for (a) or (b))

- Draft-step latency A/B, DSv3 + MTP at DCP 8: `draft_decode_context_parallel_size` 1 against 8, same batch and context, with `vllm bench serve` or `vllm bench forward`.
- For (b): FlashMLA with a 512-token replicated draft block on BLHNC (kernel-block split), and a NIXL P/D round trip with P at DCP 1 and D at DCP 8.

## Duplicate check (2026-11-20)

- `gh pr list --state open --search` with "draft decode context parallel kv cache groups", "replicated draft DCP group" and "MTP DCP kv cache group": no PR for the uniform-page split. #66503 (open) changes the packed path for a replicated DSpark draft and does not reach this case, because equal pages skip that path. #61502 and #53205 are not related.
- `gh issue list --search` with "draft_decode_context_parallel_size", "MTP DCP kv cache groups" and "replicated draft kv cache group explosion": no matching issue. Related: #66451 (packed-path capacity waste on K3) and RFC #65384 (draft execution policy under PCP/DCP).
