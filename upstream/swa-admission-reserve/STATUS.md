---
name: swa-admission-reserve
bug: 9 (admission gate under-reserves windowed groups after a long prefix hit)
date: 2026-11-20
base: e2bab26e51 (upstream main, 2026-11-20)
---

# Status: duplicate-open-PR

- status: duplicate-open-PR (vllm-project/vllm#60045 fixes this path; #66725 does not)
- commit: none. Branch `aarnphm/swa-admission-reserve` is unchanged at `e2bab26e51`.
- files changed: none in the worktree. All experiments ran on scratch copies under `trees/`.
- GPU checks still needed: none. The predictor is pure scheduler arithmetic.

## Bug at this base

Still present. At `e2bab26e51` the code is `vllm/v1/core/single_type_kv_cache_manager.py:322-377`
(the base path) and `:3176-3181` (HiSparseResident). The repro reproduces the report exactly:

```
hit=16000 prompt=20000
admission gate reserved 1258 blocks; per group (FA, SWA) = [1250, 8]
peak blocks drawn from the pool during the remaining prefill = 1380
peak non-null blocks held per group (FA, SWA) = [1250, 136]
SWA max_admission_blocks cap = 137
```

## Open PRs on the same lines

- #60045 (yqszxx, opened 2026-10-05, head `843db1462a`): rewrites the capped branch for both the
  fast path and the first-allocation path. In the capped branch it computes
  `min(required - skipped, cap) - real_held - real_hits_in_window + evictable_hits`.
  This is the fix the brief expected. State: `CONFLICTING`, mergify asked for a rebase. It was
  written before `num_unbacked_blocks` existed, and its test builds `SlidingWindowSpec` and
  `ChunkedLocalAttentionSpec`, which were removed. A Gemma 4 31B stall report in its comments
  hits the same first-allocation path, through an external (connector) hit.
- #66725 (drago1234, opened 2026-11-10, head `c963cab673`): `MERGEABLE`, blocked on review.
  For a request that already holds blocks it returns `min(uncapped, cap - recyclable_held)`,
  where recyclable means non-null with `ref_cnt == 1`. An `and blocks` guard skips requests with
  no blocks, and its description says "First-time admission is unchanged". So it does not fix
  bug 9. It fixes the in-flight and parked reservation path (issue #66724).
- Neither PR touches `HiSparseResidentManager.get_num_blocks_to_allocate`.
- Nearby open PRs, judged from title and search hit only (I did not read their diffs): #63288
  (the gate's first spec-decode step), #63136 (the async-load spec-step reserve), #66379 (parked
  reservations across allocations; its author stacked #66725 to validate it), #66738 (opt-in
  prefill backoff, stacked on #66725).

## Probe: what each rule reserves and what the request draws

`probe/admission_probe.py` (adapted from the research repro). FA + SWA, block 16,
`max_in_flight_tokens=2048`, dense retention. A warms a 16000-token prefix. B (16000-token
local hit plus a tail) goes through `allocate_slots(full_sequence_must_fit=True)`, then prefills
in 2048-token chunks. "reserved" is the gate's `apply_admission_cap=True` value. "drawn" is the
peak drop in free blocks while B finishes prefill. `exclusive` frees A first, so B's hits are
evictable. `shared` keeps A alive, so B's hits in the window are also held by A.

Trees (all from `git archive e2bab26e51`):
`base`; `pr66725` (its `vllm/` hunk applied cleanly); `pr60045` (hand port, `port_rules.py`,
because the PR conflicts); `alt` (clamp `uncapped` at `cap - recyclable_held` on both paths,
`probe/alt.patch`); `pr66725x` (#66725 with the `and blocks` guard dropped and `blocks`
defaulted to `()`, `probe/pr66725_drop_guard.patch`).

Tail 6144 tokens, so the last chunk is a full 2048 (`probe/results_fullchunk.txt`,
`probe/results_pr66725x.txt`):

| rule | window | hits | reserved | drawn | short by |
|---|---|---|---|---|---|
| base | 128 | evictable | 1392 | 1520 | 128 |
| base | 128 | shared | 384 | 520 | 136 |
| base | 1024 | evictable | 1448 | 1576 | 128 |
| base | 1024 | shared | 384 | 576 | 192 |
| #66725 | 128 | evictable | 1392 | 1520 | 128 |
| #66725 | 128 | shared | 384 | 520 | 136 |
| #66725 | 1024 | evictable | 1448 | 1576 | 128 |
| #66725 | 1024 | shared | 384 | 576 | 192 |
| #60045 | 128 | evictable | 1521 | 1520 | 0 |
| #60045 | 128 | shared | 513 | 520 | 7 |
| #60045 | 1024 | evictable | 1577 | 1576 | 0 |
| #60045 | 1024 | shared | 513 | 576 | 63 |
| alt | 128 | evictable | 1521 | 1520 | 0 |
| alt | 128 | shared | 521 | 520 | 0 |
| alt | 1024 | evictable | 1577 | 1576 | 0 |
| alt | 1024 | shared | 577 | 576 | 0 |
| #66725, guard dropped | 128 | evictable | 1521 | 1520 | 0 |
| #66725, guard dropped | 128 | shared | 521 | 520 | 0 |
| #66725, guard dropped | 1024 | evictable | 1577 | 1576 | 0 |
| #66725, guard dropped | 1024 | shared | 577 | 576 | 0 |

Window 128, tails 4000 and 8000 (`probe/results.txt`), where the last chunk is short:
base is short by 122 / 116 (evictable) and 130 / 124 (shared); #66725 gives the same numbers;
#60045 gives 1387 / 1637 vs 1380 / 1624 (evictable) and 379 / 629 vs 380 / 624 (shared, 1 short
at tail 4000); alt gives 1387 / 1637 and 387 / 637. The original repro case (tail 4000, guard
dropped) gives 1387 vs 1380.

Why #60045 is short with shared hits: a hit block that another request also holds counts against
the cap while it is in B's window, but when the window slides past it, releasing it frees no
pool block, so B's later chunks can still draw up to the full cap. #60045 reserves
`cap - shared_hits` (137 - 8 = 129, or 193 - 64 = 129) while B draws `window + chunk` (136, or
192) in its last chunk. #66725 already makes this argument (`ref_cnt == 1`) for held blocks; the
same argument applies to hit blocks at first allocation.

Why dropping the guard needs the `()` default: with only the guard removed, a fresh request
raises `TypeError: 'NoneType' object is not reversible` (`probe/pr66725x_unpatched_run.txt`).

## Tests run (CPU, this Mac)

Command, per tree `T` in `trees/` (from inside `T`):

```
PYTHONPATH=$T:<scratch>/plug \
  <scratch>/venv/bin/python -m pytest -p noclean -p no:cacheprovider \
  -q --tb=no -rfE tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_swa_inflight_window_free.py \
  tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py
```

- base: `16 failed, 324 passed, 1 warning in 44.60s`
- alt: `16 failed, 324 passed, 1 warning in 248.04s (0:04:08)` (slow from CPU contention)
- pr66725x: `16 failed, 324 passed, 6 warnings in 27.68s` (no `pyproject.toml` in that tree, so the extra warnings are unregistered-marker warnings)
- The 16 failures are the same set on every tree and are environment failures: `facebook/opt-125m`
  Hub fetch fails, and `tests/v1/kv_connector/unit/utils.py:185` raises a `VllmConfig`
  `ValidationError`. Lists: `probe/pytest_*.txt`.
- Not run: #66725's own new tests against the guard-dropped variant. Applying its test hunk to a
  scratch tree was refused at the permission prompt, so I did not retry it.

## HiSparseResident (not covered by any open PR; code reading only, not measured)

`HiSparseResidentManager.get_num_blocks_to_allocate` (`:3176-3181` at `e2bab26e51`) returns
`max(min(required, cap) - max(existing, host_pages), 0)`. `add_local_computed_blocks` pads
`host_pages` nulls into the resident row, and `allocate_new_blocks` then pulls
`cdiv(num_tokens, bs) - len(row)` real pages per chunk. So a local host-prefix hit of at least
`cap` pages reserves 0 resident pages, while the first chunk alone pulls `chunk / bs` pages.
Same shape as bug 9. A correct clamp there needs "pinned resident pages", which only
`HiSparseCoordinator` tracks (`unpinned_pages`, residency overlay), and
`_fills_admission_window` compares the same null-padded row length with the cap. I think this
needs the HiSparse owners. A one-line change in the manager would not be correct.

## What aarnphm must decide

1. Post the review comment below on #66725, linking #60045, or drop it. I posted nothing.
2. Whether to file the HiSparseResident instance as an issue for the HiSparse owners.
3. The worktree `worktrees/vllm/swa-admission-reserve` has no commits and can be removed
   (`w --rm vllm swa-admission-reserve`). I did not run `w` or `git worktree`.

## Draft review comment (aarnphm's voice; for #66725, cc #60045)

> The first allocation after a long prefix hit has the same undercount, and the `and blocks` guard skips it.
> A request with no blocks falls through to `min(required, cap) - max(skipped, hit_len, unbacked)`, which is 0 once the hit is longer than the cap.
> I think that is also the path behind the Gemma 4 report on #60045, through an external hit.
>
> Probe on main `e2bab26e51`: FA + SWA, block 16, `max_in_flight_tokens=2048`, a 16000-token local hit, 6144 tokens left in 2048-token chunks.
> "shared" keeps the request that warmed the prefix alive, so the window's hit blocks have `ref_cnt >= 1`.
>
> | | window | hits | gate reserves | request draws |
> |---|---|---|---|---|
> | main, and this PR | 128 | evictable | 1392 | 1520 |
> | main, and this PR | 1024 | shared | 384 | 576 |
> | #60045 | 128 | evictable | 1521 | 1520 |
> | #60045 | 1024 | shared | 513 | 576 |
> | this PR, guard dropped | 128 | evictable | 1521 | 1520 |
> | this PR, guard dropped | 1024 | shared | 577 | 576 |
>
> #60045 subtracts every real hit block from the cap.
> That is short when another request still holds those blocks: once the window passes them, releasing them frees nothing, and the later chunks draw the full window plus a chunk.
> Your `ref_cnt == 1` rule already handles this for held blocks.
> Applying it to requests with no blocks gives `min(uncapped, cap)` there, and it covered every case I tried:
>
> ```diff
> -        blocks = self.req_to_blocks.get(request_id)
> +        blocks = self.req_to_blocks.get(request_id, ())
>          if (
>              apply_admission_cap
>              and self.max_admission_blocks_per_request is not None
> -            and blocks
>          ):
> ```
>
> The `()` default matters: without it a fresh request raises `TypeError: 'NoneType' object is not reversible`.
> With this change the four nearest test files keep main's failure set on my machine (16 environment failures, no Hub access).
> I have not run this PR's new tests against it.
> AI assistance was used for this analysis; I reviewed the numbers.

## Reproduce

```
S=<scratch>/fixes/swa-admission-reserve
W=~/workspace/worktrees/vllm/swa-admission-reserve
PY=<scratch>/venv/bin/python
PLUG=<scratch>/plug
for t in base pr60045 pr66725 pr66725x alt; do
  mkdir -p $S/trees/$t; (cd $W && git archive HEAD vllm tests | tar -x -C $S/trees/$t)
done
gh pr diff 66725 --repo vllm-project/vllm > $S/prs/pr66725.diff   # head c963cab673
for t in pr66725 pr66725x; do (cd $S/trees/$t && git apply --include='vllm/*' $S/prs/pr66725.diff); done
(cd $S/trees/pr66725x && git apply -p2 $S/probe/pr66725_drop_guard.patch)
sed -n 322,377p $S/trees/base/vllm/v1/core/single_type_kv_cache_manager.py > $S/probe/base_block.txt
$PY $S/probe/port_rules.py $S/probe/base_block.txt $S/trees/pr60045 $S/trees/alt
cd $S/probe && PYTHONPATH=$S/trees/<t>:$PLUG $PY admission_probe.py \
  --mode {exclusive,shared} --tail {4000,6144,8000} --window {128,1024}
```
