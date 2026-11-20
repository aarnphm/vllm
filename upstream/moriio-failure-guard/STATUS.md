# moriio-failure-guard: STATUS (2026-11-20)

status: duplicate-open-PR (#65690 fixes bug 10; #65923, stacked on it, makes the failure recoverable)
commit: none. Branch `aarnphm/moriio-failure-guard` is still at base `e2bab26e51`, worktree clean.
files changed: none

## Bug 10 at base e2bab26e51: confirmed on CPU (the study had it as "read in code")

- `moriio_connector.py:3221-3231`: `_record_failed_recv` withholds block IDs only `if self._has_mamba`.
- `moriio_connector.py:3196-3215`: the READ step barrier uses `fail_fast=self._has_mamba` and re-raises only for Mamba. Other layouts log and run the forward on the failed KV.
- `moriio_connector.py:4380-4389`: for non-Mamba layouts, the reported list is every group's destination blocks, flattened.
- `scheduler.py:3922-3931`: `_handle_invalid_blocks` raises `RuntimeError` when `len(kv_cache_groups) > 1`.
- NIXL gates on `not _is_hma_required and len(kv_cache_groups) == 1` (`nixl/base_worker.py:796-798`).
- History: #56855 (2026-09-16) replaced the scheduler's multi-group recompute with this `RuntimeError`. #51052 (2026-09-21) added MoRIIO's `_has_mamba` guard five days later. The "match NIXL" in its docstring was already wrong then, because NIXL gated on `_is_hma_required`, which is True for SWA hybrids. #61499 (2026-11-18) moved NIXL to the group count. MoRIIO never moved.

Repro: `repro_moriio_failure_guard.py`. It runs MoRIIO's real barrier, `_pop_done_transfers`, `get_block_ids_with_load_errors`, `get_exchange_clipped_blocks` and `_align_read_blocks`. Their output goes into a real `Scheduler` built from `make_kv_cache_config(block_size=16, swa_enabled=True)`. The request is sync-loaded and RUNNING, the way READ admits it (`moriio_connector.py:901-907`).

```
W=~/workspace/worktrees/vllm/moriio-failure-guard
S=<scratch>/fixes/moriio-failure-guard
uv pip install --python <scratch>/venv/bin/python --target $S/extra msgpack
PYTHONPATH=$W:<scratch>/plug:$S/extra \
  <scratch>/venv/bin/python $S/repro_moriio_failure_guard.py
```

The shared CPU venv has no `msgpack`, so every MoRIIO import fails there. I installed it into `$S/extra` and left the venv untouched.

Base (`repro.base.out`):
```
A1. step barrier returned (no raise): the forward runs on the failed KV
A2. get_block_ids_with_load_errors() -> [0, 1, 2, ..., 18]
B.  scheduler.update_from_output raised RuntimeError: A KV connector reported block-level load failures (invalid_block_ids) which is not supported for models with multiple KV cache groups. ...
```

Base plus #65690 (diff at head `6ac4157894` applied to a `git archive` copy, `repro.pr65690.out`):
```
A1. step barrier raised TransferError: 1/1 transfers failed: ...
A2. get_block_ids_with_load_errors() -> []
B.  not reached: the step fails in start_load_kv before the forward
```

The engine still stops on a failed READ for multi-group layouts with #65690, and its description says so. #65923 adds `failed_sync_recving`, a request-level channel for sync loads, plus the scheduler handling to preempt or fail one request.

## Why "report the whole request via failed_recving" cannot work for READ

READ loads synchronously, so the request is RUNNING when its read fails. READ also never reports `finished_recving` (`moriio_connector.py:3060-3068`). Same output on both trees:
```
C1. KVOutputAggregator, failed_recving without finished_recving -> forwarded set(), held {'id-1'}
C2. policy=fail, failed_recving={RUNNING req} -> status=RUNNING, num_computed_tokens=160, output_token_ids=[7]
```
The aggregator forwards a failure only alongside `finished_recving` (`kv_connector/utils.py:251-252`), and the held entry never drains. `_handle_failed_recving` acts only on `WAITING_FOR_REMOTE_KVS` (`scheduler.py:3898-3911`). Writing the NIXL shape into MoRIIO would swap the crash for tokens sampled from half-written KV.

## Adjacent defect, not bug 10, present at base and with #65690

Layout: one sliding-window group (every attention layer SWA, HMA on). It passes both the `_has_mamba` guard and #65690's `single group and not Mamba` guard. `get_exchange_clipped_blocks` keeps `cdiv(sw, bs) + 1` blocks, so the clipped list can start with the null block 0. The scheduler's single-group path matches block 0 in every request whose block table starts with it. The run uses two READ requests, and only A's read fails:
```
D. 1 SWA group, policy=fail: A=[0, 0, 1, ..., 8] B=[0, 0, 9, ..., 16]; reported [0, 1, ..., 8] -> A(failed) FINISHED_ERROR; B(ok) FINISHED_ERROR computed=159 out=[]
D. 1 SWA group, policy=recompute: ... A(failed) RUNNING computed=0; B(ok) RUNNING computed=159 out=[]
```
`fail` is the default `kv_load_failure_policy` (`vllm/config/kv_transfer.py:117`). Two fixes would work, and both are for #65690 to take:
- drop `NULL_BLOCK_ID` (`vllm/v1/attention/backends/utils.py:56`, already used by `mooncake_store/scheduler.py:112`) from the reported set;
- adopt NIXL's `not _is_hma_required` term, which turns the collateral failures into a failed step.

I did not check how common all-SWA layouts are in MoRIIO deployments.

## Duplicate check

Open PRs searched: "moriio failed read", "moriio invalid_block_ids", "moriio kv cache groups", "moriio hybrid failure", "moriio load error", "_handle_invalid_blocks multiple groups", "invalid_block_ids multiple KV cache groups", "block-level load failures", "sync load failure hybrid". Issues searched: "moriio invalid_block_ids", "moriio RuntimeError multiple KV cache groups", "moriio gpt-oss", "block-level load failures".

- #65690, `[Bugfix][ROCm][MoRIIO] READ: fail only the requests of a failed handshake, not the decode engine`. amd-weisun, open, not draft, MERGEABLE, review required, updated 2026-11-19. Its "Multi-group layouts" claim is this bug: one KV cache group and no Mamba, otherwise fail the step before the forward. It applies cleanly to `e2bab26e51`. Its tests are `test_failed_read_reports_blocks_only_for_one_group` and `test_unrecoverable_step_barrier_fails_closed[has_mamba, single_group]`.
- #65923, `[ROCm][MORI IO]Moriio hybrid load failure`. Same author, stacked on #65690, open. Adds `failed_sync_recving` with per-request preempt or fail. Its MI355X validation skipped Gemma 3 (gated checkpoint), so no sliding-window hybrid was run end to end.
- #56733 (draft, 2026-09-16) preempts hybrid sync-load failures on the block-level path, which main now refuses before reaching it. RFC #47812 covers group-aware recovery. Neither duplicates #65690.

## Tests run

No code change, so no pytest runs and no failure-set comparison. Evidence is the repro on both trees above: `repro.base.out`, `repro.pr65690.out`, raw logs in `repro.*.raw`. The `Exception ignored in MoRIIOConnectorWorker.__del__` at exit is interpreter teardown (logger already None), not part of the result.

## GPU checks still needed

None for this branch. MoRIIO needs ROCm and `mori`, so the NVIDIA clusters cannot run it. For #65690 and #65923 the missing evidence is a failed READ on a sliding-window hybrid (gpt-oss or Gemma 3) on MI355X.
