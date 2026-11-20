# No PR: bug 10 is fixed by open PR #65690

#65690 changes the same guard to "one KV cache group and no Mamba" and fails the step before the forward otherwise.
#65923, stacked on it, adds per-request recovery for synchronous loads.
Opening a third PR on the same lines would duplicate them.

Below is a review comment draft for #65690, in case aarnphm wants to post it.

---

I ran this against a CPU repro on main (`e2bab26e51`). The repro chains MoRIIO's real step barrier, `_pop_done_transfers` and `get_block_ids_with_load_errors` into a real two-group `Scheduler`.

On full + sliding-window (the gpt-oss / Gemma 3 shape), main raises the `_handle_invalid_blocks` `RuntimeError` after the forward.
With this PR the barrier raises `TransferError` before the forward, and no block IDs are reported.
That matches the description.

One layout still reports block IDs that hit other requests: a single sliding-window group (every attention layer SWA, HMA on).
`_block_load_errors_supported` is True there.
`get_exchange_clipped_blocks` keeps `cdiv(sw, bs) + 1` blocks, so the clipped list can start with the null block, and `_read_blocks` passes it to `_recving_local_blocks`.
In my repro (block_size 16, sliding_window 128, two 160-token READ requests, only A's read fails), MoRIIO reports `[0, 1, ..., 8]`.
Under the default `kv_load_failure_policy=fail`, request B, whose read succeeded, also ends `FINISHED_ERROR`. Under `recompute` it loses that step's token.
Every running request whose block table starts with the null block is matched, so I think one failed read can fail most long requests in the batch.
Dropping `NULL_BLOCK_ID` from the reported set would fix it, as `mooncake_store/scheduler.py` already does.
Gating on NIXL's `not _is_hma_required` would also stop it, at the cost of failing the step.
I have not checked how common all-SWA layouts are in MoRIIO deployments, so this may be low priority.

AI assistance (Claude) was used for the repro and this comment; I reviewed both before posting.
