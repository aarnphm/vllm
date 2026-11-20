## Overview

Mooncake P/D ships the speculative scratch slots of every Mamba/GDN group along with the running state, and the P worker pairs P's and D's lists by their tails. Claude and I hit this while auditing how the P/D connectors handle KV cache groups.

- `MambaManager` keeps a request's running state at index `-(1 + num_speculative_blocks)` of its block list, with the scratch slots after it (`num_speculative_blocks = num_speculative_tokens`, or 0 under `--use-replayssm`).
- Mooncake sends those lists whole from both exchange points (`request_finished` on P, `update_state_after_alloc` on D). `_build_transfer_params` then drops null blocks and keeps the tail (`mooncake_connector.py:2126-2163` at `e2bab26e51`).

What happens depends on how many scratch slots each side keeps (`k` per Mamba group):

- **P keeps more than D** (MTP on P only, or `--use-replayssm` on D only): P's last scratch slot lands in D's running-state slot. P's prefill never writes that slot, so D decodes from whatever an earlier request left there. **IMPORTANT:** nothing errors; the output is just wrong.
- P keeps fewer than D (MTP on decode only, which seems like the natural P/D setup): every hybrid request fails with `P num blocks less than D`.
- Same `k` on both: `k + 1` states move per Mamba layer where 1 is needed, 4:1 at `k = 3`.

The silent case, in five steps (FA + GDN, P with MTP `k = 2`, D without spec decode, no prefix caching):

1. P admits the request, drops the last prompt token, and prefills `N - 1` tokens. GDN row `[2, 3, 4]`: block 2 is the running state, blocks 3 and 4 are scratch.
2. The request finishes on P; `request_finished` queues `[2, 3, 4]` for sending.
3. D admits the request with `N - 1` external tokens. GDN row `[2]`; D asks P for `[2]`.
4. P's worker tail-aligns `[2, 3, 4]` against `[2]` and copies P's block 4 into D's block 2.
5. D recomputes token `N` from block 2, which holds stale state.

## Fix

Each side now clips SSM groups to its running-state slot at both exchange points, using its own `num_speculative_blocks`. This reuses `clip_ssm_state_blocks`, the rule NIXL (`get_exchange_clipped_blocks`) and MoRIIO already apply.

- One slot per Mamba group on each side, so the worker pairs running state to running state, and P and D can run different draft lengths.
- The change lives in `MooncakeConnectorScheduler.get_sw_clipped_blocks`, which both exchange points already call. I kept the name to keep the diff small. It now clips more than the sliding window, so I'm happy to rename it to NIXL's `get_exchange_clipped_blocks` if reviewers prefer.
- Align mode follows the same rule: the manager keeps scratch slots at the tail (it relocates them, or reuses one as the next running state), and prefill checkpoints and the previous state sit before the running state.
- No handshake check: a draft-length mismatch is now a supported setup, so there is nothing to refuse.
- _Note that_ P and D both need this change. A fixed P with an unfixed D fails with `P num blocks less than D` when D drafts. A fixed D with an unfixed P still reads the wrong slot.

## Validation

New `cpu_test`: `test_hybrid_gdn_pd_exchange_only_the_running_state` in `tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py`. It runs a real P scheduler through prefill to `request_finished`, and a real D scheduler through admission. It then checks that each side sends only the GDN slot holding the request's state. Cases: P/D draft lengths 2/0 and 1/3, each with and without Mamba prefix caching (align mode).

Run on macOS, CPU only, from a source checkout (`-p noclean` skips a conftest cleanup that segfaults on macOS):

```
PYTHONPATH=<checkout>:<plugin dir> python -m pytest -p noclean \
  "tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py::test_hybrid_gdn_pd_exchange_only_the_running_state" -q
```

Base `e2bab26e51` (with the new test file copied in):

```
E       assert [2, 3, 4] == [2]
E       assert [0, 0, 0, 5, 6, 8] == [5]
E       assert [2, 3] == [2]
E       assert [0, 0, 0, 5, 7] == [5]
4 failed, 1 warning in 19.05s
```

This PR (`4967ff9ff4`):

```
4 passed, 1 warning in 5.79s
```

The three Mooncake connector test files, before and after:

```
pytest -p noclean tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py \
  tests/v1/kv_connector/unit/test_mooncake_connector_hma.py \
  tests/v1/kv_connector/unit/test_mooncake_connector.py -q
base: 66 failed, 158 passed, 1 warning in 179.60s (0:02:59)
PR:   66 failed, 162 passed, 1 warning in 45.64s
```

The 66 failures are identical before and after. All are `NotImplementedError` from `current_platform.set_device` when the worker is built on macOS, where no platform is detected. CI builds the worker on a real platform.

<details>
<summary>Pairing before and after, with block lists from real schedulers fed to the real <code>_build_transfer_params</code> (local probe)</summary>

`(P block, D block, slots)` for the GDN region, B = 16. The "none" rows use a 10-token prompt; the align rows use 53 tokens, so P prefills 48 + 4.

```
mode   kP kD  P sends (base)        D asks (base)  base pairs                 fixed pairs
none   2  0   [2, 3, 4]             [2]            (4, 2, 1)                  (2, 2, 1)
none   2  2   [2, 3, 4]             [2, 3, 4]      (2, 2, 3)                  (2, 2, 1)
none   1  3   [2, 3]                [2, 3, 4, 5]   P num blocks less than D   (2, 2, 1)
none   0  0   [2]                   [2]            (2, 2, 1)                  (2, 2, 1)
align  2  0   [0, 0, 0, 5, 6, 8]    [5]            (8, 5, 1)                  (5, 5, 1)
align  3  3   [0, 0, 0, 5, 6, 7, 9] [5, 6, 7, 8]   (5, 5, 3) + (9, 8, 1)      (5, 5, 1)
align  1  3   [0, 0, 0, 5, 7]       [5, 6, 7, 8]   P num blocks less than D   (5, 5, 1)
align  0  0   [0, 0, 0, 6]          [5]            (6, 5, 1)                  (6, 5, 1)
align  2  1   [0, 0, 0, 6, 7, 9]    [5, 6]         (7, 5, 1) + (9, 6, 1)      (6, 5, 1)   # prefill checkpoint, alignment 32
```

</details>

I haven't run this end to end on GPUs yet. The fix only changes which block ids the scheduler sends, but a P/D accuracy run with a hybrid GDN model and mismatched MTP lengths is the real proof, and I'll add it here once it lands.

## Duplicate check

- Open PRs: searched `mooncake mamba speculative`, `mooncake clip_ssm_state_blocks`, `mooncake GDN MTP`, `mooncake scratch`, `mooncake num_speculative_blocks`, `mooncake hybrid spec decode`. None strip Mooncake's scratch slots.
- I grepped the diffs of the open PRs that touch `mooncake_connector.py`: #53078, #62255, #68994, #59330, #65584, #63998, #62504. None change the Mamba pairing. #62504 adds `num_speculative_blocks` to a layout fingerprint, which would refuse mismatched P/D instead of pairing them.
- Issues: searched `mooncake mamba speculative`, `mooncake GDN wrong output`, `P num blocks less than D`. Nothing reports this bug.
- Nothing merged after `e2bab26e51` touches this path.

AI assistance (Claude) was used to find, fix and test this; I reviewed every changed line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
