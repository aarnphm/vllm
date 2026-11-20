# [Bugfix][NIXL] Refuse P/D peers whose Mamba conv states have different rows

## Overview

Claude and I hit a silent P/D corruption in the NIXL connector: when prefill and decode run different `num_speculative_tokens` on a Mamba2/GDN/KDA hybrid, decode reads its conv state from the wrong bytes and nothing errors.

- The conv state holds `conv_kernel - 1 + num_speculative_tokens` rows ([Mamba2](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/model_executor/layers/mamba/mamba_utils.py#L211-L213), [GDN](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/model_executor/layers/mamba/mamba_utils.py#L291-L294), [KDA](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/model_executor/layers/mamba/mamba_utils.py#L323-L325)).
- D places P's conv sub-projections with its **own** row count: `_build_mamba_remote` calls `remote_conv_offsets` on the local split ([base_worker.py#L2242](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L2242), [ssm_conv_transfer_utils.py#L71-L104](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py#L71-L104)).
- Nothing compares conv geometry first: `_validate_remote_agent_handshake` runs no per-region check for non-MLA Mamba hybrids ([#L3136-L3151](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L3136-L3151)), the compat hash leaves `num_speculative_tokens` out ([metadata.py#L142-L151](https://github.com/vllm-project/vllm/blob/e2bab26e51/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py#L142-L151)), and the docs list it as safe to differ ([compatibility doc](https://github.com/vllm-project/vllm/blob/e2bab26e51/docs/features/nixl_connector_compatibility.md?plain=1#L97)).
- Qwen3-Next at TP4, P on MTP n=1, D on n=3: P writes Q/K/V at `(0, 4096), (4096, 4096), (8192, 8192)`, D reads `(0, 6144), (6144, 6144), (12288, 12288)`. Both sides pad the Mamba page to the attention page, so even the page sizes match.
- D's V read runs past P's conv state into P's SSM state. The SSM state itself arrives intact (its address uses P's conv size), so only the conv window is garbage. That seems like the worst kind: plausible tokens, no error.

## Fix

- P and D advertise `conv_rows` in `NixlAgentMetadata`, and `add_remote_agent` refuses a peer with a different count before it registers anything for it. Bumped `NIXL_CONNECTOR_VERSION` to 18.
- `add_remote_agent` is shared, so this covers the pull reader and the push writer.
- Docs: `num_speculative_tokens` may still differ, except for Mamba2/GDN/KDA hybrids.
- Claude looked at reading P's conv state at P's row count instead. It would be enough semantically (prefill only writes the first `conv_kernel - 1` rows, and the first decode step reads those), but in the DS layout each conv dim's rows are contiguous, so a different row count changes the per-dim stride. One READ per sub-projection can't re-lay that; it needs one descriptor per conv dim (2048 per layer at TP4, 6 bytes each) or a staging buffer plus a kernel on D. I don't think that's worth it short term, so this PR refuses.

**IMPORTANT**: a pair that differs in `num_speculative_tokens` on these models used to "work" (with corrupted decodes) and now fails the handshake. That's the point, but it is a behaviour change for anyone running such a pair.

_Note that P and D only see each other's metadata at the handshake, so there is no earlier place to refuse; requests to that P fail through `kv_load_failure_policy` like any other handshake error._

_Note that MoRIIO also computes remote conv offsets from the local split (`moriio_layout.py:515`). I haven't checked whether anything there catches this._

## Validation

New test `test_real_model_pull_refuses_peer_with_other_conv_rows` in `tests/v1/kv_connector/unit/test_nixl_desc_geometry.py`: it plans P (conv rows 4) and D (conv rows 6) with the engine's KV planner, builds real `NixlConnectorWorker`s, and hands P's real handshake payload to D. `_make_remote_meta` in the same file now gives its hand-built peers the local conv rows, since the new field fails closed.

The Mac I ran on detects `UnspecifiedPlatform`, which reports no hybrid KV cache support, so every hybrid real-model test fails there before it reaches the connector (CUDA and CPU platforms report support, so CI runs them). I ran with a one-line pytest plugin, `hmashim`, that sets `UnspecifiedPlatform.support_hybrid_kv_cache` to `True`; the plain runs are below too.

```
HF_HUB_OFFLINE=1 PYTHONPATH=<tree>:<plugin dir> python -m pytest -p hmashim -p noclean \
  "tests/v1/kv_connector/unit/test_nixl_desc_geometry.py::test_real_model_pull_refuses_peer_with_other_conv_rows" -q
```

- fail-before (e2bab26e51 code, new test file): `E   Failed: DID NOT RAISE RuntimeError`, then `1 failed, 1 warning in 6.62s`
- pass-after: `1 passed, 1 warning in 2.31s`

Neighbouring Mamba handshake and geometry tests in the same file, base vs fix, `-k 'conv_rows or (real_model_layout_pull and (Qwen3 or Nemotron or Kimi)) or hetero_ppl or hetero_non_divisible or hetero_block_split or mla_hybrid or dcp_rejects or dcp_pull_with_block_size_ratio or csa_linear'`:

- with `hmashim`: base `300 passed, 62 deselected, 1 warning in 12.00s`, fix `301 passed, 62 deselected, 1 warning in 13.32s` (the extra one is the new test)
- without it: base `243 failed, 57 passed`, fix `244 failed, 57 passed`. The only difference is the new test, which fails like its 6 real-model Mamba neighbours (`ValueError: Failed to promote local KV cache specs to one unified type.`); the rest stop at `assert self._is_hma_required` in `NixlConnectorWorker.__init__`. Both are the missing HMA support on macOS.
- `test_nixl_connector_hma.py` and `test_nixl_push_connector.py` with `-k 'mamba or ssm or conv or handshake or validate'`: `45 passed, 154 deselected` on base and on fix.

The corruption at base, end to end on CPU: real planner, real registration, P's real handshake payload, D's real READ setup, NIXL READs replayed as memmove. Qwen3-Next GDN at TP1, P on MTP n=1, D on n=3, P's conv window filled with known values:

```
padded page bytes P/D: [2359296] [2359296]
handshake accepted
conv window dims that differ from P: 8191/8192 (first bad dim 1)
dim 1 window on P: [37.0, 52.0, 119.0]
dim 1 window on D: [119.0, 11.0, 75.0]
SSM state equal: True
```

With the fix the same script stops at the handshake:

```
handshake refused: Mamba conv state rows differ between P and D: local=6, remote=4. The conv state holds conv_kernel - 1 + num_speculative_tokens rows, so P and D must use the same num_speculative_tokens.
```

Not run here: `test_nixl_connector.py` (it imports Ray, and the laptop was out of memory). No test there builds a Mamba worker that reaches `add_remote_agent`, so I don't expect a change, but CI has to confirm. No GPU run either: the change is handshake-only, and a GSM8K P/D pair with n=1 against n=3 would now just show the refusal.

## Duplicate check

- `gh pr list --state open --search` with "nixl mamba conv num_speculative_tokens", "nixl conv state spec decode mismatch", "ssm_conv_transfer_utils" and "nixl hybrid mamba handshake"; `gh issue list --search` with "nixl mamba conv speculative" and "P/D disaggregated mamba conv_state num_speculative_tokens". Nothing fixes this.
- Same code, different bugs: #61531 (Mamba2 conv transfer across TP when B/C are replicated) and #65094 (reject P/D pairs whose KV cache groups scale differently; it also adds a `NixlAgentMetadata` field). Draft #65232 (spec-driven TP slicing) rewrites this area and also bumps the version to 18, so whichever lands second rebases.
- #52627 (K3 silent corruption with the Mooncake store) is a different path.

AI assistance was used (Claude Code); I reviewed every changed line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
