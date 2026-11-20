status: fixed (2026-11-20)

commit: af3c72b1d1da73a3fa9af168efb4e20165b5841d on aarnphm/nixl-conv-spec-mismatch
  (base e2bab26e51, one commit, Signed-off-by only, not pushed)
  [Bugfix][NIXL] Refuse P/D peers whose Mamba conv states have different rows

fix: option (b). P and D advertise `conv_rows` in NixlAgentMetadata
(NIXL_CONNECTOR_VERSION 17 -> 18); `add_remote_agent` raises RuntimeError
before registering anything when the local and remote conv rows differ.
Option (a) rejected: in the DS layout a different row count changes the
per-dim stride, so a remap needs one descriptor per conv dim or a staging
buffer plus a kernel on D.

files changed:
- vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py (field, version 18)
- vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (advertise conv_rows, check in add_remote_agent)
- tests/v1/kv_connector/unit/test_nixl_desc_geometry.py (new test; `_real_model_side` takes `layers`; `_make_remote_meta` advertises the local conv rows)
- docs/features/nixl_connector_compatibility.md (num_speculative_tokens bullet)

tests (job venv, HF_HUB_OFFLINE=1, -p noclean; `hmashim` = plug/hmashim.py, makes
macOS UnspecifiedPlatform report hybrid KV cache support so hybrid plans build):
- new test, fail-before on e2bab26e51 code: `E   Failed: DID NOT RAISE RuntimeError` / `1 failed, 1 warning in 6.62s`  (log: failbefore_newtest.log)
- new test, pass-after: `1 passed, 1 warning in 2.31s`  (log: passafter_newtest.log)
- test_nixl_desc_geometry.py -k 'conv_rows or (real_model_layout_pull and (Qwen3 or Nemotron or Kimi)) or hetero_ppl or hetero_non_divisible or hetero_block_split or mla_hybrid or dcp_rejects or dcp_pull_with_block_size_ratio or csa_linear'
  - hmashim: base `300 passed, 62 deselected, 1 warning in 12.00s`; fix `301 passed, 62 deselected, 1 warning in 13.32s` (base_geom.log, fix_geom.log)
  - plain: base `243 failed, 57 passed`; fix `244 failed, 57 passed`; only diff = the new test, failing like its 6 real-model Mamba neighbours ("Failed to promote local KV cache specs", HMA off on macOS); the rest are `assert self._is_hma_required` (base_worker.py:833). Environment, same cause. (base_geom_plain.log, fix_geom_plain.log)
- test_nixl_connector_hma.py + test_nixl_push_connector.py -k 'mamba or ssm or conv or handshake or validate': base and fix `45 passed, 154 deselected` (base_hma_push.log, fix_hma_push.log)
- end-to-end CPU evidence (silent_corruption_at_base.py): base accepts the handshake, 8191/8192 conv window dims on D differ from P, SSM state equal; fix refuses with the conv rows message (silent_corruption_base.out, silent_corruption_fix.out)
- lint: ruff check/format, typos, markdownlint-cli2, docstring style, mypy (pre-commit on commit): all passed

not run here (memory rule, Ray import): tests/v1/kv_connector/unit/test_nixl_connector.py.
No test there builds a Mamba worker that reaches add_remote_agent; expected unchanged. Run on Linux:
  .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_nixl_connector.py -k "handshake or mamba" -q
  .venv/bin/python -m pytest -m cpu_test tests/v1/kv_connector/unit/test_nixl_desc_geometry.py -k "conv_rows or real_model_layout_pull" -q

rebuild fail-before tree (deleted to free disk):
  git -C <worktree> archive e2bab26e51 vllm tests pyproject.toml | tar -x -C <tmp>
  cp <worktree>/tests/v1/kv_connector/unit/test_nixl_desc_geometry.py <tmp>/tests/v1/kv_connector/unit/

GPU checks still needed: none for correctness (handshake-only change). Optional E2E:
  two engines with NIXL, Qwen3-Next, P --speculative-config '{"method":"mtp","num_speculative_tokens":1}',
  D with 3: expect handshake_setup_failed with the conv rows message; with equal n, GSM8K through
  the P/D proxy unchanged.

duplicate check: no open PR or issue fixes this. Same code, different bugs: #61531, #65094;
draft #65232 rewrites this area and also bumps the version to 18. #52627 is a Mooncake-store path.

for aarnphm to decide:
- refusal (b) over remap (a); the doc bullet now says these models must match num_speculative_tokens
- `conv_rows` defaults to 0, so a hand-built or old-version Mamba peer is refused (fail closed)
- MoRIIO computes remote conv offsets the same way (moriio_layout.py:515); not checked or touched
