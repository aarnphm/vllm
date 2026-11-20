---
agent: mooncake-mamba-slots
bug: 4 (Mooncake P/D pairs Mamba state slots by tail, scratch slots included)
date: 2026-11-20
---

status: fixed
commit: 4967ff9ff4 on aarnphm/mooncake-mamba-slots (base e2bab26e51), not pushed

## Files changed

- vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
  (+22 -5: `MooncakeConnectorScheduler._ssm_spec_blocks`, and SSM groups go
  through `clip_ssm_state_blocks` in `get_sw_clipped_blocks`)
- tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py
  (+80 -2: new `test_hybrid_gdn_pd_exchange_only_the_running_state`, and
  `make_hybrid_gdn_kv_cache_config` takes `num_speculative_blocks`)

## Bug re-verified at e2bab26e51

- The recorded repro (research/raw/kv-groups/scripts/connectors/repro_group_bugs.py)
  prints the same output as its .out: `(23, 31, 1)`, `(21, 31, 3)`,
  `P num blocks less than D`.
- probe_pairing.py (here) takes block lists from real P and D schedulers and
  feeds them to the real `_build_transfer_params`. Output: probe_base.log
  (base) and probe_fix.log (fix). The bug shows in "none" and align mode; the
  fix pairs P's running state with D's in all 9 cases (one slot per layer),
  including P/D draft lengths 2/0, 2/2, 1/3, 3/3, 0/0 and a prefill-checkpoint
  case.

## Tests (macOS, CPU, venv c96b83d0, `-p noclean`)

New test, fail-before (git archive of HEAD, final test file copied in):
`4 failed, 1 warning in 19.05s`, with
`assert [2, 3, 4] == [2]`, `assert [0, 0, 0, 5, 6, 8] == [5]`,
`assert [2, 3] == [2]`, `assert [0, 0, 0, 5, 7] == [5]` (newtest_base.log)

New test, pass-after: `4 passed, 1 warning in 5.79s` on 4967ff9ff4 (newtest_fix.log)

Nearest files (test_mooncake_connector_hybrid_mamba.py, _hma.py, test_mooncake_connector.py):
- base: `66 failed, 158 passed, 1 warning in 179.60s (0:02:59)` (baseline_tests.log)
- fix:  `66 failed, 162 passed, 1 warning in 45.64s` (fix_tests.log)
- The failure sets are identical (baseline_failed.txt = fix_failed.txt). All 66
  are environment failures: `NotImplementedError` from
  `current_platform.set_device`, because vLLM resolves to UnspecifiedPlatform
  on this Mac when it builds a Mooncake worker. They include every
  worker-building test in the hybrid Mamba file.

Lint: `uvx ruff check` and `uvx ruff format --check` pass on both files.
The pre-commit hooks passed on commit, including mypy and the CI test-tethering check.

## CI

The new test is `cpu_test`, so `(CPU) V1 Others` runs it
(`.buildkite/test_areas/misc.yaml`, `pytest -m 'cpu_test' v1/kv_connector/unit`).
This PR triggers that step through `tests/v1`. That step and the H200 KV
connector step both exclude `vllm/distributed/kv_transfer/` from their
triggers, so a later change to the Mooncake connector alone would not re-run
this test. This gap already exists and is not changed here.

## GPU checks still needed

The fix only changes which block ids the scheduler sends, and the CPU test
proves that at the block-list level. Output correctness needs a real P/D run
with Mooncake (RDMA nodes) and a hybrid GDN model with an MTP head, for example
Qwen/Qwen3-Next-80B-A3B-Instruct (TP2 per role). The Mooncake accuracy harness
(`tests/v1/kv_connector/mooncake_integration/run_accuracy_test.sh`) applies
`VLLM_SERVE_EXTRA_ARGS` to both roles, so the mismatched cases need a manual
launch:

1. P: `VLLM_MOONCAKE_BOOTSTRAP_PORT=$BP vllm serve $MODEL --port $PP -tp 2 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'`
2. D: `vllm serve $MODEL --port $DP -tp 2 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'`
   (add `"kv_connector_extra_config":{"device_name":"<hca>"}` where the
   cluster needs it pinned to one HCA, as run_accuracy_test.sh does on b200)
3. Proxy: `python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py --port $XP --prefill http://P:$PP $BP --decode http://D:$DP`
4. `lm_eval --model local-completions --tasks gsm8k --model_args model=$MODEL,base_url=http://proxy:$XP/v1/completions,num_concurrent=32 --limit 500`
5. Repeat with the spec config moved to D only (`num_speculative_tokens: 3`),
   and with prefix caching on and off. Compare against the same model served
   without P/D.

Expected on base: lower gsm8k with MTP on P only (silent wrong state); with
MTP on D only, hybrid requests fail with "P num blocks less than D". Expected
with the fix: both match the non-P/D baseline.

## For aarnphm

- The commit carries `Signed-off-by: Aaron Pham <contact@aarnphm.xyz>` (DCO),
  like the sibling branch aarnphm/nixl-dcp-group-index. No Co-Authored-By and
  no Generated-with line in the commit.
- PR.md says "I reviewed every changed line". That line is only true once you
  have reviewed the diff.
- Name: I kept `get_sw_clipped_blocks` to keep the diff small. NIXL's name
  for the same job is `get_exchange_clipped_blocks`. A rename touches 2 tests
  in test_mooncake_connector_hma.py.

## Repeat (one Python process at a time, narrow test ids)

The fail-before tree copy was deleted after the run (coordinator memory
rule, 2026-11-20). To recreate it, run the steps one after another:

```sh
W=~/workspace/worktrees/vllm/mooncake-mamba-slots
B=<scratch>/fixes/mooncake-mamba-slots/base
PY=<scratch>/venv/bin/python
PLUG=<scratch>/plug
T="tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py::test_hybrid_gdn_pd_exchange_only_the_running_state"
# fail-before: base source, new test file
mkdir -p $B && git -C $W archive e2bab26e51 vllm tests pyproject.toml | tar -x -C $B
git -C $W show 4967ff9ff4:tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py \
  > $B/tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py
(cd $B && PYTHONPATH=$B:$PLUG $PY -m pytest -p noclean "$T" -q)   # 4 failed
rm -rf $B
# pass-after
(cd $W && PYTHONPATH=$W:$PLUG $PY -m pytest -p noclean "$T" -q)   # 4 passed
# pairing probe (real schedulers -> real _build_transfer_params); use
# PYTHONPATH=$B:$PLUG before `rm -rf $B` to get probe_base.log instead
PYTHONPATH=$W:$PLUG $PY <scratch>/fixes/mooncake-mamba-slots/probe_pairing.py
```

The three-file suite comparison above ran before the memory rule. Do not
rerun it whole on this Mac. In CI, `(CPU) V1 Others` runs the cpu_test part.
