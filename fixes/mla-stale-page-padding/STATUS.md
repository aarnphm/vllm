# STATUS: bug 16, stale MLA alignment padding after a spec rewrite

- status: fixed
- commit: `f51efcd22ff228003025df90112cc18b18fc9db5` on `aarnphm/mla-stale-page-padding` (base `e2bab26e51`), not pushed. Signed off (DCO), no Co-Authored-By, no Generated-with line.
- files changed:
  - `vllm/v1/kv_cache_interface.py` (+7 -3): `MLAAttentionSpec.__post_init__` derives `page_size_padded` from the current geometry whenever `alignment` is set, and sets it to `None` when the page is already aligned.
  - `tests/v1/core/test_kv_cache_utils.py` (+30): `test_mla_alignment_padding_follows_rewritten_geometry`, two cases, asserts `replace(spec, ...).page_size_bytes` equals a fresh spec's.

## Bug at base (re-verified)

- `repro_stale_padding.py` at base: `bs=48 ... page 8640 (fresh spec would give 6336)`, `copy == fresh? False`, `bs=96 AssertionError`. With the fix: `page 6336`, `copy == fresh? True`, `bs=96 page 12672`.
- `repro_dsv41_pp.py` at base: `copied c4 page_size_bytes -> AssertionError`. With the fix: `copied c4 page 33792`. This script uses ratios 128 and 4, which DeepSeek-V4.1 rejects (it allows 0, 1, 2), so it shows the mechanism, not the V4.1 relay itself.
- History: the conditional write came with DeepSeek-V4 (#40860, `_apply_alignment_padding`). #61452 (`43c1db8c9f`) moved it into `MLAAttentionSpec.__post_init__` unchanged. `unify_kv_cache_spec_page_size` documents the rule the fix follows: MLA pages are padded only by their own `alignment`.

## Rewrite sites at e2bab26e51 (task item 1)

Only six producers set `alignment` (all DeepSeek-V4/V4.1): `models/deepseek_v4/attention.py:827` (compressed KV), `:874` (indexer), `models/deepseek_v4/compressor.py:183` (compressor state), `models/deepseek_v41/attention.py:1306` (compressed KV), `:1378` (indexer), `v1/attention/backends/mla/sparse_swa.py:128` (sparse SWA). `scan_sites.py` lists every producer and rewrite site (`scan_sites.txt`). `site_matrix.py` replays each rewrite on the real producer geometries, blocks 16-256 (`site_matrix.txt`):

| site | rewrite | wrong at base (real geometries) |
|---|---|---|
| `models/deepseek_v41/nvidia/pipeline.py:143` `PipelineCache.get_kv_cache_config` | replica spec built at model construction, re-sized to the final block | yes: 38/408 block-size changes assert (V4.1 528 B, 288 B, 132 B, 68 B rows growing from a padded 16/32-token block) |
| `v1/core/kv_cache_planning.py:381` unify ratio branch | `block_size *= ratio` | same 38 asserts when reached; only outside the packed (block-outermost) path. With or without the fix a padded MLA spec here ends in an AssertionError at `:392` (pre-existing, not this bug) |
| `models/deepseek_v41/nvidia/pipeline.py:199` `PipelineRelay.configure` | `tokens_per_state` of the source | 18/90 assert, all from a compress-ratio-0 first layer. V4.1-Flash (ratios 0,0,2x18,1x20; sources 2/8/14/20) never starts a relaying stage on one, so latent |
| `v1/kv_cache_interface.py:217` `get_block_stride_alignment`; `v1/worker/utils.py:304` builder spec | shrink to kernel block | no (0/201): an integer shrink keeps an unaligned page unaligned |
| `v1/core/kv_cache_planning.py:768-777` local-attention promotion | block change + explicit `max(old padded, new unpadded)` | no (0/50) |
| `v1/core/kv_cache_planning.py:1059` `_ensure_min_page_size` | block and padding both scaled | explicit value; replaced at base when the scaled page is unaligned. The fix also replaces it when aligned (see decisions) |
| `v1/hisparse/binding.py:91`, `layout.py:254` | kernel/pool block | no aligned spec reaches HiSparse (DSv3.2, GLM5-Next, Qwen4-exp, Kimi-K3 only) |
| `worker/utils.py:495`, `flashmla_sparse.py:185`, `mla/sparse_utils.py:63`, `worker/gpu/model_runner.py:681` | `block_stride_alignment` / `dcp_sharded` | no: page unchanged |
| `mla_attention.py:1916`, glm5next `:135`, dots3 `:242`, minimax, kimi_k3, `kv_cache_planning.py:1148` | MLA specs without `alignment` | not affected by this bug |
| offloading `canonical_mapping.py:398`, `offloading/worker.py:136` | `replace(page_size_padded=None)` to read the unpadded page | returns the aligned page for an aligned spec, at base and with the fix (unchanged) |

No real geometry gives the 36% over-allocation; it needs a block change that is not an integer ratio (64 -> 48). Six matrix rows shrink a c128 spec below 128 tokens (0 states); every c128 backend uses 256-token kernel blocks, so that is unreachable.

## Tests (one process at a time; `P=<scratch>/venv/bin/python`, `PYTHONPATH=<worktree>:<scratch>/plug`, `-p noclean -q`)

Fail-before was run by writing `git show HEAD:vllm/v1/kv_cache_interface.py` over the file (not `git stash`, which all worktrees share), then restoring the fix.

- New test, base: `2 failed, 1 warning in 0.25s` (`assert 8640 == 6336`; `AssertionError` at `kv_cache_interface.py:512`). Fix: `2 passed, 1 warning in 0.16s`. (`fail_before.txt`, `pass_after.txt`)
- With `VLLM_TARGET_DEVICE=cpu` (platform detection fails without it on this Mac):
  - `tests/v1/core/test_kv_cache_utils.py`: base `11 failed, 105 passed`, fix `9 failed, 107 passed`. Only difference: the new test. The 9 shared failures are `test_num_hash_lookahead_tokens[*]`, rejected by the CPU platform ("does not yet support: speculative decoding").
  - `tests/v1/core/test_kv_cache_planning.py`: `75 passed` base and fix.
  - `tests/v1/core/test_contiguous_kv_packing.py`: `34 passed` base and fix.
- Without `VLLM_TARGET_DEVICE`: same failure sets base vs fix (37/43 env failures "Failed to infer device type").
- Lint: `uvx ruff check` and `uvx ruff format --check` pass; pre-commit hooks passed on commit.
- CI: the file is `pytestmark = pytest.mark.cpu_test`, selected by the `cpu-small` step `pytest -v -s -m 'cpu_test' v1/core` (`.buildkite/test_areas/misc.yaml`).
- Not run: `tests/v1/kv_connector/unit/test_nixl_connector.py -k test_register_packed_dsv4_mla_cache_as_single_region` (imports `ray`; the brief rules Ray out).

## Bug 17 overlap (coordinator request)

- This fix does not cover bug 17. `bug17/repro_planner.py` (copy of the bug 17 script, run from this dir) against this branch: `gemma3 5` and `gptoss 1` both print `PLANNING FAILED: AssertionError` at `kv_cache_planning.py:392` -> `kv_cache_interface.py:512`, the same frames as the bug 17 agent's base runs. (`bug17/fix-gemma3-5.txt`, `bug17/fix-gptoss-1.txt`)
- Overlap with #54599's third hunk (unify ratio branch scales `page_size_padded` with `block_size`): same failure class and same planner branch. No shared files.
- Not generalized: for a plain `AttentionSpec`, `page_size_padded` is an absolute page that the caller picks, and `replace` cannot tell a carried value from a chosen one. `kv_cache_planning.py:1148` deliberately pairs a smaller block with a larger absolute page, so a spec-level rescale rule would break it. Bug 17 belongs at the call site, as #54599 does.
- If both land: #54599's scaled value is ignored for aligned MLA specs (they re-derive), which matches the MLA rule.

## Duplicate check

- PRs: `gh pr list --repo vllm-project/vllm --state open --search` "page_size_padded MLAAttentionSpec" (none), "stale page_size_padded" (#62453, #54599), "MLA alignment padding copy_with_new_block_size" (none).
- Issues: "page_size_padded alignment" (#54877), "MLAAttentionSpec page_size_padded AssertionError" (none).
- #62453 edits NIXL host buffers; #54599 see above; #54877 is the padded-page reshape on a kernel-block split. None changes `MLAAttentionSpec.__post_init__`.

## GPU checks still needed

- DeepSeek-V4.1 PP relay with a construction-time block size that pads (expected: base fails at startup in `page_size_bytes`, fix starts), if the platform accepts `--block-size 16` and raises it after model load:
  `vllm serve deepseek-ai/DeepSeek-V4.1-Flash --tensor-parallel-size 4 --pipeline-parallel-size 2 --kv-cache-dtype fp8_ds_mla --block-size 16`
  The relay needs packed FlashMLA caches, PCP=DCP=1, no microbatching, no spec decode, no KV transfer. Then one greedy request at the default block size to confirm no change.
- `pytest tests/v1/kv_connector/unit/test_nixl_connector.py -k test_register_packed_dsv4_mla_cache_as_single_region` where `ray` is installed.

## Decisions for aarnphm

1. Behavior change: an explicit `page_size_padded` on a spec with `alignment` is now always replaced by the derived value. No producer or test passes one. `_ensure_min_page_size` does (hidden-state extraction with TP); for a padded MLA spec whose scaled page is aligned it used to keep the scaled common page and now gets the derived one. Reaching it needs hidden-state extraction on DeepSeek-V4 outside the packed layout.
2. Bug 17 stays separate (call-site fix, #54599 or its rebase).
3. The unify ratio branch still raises `AssertionError` (not `NotImplementedError`, so no fallback) for a padded MLA spec, at base and with the fix. Possible follow-up.
