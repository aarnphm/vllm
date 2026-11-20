# [Bugfix][Core] Re-derive MLA alignment padding when a spec is rewritten

## Overview

Claude and I found this while auditing every place vLLM rewrites a KV cache spec.

`MLAAttentionSpec.__post_init__` turns `alignment` into `page_size_padded`, but only writes the field when the page needs padding.
`dataclasses.replace` and `copy_with_new_block_size` pass the old `page_size_padded` back into `__init__`.
So when the new page happens to be aligned, it keeps the padding of the old geometry.
The page then comes out larger than a fresh spec's, or smaller than its own data, and `page_size_bytes` fails its assertion.

Only specs that set `alignment` are affected, which today means the DeepSeek-V4 and V4.1 caches (compressed KV, indexer, compressor state, sparse SWA).

Claude replayed every site that rewrites one of these specs with the real record sizes, alignments and blocks of 16 to 256 tokens:
- `PipelineCache.get_kv_cache_config` (`deepseek_v41/nvidia/pipeline.py:143`) re-sizes the cross-stage replica spec it built at model construction. 38 of 408 block changes crash, all V4.1 records growing from a padded 16- or 32-token block. The unify ratio branch (`kv_cache_planning.py:381`) does the same rewrite.
- `PipelineRelay.configure` (`pipeline.py:199`) swaps `tokens_per_state`. It crashes only when a stage starts on a ratio-0 layer, which the V4.1-Flash layout never does, so I think this one is latent.
- Kernel-block shrinks, local-attention promotion and HiSparse never produce a wrong page here.

With real geometries this is a startup `AssertionError`, not a silent over-allocation.
The 36% over-allocation needs a block change that is not an integer ratio (64 to 48 tokens of 132 B indexer rows at 576 B alignment). No current site does that, but the spec API allows it.
I have not reproduced the crash in a real DeepSeek-V4.1 PP run yet.

## Fix

Claude proposes that, when `alignment` is set, `__post_init__` always derives the padding from the current geometry and clears it when the page is already aligned.
A rewritten spec now equals a freshly built one.
This follows the rule `unify_kv_cache_spec_page_size` already states: an MLA block stride can only be padded by its own `alignment`.
The conditional write dates from the DeepSeek-V4 PR (#40860); #61452 moved it into `__post_init__` unchanged.
Specs without `alignment`, including `HiddenStateCacheSpec`, are unchanged.

One behavior change to check: an explicit `page_size_padded` on an aligned spec is now always replaced by the derived value.
It was already replaced whenever the new page was unaligned, and no producer or test passes one.
`_ensure_min_page_size` (hidden-state extraction) does pass a scaled one. For a padded MLA spec it now gets the derived page in the aligned case too. That needs hidden-state extraction on DeepSeek-V4 outside the packed layout, which I have not tried.

## Validation

New test: `test_mla_alignment_padding_follows_rewritten_geometry` in `tests/v1/core/test_kv_cache_utils.py`.
It compares `replace(spec, ...).page_size_bytes` with a fresh spec for a block change (132 B rows, 64 to 48) and a `tokens_per_state` change (528 B V4.1 records, 2 to 1 at 32-token blocks).

```bash
PYTHONPATH=$PWD:$PLUG $PY -m pytest -p noclean -q \
  "tests/v1/core/test_kv_cache_utils.py::test_mla_alignment_padding_follows_rewritten_geometry"
```

- Before (base `kv_cache_interface.py`): `2 failed, 1 warning in 0.25s`, with `assert 8640 == 6336` and the `AssertionError` at `kv_cache_interface.py:512`.
- After: `2 passed, 1 warning in 0.16s`.

Nearest test files, with `VLLM_TARGET_DEVICE=cpu` on my Mac:
- `tests/v1/core/test_kv_cache_utils.py`: base `11 failed, 105 passed`, fix `9 failed, 107 passed`. The only difference is the new test; the 9 shared failures are `test_num_hash_lookahead_tokens[*]`, which the CPU platform rejects (no spec decode).
- `tests/v1/core/test_kv_cache_planning.py`: `75 passed` on base and fix.
- `tests/v1/core/test_contiguous_kv_packing.py`: `34 passed` on base and fix.

Not run yet: a DeepSeek-V4.1 PP serve with `--block-size 16` on GPU, and the NIXL DSv4 registration test (needs `ray`).

## Overlap with #54599

#54599's third hunk scales `page_size_padded` with `block_size` in the same unify branch, for every `AttentionSpec`.
That fixes the Gemma 3 / gpt-oss fp8 skip-layer crash, and this PR does not: the planner repro for that bug still fails at `kv_cache_planning.py:392` on this branch, same as on base.
I kept this MLA-only on purpose. For a plain `AttentionSpec`, `page_size_padded` is an absolute page the caller picks, and `replace` cannot tell a carried value from a chosen one (`kv_cache_planning.py:1148` pairs a smaller block with a larger page on purpose).
If both land, #54599's scaled value is ignored for aligned MLA specs, which re-derive their padding. The two PRs touch different files.

## Duplicate check

- Open PRs searched: "page_size_padded MLAAttentionSpec", "stale page_size_padded", "MLA alignment padding copy_with_new_block_size". Hits: #62453 (NIXL host buffers for hybrid Mamba) and #54599 (above).
- Issues searched: "page_size_padded alignment", "MLAAttentionSpec page_size_padded AssertionError". Hit: #54877 (padded page reshape on a kernel-block split, different code).
- No open PR changes `MLAAttentionSpec.__post_init__`.

AI assistance (Claude) was used for the investigation, the patch and this description. I reviewed the change and ran the commands above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
