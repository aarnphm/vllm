status: fixed
branch: aarnphm/backend-per-kind-keys (worktree ~/workspace/worktrees/vllm/backend-per-kind-keys), base e2bab26e51, not pushed
commit: 73d595add177c8dfa43c192a02d0daefddc35c6c

## Bug 18 at the base (confirmed by running, not only by reading)
- Config accepts all 10 KVCacheSpecKind keys (vllm/config/attention.py:307 at base).
- Selector `get_attn_spec_kind` yields 6: cross_attention, encoder_only_attention, full_attention, mla_attention, sliding_window, sliding_window_mla (vllm/v1/attention/selector.py:70-108; lookup at :187-197).
- Accepted and never read: chunked_local_attention, mamba, sink_full_attention, unknown.
  - chunked-local: ChunkedLocalAttention calls get_attn_backend(head_size, dtype, kv_cache_dtype) (chunked_local_attention.py:103), so it resolves as full_attention.
  - mamba: Mamba layers go through get_mamba_attn_backend (layers/mamba/abstract.py:110-112) and never read the map.
  - sink: the only in-tree StaticSinkAttention (openpangu.py:625-638) pins FlashAttentionDiffKVBackend; no selection happens.
- Llama-4 effect (real ChunkedLocalAttention and Attention constructors on CPU, backend recorded at the selector's cached entry): map {"full_attention": "TRITON_ATTN", "chunked_local_attention": "FLEX_ATTENTION"} -> chunked-local layer selects TRITON_ATTN, NoPE layer selects TRITON_ATTN. The vllm serve parser accepted the chunked_local_attention entry silently.
- PR #48012 (which added backend_per_kind) listed chunked-local and sink as "not targetable" under Limitations; the validator still accepted them.

## Fix chosen: config rejects keys the selector never reads (the smaller of the two)
- Validator accepts only the 6 selector kinds; other KVCacheSpecKind values raise "backend_per_kind cannot target '<kind>': backend selection never assigns that kind. Chunked-local layers select as 'full_attention', and Mamba layers do not use attention backends. Valid kinds are: ..."; non-kinds keep "Unknown KV cache group kind ...". Field docstring lists the valid keys and states the chunked-local -> full_attention mapping.
- Not done (aarnphm to decide): make chunked_local_attention targetable via a use_chunked_local flag (ChunkedLocalAttention -> get_attn_backend -> get_attn_spec_kind). ~10 lines, but it changes what an existing full_attention entry does on Llama 4 (it would stop steering chunked-local layers). mamba/sink/unknown must be rejected either way.

## Files changed
- vllm/config/attention.py (+27 -5)
- tests/v1/attention/test_backend_per_kind.py (+25): test_backend_per_kind_accepts_only_selectable_kinds

## Tests (CPU, macOS; one process at a time)
Command prefix: cd <worktree> && VLLM_TARGET_DEVICE=cpu PYTHONPATH=<worktree>:<scratch>/plug <scratch>/venv/bin/python -m pytest -p noclean
(VLLM_TARGET_DEVICE=cpu is needed because the venv has no installed vllm package, so platform detection fails without it.)
- new test, base code: 1 failed, 1 warning in 0.06s (extra accepted: sink_full_attention, mamba, chunked_local_attention, unknown) -> fail_before.txt
- new test, fix: 1 passed, 1 warning in 0.23s -> pass_after.txt
- tests/v1/attention/test_backend_per_kind.py: base 26 passed, 1 warning in 0.06s; fix 27 passed, 1 warning in 0.06s
- tests/engine/test_arg_utils.py -k attention: base 3 passed, 117 deselected; fix 3 passed, 117 deselected
- no failures before or after, so no environment failures to report
- repro: VLLM_TARGET_DEVICE=cpu PYTHONPATH=<worktree> <venv python> repro_kinds.py -> repro_base.txt (bug) / repro_fix.txt (fix: 6 accepted, 0 never read, CLI exits with valid kinds)
- lint: uvx ruff check + ruff format --check: pass; check_docstring_style.py: pass; commit-time pre-commit (incl. mypy --python-version 3.11, test tethering): pass

## GPU checks still needed
- None required: the change is config validation only.
- Optional regression: pytest tests/v1/e2e/general/test_attention_backend_per_kind.py (gemma-3-1b-it, CUDA). It uses only full_attention and sliding_window, which stay valid.

## Before opening a PR
- Run vLLM's pr-checklist skill (AGENTS.md requires it); PR.md is the body draft.
