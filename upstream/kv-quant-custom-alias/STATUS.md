# kv-quant-custom-alias (bug 11)

status: duplicate-open-PR (vllm-project/vllm#63691)
commit sha: none. Branch `aarnphm/kv-quant-custom-alias` is still at base `e2bab26e51`, worktree clean.
files changed: none. No PR.md: vLLM AGENTS.md says do not open a second PR for a fix that has an open one.
GPU checks still needed: none. No in-tree handler returns CUSTOM, Octave keeps 12-14, and the Triton kernel treats 12 and 15 the same (plain cast).

## Bug at base e2bab26e51

Still present: `vllm/v1/kv_cache_interface.py` has `OCTAVE_K3V4 = 12` (line 61) and `CUSTOM = 12` (line 69), and no `@unique`.

Repro (`repro_custom_octave.py`, `PYTHONPATH=<worktree>`, output `repro_before_base.out`):

```
mode: 12 OCTAVE_K3V4 is_octave: True
spec == octave spec: True hash equal: True
page bytes (plugin view): 32768
after RocmOctave.customize_spec: state_content_bytes= 136 page bytes= 17408
mooncake fingerprint key str(kv_quant_mode): 12 12
```

## Duplicate check (2026-11-20)

- `gh pr list --repo vllm-project/vllm --state open --search` with "KVQuantMode CUSTOM", "KVQuantMode alias", "CUSTOM OCTAVE_K3V4", "KVQuantMode unique" and "kv quant mode custom octave": each search returns #63691.
- `gh issue list --state all` with "KVQuantMode CUSTOM", "KVQuantMode" and "CUSTOM OCTAVE_K3V4 alias": no issue for this bug.
- #63691 "[Bugfix] KVQuantMode: give CUSTOM its own value (it aliased OCTAVE_K3V4)". Author JartX (org member). Opened 2026-10-26, last updated 2026-11-09. Labels `bug`, `needs-rebase`, `quantization`. Review required; no human review yet. Diff: `@unique` on `KVQuantMode`, `CUSTOM = 15`, and three asserts added to `test_backend_sentinel_properties` (`is_turboquant`, `is_octave`, `not any(m.is_custom_mode ...)`).

## What #63691 needs to land on today's main (checked in scratch, nothing pushed)

1. The rebase conflict is only the import line. The PR expects `from enum import Enum, IntEnum`; main has `from enum import IntEnum, StrEnum`. `git apply --check pr63691.diff` on a `git archive` of `e2bab26e51` fails only at `vllm/v1/kv_cache_interface.py:8`. The test hunk applies cleanly.
2. Its third new assert reads `KVQuantMode.is_custom_mode`. #64076 (`6044ab0609`) deleted that property as dead code. After a plain rebase the test fails with `AttributeError: 'KVQuantMode' object has no attribute 'is_custom_mode'`. The assert `mode.name == "CUSTOM"` keeps the intent and passes. `tests/test_attention_backend_registry.py::test_custom_is_not_alias_of_any_backend` already uses that check for `AttentionBackendEnum`.
3. Another open PR also wants value 15. #68945 (Kitty, WIP, opened 2026-11-20) adds `KITTY = 15`. #64358 (mxfp4, draft) adds `MXFP4 = 17`. #47618 and #53979 also mention `KVQuantMode` but add no member.
   - #68945's `kv_cache_interface.py` hunks apply cleanly on top of the rebased #63691 (`git apply`, offset 1 line; `merge_probe/`), so git reports no conflict.
   - Importing that merged file fails: `ValueError: duplicate values found in <enum 'KVQuantMode'>: CUSTOM -> KITTY`. Whichever of the two PRs lands second breaks every vLLM import.
   - Without `@unique` (simulated, `kitty_collision.py`), CUSTOM becomes `<KVQuantMode.KITTY: 15>`: the same bug with a different victim.
   - `@unique` alone on today's main (CUSTOM still 12) fails at import with `CUSTOM -> OCTAVE_K3V4`. So the guard would have stopped #59774 in CI.
   - Option: give CUSTOM a value well above the in-use range (for example 255), so that the next "next free value" member cannot collide with it. Nothing indexes a table by the mode value, so any unused int works.

## Integer persistence sweep (checks #63691's claim "Nothing persists these values")

- `csrc/`, `rust/`: nothing mirrors the enum (`rg -i "kv_?quant_?mode|KVQuantMode" csrc rust` finds nothing).
- Triton unified attention: the `KV_QUANT_MODE` constexpr branches only on `== 1` and on `2..3` (`triton_unified_attention.py:167, 415-418`). Both 12 and 15 take the plain-cast path.
- Octave kernels take `OctaveFormat.kernel_code = v_bits + 16 * compact` (`ops/rocm_octave.py:131-133`), which does not depend on the enum value.
- Connectors (NIXL, Mooncake, MoRIIO, offloading, decode-bench) branch on `is_nvfp4`, `is_per_token_head` or `== FP8_PER_TENSOR`. No handshake carries the int.
- The code never calls `KVQuantMode(<int>)` and never indexes by the mode value.
- The one persisted use is the Mooncake store `schema_fingerprint` (`mooncake_store/layout.py:561`, `str(spec.kv_quant_mode)`). Renumbering changes only CUSTOM's key ("12" to "15"); Octave keeps "12". A CUSTOM deployment that already has KV in the store gets cache misses, never wrong bytes.
- The claim holds, with that one exception, and the exception is the separation the fix is for.

## Tests

Interpreter: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>:<scratch>/plug <scratch>/venv/bin/python -m pytest -p noclean ...`. Each run sets PYTHONPATH to the tree it runs from. `tests/` is a package, so pytest puts that tree's root first on `sys.path`.

Trees: `base/` is `git archive e2bab26e51` plus only the #63691 test hunk. `pr63691/` is #63691 rebased by hand onto `e2bab26e51` (3 lines in the enum plus the test hunk).

| run | result |
|---|---|
| fail-before: `base/`, `tests/config/test_kv_cache_dtype.py` | `1 failed, 5 passed, 1 warning in 23.42s` at `assert mode.is_octave is False` (`<KVQuantMode.OCTAVE_K3V4: 12>.is_octave`), file `before_pr_test_on_base.txt` |
| #63691 rebased as written | `1 failed, 5 passed, 1 warning in 17.54s`, `AttributeError: ... 'is_custom_mode'`, file `after_rebased_pr_as_written.txt` |
| #63691 rebased, third assert replaced by `mode.name == "CUSTOM"` | `6 passed, 1 warning in 43.61s`, file `after_rebased_pr_name_check.txt` |
| repro on `pr63691/` | `mode: 15 CUSTOM is_octave: False`; `spec == octave spec: False`; page stays 32768 after Octave `customize_spec`; fingerprint keys `15` vs `12` (`repro_after_pr63691.out`) |

CI blind spot: the unmodified `tests/config/test_kv_cache_dtype.py` on the buggy base gives `6 passed, 1 warning in 28.58s` (`pristine_base_existing_test.txt`).

The 16 test files that import `KVQuantMode` (`kvquantmode_tests.txt`; `tests/v1/core/utils.py` is a helper and is left out) were run with `run_suite.sh` and compared with `compare.sh`. The runs use `HF_HUB_OFFLINE=1` and `--continue-on-collection-errors`. The first online attempt hung on a Hugging Face Hub HTTPS connection in both trees; its partial output is `suite_base.online_hung.txt.partial`.

- base (pristine worktree `e2bab26e51`): `2054 failed, 266 passed, 669 skipped, 1 warning, 188 errors in 77.88s (0:01:17)`
- `pr63691/` (rebased, with the `name` assert): `2054 failed, 266 passed, 669 skipped, 1 warning, 188 errors in 102.87s (0:01:42)`
- The failure sets are identical: 0 failures only in base, 0 only in the PR tree (`suite_base.fails`, `suite_pr63691.fails`).
- Every failure comes from the macOS CPU environment and appears in both trees:

  | count | error |
  |---|---|
  | 1937 | `RuntimeError: Device string must not be empty` |
  | 159 | `Failed to infer device type` |
  | 119 | `NotImplementedError` |
  | 24 | `triton` has no `next_power_of_2` |
  | 1 | missing `ray` (nixl test collection) |
  | 1 | missing `msgpack` (moriio test collection) |
  | 1 | offline Hub `ModelConfig` |

Lint of the rebased PR files, run from `pr63691/` against the repo's `pyproject.toml`: `uvx ruff check` gives `All checks passed!` and `uvx ruff format --check` gives `2 files already formatted`.

#63691's CI state: only the pre-commit job has run, and it passed. Buildkite never ran because the PR has no `ready` label.

## Review comment draft for #63691 (for aarnphm to edit and post)

```
I rebased this onto main (e2bab26e51) locally to see if it still holds.
The conflict is only the enum import line; main now has `from enum import IntEnum, StrEnum`.
After the rebase, test_backend_sentinel_properties fails with `AttributeError: 'KVQuantMode' object has no attribute 'is_custom_mode'`, because #64076 removed that property as dead code.
`assert mode.name == "CUSTOM"` keeps the intent and passes (6 passed).
Heads-up: #68945 adds `KITTY = 15`. The hunks don't overlap, so git won't flag it. With `@unique`, whichever lands second fails at import, which I think is the right failure.
It might be worth parking CUSTOM far outside the range the real modes use, so the next mode someone appends can't take its value.
I checked the "nothing persists" claim too. The only place the int leaves the process is the Mooncake store schema fingerprint, and only CUSTOM's key changes (12 -> 15). Octave keeps 12.
```
