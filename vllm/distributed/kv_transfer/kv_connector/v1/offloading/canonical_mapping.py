# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP); everything downstream consumes byte mappings. The canonical
page of a layer is the full offloaded block without parallelism: all KV
heads, all block_size * dcp * pcp tokens, in the worker's page encoding.
Uncertifiable layers get an opaque fallback mapping (fail closed).
"""

from dataclasses import dataclass, replace
from math import prod
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change
CANONICAL_FORMAT_VERSION = 1


def canonical_format_id(kv_cache_layout: str) -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    format version with that family; consumers must match it exactly. The family
    keeps its historical NHD/HND spelling so ids stay stable for KV persisted
    before the layout enum existed."""
    layout = KVCacheLayout[kv_cache_layout]
    legacy = {KVCacheLayout.LBNHC: "nhd", KVCacheLayout.LBHNC: "hnd"}
    family = legacy.get(layout, layout.name.lower())
    return f"v{CANONICAL_FORMAT_VERSION}-{family}"


@dataclass(frozen=True, slots=True)
class _RankContext:
    """Sharding parameters of one worker rank within the offload group."""

    tp_size: int
    dcp_size: int
    pcp_size: int
    interleave: int
    total_kv_heads: int
    rank: int

    @property
    def cp_size(self) -> int:
        return self.dcp_size * self.pcp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def total_cp_rank(self) -> int:
        pcp_rank = self.rank // self.tp_size
        return pcp_rank * self.dcp_size + self.tp_rank % self.dcp_size


@dataclass(frozen=True, slots=True)
class ByteRegion:
    """A byte region within a page that repeats once per token."""

    local_offset: int
    canonical_offset: int
    bytes_per_token: int
    canonical_token_stride: int


def _coalesce_runs(runs: list[CopyRun]) -> tuple[CopyRun, ...]:
    """Collapse contiguous fragments within and across runs to minimize the
    number of copy ops (e.g. a single-rank mapping becomes one whole-page run).
    """
    out: list[CopyRun] = []
    for run in runs:
        if (
            run.num_fragments > 1
            and run.local_stride == run.fragment_size
            and run.canonical_stride == run.fragment_size
        ):
            size = run.fragment_size * run.num_fragments
            run = CopyRun(run.local_offset, run.canonical_offset, size, 1, size, size)
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.num_fragments == 1
            and run.num_fragments == 1
            and prev.local_offset + prev.fragment_size == run.local_offset
            and prev.canonical_offset + prev.fragment_size == run.canonical_offset
        ):
            size = prev.fragment_size + run.fragment_size
            out[-1] = CopyRun(
                prev.local_offset, prev.canonical_offset, size, 1, size, size
            )
        else:
            out.append(run)
    return tuple(out)


def _local_to_canonical_token(local_idx: int, ctx: _RankContext) -> int:
    """Canonical position of one of this rank's local token indices."""
    chunk, pos_in_chunk = divmod(local_idx, ctx.interleave)
    return (chunk * ctx.cp_size + ctx.total_cp_rank) * ctx.interleave + pos_in_chunk


def _interleave_cp_tokens(
    regions: list[ByteRegion],
    num_tokens: int,
    ctx: _RankContext,
) -> tuple[CopyRun, ...]:
    """Place each region's num_tokens rows at their canonical token positions,
    one run per chunk of interleaved tokens."""
    runs: list[CopyRun] = []
    for region in regions:
        if ctx.cp_size == 1:
            runs.append(
                CopyRun(
                    region.local_offset,
                    region.canonical_offset,
                    region.bytes_per_token,
                    num_tokens,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
            continue
        for chunk_start in range(0, num_tokens, ctx.interleave):
            canonical_token = _local_to_canonical_token(chunk_start, ctx)
            runs.append(
                CopyRun(
                    region.local_offset + chunk_start * region.bytes_per_token,
                    region.canonical_offset
                    + canonical_token * region.canonical_token_stride,
                    region.bytes_per_token,
                    ctx.interleave,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
    return _coalesce_runs(runs)


def _attention_runs(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    num_blocks: int,
    head_shard: int,
    num_head_shards: int,
    ctx: _RankContext,
) -> tuple[CopyRun, ...] | None:
    """Runs from this rank's page to the canonical page, as a layout composition.

    The local page and the canonical page are two strided layouts over the same
    named coordinates: head h, token (chunk, pos) and the inner element axes.
    Local strides come from the tensor; canonical strides come from the same
    axis order with every head and cp_size * block_size tokens. The runs are
    the joint coalesce of the two: merge axes, innermost first, while both
    sides stay contiguous. Any stride order is covered; None when the tensor
    is not one block-major page of the spec (fail closed).
    """
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    elem = kv_cache.element_size()
    shape, stride = tuple(kv_cache.shape), kv_cache.stride()
    names: tuple[str, ...]
    if shape == (num_blocks, heads, bs, 2 * head_size):  # K and V packed per head
        names = ("B", "H", "N", "E")
    elif shape == (num_blocks, 2, bs, heads, head_size):  # K and V page halves
        names = ("B", "C", "N", "H", "E")
    else:
        return None
    if stride[-1] != 1 or prod(shape[1:]) * elem != spec.real_page_size_bytes:
        return None
    local = {n: (shape[i], stride[i] * elem) for i, n in enumerate(names) if n != "B"}
    # physical order, innermost first; on a stride tie (a size-1 axis) the head
    # axis sits inside the token axis, so NHD wins, as in the format id
    order = sorted(local, key=lambda n: (local[n][1], n != "E", n != "H"))
    acc = elem
    for n in order:
        size, st = local[n]
        if size > 1 and st != acc:
            return None  # not a dense page
        acc *= size
    # the canonical page has a fixed format (NHD or HND), not "the local order"
    formats = {
        ("E", "H", "N"),
        ("E", "N", "H"),
        ("E", "H", "N", "C"),
        ("E", "N", "H", "C"),
    }
    if tuple(order) not in formats:
        return None
    # canonical page: same axis order, all heads, cp_size * block_size tokens
    csize = dict(local)
    csize["H"] = (num_head_shards * heads, 0)
    csize["N"] = (ctx.cp_size * bs, 0)
    canon, acc = {}, elem  # canonical axes in the same format as the local page
    for n in order:
        canon[n] = acc
        acc *= csize[n][0]
    il = ctx.interleave if ctx.cp_size > 1 else bs
    # local token n = chunk * il + pos; canonical t = (chunk * cp + rank) * il + pos
    modes = [
        (n, local[n][0], local[n][1], canon[n])
        for n in names[1:]
        if n not in ("H", "N")
    ]
    modes += [
        ("H", heads, local["H"][1], canon["H"]),
        ("pos", il, local["N"][1], canon["N"]),
        ("chunk", bs // il, il * local["N"][1], ctx.cp_size * il * canon["N"]),
    ]
    src0 = 0
    dst0 = head_shard * heads * canon["H"] + ctx.total_cp_rank * il * canon["N"]
    # joint coalesce: innermost (by canonical stride) axes that stay contiguous on
    # both sides form one fragment; the rest enumerate fragments
    modes.sort(key=lambda m: m[3])
    frag, i = elem, 0
    while i < len(modes) and modes[i][2] == frag and modes[i][3] == frag:
        frag *= modes[i][1]
        i += 1
    starts = [(src0, dst0)]
    for _, size, ls, cs in modes[i:]:
        starts = [(a + k * ls, b + k * cs) for a, b in starts for k in range(size)]
    return _coalesce_runs(
        [CopyRun(a, b, frag, 1, frag, frag) for a, b in sorted(starts)]
    )


def _layer_mapping(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified mapping for one layer at one rank, or None (fail closed)."""
    if not isinstance(spec, AttentionSpec):
        return None
    bs = spec.block_size
    page = spec.real_page_size_bytes
    if ctx.cp_size > 1 and (ctx.interleave > bs or bs % ctx.interleave):
        return None

    if isinstance(spec, MLAAttentionSpec):
        # TP-replicated latent; CP shards its tokens across the DCP groups
        if (
            spec.tokens_per_state != 1
            or page % bs
            or ctx.tp_size % ctx.dcp_size
            or spec.kv_quant_mode.is_per_token_head
        ):
            return None
        row = page // bs
        return CanonicalPageMapping(
            canonical_page_size_bytes=ctx.cp_size * page,
            local_page_size_bytes=page,
            runs=_interleave_cp_tokens([ByteRegion(0, 0, row, row)], bs, ctx),
            num_writers=ctx.tp_size // ctx.dcp_size,
            writer_index=ctx.tp_rank // ctx.dcp_size,
            parallelism_agnostic=ctx.cp_size == 1,
        )

    if spec.kv_quant_mode.is_per_token_head or not isinstance(kv_cache, torch.Tensor):
        return None
    total, tp = ctx.total_kv_heads, ctx.tp_size
    if spec.num_kv_heads != max(1, total // tp):
        return None
    if total >= tp:
        if total % tp:
            return None
        num_head_shards, replication = tp, 1
    else:
        if tp % total:
            return None
        num_head_shards, replication = total, tp // total
    # DCP shards tokens across ranks holding replicated KV
    if replication % ctx.dcp_size:
        return None

    head_shard = ctx.tp_rank // replication
    runs = _attention_runs(kv_cache, spec, num_blocks, head_shard, num_head_shards, ctx)
    if runs is None:
        return None
    return CanonicalPageMapping(
        canonical_page_size_bytes=ctx.cp_size * num_head_shards * page,
        local_page_size_bytes=page,
        runs=runs,
        num_writers=replication // ctx.dcp_size,
        writer_index=(ctx.tp_rank % replication) // ctx.dcp_size,
        parallelism_agnostic=ctx.cp_size == 1,
    )


def _opaque_fallback_mapping(
    page_size_bytes: int, num_ranks: int, rank: int
) -> CanonicalPageMapping:
    """Fallback: place the worker's page whole at a worker-exclusive offset."""
    run = CopyRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        runs=(run,),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=False,
    )


def _run_intervals(runs: tuple[CopyRun, ...], canonical: bool) -> list[tuple[int, int]]:
    intervals = []
    for run in runs:
        offset = run.canonical_offset if canonical else run.local_offset
        stride = run.canonical_stride if canonical else run.local_stride
        for i in range(run.num_fragments):
            start = offset + i * stride
            intervals.append((start, start + run.fragment_size))
    return sorted(intervals)


def _is_exact_partition(intervals: list[tuple[int, int]], size: int) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == 0
        and intervals[-1][1] == size
        and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    )


def _verify_tiling(layer_name: str, per_rank: list[CanonicalPageMapping]) -> None:
    """Whichever ranks a block elects as writers must tile the canonical page
    exactly once, and each rank's runs must cover exactly its local page."""
    size = per_rank[0].canonical_page_size_bytes
    num_writers = per_rank[0].num_writers
    for mapping in per_rank:
        assert mapping.canonical_page_size_bytes == size
        assert mapping.num_writers == num_writers
        local = _run_intervals(mapping.runs, canonical=False)
        assert _is_exact_partition(local, mapping.local_page_size_bytes), (
            f"runs do not cover the local page of layer {layer_name}"
        )
    for block_id in range(num_writers):
        stored: list[tuple[int, int]] = []
        for mapping in per_rank:
            if mapping.is_writer(block_id):
                stored += _run_intervals(mapping.runs, canonical=True)
        stored.sort()
        assert _is_exact_partition(stored, size), (
            f"writers of block {block_id} do not tile the canonical page "
            f"of layer {layer_name}"
        )


def _unpadded_page_size(spec: KVCacheSpec) -> int | None:
    if isinstance(spec, AttentionSpec):
        return spec.unpadded_page_size_bytes
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None).page_size_bytes
    return None


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, CanonicalPageMapping]:
    """Per-layer canonical page mappings for this worker.

    Empty when the worker group is not exactly the TP x PCP grid; layers
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size
    if parallel_config.world_size != group_size:
        return {}

    def ctx(rank: int) -> _RankContext:
        return _RankContext(
            tp_size=tp_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pcp_size=pcp_size,
            interleave=parallel_config.cp_kv_cache_interleave_size,
            total_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
            rank=rank,
        )

    my_rank = parallel_config.rank
    num_blocks = kv_cache_config.num_blocks

    mappings: dict[str, CanonicalPageMapping] = {}
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        for layer_name, spec in kv_cache_group.kv_cache_specs.items():
            per_rank: list[CanonicalPageMapping] = []
            for rank in range(group_size):
                mapping = _layer_mapping(
                    spec, kv_caches.get(layer_name), num_blocks, ctx(rank)
                )
                if mapping is None:
                    break
                per_rank.append(mapping)
            if len(per_rank) != group_size:
                page = _unpadded_page_size(spec)
                if page is None:
                    continue
                per_rank = [
                    _opaque_fallback_mapping(page, group_size, rank)
                    for rank in range(group_size)
                ]
            _verify_tiling(layer_name, per_rank)
            mappings[layer_name] = per_rank[my_rank]
    return mappings
