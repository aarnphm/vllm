# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`StaticSinkAttention` keeps each layer's sink KV in blocks
1..sink_len // block_size, which the scheduler never hands to a request.

The sink is written with every KV update, through the same writer as the
request's KV. The checks are the ids the read table and the device pool agree
on, and the bytes in the cache: the sink lands in its own blocks in the
worker's `[blocks, kv_heads, block_tokens, K | V]` layout, every other block
keeps its contents, and a captured step rewrites a cleared sink.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SinkFullAttentionManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

HEAD_DIM = 192
V_HEAD_DIM = 128
BLOCK_SIZE = 16
SINK_LEN = 2 * BLOCK_SIZE
NUM_BLOCKS = 8
CANARY = 7.0


def test_decode_context_parallelism_is_refused_when_the_model_is_built():
    """Each DCP rank keeps the whole sink in its blocks 1..sink_len/block_size,
    which DCP-scaled scheduler blocks would not reserve. The layer refuses it
    when the model is built, before profiling and warmup."""
    from vllm.model_executor.layers.attention.static_sink_attention import (
        StaticSinkAttention,
    )

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.parallel_config.decode_context_parallel_size = 2
    with (
        set_current_vllm_config(config),
        pytest.raises(ValueError, match="decode context parallelism"),
    ):
        StaticSinkAttention(
            num_heads=1,
            head_size=HEAD_DIM,
            scale=HEAD_DIM**-0.5,
            sink_len=SINK_LEN,
            num_kv_heads=1,
            prefix="model.layers.0.self_attn.attn",
        )


def test_weight_reload_refreshes_the_sink_tensors_in_place():
    """Captured CUDA graphs write the sink from the tensors bound when they were
    captured. A weight reload hands the layer new sink K/V (OpenPangu's
    `post_weight_load` builds a fresh normalized K), so the layer must copy them
    into the bound tensors rather than rebind."""
    from vllm.model_executor.layers.attention.static_sink_attention import (
        StaticSinkAttention,
    )
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
    )

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=BLOCK_SIZE),
    )
    with set_current_vllm_config(config):
        layer = StaticSinkAttention(
            num_heads=1,
            head_size=HEAD_DIM,
            scale=HEAD_DIM**-0.5,
            sink_len=SINK_LEN,
            num_kv_heads=1,
            cache_config=config.cache_config,
            prefix="model.layers.0.self_attn.attn",
            attn_backend=FlashAttentionDiffKVBackend,
            head_size_v=V_HEAD_DIM,
        )
    layer.update_sink_kv(
        torch.zeros(SINK_LEN, 1, HEAD_DIM), torch.zeros(SINK_LEN, 1, V_HEAD_DIM)
    )
    assert layer.sink_key is not None and layer.sink_value is not None
    addresses = (layer.sink_key.data_ptr(), layer.sink_value.data_ptr())

    sink_key = torch.randn(SINK_LEN, 1, HEAD_DIM)
    sink_value = torch.randn(SINK_LEN, 1, V_HEAD_DIM)
    layer.update_sink_kv(sink_key, sink_value)

    assert (layer.sink_key.data_ptr(), layer.sink_value.data_ptr()) == addresses
    torch.testing.assert_close(layer.sink_key, sink_key, atol=0, rtol=0)
    torch.testing.assert_close(layer.sink_value, sink_value, atol=0, rtol=0)


def test_sink_read_ids_are_the_blocks_the_device_pool_keeps():
    """Beside a drafter with twice the page, the planner grows the sink group's
    blocks to 32 tokens while `CacheConfig.block_size` stays 16. The read table
    must prepend the one 32-token block the pool keeps, not two 16-token ids
    that reach into a request's block."""
    from vllm.model_executor.layers.attention.static_sink_attention import (
        create_static_sink_attention_backend,
    )
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
    )

    config = VllmConfig(
        model_config=ModelConfig(max_model_len=64),
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=BLOCK_SIZE),
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.kv_cache_layout = "LBNHC"

    def spec(num_kv_heads: int) -> AttentionSpec:
        return AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=64,
            dtype=torch.bfloat16,
        )

    groups = get_kv_cache_groups(
        config,
        {
            "sink": SinkFullAttentionManager.layer_config(spec(1), sink_len=SINK_LEN),
            "draft": FullAttentionManager.layer_config(spec(2)),
        },
    )
    (sink_group,) = [g for g in groups if "sink" in g.kv_cache_specs]
    assert sink_group.block_size == 2 * BLOCK_SIZE

    builder = create_static_sink_attention_backend(
        FlashAttentionDiffKVBackend, sink_len=SINK_LEN
    ).get_builder_cls()(
        sink_group.kv_cache_specs["sink"], ["sink"], config, torch.device("cpu")
    )
    sink_ids = builder.block_table_with_sink[0, : builder.num_sink_blocks].tolist()

    pool = KVCacheManager(
        KVCacheConfig(
            num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=64,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
        scheduler_block_size=2 * BLOCK_SIZE,
    ).block_pool
    handed_out = {b.block_id for b in pool.get_new_blocks(pool.get_num_free_blocks())}
    assert sink_ids == [1]
    assert handed_out == set(range(1, NUM_BLOCKS)) - set(sink_ids)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the KV cache writer is a Triton kernel"
)
@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_sink_write_lands_in_sink_blocks_and_replays(
    dist_init, monkeypatch, num_kv_heads
):
    from vllm.model_executor.layers.attention.static_sink_attention import (
        StaticSinkAttention,
    )
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
    )

    if not FlashAttentionDiffKVBackend.is_supported_on_current_device(
        HEAD_DIM, V_HEAD_DIM, has_sinks=False
    ):
        pytest.skip("FlashAttention DiffKV needs FA3 or FA4")
    monkeypatch.setattr(FlashAttentionDiffKVBackend, "head_size_v", V_HEAD_DIM)

    config = VllmConfig(cache_config=CacheConfig(block_size=BLOCK_SIZE))
    with set_current_vllm_config(config):
        layer = StaticSinkAttention(
            num_heads=num_kv_heads,
            head_size=HEAD_DIM,
            scale=HEAD_DIM**-0.5,
            sink_len=SINK_LEN,
            num_kv_heads=num_kv_heads,
            cache_config=config.cache_config,
            prefix="model.layers.0.self_attn.attn",
            attn_backend=FlashAttentionDiffKVBackend,
            head_size_v=V_HEAD_DIM,
        ).to("cuda")

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    layer.update_sink_kv(
        randn(SINK_LEN, num_kv_heads, HEAD_DIM),
        randn(SINK_LEN, num_kv_heads, V_HEAD_DIM),
    )
    kv_cache = torch.full(
        (NUM_BLOCKS, num_kv_heads, BLOCK_SIZE, HEAD_DIM + V_HEAD_DIM),
        CANARY,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # A request in block 3, the first block after the null and sink blocks.
    num_tokens = 5
    key = randn(num_tokens, num_kv_heads, HEAD_DIM)
    value = randn(num_tokens, num_kv_heads, V_HEAD_DIM)
    slot_mapping = torch.arange(
        3 * BLOCK_SIZE, 3 * BLOCK_SIZE + num_tokens, dtype=torch.long, device="cuda"
    )

    expected = kv_cache.clone()
    writes = [
        (BLOCK_SIZE + token, layer.sink_key[token], layer.sink_value[token])
        for token in range(SINK_LEN)
    ] + [
        (slot, key[token], value[token])
        for token, slot in enumerate(slot_mapping.tolist())
    ]
    for slot, k, v in writes:
        block, offset = divmod(slot, BLOCK_SIZE)
        expected[block, :, offset, :HEAD_DIM] = k
        expected[block, :, offset, HEAD_DIM:] = v

    layer.impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
    torch.testing.assert_close(kv_cache, expected, atol=0, rtol=0)

    # A remapped cache comes back zeroed: a captured step must restore the sink.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
    kv_cache[1 : 1 + SINK_LEN // BLOCK_SIZE].zero_()
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(kv_cache, expected, atol=0, rtol=0)
