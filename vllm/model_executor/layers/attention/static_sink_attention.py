# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.core.single_type_kv_cache_manager import SinkFullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    get_kv_quant_mode,
)

logger = init_logger(__name__)


@functools.lru_cache
def create_static_sink_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    sink_len: int = 0,
) -> type[AttentionBackend]:
    prefix = "StaticSink_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class StaticSinkAttentionBuilder(underlying_builder):  # type: ignore
        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            model_config = vllm_config.model_config
            scheduler_config = vllm_config.scheduler_config
            self.sink_len = sink_len
            # The group's resolved block size, which the planner may have grown
            # past `CacheConfig.block_size` and the scheduler reserves in.
            self.block_size = kv_cache_spec.block_size
            sink_block_ids = SinkFullAttentionManager.sink_block_ids(
                SinkFullAttentionManager.config(sink_len=sink_len), self.block_size
            )
            self.num_sink_blocks = len(sink_block_ids)
            self.max_num_blocks = cdiv(model_config.max_model_len, self.block_size)
            self.block_table_with_sink = torch.zeros(
                (
                    scheduler_config.max_num_seqs,
                    self.max_num_blocks + self.num_sink_blocks,
                ),
                device=device,
                dtype=torch.int32,
            )
            self.block_table_with_sink[:, : self.num_sink_blocks] = torch.arange(
                sink_block_ids.start,
                sink_block_ids.stop,
                device=device,
                dtype=torch.int32,
            )
            # The caller's seq_lens is the runner's batch tensor, which the
            # sampler and drafters read after this build, so the sink lengths
            # live in a buffer of their own. It is persistent for CUDA graphs.
            self.seq_lens_with_sink = torch.zeros(
                scheduler_config.max_num_seqs, device=device, dtype=torch.int32
            )
            # build() restages seq_lens, which the in-place
            # update_draft_decode_metadata() does not refresh.
            self.supports_draft_decode_metadata_update = False

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            num_reqs = common_attn_metadata.num_reqs
            seq_lens = common_attn_metadata.seq_lens
            seq_lens_with_sink = self.seq_lens_with_sink[:num_reqs]
            torch.add(seq_lens, self.sink_len, out=seq_lens_with_sink)
            # Padded requests keep a zero length.
            seq_lens_with_sink.masked_fill_(seq_lens == 0, 0)
            max_seq_len = common_attn_metadata.max_seq_len + self.sink_len
            max_num_blocks = cdiv(max_seq_len, self.block_size)
            self.block_table_with_sink[
                :num_reqs, self.num_sink_blocks : self.num_sink_blocks + max_num_blocks
            ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
            common_attn_metadata = common_attn_metadata.replace(
                seq_lens=seq_lens_with_sink,
                max_seq_len=max_seq_len,
                block_table_tensor=self.block_table_with_sink[:num_reqs],
                _num_computed_tokens_cache=None,
            )

            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    # The sink write rides on the KV update op, which attention depends on.
    assert not underlying_attn_backend.forward_includes_kv_cache_update
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class StaticSinkAttentionImpl(underlying_impl):  # type: ignore
        def write_sink(
            self, layer: "StaticSinkAttention", kv_cache: torch.Tensor
        ) -> None:
            """Write the sink KV into blocks 1..sink_len // block_size, which no
            request owns, with the backend's own KV cache writer."""
            assert layer.sink_key is not None and layer.sink_value is not None
            # The writer's own block size: the view is `[blocks, kv_heads,
            # block_tokens, K | V]` in the group's resolved blocks.
            block_size = kv_cache.shape[2]
            slot_mapping = torch.arange(
                block_size,
                block_size + sink_len,
                device=kv_cache.device,
                dtype=torch.long,
            )
            super().do_kv_cache_update(
                layer, layer.sink_key, layer.sink_value, kv_cache, slot_mapping
            )

        def do_kv_cache_update(
            self,
            layer: torch.nn.Module,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            # Rewrite the sink with every KV update, so it is in place before
            # this step's attention even if the cache was remapped or the
            # weights reloaded since the last step.
            assert isinstance(layer, StaticSinkAttention)
            if kv_cache.numel() > 0:
                self.write_sink(layer, kv_cache)
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    attn_backend = subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": staticmethod(lambda: StaticSinkAttentionBuilder),
            "get_impl_cls": staticmethod(lambda: StaticSinkAttentionImpl),
        },
    )

    return attn_backend


@CustomOp.register("static_sink_attention")
class StaticSinkAttention(Attention, CustomOp):
    """Attention with static sink tokens."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sink_len: int,
        attn_backend: type[AttentionBackend] | None = None,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ):
        if get_current_vllm_config().parallel_config.decode_context_parallel_size > 1:
            # Each rank keeps the whole sink in its blocks 1..sink_len //
            # block_size, but DCP-scaled scheduler blocks would reserve fewer.
            raise ValueError(
                "StaticSinkAttention does not support decode context parallelism."
            )
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        if attn_backend is not None:
            underlying_attn_backend = attn_backend
        else:
            underlying_attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype)
        attn_backend = create_static_sink_attention_backend(
            underlying_attn_backend,  # type: ignore[arg-type]
            sink_len=sink_len,
        )
        Attention.__init__(
            self=self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            **kwargs,
        )
        self.sink_len = sink_len
        self.sink_key: torch.Tensor | None = None
        self.sink_value: torch.Tensor | None = None

    def update_sink_kv(self, sink_key, sink_value) -> None:
        self.sink_key = sink_key
        self.sink_value = sink_value

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        assert self.sink_key is not None and self.sink_value is not None, (
            "sink_key and sink_value have not been prepared"
        )
        return super().forward(query, key, value, output_shape)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, output_shape)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        # Block size may get updated after model loading, refresh it
        self.block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER

        spec = AttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        return SinkFullAttentionManager.layer_config(spec, sink_len=self.sink_len)
