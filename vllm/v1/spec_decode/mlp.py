# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.config import SpeculativeConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models import ModelRegistry
from vllm.utils import LazyLoader
from vllm.v1.sample.metadata import SamplingMetadata

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = LazyLoader("torch", globals(), "torch")
    nn = LazyLoader("nn", globals(), "torch.nn")

logger = init_logger(__name__)


@dataclass
class MlpProposer:
    vllm_config: VllmConfig
    device: torch.device

    speculative_config: SpeculativeConfig = field(init=False, repr=False)
    model: nn.Module = field(init=False, repr=False)

    def __post_init__(self):
        if self.vllm_config.speculative_config is None:
            raise ValueError(
                "'speculative_config' cannot be None when using 'mlp_speculator'"  # noqa: E501
            )
        self.speculative_config = self.vllm_config.speculative_config

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        ...

    def load_model(self, target_model: nn.Module) -> None:
        loader = get_model_loader(self.vllm_config.load_config)
        target_layer_num = self.vllm_config.model_config.get_num_layers(
            self.vllm_config.parallel_config)

        draft_model_config = \
            self.speculative_config.draft_model_config
        # FIXME: This does not handle with distributed inference.
        target_device = self.vllm_config.device_config.device
        # We need to set the vllm_config here to register attention
        # layers in the forward context.
        with set_default_torch_dtype(draft_model_config.dtype), \
                set_current_vllm_config(self.vllm_config):
            draft_model_cls, _ = ModelRegistry.resolve_model_cls(
                draft_model_config.architectures)
            self.model = draft_model_cls(
                vllm_config=self.vllm_config,
                start_layer_id=target_layer_num,
            ).to(target_device)

        weights = loader.get_all_weights(draft_model_config, self.model)
        self.model.load_weights(weights)
