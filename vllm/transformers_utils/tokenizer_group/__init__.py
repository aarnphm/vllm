from typing import Optional, Type

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         TokenizerPoolConfig)

from .base_tokenizer_group import AnyTokenizer, BaseTokenizerGroup
from .tokenizer_group import TokenizerGroup

def init_tokenizer_from_configs(model_config: ModelConfig,
                                scheduler_config: SchedulerConfig,
                                parallel_config: ParallelConfig,
                                ):
    init_kwargs = dict(tokenizer_id=model_config.tokenizer,
                       max_num_seqs=scheduler_config.max_num_seqs,
                       max_input_length=None,
                       tokenizer_mode=model_config.tokenizer_mode,
                       trust_remote_code=model_config.trust_remote_code,
                       revision=model_config.tokenizer_revision)

    return get_tokenizer_group(parallel_config.tokenizer_pool_config,
                               **init_kwargs)


def get_tokenizer_group(tokenizer_pool_config: Optional[TokenizerPoolConfig],
                        **init_kwargs) -> BaseTokenizerGroup:
    tokenizer_cls: Type[BaseTokenizerGroup]
    if tokenizer_pool_config is None:
        tokenizer_cls = TokenizerGroup
    elif isinstance(tokenizer_pool_config.pool_type, type) and issubclass(
            tokenizer_pool_config.pool_type, BaseTokenizerGroup):
        tokenizer_cls = tokenizer_pool_config.pool_type
    else:
        raise ValueError(
            f"Unknown pool type: {tokenizer_pool_config.pool_type}")
    return tokenizer_cls.from_config(tokenizer_pool_config, **init_kwargs)


__all__ = ["AnyTokenizer", "get_tokenizer_group", "BaseTokenizerGroup"]
