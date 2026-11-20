# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.config import ParallelConfig
from vllm.config.fault_tolerance import FaultToleranceConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVConnectorOutput
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.outputs import ModelRunnerOutput


def _model_output(kv_connector_output: KVConnectorOutput | None) -> ModelRunnerOutput:
    output = ModelRunnerOutput(req_ids=[])
    output.kv_connector_output = kv_connector_output
    return output


def _done_future(result=None, exception=None) -> Future:
    future: Future = Future()
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(result)
    return future


def _make_sentinel(batch_queue: deque) -> tuple[EngineCoreSentinel, MagicMock]:
    scheduler = MagicMock()
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine = SimpleNamespace(
        batch_queue=batch_queue, scheduler=scheduler, deferred_step=None
    )
    return sentinel, scheduler


def test_drain_batch_queue_surfaces_kv_connector_output():
    """Pre-fault KV output must reach the scheduler, not be lost on clear."""
    kv_output = KVConnectorOutput(finished_sending={"req1"})
    batch_queue = deque([(_done_future(_model_output(kv_output)), None, None)])
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_called_once()
    sched_output, model_output = scheduler.update_from_output.call_args[0]
    assert isinstance(sched_output, SchedulerOutput)
    assert sched_output.num_scheduled_tokens == {}
    assert model_output.kv_connector_output is kv_output


def test_drain_batch_queue_drops_failures_and_empty_outputs():
    """Failed futures and empty KV outputs are dropped without scheduler calls."""
    batch_queue = deque(
        [
            (_done_future(exception=RuntimeError("boom")), None, None),
            (_done_future(_model_output(None)), None, None),
            (_done_future(_model_output(KVConnectorOutput())), None, None),
        ]
    )
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_not_called()


def test_drain_batch_queue_clears_deferred_step():
    """A step deferred before the fault would block all further scheduling."""
    sentinel, _ = _make_sentinel(deque())
    sentinel.engine.deferred_step = (SchedulerOutput.make_empty(), Future())

    sentinel._drain_batch_queue()

    assert sentinel.engine.deferred_step is None


def test_fault_evicts_only_failed_pages_with_hisparse_host_pool(tmp_path):
    """HiSparse numbers host blocks in their own pool, so the failed step's
    host block IDs also name unrelated device blocks. Recovery must keep the
    device pages that the failed step never wrote."""
    from tests.v1.core.test_prefix_caching import (
        HISPARSE_BLOCK_SIZE,
        make_hisparse_kv_cache_config,
        make_kv_cache_manager,
        make_request,
    )
    from tests.v1.core.utils import create_scheduler

    page = HISPARSE_BLOCK_SIZE
    init_none_hash(sha256)
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    scheduler = create_scheduler(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=128
    )
    # Groups: host source, device indexer, device resident, device hot.
    manager = make_kv_cache_manager(
        make_hisparse_kv_cache_config(32, 16),
        max_model_len=128,
        enable_caching=True,
        hash_block_size=page,
        log_stats=True,
    )
    scheduler.kv_cache_manager = manager
    scheduler.kv_cache_config = manager.kv_cache_config
    engine = SimpleNamespace(
        engine_index=0,
        scheduler=scheduler,
        model_executor=SimpleNamespace(collective_rpc=lambda *args, **kwargs: []),
        output_queue=queue.Queue(),
    )
    ft_config = FaultToleranceConfig(request_failure_policy="recompute")
    sentinel = EngineCoreSentinel(
        engine, ParallelConfig(fault_tolerance_config=ft_config)
    )

    a = make_request("a", list(range(page + page // 2)), page, sha256)
    b = make_request("b", list(range(100, 100 + 3 * page + page // 2)), page, sha256)
    c = make_request("c", list(range(200, 200 + 4 * page)), page, sha256)
    scheduler.add_request(a)
    scheduler.add_request(b)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, ModelRunnerOutput(req_ids=["a", "b"], sampled_token_ids=[[7], [7]])
    )
    # The failing step writes the partial last pages of A and B, and all of C.
    scheduler.add_request(c)
    scheduler.schedule()
    # Its host block IDs name some of B's cached device pages.
    host_b, indexer_b = manager.get_blocks("b").blocks[:2]
    failed_host = [host_b[3], *manager.get_blocks("c").blocks[0]]
    assert {blk.block_id for blk in failed_host} & {
        blk.block_id for blk in indexer_b[:3]
    }

    sentinel.on_fault(RuntimeError("step failed"))

    assert sentinel.status_type == EngineStatusType.UNHEALTHY
    pool = manager.block_pool
    assert all(pool.get_cached_block(h, [1]) for h in b.block_hashes[:3])
    assert not any(pool.get_cached_block(h, [1]) for h in c.block_hashes)
