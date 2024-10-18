from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerBase, WorkerWrapperBase

logger = init_logger(__name__)


def create_worker(worker_module_name: str, worker_class_name: str,
                  worker_class_fn: Optional[Callable[[], Type[WorkerBase]]],
                  **kwargs):
    wrapper = WorkerWrapperBase(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
        worker_class_fn=worker_class_fn,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
        )

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.scheduler_config.is_multi_step:
            worker_module_name = "vllm.worker.multi_step_worker"
            worker_class_name = "MultiStepWorker"
        else:
            worker_module_name = "vllm.worker.worker"
            worker_class_name = "Worker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    def _get_create_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict:
        worker_kwargs = self._get_worker_kwargs(local_rank, rank,
                                                distributed_init_method)

        (worker_module_name, worker_class_name,
         worker_class_fn) = self._get_worker_module_and_class()
        worker_kwargs.update(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            worker_class_fn=worker_class_fn,
        )

        return worker_kwargs

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        return create_worker(**self._get_create_worker_kwargs(
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)
        max_concurrency = (num_gpu_blocks * self.cache_config.block_size /
                           self.model_config.max_model_len)
        logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                    self.model_config.max_model_len, max_concurrency)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return

    def start_profile(self) -> None:
        self.driver_worker.start_profile()

    def stop_profile(self) -> None:
        self.driver_worker.stop_profile()


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output
