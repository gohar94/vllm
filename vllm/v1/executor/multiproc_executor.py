# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle

import vllm.envs as envs
import torch

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_pp_group, get_tp_group)
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from vllm.utils import (decorate_logs, get_distributed_init_method,
                        get_loopback_ip, get_mp_context, get_open_port,
                        set_process_title)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.utils import get_and_update_mm_cache
from vllm.v1.outputs import (AsyncModelRunnerOutput, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        # RPC dispatch bookkeeping for concurrent requests.
        self._rpc_id_counter = itertools.count()
        self._pending_lock = threading.Lock()
        self._pending_rpcs: dict[int, "_PendingRPC"] = {}
        self._rpc_enqueue_lock = threading.Lock()
        self._response_threads: list[Thread] = []
        self._start_response_listeners()

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def _start_response_listeners(self):
        for handle in self.workers:
            thread = Thread(target=self._response_listener,
                            args=(handle, ),
                            daemon=True,
                            name=f"mp_exec_resp_{handle.rank}")
            thread.start()
            self._response_threads.append(thread)

    def _response_listener(self, handle: "WorkerProcHandle") -> None:
        mq = handle.worker_response_mq
        while not self.shutdown_event.is_set():
            try:
                message = mq.dequeue(timeout=0.1,
                                     cancel=self.shutdown_event)
            except TimeoutError:
                continue
            except Exception:
                if not self.shutdown_event.is_set():
                    logger.exception("Failed to dequeue worker response.")
                break
            if message is None:
                continue
            if not isinstance(message, tuple) or len(message) != 3:
                logger.error("Unexpected worker response format: %s", message)
                continue
            rpc_id, status, payload = message
            self._dispatch_worker_response(rpc_id, handle.rank, status,
                                           payload)

    def _dispatch_worker_response(self, rpc_id: int, worker_rank: int,
                                  status, payload) -> None:
        with self._pending_lock:
            pending = self._pending_rpcs.get(rpc_id)
        if pending is None:
            logger.debug("Received response for unknown rpc_id %s", rpc_id)
            return

        if status != WorkerProc.ResponseStatus.SUCCESS:
            error = RuntimeError(
                f"Worker failed with error '{payload}', please check the"
                " stack trace above for the root cause")
            pending.fail(error)
            return

        pending.add_success(worker_rank, payload)

    def _remove_pending_rpc(self, rpc_id: int) -> None:
        with self._pending_lock:
            self._pending_rpcs.pop(rpc_id, None)

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch",
                            unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        outputs = self.collective_rpc("take_draft_token_ids",
                                      unique_reply_rank=self.output_rank)
        return outputs[0]

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(
                method, protocol=pickle.HIGHEST_PROTOCOL)

        workers = (self.workers[unique_reply_rank],
                   ) if unique_reply_rank is not None else self.workers
        expected_ranks = tuple(w.rank for w in workers)

        rpc_id = next(self._rpc_id_counter)
        pending = _PendingRPC(rpc_id, expected_ranks, self._remove_pending_rpc)
        with self._pending_lock:
            self._pending_rpcs[rpc_id] = pending

        try:
            with self._rpc_enqueue_lock:
                self.rpc_broadcast_mq.enqueue(
                    (rpc_id, send_method, args, kwargs, unique_reply_rank))
        except Exception:
            self._remove_pending_rpc(rpc_id)
            pending.fail(RuntimeError("Failed to enqueue RPC request"))
            raise

        if non_block:
            return [pending.futures[rank] for rank in expected_ranks]

        try:
            responses = pending.result(deadline)
        except TimeoutError as e:
            pending.fail(e)
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception:
            raise

        return responses

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, 'workers', None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

            self.shutdown_event.set()
            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                del self.io_thread_pool

        self.rpc_broadcast_mq = None
        for thread in self._response_threads:
            thread.join(timeout=1.0)
        self._response_threads.clear()

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size


class _PendingRPC:
    """Tracks outstanding RPC responses across workers."""

    def __init__(self, rpc_id: int, expected_ranks: tuple[int, ...],
                 on_complete: Callable[[int], None]) -> None:
        self.rpc_id = rpc_id
        self.expected_ranks = expected_ranks
        self._on_complete = on_complete
        self._lock = threading.Lock()
        self._completed = False
        self.futures: dict[int, Future] = {
            rank: Future() for rank in expected_ranks
        }
        for fut in self.futures.values():
            fut.add_done_callback(self._future_done)

    def _future_done(self, _: Future) -> None:
        with self._lock:
            if self._completed:
                return
            if all(f.done() for f in self.futures.values()):
                self._completed = True
        if self._completed:
            self._on_complete(self.rpc_id)

    def add_success(self, worker_rank: int, value: Any) -> None:
        future = self.futures.get(worker_rank)
        if future is None or future.done():
            return
        future.set_result(value)

    def fail(self, exc: Exception) -> None:
        with self._lock:
            if self._completed:
                return
            self._completed = True
        for future in self.futures.values():
            if not future.done():
                future.set_exception(exc)
        self._on_complete(self.rpc_id)

    def result(self, deadline: Optional[float]) -> list[Any]:
        results: list[Any] = []
        for rank in self.expected_ranks:
            future = self.futures[rank]
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            results.append(future.result(timeout=timeout))
        return results


@dataclass
class _WorkerTask:
    rpc_id: int
    method: Union[str, bytes]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output_rank: Optional[int]


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ
    death_writer: Optional[Connection] = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy")
            self.async_output_copy_thread.start()
        else:
            self.async_output_queue = queue.Queue()

        # Track CUDA device for execution threads.
        self.device = torch.cuda.current_device()

        # Dedicated execution queues/threads for primary and secondary work.
        self.primary_task_queue: queue.Queue[
            Optional[_WorkerTask]] = queue.Queue()
        self.secondary_task_queue: queue.Queue[
            Optional[_WorkerTask]] = queue.Queue()
        self._execution_threads: list[Thread] = []

        primary_thread = Thread(target=self._execution_loop,
                                args=(self.primary_task_queue, "primary"),
                                daemon=True,
                                name=f"WorkerPrimaryExec-{rank}")
        primary_thread.start()
        self._execution_threads.append(primary_thread)

        secondary_thread = Thread(target=self._execution_loop,
                                  args=(self.secondary_task_queue, "secondary"),
                                  daemon=True,
                                  name=f"WorkerSecondaryExec-{rank}")
        secondary_thread.start()
        self._execution_threads.append(secondary_thread)

        # Initialize multimodal receiver cache if needed
        self.mm_receiver_cache = worker_receiver_cache_from_config(
            vllm_config, MULTIMODAL_REGISTRY, shared_worker_lock)

        # Initialize device
        self.worker.init_device()

        # Set process title and log prefix
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel)

        # Load model
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    # Extract the message queue handle.
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        # Stop execution threads first to avoid dangling GPU work.
        try:
            self.primary_task_queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        try:
            self.secondary_task_queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        for thread in getattr(self, "_execution_threads", ()):
            thread.join(timeout=1.0)

        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()
        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, rpc_id: int, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            status = WorkerProc.ResponseStatus.FAILURE
            payload = str(output)
        else:
            status = WorkerProc.ResponseStatus.SUCCESS
            payload = output
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue((rpc_id, status, payload))

    def handle_output(self, rpc_id: int, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put((rpc_id, output))
        else:
            self.enqueue_output(rpc_id, output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            rpc_id, output = self.async_output_queue.get()
            self.enqueue_output(rpc_id, output)

    def _execution_loop(self, task_queue: "queue.Queue[Optional[_WorkerTask]]",
                        role: str) -> None:
        """Executes tasks associated with a particular stream role."""
        torch.cuda.set_device(self.device)
        while True:
            task = task_queue.get()
            if task is None:
                break
            if isinstance(task, _WorkerTask):
                rpc_id = task.rpc_id
                method = task.method
                args = task.args
                kwargs = task.kwargs
                output_rank = task.output_rank
            else:
                rpc_id, method, args, kwargs, output_rank = task
            logger.debug("Worker rank %s executing rpc_id=%s on %s stream",
                        self.rank, rpc_id, role)
            try:
                func = self._resolve_method(method)
                if (self.mm_receiver_cache is not None
                        and getattr(func, "__name__", "") == "execute_model"):
                    get_and_update_mm_cache(self.mm_receiver_cache, args)
                if isinstance(method, str) and hasattr(self.worker,
                                                       "use_runner"):
                    context = self.worker.use_runner(role)
                else:
                    context = nullcontext()
                with context:
                    output = func(*args, **kwargs)
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                output = e

            if output_rank is None or self.rank == output_rank:
                self.handle_output(rpc_id, output)

    def _resolve_method(self, method: Union[str, bytes]) -> Callable:
        if isinstance(method, str):
            return getattr(self.worker, method)
        if isinstance(method, bytes):
            return partial(cloudpickle.loads(method), self.worker)
        raise TypeError(f"Unsupported RPC method type: {type(method)}")

    # TODO(girfan): Clean this up a bit. Maybe add a field to the RPC itself to indicate the stream it should be executed on?
    def _select_task_queue(
            self, method: Union[str, bytes],
            args: tuple[Any, ...]) -> "queue.Queue[Optional[_WorkerTask]]":
        if not self.use_async_scheduling:
            return self.primary_task_queue
        if isinstance(method, bytes):
            return self.primary_task_queue
        if method != "execute_model" or not args:
            return self.primary_task_queue
        scheduler_output = args[0]
        queue_label = getattr(scheduler_output, "queue", None)
        if queue_label == "secondary":
            return self.secondary_task_queue
        try:
            if self._requires_secondary_stream(scheduler_output):
                return self.secondary_task_queue
        except Exception:
            logger.debug("Failed to inspect scheduler_output for stream "
                         "selection; defaulting to primary stream.",
                         exc_info=True)
        return self.primary_task_queue

    def _requires_secondary_stream(self,
                                   scheduler_output: "SchedulerOutput") -> bool:
        def is_skip_kv(sampling_params) -> bool:
            if sampling_params is None:
                return False
            extra_args = getattr(sampling_params, "extra_args", None)
            if not extra_args:
                return False
            return bool(extra_args.get("skip_kv_cache", False))

        has_training = any(req.is_training
                           for req in scheduler_output.scheduled_new_reqs)
        has_skip_kv = any(is_skip_kv(req.sampling_params)
                          for req in scheduler_output.scheduled_new_reqs)
        if has_training or has_skip_kv:
            return True

        cached_ids = set(scheduler_output.num_scheduled_tokens.keys())
        cached_ids.update(scheduler_output.scheduled_cached_reqs.req_ids)
        for req_id in cached_ids:
            cached_state = self._get_request_state(req_id)
            if cached_state is None:
                continue
            if getattr(cached_state, "is_training", False):
                return True
            if is_skip_kv(getattr(cached_state, "sampling_params", None)):
                return True
        return False

    def _get_request_state(self, req_id: str) -> Optional[Any]:
        model_runner = getattr(self.worker, "model_runner", None)
        if model_runner is None:
            return None
        requests = getattr(model_runner, "requests", None)
        if isinstance(requests, dict):
            return requests.get(req_id)
        return None

    def worker_busy_loop(self, cancel: Optional[threading.Event] = None):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            if cancel is not None and cancel.is_set():
                break
            try:
                message = self.rpc_broadcast_mq.dequeue(cancel=cancel)
            except TimeoutError:
                if cancel is not None and cancel.is_set():
                    break
                continue

            if message is None:
                if cancel is not None and cancel.is_set():
                    break
                continue

            rpc_id, method, args, kwargs, output_rank = message
            task = _WorkerTask(rpc_id=rpc_id,
                               method=method,
                               args=args,
                               kwargs=kwargs,
                               output_rank=output_rank)
            target_queue = self._select_task_queue(method, args)
            target_queue.put(task)
            logger.debug("Worker rank %s dispatched rpc_id=%s to %s queue",
                        self.rank, rpc_id,
                        "secondary" if target_queue
                        is self.secondary_task_queue else "primary")

        # Signal execution threads to terminate.
        self.primary_task_queue.put(None)
        self.secondary_task_queue.put(None)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)
