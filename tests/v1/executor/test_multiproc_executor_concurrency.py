# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import queue
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest
import os

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS_VLLM_V1"

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.multiproc_executor import (MultiprocExecutor,
                                                 WorkerProc)


class _DummyRPCQueue:

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def enqueue(self, message):
        self._queue.put(message)

    def get(self, timeout: float = 1.0):
        return self._queue.get(timeout=timeout)


def _make_executor(num_workers: int = 2) -> MultiprocExecutor:
    executor = object.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.shutdown_event = threading.Event()
    executor.rpc_broadcast_mq = _DummyRPCQueue()
    executor.workers = [
        SimpleNamespace(rank=i, worker_response_mq=None)
        for i in range(num_workers)
    ]
    executor._rpc_id_counter = itertools.count()
    executor._pending_lock = threading.Lock()
    executor._pending_rpcs = {}
    executor._rpc_enqueue_lock = threading.Lock()
    executor._response_threads = []
    executor.io_thread_pool = None
    executor.has_connector = False
    return executor


def _cleanup_executor(executor: MultiprocExecutor) -> None:
    executor.shutdown_event.set()
    executor.workers = []


def _drain_messages(executor: MultiprocExecutor, count: int):
    messages = []
    for _ in range(count):
        messages.append(executor.rpc_broadcast_mq.get(timeout=1.0))
    return messages


def test_collective_rpc_concurrent_calls():
    executor = _make_executor(num_workers=2)
    results: dict[int, list[Any]] = {}

    def call(idx: int):
        res = executor.collective_rpc("execute_model", timeout=2.0)
        results[idx] = res

    threads = [
        threading.Thread(target=call, args=(0, )),
        threading.Thread(target=call, args=(1, )),
    ]
    for thread in threads:
        thread.start()

    messages = _drain_messages(executor, count=2)

    # Respond to the second RPC first to ensure out-of-order completion works.
    for rpc_id, _, _, _, _ in reversed(messages):
        executor._dispatch_worker_response(
            rpc_id, worker_rank=0,
            status=WorkerProc.ResponseStatus.SUCCESS,
            payload=f"{rpc_id}-w0")
        executor._dispatch_worker_response(
            rpc_id, worker_rank=1,
            status=WorkerProc.ResponseStatus.SUCCESS,
            payload=f"{rpc_id}-w1")

    for thread in threads:
        thread.join(timeout=2.0)

    try:
        assert results[0] == [f"0-w0", f"0-w1"]
        assert results[1] == [f"1-w0", f"1-w1"]
        assert executor._pending_rpcs == {}
    finally:
        _cleanup_executor(executor)


def test_collective_rpc_non_block_returns_futures():
    executor = _make_executor(num_workers=2)

    futures = executor.collective_rpc("execute_model",
                                      non_block=True,
                                      timeout=2.0)

    assert all(isinstance(fut, Future) for fut in futures)

    rpc_id, *_ = executor.rpc_broadcast_mq.get(timeout=1.0)
    executor._dispatch_worker_response(
        rpc_id,
        worker_rank=0,
        status=WorkerProc.ResponseStatus.SUCCESS,
        payload="primary",
    )
    executor._dispatch_worker_response(
        rpc_id,
        worker_rank=1,
        status=WorkerProc.ResponseStatus.SUCCESS,
        payload="secondary",
    )

    try:
        assert [future.result(timeout=1.0) for future in futures] == [
            "primary", "secondary"
        ]
        assert executor._pending_rpcs == {}
    finally:
        _cleanup_executor(executor)


def test_collective_rpc_failure_propagates():
    executor = _make_executor(num_workers=2)
    caught: list[BaseException] = []

    def call():
        with pytest.raises(RuntimeError) as exc_info:
            executor.collective_rpc("execute_model", timeout=2.0)
        caught.append(exc_info.value)

    thread = threading.Thread(target=call)
    thread.start()

    rpc_id, *_ = executor.rpc_broadcast_mq.get(timeout=1.0)
    executor._dispatch_worker_response(
        rpc_id,
        worker_rank=0,
        status=WorkerProc.ResponseStatus.FAILURE,
        payload="boom",
    )

    thread.join(timeout=2.0)
    try:
        assert len(caught) == 1
        assert "boom" in str(caught[0])
        assert executor._pending_rpcs == {}
    finally:
        _cleanup_executor(executor)


def _make_worker_stub(use_async_scheduling: bool = True) -> WorkerProc:
    worker = object.__new__(WorkerProc)
    worker.use_async_scheduling = use_async_scheduling
    worker.secondary_task_queue = queue.Queue()
    worker.primary_task_queue = queue.Queue()
    worker.worker = SimpleNamespace(
        model_runner=SimpleNamespace(requests={}))
    worker.mm_receiver_cache = None
    return worker


def test_requires_secondary_stream_detects_training_new_request():
    worker = _make_worker_stub()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(is_training=True,
                                            sampling_params=None)],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    assert worker._requires_secondary_stream(scheduler_output)


def test_requires_secondary_stream_detects_skip_kv_request():
    worker = _make_worker_stub()
    sampling_params = SimpleNamespace(extra_args={"skip_kv_cache": True})
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(is_training=False,
                                            sampling_params=sampling_params)],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    assert worker._requires_secondary_stream(scheduler_output)


def test_requires_secondary_stream_detects_cached_training_request():
    worker = _make_worker_stub()
    worker.worker.model_runner.requests["cached"] = SimpleNamespace(
        is_training=True, sampling_params=None)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"cached": 4},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"]),
    )
    assert worker._requires_secondary_stream(scheduler_output)


def test_requires_secondary_stream_inference_defaults_primary():
    worker = _make_worker_stub()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(is_training=False,
                                            sampling_params=None)],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    assert not worker._requires_secondary_stream(scheduler_output)


def test_select_task_queue_respects_async_flag():
    worker = _make_worker_stub(use_async_scheduling=True)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(is_training=True,
                                            sampling_params=None)],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    queue_selected = worker._select_task_queue("execute_model",
                                               (scheduler_output, ))
    assert queue_selected is worker.secondary_task_queue

    worker_no_async = _make_worker_stub(use_async_scheduling=False)
    queue_selected = worker_no_async._select_task_queue(
        "execute_model", (scheduler_output, ))
    assert queue_selected is worker_no_async.primary_task_queue


def test_worker_busy_loop_dispatches_to_streams(monkeypatch):
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)

    outputs: list[tuple[int, Any]] = []
    primary_started = threading.Event()
    secondary_started = threading.Event()
    primary_continue = threading.Event()
    secondary_continue = threading.Event()

    def execute_model(scheduler_output, intermediate_tensors=None):
        if scheduler_output.tag == "primary":
            primary_started.set()
            secondary_started.wait(timeout=1.0)
            primary_continue.wait(timeout=1.0)
            return "primary-output"
        secondary_started.set()
        primary_started.wait(timeout=1.0)
        secondary_continue.wait(timeout=1.0)
        return "secondary-output"

    worker = object.__new__(WorkerProc)
    worker.rank = 0
    worker.device = 0
    worker.use_async_scheduling = True
    worker.mm_receiver_cache = None
    worker.worker_response_mq = None
    worker.worker = SimpleNamespace(execute_model=execute_model,
                                    model_runner=SimpleNamespace(requests={}))
    worker.primary_task_queue = queue.Queue()
    worker.secondary_task_queue = queue.Queue()
    worker.async_output_queue = queue.Queue()
    worker._execution_threads = []

    def handle_output(self, rpc_id, output):
        outputs.append((rpc_id, output))

    worker.handle_output = handle_output.__get__(worker, WorkerProc)

    primary_thread = threading.Thread(target=worker._execution_loop,
                                      args=(worker.primary_task_queue,
                                            "primary"),
                                      daemon=True)
    secondary_thread = threading.Thread(target=worker._execution_loop,
                                        args=(worker.secondary_task_queue,
                                              "secondary"),
                                        daemon=True)
    primary_thread.start()
    secondary_thread.start()
    worker._execution_threads.extend([primary_thread, secondary_thread])

    msg_queue: queue.Queue = queue.Queue()
    scheduler_output_primary = SimpleNamespace(
        tag="primary",
        scheduled_new_reqs=[
            SimpleNamespace(is_training=False, sampling_params=None)
        ],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    scheduler_output_secondary = SimpleNamespace(
        tag="secondary",
        scheduled_new_reqs=[
            SimpleNamespace(is_training=True, sampling_params=None)
        ],
        num_scheduled_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    msg_queue.put(
        (1, "execute_model", (scheduler_output_primary, None), {}, None))
    msg_queue.put(
        (2, "execute_model", (scheduler_output_secondary, None), {}, None))

    def dequeue(cancel=None):
        try:
            return msg_queue.get(timeout=0.1)
        except queue.Empty:
            if cancel is not None and cancel.is_set():
                raise TimeoutError
            raise TimeoutError

    worker.rpc_broadcast_mq = SimpleNamespace(dequeue=dequeue)
    cancel_event = threading.Event()

    busy_thread = threading.Thread(target=worker.worker_busy_loop,
                                   args=(cancel_event, ),
                                   daemon=True)
    busy_thread.start()

    assert primary_started.wait(timeout=1.0)
    assert secondary_started.wait(timeout=1.0)

    primary_continue.set()
    secondary_continue.set()

    deadline = time.time() + 1.0
    while len(outputs) < 2 and time.time() < deadline:
        time.sleep(0.01)

    assert dict(outputs) == {
        1: "primary-output",
        2: "secondary-output",
    }

    cancel_event.set()
    busy_thread.join(timeout=1.0)
    worker.primary_task_queue.put(None)
    worker.secondary_task_queue.put(None)
    for thread in worker._execution_threads:
        thread.join(timeout=1.0)


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_MULTIPROC_TEST"),
    reason="requires RUN_E2E_MULTIPROC_TEST=1 to run integration scenario",
)
def test_multiproc_executor_end_to_end_primary_secondary(tmp_path):
    # models_dir = os.getenv("LAAL_MODELS_DIR", "/home/girfan/models")
    # if not os.path.exists(models_dir):
    #     pytest.skip(f"Models dir {models_dir} not found")

    models_dir = "/home/girfan/models"
    model_path = os.path.join(models_dir, "Llama-3.2-1B-Instruct")

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=1024,
        enforce_eager=True,
        async_scheduling=True,
        training_token_budget_ratio=0.5,
        dtype="bfloat16",
        disable_log_stats=True,
        block_size=64,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    try:
        sampling_primary = SamplingParams(max_tokens=1,
                                          temperature=0.0,
                                          seed=123)
        sampling_secondary = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            seed=321,
            extra_args={"skip_kv_cache": True},
        )

        engine.add_request("primary", "Hello world", sampling_primary)
        engine.add_request("secondary", "Hello world", sampling_secondary)

        outputs = {}
        step_count = 0
        while engine.has_unfinished_requests():
            step_count += 1
            for output in engine.step():
                outputs[output.request_id] = output
        assert step_count > 0
        assert set(outputs.keys()) == {"primary", "secondary"}
        assert outputs["primary"].outputs
        assert outputs["secondary"].outputs
    finally:
        engine.engine_core.shutdown()

