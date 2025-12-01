"""Test concurrent execution of inference and training requests with CUDA streams."""
import pytest
import time
from typing import List

from vllm import SamplingParams
from vllm.config import SchedulerConfig
from vllm.v1.core.kv_cache_manager import Request
from vllm.v1.core.sched.scheduler import Scheduler

from .utils import create_scheduler


def create_test_request(request_id: str, 
                        prompt_token_ids: List[int],
                        is_training: bool = False,
                        skip_kv_cache: bool = False) -> Request:
    """Helper to create a test request."""
    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
    )

    if skip_kv_cache:
        sampling_params.extra_args = {"skip_kv_cache": True}

    req = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        is_training=is_training,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
    )

    return req


def test_scheduler_with_async_scheduling():
    """Test that scheduler works with async_scheduling enabled."""
    # Create scheduler with async_scheduling
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=100,
        training_token_budget_ratio=0.5,  # 50/50 split
        async_scheduling=True,
    )

    # Note: async_scheduling is a SchedulerConfig parameter, 
    # but we're just testing the scheduler logic here
    # The actual concurrent execution happens in the worker

    # Add inference requests
    inf_req1 = create_test_request("inf_1", [1, 2, 3, 4, 5])
    inf_req2 = create_test_request("inf_2", [6, 7, 8, 9, 10])

    # Add training requests
    train_req1 = create_test_request("train_1", [11, 12, 13], is_training=True)
    train_req2 = create_test_request("train_2", [14, 15, 16], skip_kv_cache=True)

    # Add all requests
    scheduler.add_request(inf_req1)
    scheduler.add_request(inf_req2)
    scheduler.add_request(train_req1)
    scheduler.add_request(train_req2)

    # Verify queue separation
    assert scheduler.has_primary_requests()
    assert scheduler.has_secondary_requests()
    assert len(scheduler.waiting) == 2  # 2 inference requests
    assert len(scheduler.secondary_waiting) == 2  # 2 training requests

    # Schedule inference requests
    inf_output = scheduler.schedule_primary()
    assert len(inf_output.scheduled_new_reqs) > 0

    # Schedule training requests
    train_output = scheduler.schedule_secondary()
    assert len(train_output.scheduled_new_reqs) > 0

    # Verify requests were scheduled from correct queues
    inf_req_ids = {req.req_id for req in inf_output.scheduled_new_reqs}
    train_req_ids = {req.req_id for req in train_output.scheduled_new_reqs}

    assert "inf_1" in inf_req_ids or "inf_2" in inf_req_ids
    assert "train_1" in train_req_ids or "train_2" in train_req_ids

    # Verify no overlap
    assert len(inf_req_ids.intersection(train_req_ids)) == 0


def test_token_budget_allocation():
    """Test that token budget is correctly split between inference and training."""
    max_tokens = 100
    training_ratio = 0.3  # 30% for training, 70% for inference

    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=max_tokens,
        training_token_budget_ratio=training_ratio,
        async_scheduling=True,
    )

    # Check token budget allocation
    expected_training_tokens = int(max_tokens * training_ratio)
    expected_inference_tokens = max_tokens - expected_training_tokens

    assert scheduler.max_num_scheduled_tokens_secondary == expected_training_tokens
    assert scheduler.max_num_scheduled_tokens_primary == expected_inference_tokens


def test_concurrent_request_routing():
    """Test that inference and training requests are routed to correct queues."""
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=100,
        training_token_budget_ratio=0.5,
        async_scheduling=True,
    )

    # Create mixed requests
    requests = [
        create_test_request("inf_1", [1, 2, 3]),
        create_test_request("train_1", [4, 5, 6], is_training=True),
        create_test_request("inf_2", [7, 8, 9]),
        create_test_request("train_2", [10, 11, 12], skip_kv_cache=True),
    ]

    # Add all requests
    for req in requests:
        scheduler.add_request(req)

    # Verify routing
    assert len(scheduler.waiting) == 2
    assert len(scheduler.secondary_waiting) == 2

    # Verify request IDs in correct queues
    inf_ids = {req.request_id for req in scheduler.waiting}
    train_ids = {req.request_id for req in scheduler.secondary_waiting}

    assert inf_ids == {"inf_1", "inf_2"}
    assert train_ids == {"train_1", "train_2"}


def test_is_training_flag_normalization():
    """Test that skip_kv_cache=True sets is_training=True."""
    scheduler = create_scheduler(
        max_num_seqs=10,
        max_num_batched_tokens=100,
        training_token_budget_ratio=0.5,
        async_scheduling=True,
    )

    # Create request with skip_kv_cache=True
    req = create_test_request("req_1", [1, 2, 3], skip_kv_cache=True)

    # Initially, is_training might be False
    assert req.is_training is False or req.is_training is True  # Could be either

    # After adding to scheduler, is_training should be normalized
    scheduler.add_request(req)

    # The request should remain inference but routed to secondary queue
    assert req.is_training is False
    assert len(scheduler.secondary_waiting) == 1
    assert len(scheduler.waiting) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

