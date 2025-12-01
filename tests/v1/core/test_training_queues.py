# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test training queue functionality in the scheduler."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler, EOS_TOKEN_ID


def test_training_queue_routing_primary_only():
    """When async scheduling is disabled, all requests route to primary queue."""
    scheduler = create_scheduler(async_scheduling=False)

    regular_requests = create_requests(num_requests=3)
    for request in regular_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 3
    assert len(scheduler.secondary_waiting) == 0  # single queue mode

    # Add training requests; they should also live in the primary queue.
    training_requests = [
        Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=None,
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=True,
        ) for i in range(2)
    ]
    for request in training_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 5  # All enqueued in primary queue
    assert len(scheduler.secondary_waiting) == 0

    # Requests with skip_kv_cache should also remain in the primary queue.
    skip_kv_requests = [
        Request(
            request_id=f"skip_kv_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(
                max_tokens=1, extra_args={"skip_kv_cache": True}),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=False,
        ) for i in range(2)
    ]
    for request in skip_kv_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 7
    assert len(scheduler.secondary_waiting) == 0
    assert scheduler.get_num_unfinished_requests() == 7


def test_training_queue_routing_with_separate_queues():
    """When async scheduling is enabled, training routes to secondary queue."""
    scheduler = create_scheduler(async_scheduling=True)

    regular_requests = create_requests(num_requests=3)
    for request in regular_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 3
    assert len(scheduler.secondary_waiting) == 0

    training_requests = [
        Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=None,
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=True,
        ) for i in range(2)
    ]
    for request in training_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 3
    assert len(scheduler.secondary_waiting) == 2

    skip_kv_requests = [
        Request(
            request_id=f"skip_kv_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(
                max_tokens=1, extra_args={"skip_kv_cache": True}),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=False,
        ) for i in range(2)
    ]
    for request in skip_kv_requests:
        scheduler.add_request(request)

    assert len(scheduler.waiting) == 3
    assert len(scheduler.secondary_waiting) == 4
    assert scheduler.get_num_unfinished_requests() == 7


def test_is_training_request_helper():
    """Test the _is_training_request helper method."""
    scheduler = create_scheduler()
    
    # Regular inference request
    regular_request = Request(
        request_id="regular",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=False,
    )
    assert not scheduler._is_training_request(regular_request)
    
    # Training request (is_training=True)
    training_request = Request(
        request_id="training",
        prompt_token_ids=list(range(10)),
        sampling_params=None,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=True,
    )
    assert scheduler._is_training_request(training_request)
    
    # Request with skip_kv_cache=True
    skip_kv_request = Request(
        request_id="skip_kv",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"skip_kv_cache": True}
        ),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=False,
    )
    assert scheduler._is_secondary_request(skip_kv_request)


def test_finish_training_requests():
    """Test training request removal respects queue separation settings."""
    scheduler = create_scheduler(async_scheduling=True)

    # Add regular and training requests
    regular_requests = create_requests(num_requests=2)
    for request in regular_requests:
        scheduler.add_request(request)
    
    training_requests = []
    for i in range(2):
        request = Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=None,
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=True,
        )
        training_requests.append(request)
        scheduler.add_request(request)
    
    # Verify initial state
    assert len(scheduler.waiting) == 2
    assert len(scheduler.secondary_waiting) == 2
    
    # Finish one training request
    scheduler.finish_requests(
        training_requests[0].request_id,
        RequestStatus.FINISHED_STOPPED
    )
    
    # Verify the training request was removed
    assert len(scheduler.waiting) == 2  # regular requests unchanged
    assert len(scheduler.secondary_waiting) == 1  # one training request removed
    assert training_requests[0].request_id not in scheduler.requests
    
    # Finish one regular request
    scheduler.finish_requests(
        regular_requests[0].request_id,
        RequestStatus.FINISHED_STOPPED
    )
    
    # Verify the regular request was removed
    assert len(scheduler.waiting) == 1  # one regular request removed
    assert len(scheduler.secondary_waiting) == 1  # training queue unchanged
    assert regular_requests[0].request_id not in scheduler.requests
    
    # Test get_num_unfinished_requests
    assert scheduler.get_num_unfinished_requests() == 2  # 1 regular + 1 training


def test_training_queue_stats():
    """Stats should report training queues when separate queues enabled."""
    scheduler = create_scheduler(skip_tokenizer_init=True, async_scheduling=True)
    
    # Add various requests
    regular_requests = create_requests(num_requests=3)
    for request in regular_requests:
        scheduler.add_request(request)
    
    training_requests = []
    for i in range(2):
        request = Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=None,
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=True,
        )
        training_requests.append(request)
        scheduler.add_request(request)
    
    # Get stats
    stats = scheduler.make_stats()
    
    # Verify stats include secondary queues
    assert stats is not None
    assert stats.num_waiting_reqs == 3
    assert stats.num_running_reqs == 0
    assert stats.num_secondary_waiting_reqs == 2
    assert stats.num_secondary_running_reqs == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

