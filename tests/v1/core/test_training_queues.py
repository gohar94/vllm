# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test training queue functionality in the scheduler."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler, EOS_TOKEN_ID


def test_training_queue_routing():
    """Test that training requests are correctly routed to training queues."""
    scheduler = create_scheduler()
    
    # Create regular inference requests
    regular_requests = create_requests(num_requests=3)
    
    # Add regular requests
    for request in regular_requests:
        scheduler.add_request(request)
    
    # Verify regular requests are in waiting queue
    assert len(scheduler.waiting) == 3
    assert len(scheduler.training_waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.training_running) == 0
    
    # Create training requests (is_training=True)
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
    
    # Verify training requests are in training_waiting queue
    assert len(scheduler.waiting) == 3
    assert len(scheduler.training_waiting) == 2
    assert len(scheduler.running) == 0
    assert len(scheduler.training_running) == 0
    
    # Create requests with skip_kv_cache=True
    skip_kv_requests = []
    for i in range(2):
        request = Request(
            request_id=f"skip_kv_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"skip_kv_cache": True}
            ),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=False,
        )
        skip_kv_requests.append(request)
        scheduler.add_request(request)
    
    # Verify skip_kv_cache requests are also in training_waiting queue
    assert len(scheduler.waiting) == 3
    assert len(scheduler.training_waiting) == 4  # 2 training + 2 skip_kv
    assert len(scheduler.running) == 0
    assert len(scheduler.training_running) == 0
    
    # Test get_num_unfinished_requests
    total_unfinished = scheduler.get_num_unfinished_requests()
    assert total_unfinished == 7  # 3 regular + 2 training + 2 skip_kv


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
    assert scheduler._is_training_request(skip_kv_request)


def test_finish_training_requests():
    """Test that training requests can be properly finished and removed from queues."""
    scheduler = create_scheduler()
    
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
    assert len(scheduler.training_waiting) == 2
    
    # Finish one training request
    scheduler.finish_requests(
        training_requests[0].request_id,
        RequestStatus.FINISHED_STOPPED
    )
    
    # Verify the training request was removed
    assert len(scheduler.waiting) == 2  # regular requests unchanged
    assert len(scheduler.training_waiting) == 1  # one training request removed
    assert training_requests[0].request_id not in scheduler.requests
    
    # Finish one regular request
    scheduler.finish_requests(
        regular_requests[0].request_id,
        RequestStatus.FINISHED_STOPPED
    )
    
    # Verify the regular request was removed
    assert len(scheduler.waiting) == 1  # one regular request removed
    assert len(scheduler.training_waiting) == 1  # training queue unchanged
    assert regular_requests[0].request_id not in scheduler.requests
    
    # Test get_num_unfinished_requests
    assert scheduler.get_num_unfinished_requests() == 2  # 1 regular + 1 training


def test_training_queue_stats():
    """Test that scheduler stats correctly track training queues."""
    scheduler = create_scheduler(skip_tokenizer_init=True)
    
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
    
    # Verify stats include training queues
    assert stats is not None
    assert stats.num_waiting_reqs == 3
    assert stats.num_running_reqs == 0
    assert stats.num_training_waiting_reqs == 2
    assert stats.num_training_running_reqs == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

