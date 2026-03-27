# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that training requests are properly detected by workers."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from .utils import create_scheduler, EOS_TOKEN_ID


def test_skip_kv_cache_sets_is_training_flag():
    """Test that skip_kv_cache=True routes request to secondary queue."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5,
                                 async_scheduling=True)

    # Create request with skip_kv_cache but is_training=False
    request = Request(
        request_id="test_1",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"skip_kv_cache": True}
        ),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=False,  # Explicitly False
    )

    # Before adding to scheduler
    assert request.is_training == False

    # Add to scheduler
    scheduler.add_request(request)

    # The scheduler should not mutate is_training, but should route to secondary queue
    assert request.is_training == False
    assert len(scheduler.secondary_waiting) == 1
    assert len(scheduler.waiting) == 0


def test_worker_receives_correct_is_training_flag():
    """Test that NewRequestData reflects original is_training flag."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5,
                                 async_scheduling=True)

    # Add request with skip_kv_cache
    request = Request(
        request_id="test_1",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"skip_kv_cache": True}
        ),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=False,
    )
    scheduler.add_request(request)

    # Schedule training requests
    output = scheduler.schedule_secondary()

    # Verify NewRequestData preserves request.is_training flag
    assert len(output.scheduled_new_reqs) == 1
    new_req_data = output.scheduled_new_reqs[0]
    assert new_req_data.is_training == False


def test_explicit_is_training_unchanged():
    """Test that requests with explicit is_training=True remain unchanged."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5,
                                 async_scheduling=True)

    # Create request with explicit is_training=True
    request = Request(
        request_id="test_1",
        prompt_token_ids=list(range(10)),
        sampling_params=None,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=True,
    )

    # Add to scheduler
    scheduler.add_request(request)

    # is_training should still be True
    assert request.is_training == True

    # Verify it's in secondary queue
    assert len(scheduler.secondary_waiting) == 1

    # Schedule and verify
    output = scheduler.schedule_secondary()
    assert output.scheduled_new_reqs[0].is_training == True


def test_inference_request_unchanged():
    """Test that regular inference requests don't have is_training modified."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5,
                                 async_scheduling=True)

    # Create regular inference request
    request = Request(
        request_id="test_1",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=False,
    )

    # Add to scheduler
    scheduler.add_request(request)

    # is_training should still be False
    assert request.is_training == False

    # Verify it's in inference queue
    assert len(scheduler.waiting) == 1
    assert len(scheduler.secondary_waiting) == 0

    # Schedule and verify
    output = scheduler.schedule_primary()
    assert output.scheduled_new_reqs[0].is_training == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


