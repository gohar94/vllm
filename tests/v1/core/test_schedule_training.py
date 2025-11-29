# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test schedule_training() implementation."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from .utils import create_scheduler, EOS_TOKEN_ID


def test_schedule_training_basic():
    """Test basic training request scheduling."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5)
    
    # Add training requests
    training_requests = []
    for i in range(3):
        request = Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),  # 10 tokens
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"skip_kv_cache": True}
            ),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
            is_training=False,  # Using skip_kv_cache flag
        )
        training_requests.append(request)
        scheduler.add_request(request)
    
    # Verify they're in training_waiting queue
    assert len(scheduler.training_waiting) == 3
    assert len(scheduler.waiting) == 0
    
    # Schedule training requests
    output = scheduler.schedule_training()
    
    # Verify scheduling worked
    assert len(output.scheduled_new_reqs) == 3
    assert output.total_num_scheduled_tokens == 30  # 3 requests * 10 tokens each
    assert len(scheduler.training_running) == 3
    assert len(scheduler.training_waiting) == 0
    
    # Verify no KV cache blocks allocated
    for req_data in output.scheduled_new_reqs:
        block_ids = req_data.block_ids
        # Should be empty tuple for training requests
        assert block_ids == ([], ) or all(len(ids) == 0 for ids in block_ids)


def test_schedule_training_with_is_training_flag():
    """Test scheduling with is_training=True flag."""
    scheduler = create_scheduler(training_token_budget_ratio=0.5)
    
    # Add training request with is_training=True
    request = Request(
        request_id="training_1",
        prompt_token_ids=list(range(20)),
        sampling_params=None,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        is_training=True,
    )
    scheduler.add_request(request)
    
    assert len(scheduler.training_waiting) == 1
    
    # Schedule
    output = scheduler.schedule_training()
    
    assert len(output.scheduled_new_reqs) == 1
    assert output.total_num_scheduled_tokens == 20
    assert len(scheduler.training_running) == 1


def test_schedule_training_token_budget():
    """Test that training scheduling respects token budget."""
    scheduler = create_scheduler(max_num_batched_tokens=100, 
                                 training_token_budget_ratio=0.5)
    
    # Add training requests that exceed token budget
    for i in range(5):
        request = Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(30)),  # 30 tokens each = 150 total
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"skip_kv_cache": True}
            ),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
        )
        scheduler.add_request(request)
    
    # Schedule - should respect 50% budget (50 tokens for training)
    output = scheduler.schedule_training()
    
    # Should only schedule requests that fit in the budget (50 tokens)
    # 1 request = 30 tokens, 2 requests = 50 tokens (30 + 20 due to budget limit)
    # Scheduler is greedy and will fit as many as possible
    assert output.total_num_scheduled_tokens <= 50
    # Can schedule 2 requests: first gets 30 tokens, second gets 20 tokens (limited by budget)
    assert len(output.scheduled_new_reqs) == 2  
    assert len(scheduler.training_running) == 2
    assert len(scheduler.training_waiting) == 3  # 3 still waiting


def test_schedule_training_and_inference_separate():
    """Test that training and inference scheduling are independent."""
    scheduler = create_scheduler(max_num_batched_tokens=100,
                                 training_token_budget_ratio=0.5)
    
    # Add inference requests
    for i in range(2):
        request = Request(
            request_id=f"inference_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
        )
        scheduler.add_request(request)
    
    # Add training requests
    for i in range(2):
        request = Request(
            request_id=f"training_{i}",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"skip_kv_cache": True}
            ),
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
        )
        scheduler.add_request(request)
    
    # Verify separate queues
    assert len(scheduler.waiting) == 2
    assert len(scheduler.training_waiting) == 2
    
    # Schedule inference
    inference_output = scheduler.schedule_inference()
    assert len(inference_output.scheduled_new_reqs) == 2
    assert len(scheduler.running) == 2
    assert len(scheduler.training_waiting) == 2  # Training unaffected
    
    # Schedule training
    training_output = scheduler.schedule_training()
    assert len(training_output.scheduled_new_reqs) == 2
    assert len(scheduler.training_running) == 2
    assert len(scheduler.running) == 2  # Inference unaffected


def test_schedule_training_chunked_prefill():
    """Test training scheduling with chunked prefill."""
    scheduler = create_scheduler(max_num_batched_tokens=100,
                                 training_token_budget_ratio=0.5)
    
    # Add large training request
    request = Request(
        request_id="training_large",
        prompt_token_ids=list(range(200)),  # Larger than budget
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"skip_kv_cache": True}
        ),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    scheduler.add_request(request)
    
    # First schedule - should schedule partial tokens (50 from training budget)
    output1 = scheduler.schedule_training()
    assert output1.total_num_scheduled_tokens == 50  # Training budget
    assert len(scheduler.training_running) == 1
    
    # Request should still be running (not finished)
    request_state = scheduler.requests_training["training_large"]
    assert request_state.num_computed_tokens == 50
    
    # Second schedule - should schedule more tokens
    # Note: Due to max_model_len constraint, the last token position is reserved
    # so we get 49 tokens instead of 50
    output2 = scheduler.schedule_training()
    assert output2.total_num_scheduled_tokens <= 50
    assert output2.total_num_scheduled_tokens >= 49  # Could be 49 or 50
    assert request_state.num_computed_tokens >= 99  # Should be close to 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

