# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TrainingManager for managing LoRA adapters during training.

This module provides functionality to create and manage LoRA adapters
specifically for training purposes.
"""

import torch
from typing import Optional
import torch.nn as nn

from vllm.model_executor.models import SupportsLoRA
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.lora import LoRALayerWeights
from vllm.lora.worker_manager import WorkerLoRAManager


from safetensors.torch import save_file
import json

import os


logger = init_logger(__name__)


# TODO(girfan): Make these configurable.
RANK = 8
ALPHA = 16


class TrainingManager:
    LoRA_PATH = "./llama3_dummy_lora"

    def __init__(
        self,
        model: SupportsLoRA,
        lora_manager: WorkerLoRAManager,
        lora_config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
        sub_modules: Optional[list[str]] = None,
    ):
        self.model = model
        self.lora_manager = lora_manager
        self.lora_config = lora_config
        self.device = device
        self.dtype = dtype
        self.sub_modules = sub_modules
        self._next_lora_id = 1


    def create_lora(self, model: nn.Module, sub_modules: list[str],
                    device: torch.device) -> LoRAModel:
        loras: dict[str, LoRALayerWeights] = {}
        for name in sub_modules:
            w = model.get_submodule(name).weight
            
            # Initialize LoRA weights following PEFT's default initialization:
            # - lora_a (Matrix A): Kaiming uniform (same as nn.Linear)
            # - lora_b (Matrix B): zeros
            # This ensures ΔW = B @ A = 0 initially (zero-init principle)
            lora_a = torch.empty([w.shape[1], RANK], 
                                 dtype=self.dtype,
                                 device=device)
            torch.nn.init.kaiming_uniform_(lora_a, a=torch.sqrt(torch.tensor(5.0)))
            
            lora_b = torch.zeros([RANK, w.shape[0]], 
                                 dtype=self.dtype,
                                 device=device)
            
            loras[name] = LoRALayerWeights(
                name,
                RANK,
                ALPHA,
                lora_a,
                lora_b,
            )
        lora_id = self._next_lora_id
        self._next_lora_id += 1
        return LoRAModel(lora_id, RANK, loras)
