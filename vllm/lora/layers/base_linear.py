# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device

# DEBUG FLAG: Force PyTorch path instead of Punica for ALL forward passes
# This helps debug if Punica kernels are preventing evaluation from seeing updated weights


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_lora_device(self.base_layer)
        self.lora_bias_stacked: Optional[tuple[torch.Tensor, ...]] = None

        self.output_slices: tuple[int, ...]
        self.tp_size: int
        self.output_size: int
        self.n_slices: int

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (lora_config.max_lora_rank if
                               not lora_config.fully_sharded_loras else divide(
                                   lora_config.max_lora_rank, self.tp_size))
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (self.output_size if
                               not lora_config.fully_sharded_loras else divide(
                                   self.output_size, self.tp_size))
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        if lora_config.bias_enabled:
            lora_bias_out_size = lora_b_out_size
            self.lora_bias_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    lora_bias_out_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                ) for _ in range(self.n_slices))
        self.output_slices = (self.lora_b_stacked[0].shape[2], )

    def reset_lora(self, index: int):
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if we should skip reset for trained LoRAs
        # Look for a TrainingManager that might have trained parameters for this LoRA
        try:
            # Try to access the model's lora_manager to check for TrainingManager
            if hasattr(self, '_parent_model'):
                model = self._parent_model
            else:
                # Try to find the model through the module hierarchy
                model = self
                while hasattr(model, '_parent') and model._parent is not None:
                    model = model._parent
                if not hasattr(model, 'lora_manager'):
                    # Try to find lora_manager in the model
                    for attr_name in dir(model):
                        attr = getattr(model, attr_name, None)
                        if attr and hasattr(attr, '_adapter_manager'):
                            model.lora_manager = attr
                            break
            
            if hasattr(model, 'lora_manager') and hasattr(model.lora_manager, '_adapter_manager'):
                adapter_manager = model.lora_manager._adapter_manager
                training_manager = getattr(adapter_manager, '_training_manager', None)
                
                if training_manager and hasattr(training_manager, 'current_lora_id'):
                    # Get the LoRA ID for this index
                    lora_index_to_id = adapter_manager.lora_index_to_id
                    if index < len(lora_index_to_id) and lora_index_to_id[index] is not None:
                        lora_id = lora_index_to_id[index]
                        
                        # Check if this is the LoRA being trained
                        if (training_manager.current_lora_id == lora_id and
                            hasattr(training_manager, 'trainable_lora_params') and
                            training_manager.trainable_lora_params):
                            logger.info(f"[SKIP RESET] Skipping reset_lora for trained LoRA {lora_id} at index {index}")
                            return
        except Exception as e:
            # If anything goes wrong with the check, proceed with normal reset
            logger.debug(f"[reset_lora] Error checking for trained LoRA: {e}")

        # FINAL FIX: Skip reset during evaluation to preserve trained weights
        # Check if we're in evaluation mode by looking at tensor requires_grad
        is_eval_mode = False
        try:
            # If any stacked tensor has requires_grad=False, we're likely in eval mode
            if (len(self.lora_a_stacked) > 0 and 
                hasattr(self.lora_a_stacked[0], 'requires_grad') and
                not self.lora_a_stacked[0].requires_grad):
                is_eval_mode = True
        except:
            pass

        if is_eval_mode:
            return
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
            if self.lora_config.bias_enabled:
                # Make mypy happy
                self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                              self.lora_bias_stacked)
                self.lora_bias_stacked[s_index][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.lora_a_stacked) == len(self.lora_b_stacked) ==
                self.n_slices == 1)

        # Check if we should skip reset for trained parameters
        should_skip_reset = False
        try:
            # Check if we have a TrainingManager with trained parameters for this LoRA
            if hasattr(self, '_lora_manager') and self._lora_manager:
                training_manager = getattr(self._lora_manager._adapter_manager, '_training_manager', None)
                if (training_manager and 
                    hasattr(training_manager, 'current_lora_id') and
                    hasattr(training_manager, 'trainable_lora_params') and
                    training_manager.trainable_lora_params):
                    # Get the lora_id for this index
                    lora_index_to_id = self._lora_manager._adapter_manager.lora_index_to_id
                    if index < len(lora_index_to_id):
                        lora_id = lora_index_to_id[index]
                        if lora_id == training_manager.current_lora_id:
                            should_skip_reset = True
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(f"[FIX] Skipping reset_lora in set_lora for trained LoRA {lora_id}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[set_lora] Error checking for trained LoRA: {e}")
        
        if not should_skip_reset:
            self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_bias is not None:
                lora_bias = self.slice_bias(lora_bias)

        self.lora_a_stacked[0][index,
                               0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                   lora_a.T, non_blocking=True)
        self.lora_b_stacked[0][index,
                               0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                   lora_b.T, non_blocking=True)
        if lora_bias is not None:

            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            assert len(self.lora_bias_stacked)
            self.lora_bias_stacked[0][index, 0, :lora_bias.shape[0]].copy_(
                lora_bias.T, non_blocking=True)

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        # TRAINING MODE: Punica kernels don't support autograd, use PyTorch ops instead
        # For training, tensors will have requires_grad=True
        # Use PyTorch path for training (requires_grad=True) to support autograd
        if x.requires_grad or output.requires_grad:
            
            # Training path: bypass Punica, use plain PyTorch matmul for gradient flow
            # Always apply LoRA during training (even if weights are zero) to enable gradient flow
            
            if len(self.lora_a_stacked) > 0:
                max_loras = self.lora_a_stacked[0].shape[0]
                
                # Apply ALL LoRAs in the stacked tensors (supports parallel training)
                # We apply even zero-initialized LoRAs to enable gradient flow from scratch
                for lora_idx in range(max_loras):
                    # Check if this slot might be active by looking at tensor shapes
                    # We don't check for zero values because LoRAs can start from zero during training
                    first_lora_a = self.lora_a_stacked[0][lora_idx, 0, :, :]
                    
                    # Apply LoRA if the slot is properly shaped (not uninitialized)
                    if first_lora_a.numel() > 0:
                        # Apply this LoRA using PyTorch ops (supports autograd)
                        # For packed layers (e.g., QKV), apply each component to its output slice
                        output_offset = 0
                        for i in range(len(self.lora_a_stacked)):
                            lora_a = self.lora_a_stacked[i][lora_idx, 0, :, :]  # [rank, input_size]
                            lora_b = self.lora_b_stacked[i][lora_idx, 0, :, :]  # [output_size, rank]
                            
                            # LoRA: lora_output = x @ lora_a.T @ lora_b.T
                            # This creates a computation graph even if weights are zero
                            lora_output = x @ lora_a.T @ lora_b.T  # [seq_len, output_size]
                            
                            # Get the output slice size for this component
                            slice_size = self.output_slices[i]
                            
                            # Add LoRA output to the corresponding slice of the base output
                            # CRITICAL: Use in-place addition to ensure gradients flow back
                            output[:, output_offset:output_offset + slice_size] = output[:, output_offset:output_offset + slice_size] + lora_output
                            output_offset += slice_size
            
            return output

        # INFERENCE MODE: Use Punica kernels (fast but no autograd)

        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_linear(
                output, x, self.lora_a_stacked, self.lora_b_stacked,
                self.lora_bias_stacked, 1.0, self.output_slices)
        if not current_platform.can_update_inplace():
            output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
