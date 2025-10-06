# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import Optional, Union, final

import torch

import vllm.envs as envs
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

# FORCE TORCH_OPS: Set to False to disable Triton and use PyTorch SGMV/BGMV
USE_TRITON = False  # Changed from HAS_TRITON

if USE_TRITON:
    from vllm.lora.ops.triton_ops import (LoRAKernelMeta, lora_expand,
                                          lora_shrink)
else:
    # Use PyTorch fallback operations (SGMV/BGMV)
    from vllm.lora.ops.torch_ops import (bgmv_expand, bgmv_shrink,
                                         sgmv_expand, sgmv_shrink)

from .punica_base import PunicaWrapperBase

# Initialize logger at module level to avoid torch.compile issues
from vllm.logger import init_logger
logger = init_logger(__name__)


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        if USE_TRITON:
            self.token_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                          max_num_batched_tokens,
                                                          device=device)

            # When cudagraph capture size is greater than max_num_seqs (max_batches,
            # here), V0 captures the graph as if max_num_seqs is set to
            # the capture size.
            # V1 doesn't have this problem and always respects max_num_seqs.
            max_num_prompts = (max_batches
                               if envs.VLLM_USE_V1 else max_num_batched_tokens)
            self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                           max_num_prompts,
                                                           device=device)
        else:
            # For torch_ops (BGMV/SGMV), we don't need kernel metadata
            self.token_mapping_meta = None
            self.prompt_mapping_meta = None
            logger.info("Using PyTorch BGMV/SGMV operations (torch_ops) for LoRA")

    def update_metadata(self, mapping: LoRAMapping,
                        lora_index_to_id: list[Optional[int]], max_loras: int,
                        vocab_size: int, extra_vocab_size: int, **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size)

        # Prepare cuda kernel metadata tensors (only for Triton)
        if USE_TRITON:
            self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
            self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: tuple[torch.Tensor,
                                         ...], scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        if USE_TRITON:
            lora_shrink(
                x,
                lora_a_stacked,
                y,
                *self.token_mapping_meta.meta_args(x.size(0)),
                scale,
            )
        else:
            # Use PyTorch BGMV implementation
            # For torch_ops, we process each slice separately
            for i, lora_a in enumerate(lora_a_stacked):
                bgmv_shrink(x, lora_a, y[i], self.token_lora_indices, scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            self._apply_bias(token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        if USE_TRITON:
            lora_expand(
                x,
                lora_b_stacked,
                y,
                *self.token_mapping_meta.meta_args(num_tokens),
                offset_start=offset_start,
                add_inputs=True,
            )
        else:
            # Use PyTorch BGMV implementation
            # Process each slice separately
            offset = offset_start
            for i, (lora_b, slice_size) in enumerate(zip(lora_b_stacked, output_slices)):
                x_slice = x[i]  # Shape: (num_tokens, rank)
                y_slice = y[:, offset:offset + slice_size]
                bgmv_expand(x_slice, lora_b, y_slice, self.token_lora_indices, add_inputs=add_inputs)
                offset += slice_size

        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        if USE_TRITON:
            lora_expand(
                x.unsqueeze(dim=0),
                (lora_b_stacked, ),
                y,
                *self.token_mapping_meta.meta_args(x.size(0)),
                offset_start=0,
                add_inputs=add_inputs,
            )
        else:
            # Use PyTorch BGMV implementation
            bgmv_expand(x, lora_b_stacked, y, self.token_lora_indices, add_inputs=add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        # DEBUG: Log punica wrapper call
        if not hasattr(self, '_punica_logged'):
            self._punica_logged = True
            logger.info(f"[Punica GPU Debug] add_lora_linear called")
            logger.info(f"  y.shape: {y.shape}, y sample: {y[0, :5].tolist()}")
            logger.info(f"  x.shape: {x.shape}")
            logger.info(f"  scale: {scale}")
            logger.info(f"  token_lora_indices: {self._token_lora_indices[:min(10, len(self._token_lora_indices))].tolist()}")
            for i, (lora_a, lora_b) in enumerate(zip(lora_a_stacked, lora_b_stacked)):
                logger.info(f"  Stack {i}: lora_a.shape={lora_a.shape}, lora_b.shape={lora_b.shape}")
                logger.info(f"  Stack {i}: lora_a norm={lora_a.norm().item():.10f}, lora_b norm={lora_b.norm().item():.10f}")

        y_before = y.clone() if not hasattr(self, '_punica_logged_2') else None

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            y = self._apply_bias(token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=torch.float32,
                device=x.device,
            )
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs)
        
        # DEBUG: Check buffer after shrink (x @ lora_a * scale)
        if y_before is not None:
            logger.info(f"  After shrink: buffer.shape={buffer.shape}, buffer norm={buffer.norm().item():.10f}")
            logger.info(f"  buffer sample: {buffer[0, 0, :min(5, buffer.shape[-1])].tolist()}")
        
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            None,
            output_slices,
            add_inputs=True,
            **kwargs)
        
        # DEBUG: Check what changed
        if y_before is not None:
            self._punica_logged_2 = True
            delta = (y - y_before).abs().max().item()
            logger.info(f"  After expand: y changed by {delta:.10f}")
            logger.info(f"  y sample after: {y[0, :5].tolist()}")

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        if USE_TRITON:
            lora_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                        *self.prompt_mapping_meta.meta_args(x.size(0)), scale)

            lora_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                        y,
                        *self.prompt_mapping_meta.meta_args(buffer.size(0)),
                        add_inputs=True)
        else:
            # Use PyTorch BGMV implementation
            # buffer = (x @ lora_a) * scale
            bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices, scale)
            # y += buffer @ lora_b
            bgmv_expand(buffer, lora_b_stacked, y, self.sampler_indices, add_inputs=True)
        y = y.view_as(y_org)
