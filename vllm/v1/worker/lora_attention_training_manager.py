# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LoRA Attention Training Manager

Manages LoRA adapters for attention layers (q_proj, v_proj) and lm_head.
Uses homogeneous batch approach - training requests processed separately.
Compatible with vLLM's LoRA inference format.
"""

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from vllm.logger import init_logger

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA adapter layer: output = input + scaling * (B @ A @ input)
    
    Compatible with vLLM's LoRA format and PEFT library.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize like PEFT: A with small random values, B with zeros
        self.lora_A = nn.Parameter(
            torch.randn(rank, input_dim, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(output_dim, rank, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adapter.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            LoRA delta [..., output_dim]
        """
        # x @ A.T -> [..., rank]
        h = F.linear(x, self.lora_A)
        # h @ B.T -> [..., output_dim]
        delta = F.linear(h, self.lora_B)
        return delta * self.scaling


class LoRAAttention(torch.autograd.Function):
    """
    Custom autograd function for attention with LoRA and gradients.
    
    Based on xformers gradient support (see test_xf.py).
    Key features:
    - Apply LoRA to Q and V (not K)
    - Use gradient-enabled xformers attention
    - No KV cache (direct Q, K, V tensors)
    - Support backward pass
    """
    
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        base_qkv_proj: nn.Module,
        base_rotary_emb: nn.Module,
        base_o_proj: nn.Module,
        q_lora: LoRALayer,
        v_lora: LoRALayer,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
    ):
        """
        Forward pass with LoRA-enabled attention.
        
        Args:
            hidden_states: [seq_len, hidden_size]
            positions: [seq_len]
            base_qkv_proj: Base QKV projection (frozen)
            base_rotary_emb: RoPE layer
            base_o_proj: Output projection (frozen)
            q_lora, v_lora: LoRA adapters
            num_heads, num_kv_heads, head_dim: Attention config
            scale: Attention scale (1/sqrt(head_dim))
        
        Returns:
            output: [seq_len, hidden_size]
        """
        seq_len = hidden_states.shape[0]
        
        # 1. Base QKV projection (no gradients)
        with torch.no_grad():
            qkv, _ = base_qkv_proj(hidden_states)
            q_size = num_heads * head_dim
            kv_size = num_kv_heads * head_dim
            q_base, k, v_base = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # 2. Apply LoRA to Q and V (with gradients!)
        q_lora_delta = q_lora(hidden_states)
        v_lora_delta = v_lora(hidden_states)
        
        q = q_base + q_lora_delta  # Gradients flow through q_lora
        v = v_base + v_lora_delta  # Gradients flow through v_lora
        # k stays as base (no LoRA)
        
        # For GQA, we need to ensure q has the same number of heads as k/v
        # If num_heads > num_kv_heads, we need to repeat k and v to match q
        if num_heads > num_kv_heads:
            # Repeat k and v to match the number of query heads
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=-1)
            v = v.repeat_interleave(repeat_factor, dim=-1)
        
        # 3. Apply RoPE (frozen)
        with torch.no_grad():
            q, k = base_rotary_emb(positions, q, k)
        
        # 4. Reshape for xformers attention
        q = q.view(seq_len, num_heads, head_dim)
        # For GQA, k and v should have the same number of heads as q after repetition
        k = k.view(seq_len, num_heads, head_dim)
        v = v.view(seq_len, num_heads, head_dim)
        
        # 5. Attention with gradient support (no KV cache!)
        # Add batch dimension for xformers: [1, seq_len, num_heads, head_dim]
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        
        if not XFORMERS_AVAILABLE:
            raise ImportError("xformers is required for LoRA attention training")
        
        output, lse = xops.memory_efficient_attention_forward_requires_grad(
            query=q,
            key=k,
            value=v,
            attn_bias=None,  # Causal mask handled automatically
            p=0.0,  # No dropout
            scale=scale,
        )
        
        # 6. Remove batch dimension and reshape to [seq_len, num_heads * head_dim]
        output = output.squeeze(0).view(seq_len, num_heads * head_dim)
        
        # 7. Output projection (frozen)
        with torch.no_grad():
            final_output, _ = base_o_proj(output)
        
        # Save for backward - need to reshape output to match query shape
        output_reshaped = output.view(1, seq_len, num_heads, head_dim)
        ctx.save_for_backward(q, k, v, output_reshaped, lse, hidden_states)
        ctx.q_lora = q_lora
        ctx.v_lora = v_lora
        ctx.scale = scale
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.seq_len = seq_len
        
        return final_output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using xformers."""
        q, k, v, output, lse, hidden_states = ctx.saved_tensors
        
        # Reshape grad_output for xformers
        seq_len = grad_output.shape[0]
        grad_output_reshaped = grad_output.view(1, seq_len, ctx.num_heads, ctx.head_dim)
        
        # Backward through attention
        grad_q, grad_k, grad_v = xops.memory_efficient_attention_backward(
            grad_output_reshaped,
            output,  # Already has batch dim
            lse,
            q, k, v,
            attn_bias=None,
            p=0.0,
            scale=ctx.scale,
        )
        
        # Remove batch dimension
        grad_q = grad_q.squeeze(0).view(seq_len, -1)  # [seq_len, num_heads * head_dim]
        grad_v = grad_v.squeeze(0).view(seq_len, -1)
        # grad_k not used (no LoRA on k)
        
        # Backward through LoRA layers (PyTorch autograd handles this)
        # We need gradients w.r.t. hidden_states for chain rule
        grad_hidden_states = None
        
        # Return gradients for all forward args (match signature)
        return grad_hidden_states, None, None, None, None, None, None, None, None, None, None


class LoRAAttentionTrainingManager:
    """
    Manages LoRA training for attention layers (q_proj, v_proj) and lm_head.
    
    Uses homogeneous batches - training requests processed separately from inference.
    Compatible with vLLM's LoRA inference format (PackedLoRALayerWeights).
    """
    
    def __init__(
        self,
        model_runner: "GPUModelRunner",
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        learning_rate: float = 2e-4,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize LoRA attention training manager.
        
        Args:
            model_runner: The GPUModelRunner instance
            num_layers: Number of transformer layers
            hidden_size: Model hidden dimension
            vocab_size: Vocabulary size
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            lora_rank: Rank of LoRA matrices
            lora_alpha: LoRA scaling factor
            learning_rate: Learning rate for optimizer
            dtype: Data type for LoRA parameters
        """
        self.model_runner = model_runner
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Create LoRA adapters for Q projection (each layer)
        self.q_loras = nn.ModuleList([
            LoRALayer(
                hidden_size,
                num_heads * head_dim,
                lora_rank,
                lora_alpha,
                dtype
            ).cuda()
            for _ in range(num_layers)
        ])
        
        # Create LoRA adapters for V projection (each layer)
        self.v_loras = nn.ModuleList([
            LoRALayer(
                hidden_size,
                num_kv_heads * head_dim,
                lora_rank,
                lora_alpha,
                dtype
            ).cuda()
            for _ in range(num_layers)
        ])
        
        # Create LoRA adapter for LM head
        self.lm_head_lora = LoRALayer(
            hidden_size,
            vocab_size,
            lora_rank,
            lora_alpha,
            dtype
        ).cuda()
        
        # Optimizer for all LoRA parameters
        all_params = (
            list(self.q_loras.parameters()) +
            list(self.v_loras.parameters()) +
            list(self.lm_head_lora.parameters())
        )
        self.optimizer = AdamW(all_params, lr=learning_rate)
        
        total_params = sum(p.numel() for p in all_params)
        logger.info(
            f"LoRA Attention Training Manager initialized:\n"
            f"  Layers: {num_layers}\n"
            f"  Rank: {lora_rank}, Alpha: {lora_alpha}\n"
            f"  Trainable parameters: {total_params:,}\n"
            f"  Learning rate: {learning_rate}"
        )
    
    def forward_layer_with_lora(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        base_layer: nn.Module,
    ) -> torch.Tensor:
        """
        Forward pass through one attention layer with LoRA.
        
        Args:
            layer_idx: Which layer (0 to num_layers-1)
            hidden_states: [seq_len, hidden_size]
            positions: [seq_len]
            base_layer: The base transformer layer (frozen)
        
        Returns:
            output: [seq_len, hidden_size]
        """
        return LoRAAttention.apply(
            hidden_states,
            positions,
            base_layer.self_attn.qkv_proj,
            base_layer.self_attn.rotary_emb,
            base_layer.self_attn.o_proj,
            self.q_loras[layer_idx],
            self.v_loras[layer_idx],
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.scale,
        )
    
    def apply_lm_head_lora(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply lm_head LoRA to get final logits.
        
        Args:
            hidden_states: [seq_len, hidden_size]
        
        Returns:
            logits: [seq_len, vocab_size]
        """
        # Get base logits (frozen model)
        with torch.no_grad():
            base_logits = self.model_runner.model.compute_logits(
                hidden_states, None
            )
        
        # Add LoRA delta
        lora_delta = self.lm_head_lora(hidden_states)
        return base_logits + lora_delta
    
    def export_to_vllm_format(self) -> Dict[str, Dict]:
        """
        Export trained LoRA weights to vLLM inference format.
        
        Returns:
            Dictionary mapping module names to LoRA weight dicts
        """
        lora_weights = {}
        
        # Export attention LoRAs (q and v, k is None)
        for layer_idx in range(self.num_layers):
            module_name = f"model.layers.{layer_idx}.self_attn.qkv_proj"
            
            q_lora = self.q_loras[layer_idx]
            v_lora = self.v_loras[layer_idx]
            
            # Store in format compatible with PackedLoRALayerWeights
            lora_weights[module_name] = {
                'rank': self.lora_rank,
                'lora_alphas': [self.lora_alpha, None, self.lora_alpha],
                'lora_a': [
                    q_lora.lora_A.detach().cpu(),
                    None,
                    v_lora.lora_A.detach().cpu(),
                ],
                'lora_b': [
                    q_lora.lora_B.detach().cpu(),
                    None,
                    v_lora.lora_B.detach().cpu(),
                ],
            }
        
        # Export lm_head LoRA
        module_name = "lm_head"
        lora_weights[module_name] = {
            'rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_a': self.lm_head_lora.lora_A.detach().cpu(),
            'lora_b': self.lm_head_lora.lora_B.detach().cpu(),
        }
        
        return lora_weights
    
    def get_training_results(self) -> Optional[Dict[str, float]]:
        """Get the latest training results."""
        if hasattr(self, '_last_training_stats') and self._last_training_stats:
            return self._last_training_stats[-1]
        return None
    
    def save_checkpoint(self, path: str):
        """Save LoRA checkpoint to disk."""
        checkpoint = {
            'q_loras': [lora.state_dict() for lora in self.q_loras],
            'v_loras': [lora.state_dict() for lora in self.v_loras],
            'lm_head_lora': self.lm_head_lora.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
                'num_heads': self.num_heads,
                'num_kv_heads': self.num_kv_heads,
                'head_dim': self.head_dim,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"LoRA checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load LoRA checkpoint from disk."""
        checkpoint = torch.load(path)
        
        for i, state_dict in enumerate(checkpoint['q_loras']):
            self.q_loras[i].load_state_dict(state_dict)
        
        for i, state_dict in enumerate(checkpoint['v_loras']):
            self.v_loras[i].load_state_dict(state_dict)
        
        self.lm_head_lora.load_state_dict(checkpoint['lm_head_lora'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        logger.info(f"LoRA checkpoint loaded from {path}")

