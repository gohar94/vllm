# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from collections.abc import Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionType
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor",
                                             1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config,
                              rope_scaling=rope_scaling,
                              quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            if hasattr(config, 'target_layer_count'):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), \
                f"effective_layer_idx: {effective_layer_idx} \
                is out of bounds for layer_types: {layer_types}"

            is_sliding = layer_types[
                effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (EncoderOnlyAttention
                    if attn_type == AttentionType.ENCODER_ONLY else Attention)

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )
        
        # DEBUG: Log attention class being used and set layer index
        layer_idx = extract_layer_index(prefix)
        self._layer_idx = layer_idx  # Store for forward pass logging
        if layer_idx == 0:
            print(f"\n[LlamaAttention Debug] Layer 0 Attention Setup")
            print(f"  Attention class: {attn_cls.__name__}")
            print(f"  Attention instance type: {type(self.attn)}")
            print(f"  cache_config: {cache_config}")
            print(f"  Actual attn object: {self.attn}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # DEBUG: Log attention forward pass details
        is_real_training = hidden_states.shape[0] < 100
        debug_key = f'_debug_attn_forward_{hidden_states.shape[0]}'
        layer_idx = getattr(self, '_layer_idx', -1)
        
        if is_real_training and layer_idx == 0 and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            print(f"\n[LlamaAttention Debug] Layer 0 Forward Pass")
            print(f"  hidden_states.shape: {hidden_states.shape}")
            print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
            print(f"  QKV projection type: {type(self.qkv_proj)}")
        
        qkv, _ = self.qkv_proj(hidden_states)
        
        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_2'):
            setattr(self, f'{debug_key}_2', True)
            print(f"  qkv.shape: {qkv.shape}")
            print(f"  qkv[0, :5]: {qkv[0, :5].tolist()}")
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        
        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_3'):
            setattr(self, f'{debug_key}_3', True)
            print(f"  After rotary - q[0, :5]: {q[0, :5].tolist()}")
            print(f"  After rotary - k[0, :5]: {k[0, :5].tolist()}")
            print(f"  After rotary - v[0, :5]: {v[0, :5].tolist()}")
            print(f"  Calling attn: {type(self.attn)}")
            # DEBUG: Check KV cache and metadata
            if hasattr(self.attn, 'kv_cache'):
                print(f"  KV cache type: {type(self.attn.kv_cache)}")
                if isinstance(self.attn.kv_cache, list) and len(self.attn.kv_cache) > 0:
                    kv = self.attn.kv_cache[0]
                    print(f"  KV cache[0] shape: {kv.shape if hasattr(kv, 'shape') else 'N/A'}")
                    print(f"  KV cache[0] numel: {kv.numel() if hasattr(kv, 'numel') else 'N/A'}")
                    if kv.numel() > 0:
                        print(f"  KV cache[0] norm: {kv.norm().item():.10f}")
        
        attn_output = self.attn(q, k, v, log=is_real_training and layer_idx == 0)

        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_4'):
            setattr(self, f'{debug_key}_4', True)
            torch.save(q, "q.pt")
            torch.save(k, "k.pt")
            torch.save(v, "v.pt")
            torch.save(hidden_states, "hidden_states.pt")
            torch.save(attn_output, "attn_output.pt")
            torch.save(self.attn.kv_cache, "kv_cache.pt")
            print(f"  Saved q, k, v to q.pt, k.pt, v.pt, attn_output.pt, hidden_states.pt, kv_cache.pt")
        
        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_5'):
            setattr(self, f'{debug_key}_5', True)
            print(f"  attn_output.shape: {attn_output.shape}")
            print(f"  attn_output[0, :5]: {attn_output[0, :5].tolist()}")
        
        # DEBUG: Before o_proj
        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_6'):
            setattr(self, f'{debug_key}_6', True)
            print(f"  Before o_proj: attn_output[0, :5]: {attn_output[0, :5].tolist()}")
            print(f"  o_proj type: {type(self.o_proj)}")
        
        output, _ = self.o_proj(attn_output)
        
        if is_real_training and layer_idx == 0 and not hasattr(self, f'{debug_key}_7'):
            setattr(self, f'{debug_key}_7', True)
            print(f"  After o_proj: output[0, :5]: {output[0, :5].tolist()}")
        
        return output

    def _init_rotary_emb(self, config: LlamaConfig,
                         rope_scaling: Optional[dict[str, Any]],
                         quant_config: Optional[QuantizationConfig]) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            partial_rotary_factor=self.partial_rotary_factor,
        )


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DEBUG: Log first layer details
        is_real_training = hidden_states.shape[0] < 100
        debug_key = f'_debug_layer_forward_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            layer_idx = getattr(self, '_layer_idx', -1)  # Will be set externally
            if layer_idx == 0:
                print(f"\n[LlamaDecoderLayer Debug] Layer 0 Forward")
                print(f"  positions: {positions.tolist()}")
                print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
                print(f"  residual is None: {residual is None}")
        
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        # DEBUG: Log after layernorm
        debug_key = f'_debug_after_norm_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            layer_idx = getattr(self, '_layer_idx', -1)
            if layer_idx == 0:
                print(f"  After input_layernorm[0, :5]: {hidden_states[0, :5].tolist()}")
        
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # DEBUG: Log after attention
        debug_key = f'_debug_after_attn_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            layer_idx = getattr(self, '_layer_idx', -1)
            if layer_idx == 0:
                print(f"  After self_attn[0, :5]: {hidden_states[0, :5].tolist()}")

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # DEBUG: Set layer indices for debugging
        for idx, layer in enumerate(self.layers):
            layer._layer_idx = idx
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor,
                                                        list[torch.Tensor]]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # DEBUG: Log input embeddings (for both profiling and actual runs)
        is_real_training = hidden_states.shape[0] < 100  # Real training has fewer tokens
        debug_key = f'_debug_embeddings_logged_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            print(f"\n[LlamaModel Debug] Input Embeddings (shape={hidden_states.shape})")
            print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
            print(f"  hidden_states.norm(): {hidden_states.norm().item():.10f}")

        aux_hidden_states = []
        layers_to_log = {0, 1, 2, len(self.layers)-1}  # Log first 3 and last layer
        
        for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            
            debug_key = f'_debug_layer_{idx}_logged_{hidden_states.shape[0]}'
            hidden_states_before = hidden_states.clone() if idx in layers_to_log and is_real_training and not hasattr(self, debug_key) else None
            hidden_states, residual = layer(positions, hidden_states, residual)
            
            # DEBUG: Log layer outputs
            if hidden_states_before is not None:
                setattr(self, debug_key, True)
                print(f"\n[LlamaModel Debug] Layer {idx} Output (shape={hidden_states.shape})")
                print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
                print(f"  hidden_states.norm(): {hidden_states.norm().item():.10f}")
                delta = (hidden_states - hidden_states_before).abs().max().item()
                print(f"  max change from input: {delta:.10f}")

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        # DEBUG: Log before norm
        debug_key = f'_debug_pre_norm_logged_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            print(f"\n[LlamaModel Debug] Before Norm (shape={hidden_states.shape})")
            print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
            print(f"  hidden_states.norm(): {hidden_states.norm().item():.10f}")
            if residual is not None:
                print(f"  residual[0, :5]: {residual[0, :5].tolist()}")

        hidden_states, _ = self.norm(hidden_states, residual)

        # DEBUG: Log after norm
        debug_key = f'_debug_post_norm_logged_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            print(f"\n[LlamaModel Debug] After Norm (final hidden states) (shape={hidden_states.shape})")
            print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
            print(f"  hidden_states.norm(): {hidden_states.norm().item():.10f}")

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      layer_type=layer_type)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            # FIX: Use config.vocab_size for LogitsProcessor, not unpadded_vocab_size
            # The unpadded_vocab_size includes lora_extra_vocab_size which is for
            # additional vocabulary tokens, but LogitsProcessor should trim to the
            # actual vocab size (config.vocab_size)
            self.logits_processor = LogitsProcessor(config.vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaDecoderLayer):
        return LlamaModel(vllm_config=vllm_config,
                          prefix=prefix,
                          layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # DEBUG: Log before logits computation
        is_real_training = hidden_states.shape[0] < 100
        debug_key = f'_debug_compute_logits_logged_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            print(f"\n[LlamaForCausalLM Debug] compute_logits Input (shape={hidden_states.shape})")
            print(f"  hidden_states[0, :5]: {hidden_states[0, :5].tolist()}")
            print(f"  hidden_states.norm(): {hidden_states.norm().item():.10f}")
        
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        
        # DEBUG: Log after logits computation
        debug_key = f'_debug_compute_logits_logged_2_{hidden_states.shape[0]}'
        if is_real_training and not hasattr(self, debug_key):
            setattr(self, debug_key, True)
            if logits is not None:
                print(f"\n[LlamaForCausalLM Debug] compute_logits Output (shape={logits.shape})")
                print(f"  logits[0, :5]: {logits[0, :5].tolist()}")
                print(f"  logits.norm(): {logits.norm().item():.10f}")
        
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int, attn_out: int):
            attn_in = self.config.head_dim * n_heads

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        # If using quantized model in mistral format,
        # quantization scales (qscale_weight) also need to be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads,
                                    self.config.hidden_size)
        elif "wk" in modules and modules[
                -1] == "qscale_weight" and loaded_weight.numel() > 1:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads, 1)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads,
                                    self.config.hidden_size)
        elif "wq" in modules and modules[
                -1] == "qscale_weight" and loaded_weight.numel() > 1:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads, 1)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = (f"{item}.{next_item}"
                             if next_item is not None else None)

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
