import sys
import torch
import pickle
from transformers import LlamaConfig
from vllm import ModelRegistry
from vllm.config import CacheConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


DEVICE = 'cuda'


def get_cache_config():
    block_size = 16
    gpu_memory_utilization = 0.9
    swap_space = 16
    cache_dtype = "auto"
    num_gpu_blocks_override = None
    sliding_window = None
    enable_prefix_caching = False
    cpu_offload_gb = 0
    config = CacheConfig(
        block_size,
        gpu_memory_utilization,
        swap_space,
        cache_dtype,
        num_gpu_blocks_override,
        sliding_window,
        enable_prefix_caching,
        cpu_offload_gb,
    )
    return config

def get_llama_config():
    llama_config_dict = {
        "_name_or_path": "meta-llama/Meta-Llama-3-70B-Instruct",
        "architectures": [
          "LlamaForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 28672,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": torch.bfloat16,
        "transformers_version": "4.43.4",
        "use_cache": True,
        "vocab_size": 128256
    }
    llama_config = LlamaConfig(**llama_config_dict)
    return llama_config


def get_llama_attention(cache_config, llama_config):
    hidden_size = 8192
    num_heads = 64
    num_kv_heads = 8
    rope_theta = 500000.0
    rope_scaling = None
    max_position_embeddings = 8192
    quant_config = None
    bias = False
    prefix = "model.layers.79.self_attn"
    tp_rank = 0 # Dummy
    tp_size = 1
    cls = ModelRegistry.load_model_cls("LlamaAttention")
    attn = cls(
        config=llama_config,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
        quant_config=quant_config,
        bias=bias,
        cache_config=cache_config,
        prefix=prefix,
        tp_rank=tp_rank,
        tp_size=tp_size,
    ).to(DEVICE)
    return attn


def get_fwd_inputs(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    positions = data['positions'].to(DEVICE)
    hidden_states = data['hidden_states'].to(DEVICE)
    kv_cache = data['kv_cache'].to(DEVICE)
    attn_metadata = data['attn_metadata']
    print(f"positions: {positions.shape}", file=sys.stderr)
    print(f"hidden_states: {hidden_states.shape}", file=sys.stderr)
    print(f"k_cache: {kv_cache[0].shape}", file=sys.stderr)
    print(f"v_cache: {kv_cache[1].shape}", file=sys.stderr)
    if attn_metadata.query_start_loc is not None:
        attn_metadata.query_start_loc = attn_metadata.query_start_loc.to(DEVICE)
    if attn_metadata.seq_start_loc is not None:
        attn_metadata.seq_start_loc = attn_metadata.seq_start_loc.to(DEVICE)
    if attn_metadata.context_lens_tensor is not None:
        attn_metadata.context_lens_tensor = attn_metadata.context_lens_tensor.to(DEVICE)
    if attn_metadata.block_tables is not None:
        attn_metadata.block_tables = attn_metadata.block_tables.to(DEVICE)
    print(f"positions: {positions.shape} {hex(positions.data_ptr())}", file=sys.stderr)
    print(f"hidden_states: {hidden_states.shape} {hex(hidden_states.data_ptr())}", file=sys.stderr)
    print(f"k_cache: {kv_cache[0].shape} {hex(kv_cache[0].data_ptr())}", file=sys.stderr)
    print(f"v_cache: {kv_cache[1].shape} {hex(kv_cache[1].data_ptr())}", file=sys.stderr)
    return positions, hidden_states, kv_cache, attn_metadata


def run_benchmark():
    fpath = "prefill_1413833.pkl"
    positions, hidden_states, kv_cache, attn_metadata = get_fwd_inputs(fpath)
    with set_default_torch_dtype(torch.bfloat16):
        cache_config = get_cache_config()
        llama_config = get_llama_config()
        attn = get_llama_attention(cache_config, llama_config)
        output = attn(positions, hidden_states, kv_cache, attn_metadata)
        print(f"output: {output.shape} {output.data_ptr()}", file=sys.stderr)


def main():
    run_benchmark()


if __name__ == '__main__':
    main()
