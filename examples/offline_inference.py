from vllm import LLM, SamplingParams
import torch

# Sample prompts.
prompts = [
    "The president of the United States is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", disable_log_stats=False, tensor_parallel_size=2)

llm = LLM(
    model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    quantization="gptq",
    dtype=torch.float16,
    tensor_parallel_size=2,
    max_model_len=16384,
    revision="gptq-4bit-32g-actorder_True",
    gpu_memory_utilization=0.75,
    disable_custom_all_reduce=True,
    enforce_eager=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
