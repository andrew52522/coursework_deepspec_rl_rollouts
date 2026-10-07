import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/model"


def main():
    print("=== ENV ===")
    print(
        "VLLM_USE_V2_MODEL_RUNNER =",
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "<UNSET>"),
    )

    print("=== TOKENIZER ===")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
    )

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2? Answer briefly.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("=== CREATE VLLM ENGINE ===")
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        enforce_eager=True,
    )

    print("=== GENERATE ===")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    text = outputs[0].outputs[0].text

    print("=== OUTPUT ===")
    print(text)

    print("=== VLLM NATIVE SMOKE: OK ===")


if __name__ == "__main__":
    main()
