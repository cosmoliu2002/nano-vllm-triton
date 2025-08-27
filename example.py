import argparse
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm triton example")
    argparse.add_argument("--model-path", type=str, default="/path/to/your/model")
    argparse.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    argparse.add_argument("--enforce-eager", type=bool, default=True)
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--max-tokens", type=int, default=256)
    args = argparse.parse_args()

    main(args)
