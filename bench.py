import argparse
import os
import time
from random import randint, seed

def main(args):
    inference_engine = args.engine
    if inference_engine == "nanovllmtriton":
        from nanovllm import LLM, SamplingParams
            
    elif inference_engine == "vllm":
        from vllm import LLM, SamplingParams
    else:
        raise ValueError(f"Unsupported engine: {inference_engine}. Choose 'nanovllmtriton' or 'vllm'")
    
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    
    path = os.path.expanduser(args.model_path)
    model_name = os.path.basename(path)
    
    llm = LLM(
        path, 
        enforce_eager=False,
        max_model_len=4096
        )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]

    if inference_engine == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Inference Engine: {inference_engine}, Model: {model_name}, Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm benchmark")
    argparse.add_argument("--model-path", type=str, default="/path/to/your/model")
    argparse.add_argument("--engine", type=str, choices=["nanovllmtriton", "vllm"], default="nanovllmtriton",help="Inference engine to use: nanovllmtriton (with Triton optimizations) or vllm")
    args = argparse.parse_args()

    main(args)

