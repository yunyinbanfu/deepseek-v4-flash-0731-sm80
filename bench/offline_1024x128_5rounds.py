import gc
import os
import statistics as st
import time

from vllm import LLM, SamplingParams
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

MODEL = os.environ.get("DSV4_MODEL", "/srv/models/deepseek-ai/DeepSeek-V4-Flash-0731")

CONCURRENCY_TO_REQUESTS = {
    1: 16,
    4: 32,
    8: 64,
}

ROUNDS = 5
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 128

def make_prompt(tok, input_tokens, seed):
    unit = f"这是一个用于离线并发吞吐测试的中文文本片段，编号为{seed}，内容本身不重要，只用于构造固定长度输入。"
    ids = []
    while len(ids) < input_tokens:
        ids.extend(tok.encode(unit, add_special_tokens=False))
    ids = ids[:input_tokens]
    return tok.decode(ids)

def run_one_group(conc, req_per_round):
    tok = DeepseekV4Tokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = [
        make_prompt(tok, INPUT_TOKENS, seed=i)
        for i in range(req_per_round)
    ]

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        max_model_len=2048,
        max_num_seqs=conc,
        max_num_batched_tokens=conc * INPUT_TOKENS,
        gpu_memory_utilization=0.90,
        kv_cache_dtype="fp8",
        enforce_eager=True,
    )

    sampling = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        temperature=0.0,
        ignore_eos=True,
    )

    vals = []

    for r in range(ROUNDS):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        tok_s = out_tokens / elapsed
        vals.append(tok_s)

        print(
            f"conc={conc} req_per_round={req_per_round} round={r + 1}/{ROUNDS} "
            f"elapsed_s={elapsed:.3f} output_tokens={out_tokens} decode_tok_s={tok_s:.2f}",
            flush=True,
        )

    med = st.median(vals)
    lo, hi = min(vals), max(vals)
    cv = (st.stdev(vals) / st.mean(vals)) * 100 if len(vals) > 1 else 0.0

    print(
        f"SUMMARY conc={conc} req_per_round={req_per_round} "
        f"values=" + " / ".join(f"{v:.2f}" for v in vals) +
        f" median={med:.2f} range={lo:.2f}-{hi:.2f} cv={cv:.2f}%",
        flush=True,
    )

    del llm
    gc.collect()

def main():
    for conc, req_per_round in CONCURRENCY_TO_REQUESTS.items():
        run_one_group(conc, req_per_round)

if __name__ == "__main__":
    main()
