import os
import time

os.environ["VLLM_MARLIN_FP8_DEQUANT_BF16"] = "1"
os.environ["VLLM_MARLIN_FP8_DEQUANT_INCLUDE"] = "shared_experts.gate_up_proj,shared_experts.down_proj"

from vllm import LLM, SamplingParams
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

MODEL = os.environ.get("DSV4_MODEL", "/models/DeepSeek-V4-Flash-0731")
CONC = 8
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 64

def main():
    tok = DeepseekV4Tokenizer.from_pretrained(MODEL, trust_remote_code=True)

    unit = "这是一个用于离线并发吞吐测试的中文文本片段，内容本身不重要，只用于构造固定长度输入。"
    ids = []
    while len(ids) < INPUT_TOKENS:
        ids.extend(tok.encode(unit, add_special_tokens=False))
    ids = ids[:INPUT_TOKENS]
    prompt = tok.decode(ids)

    prompts = [prompt for _ in range(CONC)]

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        max_model_len=2048,
        max_num_seqs=CONC,
        max_num_batched_tokens=CONC * INPUT_TOKENS,
        gpu_memory_utilization=0.90,
        kv_cache_dtype="fp8",
        enforce_eager=True,
    )

    sampling = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        temperature=0.0,
        ignore_eos=True,
    )

    print(f"START conc={CONC} input_tokens={INPUT_TOKENS} output_tokens={OUTPUT_TOKENS}", flush=True)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tokens = CONC * INPUT_TOKENS

    print("DONE")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"input_tokens={in_tokens}")
    print(f"output_tokens={out_tokens}")
    print(f"prefill_plus_decode_tok_s={(in_tokens + out_tokens) / elapsed:.2f}")
    print(f"decode_tok_s={out_tokens / elapsed:.2f}")
    print(f"per_request_decode_tok_s={(out_tokens / CONC) / elapsed:.2f}")

    for i, o in enumerate(outs):
        print(f"REQ {i} out_tokens={len(o.outputs[0].token_ids)} text={o.outputs[0].text[:80]!r}")

if __name__ == "__main__":
    main()
