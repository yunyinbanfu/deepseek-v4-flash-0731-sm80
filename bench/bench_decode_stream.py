#!/usr/bin/env python3
"""Decode throughput vs context, measured by STREAMING.

The subtract-two-calls method (bench_decode_ctx.py) assumes the prefill cost is
identical across the two calls. At long context prefill dominates and varies by
seconds run to run, so the subtraction produces garbage -- it gave 135 t/s at
50k, out of line with its own neighbours.

Streaming avoids the assumption entirely: timestamp the first token and the
last, and divide by the tokens in between. Time-to-first-token comes out of the
same run for free, which is the number a user actually feels on a long prompt.
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"
MODEL = "dsv4s"
N_DECODE = 192

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def make_prompt(approx_tokens, seed):
    rng = random.Random(seed)
    return f"[run {seed}] " + " ".join(
        rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3))
    )


def stream_once(prompt, max_tokens, timeout=3600):
    req = urllib.request.Request(
        URL,
        json.dumps(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                # Force exactly max_tokens. Without this, DSpark's output
                # diverges and hits EOS early (~50 tokens vs the 192 requested),
                # so a spec-vs-plain comparison silently compares different
                # generation lengths -- and short generations are dominated by
                # ramp-up. This was the confound that made an earlier run read
                # TP4+DSpark at 24 t/s when it was really 79.5.
                "ignore_eos": True,
                # ★ Required for a correct rate under speculative decoding: each
                # SSE chunk can carry SEVERAL tokens (roughly the acceptance
                # length), so counting chunks under-reports by that factor. Use
                # the server's own completion_tokens instead.
                "stream_options": {"include_usage": True},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                n_tokens = obj["usage"].get("completion_tokens", n_tokens)
            if not obj.get("choices"):
                continue
            if obj["choices"][0].get("text"):
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first else float("nan")
    dec_s = (t_end - t_first) if t_first else float("nan")
    # Fall back to chunk count only if the server sent no usage block (then the
    # two are equal anyway, because that means one token per chunk).
    n = n_tokens or n_chunks
    # The first token came out of prefill, so it is not a decode step.
    rate = (n - 1) / dec_s if t_first and dec_s > 0 and n > 1 else float("nan")
    return ttft, dec_s, n, rate


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    targets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [
        2000, 8000, 32000, 65000, 100000
    ]
    print(f"########## {label} :: DECODE vs CONTEXT (streaming) ##########")
    stream_once(make_prompt(2000, 1), 8)  # warm-up discard
    print(f"{'ctx tok':>9} {'TTFT s':>8} {'decode s':>9} {'gen':>5} {'decode t/s':>11}")
    for i, t in enumerate(targets):
        try:
            ttft, dec_s, n, rate = stream_once(make_prompt(t, 4242 + i), N_DECODE)
            approx_ctx = int(t / 1.3 * 1.3)
            print(f"{approx_ctx:>9} {ttft:>8.2f} {dec_s:>9.2f} {n:>5} {rate:>11.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{t:>9}  FAILED: {str(e)[:110]}")


if __name__ == "__main__":
    main()
