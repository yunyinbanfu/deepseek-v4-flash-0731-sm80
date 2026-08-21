#!/usr/bin/env python3
"""Prefill + decode benchmark against a vLLM OpenAI endpoint.

Prefill is measured as: send a long prompt with max_tokens=1 and time the call.
That is prefill + one decode step; at these prompt lengths the single decode
step is noise. prompt_tokens comes from the server's own usage block, so the
rate is against the real tokenizer count, not a guess.

Every prompt is unique (a random preamble + a distinct body) so a stray prefix
cache cannot serve it. Run the server with --no-enable-prefix-caching anyway.
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"
MODEL = "dsv4s"

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def make_prompt(approx_tokens: int, seed: int) -> str:
    """~1.3 tokens per word for this kind of text; overshoot then let usage tell us."""
    rng = random.Random(seed)
    n_words = int(approx_tokens / 1.3)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"[run {seed}] Notes: {body}\nSummarize the notes above."


def post(payload: dict, timeout: int = 1800):
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.perf_counter() - t0, resp


def bench_prefill(targets, reps, seed_base):
    print(f"{'prompt tok':>11} {'rep':>4} {'elapsed s':>10} {'prefill tok/s':>14}")
    results = {}
    for t in targets:
        rates = []
        for r in range(reps):
            seed = seed_base + t * 100 + r
            dt, resp = post(
                {
                    "model": MODEL,
                    "prompt": make_prompt(t, seed),
                    "max_tokens": 1,
                    "temperature": 0,
                }
            )
            ptok = resp["usage"]["prompt_tokens"]
            rate = ptok / dt
            rates.append((ptok, rate))
            print(f"{ptok:>11} {r:>4} {dt:>10.3f} {rate:>14.1f}")
        best = max(rates, key=lambda x: x[1])
        results[t] = best
    print("\nBEST PER LENGTH (prefill):")
    for t, (ptok, rate) in results.items():
        print(f"  ~{t:>5} target -> {ptok:>6} real tokens : {rate:>8.1f} tok/s")
    return results


def bench_decode(seed_base, max_tokens=400):
    dt, resp = post(
        {
            "model": MODEL,
            "prompt": f"[run {seed_base}] Explain in careful detail how a modern "
            "operating system kernel implements virtual memory: page tables, the "
            "TLB, page faults, demand paging, copy-on-write, and swapping.",
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    )
    ctok = resp["usage"]["completion_tokens"]
    print(f"\nDECODE end-to-end: {ctok} tokens in {dt:.2f}s = {ctok / dt:.1f} tok/s")
    print("  (end-to-end incl. prefill of a short prompt; compare to the")
    print("   engine's own steady-state 'Avg generation throughput' log line)")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    seed_base = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"===== {tag} =====")
    bench_prefill([1000, 2000, 4000, 7000], reps=2, seed_base=seed_base)
    bench_decode(seed_base)
