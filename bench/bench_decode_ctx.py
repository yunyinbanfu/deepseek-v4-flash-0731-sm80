#!/usr/bin/env python3
"""Decode throughput as a function of CONTEXT LENGTH.

Every decode number in this project so far came from a ~60-token prompt, so we
have never measured what a long conversation actually feels like. On the OLD
stack decode fell 21.7 -> 18.7 -> 13.1 t/s at 0 / 1.2k / 4.2k tokens, which was
the evidence that the sparse-attention/indexer path degrades with context. This
re-tests that on the current stack, at real context lengths.

Method: for each context length, time the same prompt twice -- once with
max_tokens=1 and once with max_tokens=N+1 -- and take the difference. That
subtracts prefill (and everything else fixed) and leaves N pure decode steps.
Prefix caching must be OFF or the second call skips prefill and the subtraction
is wrong.
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


def post(prompt, max_tokens, timeout=3600):
    req = urllib.request.Request(
        URL,
        json.dumps(
            {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
        ).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.perf_counter() - t0, resp


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    targets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [
        2000, 8000, 32000, 65000, 100000
    ]
    print(f"########## {label} :: DECODE vs CONTEXT ##########")
    # Warm-up discard: the first request after boot carries Triton JIT and reads
    # roughly 4x low. Never let it into a recorded sample.
    post(make_prompt(2000, 1), 8)
    print(f"{'ctx tok':>9} {'prefill s':>10} {'decode s':>9} {'gen':>5} {'decode t/s':>11}")
    for i, t in enumerate(targets):
        p = make_prompt(t, 777 + i)
        try:
            t1, r1 = post(p, 1)
            t2, r2 = post(p, N_DECODE + 1)
            n_ctx = r1["usage"]["prompt_tokens"]
            gen = r2["usage"]["completion_tokens"] - r1["usage"]["completion_tokens"]
            dt = t2 - t1
            rate = gen / dt if dt > 0 and gen > 0 else float("nan")
            print(f"{n_ctx:>9} {t1:>10.2f} {dt:>9.2f} {gen:>5} {rate:>11.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{t:>9}  FAILED: {str(e)[:110]}")


if __name__ == "__main__":
    main()
