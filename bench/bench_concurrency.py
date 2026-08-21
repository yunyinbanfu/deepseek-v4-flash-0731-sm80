#!/usr/bin/env python3
"""Concurrency sweep: aggregate decode AND aggregate prefill.

Single-stream numbers understate a serving box. Both phases behave differently
under load:
  - decode batches many 1-token steps into one pass -> aggregate should scale
    well until memory bandwidth or the KV pool runs out;
  - prefill is already compute-dense, so concurrency mostly fattens the GEMMs
    and fills SMs. If single-request prefill is running at a few percent of peak
    FLOPs, concurrent prefill is where the headroom shows up (or doesn't).

Each request gets a unique prompt so nothing can be served from a prefix cache.
"""
import json
import random
import sys
import threading
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


def make_prompt(approx_tokens, seed):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3)))
    return f"[run {seed}] Notes: {body}\nSummarize the notes above."


def post(payload, timeout=3600):
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def sweep(concurrencies, prompt_tokens, max_tokens, label, seed_base):
    print(f"\n===== {label} =====")
    print(f"{'conc':>5} {'wall s':>9} {'prompt tok':>11} {'gen tok':>9} "
          f"{'prefill t/s':>12} {'decode t/s':>11}")
    for c in concurrencies:
        results = [None] * c
        prompts = [make_prompt(prompt_tokens, seed_base + c * 1000 + i) for i in range(c)]

        def work(i):
            try:
                results[i] = post(
                    {
                        "model": MODEL,
                        "prompt": prompts[i],
                        "max_tokens": max_tokens,
                        "temperature": 0,
                    }
                )
            except Exception as e:  # noqa: BLE001
                results[i] = {"error": str(e)}

        threads = [threading.Thread(target=work, args=(i,)) for i in range(c)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0

        ok = [r for r in results if r and "usage" in r]
        if not ok:
            print(f"{c:>5}  ALL FAILED: {results[0]}")
            continue
        ptok = sum(r["usage"]["prompt_tokens"] for r in ok)
        gtok = sum(r["usage"]["completion_tokens"] for r in ok)
        # With max_tokens=1 the wall clock is essentially all prefill; with a
        # long generation it is essentially all decode. Report both rates but
        # only one is meaningful per mode.
        print(f"{c:>5} {wall:>9.2f} {ptok:>11} {gtok:>9} "
              f"{ptok / wall:>12.1f} {gtok / wall:>11.1f}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    print(f"########## {label} ##########")
    # Prefill: long prompts, 1 token out.
    sweep([1, 2, 4, 8], prompt_tokens=3000, max_tokens=1,
          label=f"{label} :: PREFILL (3k-token prompts, max_tokens=1)", seed_base=10)
    # Decode: short prompt, long generation.
    sweep([1, 4, 8, 16], prompt_tokens=60, max_tokens=300,
          label=f"{label} :: DECODE (300 tokens out)", seed_base=90)
