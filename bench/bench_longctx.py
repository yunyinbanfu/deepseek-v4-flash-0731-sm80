#!/usr/bin/env python3
"""Long-context prefill sweep — does DSv4 prefill rise with context like GLM-5.2 did?

GLM-5.2 on PP8 measured 665 / 1,497 / 2,342 / 2,675 t/s at 4k / 32k / 65k / 131k:
prefill throughput climbs steeply with sequence length because longer sequences
give fatter GEMMs and amortise the fixed per-layer overheads.

Every DSv4 prefill number we have was taken at <= 5.4k tokens (max-model-len was
8192), i.e. the flat bottom of that curve. This walks the same ladder.

Watch for the opposite outcome too: DSv4 carries a DSA sparse indexer whose cost
grows with context, so it is entirely possible prefill FALLS instead of rising.
That would be an architectural difference from GLM, not a bug.
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


def make_prompt(approx_tokens, seed):
    rng = random.Random(seed)
    return f"[run {seed}] " + " ".join(
        rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3))
    )


def post(payload, timeout=3600):
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.perf_counter() - t0, resp


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    targets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [
        4000, 8000, 16000, 32000, 65000, 100000
    ]
    print(f"########## {label} :: single-request prefill vs context ##########")
    print(f"{'target':>8} {'real tok':>9} {'wall s':>9} {'prefill t/s':>12}")
    for i, t in enumerate(targets):
        try:
            dt, resp = post(
                {
                    "model": MODEL,
                    "prompt": make_prompt(t, 31337 + i),
                    "max_tokens": 1,
                    "temperature": 0,
                }
            )
            n = resp["usage"]["prompt_tokens"]
            print(f"{t:>8} {n:>9} {dt:>9.2f} {n / dt:>12.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{t:>8}  FAILED: {str(e)[:120]}")


if __name__ == "__main__":
    main()
