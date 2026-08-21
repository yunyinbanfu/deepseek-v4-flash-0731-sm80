#!/usr/bin/env python3
"""End-to-end decode benchmark over three content types.

End-to-end wall clock for a single request is what a user actually feels, and
speculative decoding's benefit is strongly content-dependent, so one prompt is
not a measurement. Technical exposition is highly predictable (draft hits),
open-ended prose is not (draft misses), code sits in between.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"
MODEL = "dsv4s"

PROMPTS = {
    "technical": (
        "Explain in careful detail how a modern operating system kernel implements "
        "virtual memory: page tables, the TLB, page faults, demand paging, "
        "copy-on-write, and swapping."
    ),
    "open-prose": (
        "Write an original short story about a lighthouse keeper who discovers "
        "something unexpected washed up on the shore one winter morning."
    ),
    "code": (
        "Write a complete Python implementation of a red-black tree with insert, "
        "delete, and search, including the rebalancing cases, with comments."
    ),
}


def run(tag, max_tokens=400):
    print(f"===== {tag} =====")
    print(f"{'prompt':>12} {'tokens':>7} {'elapsed s':>10} {'tok/s':>8}")
    total_tok = total_s = 0
    for name, p in PROMPTS.items():
        req = urllib.request.Request(
            URL,
            json.dumps(
                {"model": MODEL, "prompt": p, "max_tokens": max_tokens, "temperature": 0}
            ).encode(),
            {"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        resp = json.load(urllib.request.urlopen(req, timeout=1800))
        dt = time.perf_counter() - t0
        ctok = resp["usage"]["completion_tokens"]
        total_tok += ctok
        total_s += dt
        print(f"{name:>12} {ctok:>7} {dt:>10.2f} {ctok / dt:>8.1f}")
    print(f"{'AGGREGATE':>12} {total_tok:>7} {total_s:>10.2f} {total_tok / total_s:>8.1f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "run")
