#!/usr/bin/env python3
"""Extreme-context test: does it LOAD, how fast, and is the attention still real?

"It didn't crash" is not a context test. Each run buries a distinctive
passphrase at ~10% depth in a haystack of random words and then asks for it
back, so a pass means attention genuinely reached across the whole window --
which is the thing that actually breaks at long context on a sparse-attention
model like DSv4 (the indexer has to select the right blocks).

Reports time-to-first-token so the ingest cost at each size is visible.
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"
MODEL = "dsv4s"
NEEDLE_VALUE = "quantum-lighthouse-4471"

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def build(approx_tokens, seed, depth=0.10):
    rng = random.Random(seed)
    n_words = int(approx_tokens / 1.3)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    at = int(n_words * depth)
    needle = f". IMPORTANT FACT: the secret passphrase is {NEEDLE_VALUE} . "
    body = " ".join(words[:at]) + needle + " ".join(words[at:])
    return (
        "Read the following notes carefully.\n\n" + body +
        "\n\nQuestion: What is the secret passphrase mentioned in the notes?\n"
        "Answer: The secret passphrase is"
    )


def run(approx_tokens, seed):
    prompt = build(approx_tokens, seed)
    req = urllib.request.Request(
        URL,
        json.dumps(
            {"model": MODEL, "prompt": prompt, "max_tokens": 24,
             "temperature": 0, "stream": True,
             # Real prompt_tokens, not my ~1.3 words/token estimate -- the
             # effective t/s is meaningless without the true count.
             "stream_options": {"include_usage": True}}
        ).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    out = []
    n_prompt = 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                n_prompt = obj["usage"].get("prompt_tokens", n_prompt)
            if obj.get("choices") and obj["choices"][0].get("text"):
                if t_first is None:
                    t_first = time.perf_counter()
                out.append(obj["choices"][0]["text"])
    text = "".join(out)
    ttft = (t_first - t0) if t_first else float("nan")
    return ttft, text, n_prompt


def main():
    targets = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [
        130000, 250000, 500000
    ]
    print(f"{'real ctx':>10} {'TTFT s':>9} {'prefill t/s':>12} {'needle':>7}  output")
    for i, t in enumerate(targets):
        try:
            ttft, text, n_prompt = run(t, 606 + i)
            found = "PASS" if NEEDLE_VALUE in text else "FAIL"
            rate = n_prompt / ttft if n_prompt and ttft == ttft else float("nan")
            print(f"{n_prompt:>10} {ttft:>9.1f} {rate:>12.0f} {found:>7}  "
                  f"{text.strip()[:55]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"{t:>11}  FAILED: {str(e)[:130]}")


if __name__ == "__main__":
    main()
