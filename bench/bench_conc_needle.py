#!/usr/bin/env python3
"""Concurrent long-context needle test.

The row-chunked indexer path slices rows out of a prefill chunk that may hold
tokens from SEVERAL requests at once. Single-request tests never exercise that.
This fires N distinct long needle prompts simultaneously -- each with its own
passphrase -- and checks every one comes back with ITS OWN needle, which would
fail if row offsets bled across requests.
"""
import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def build(approx_tokens, seed, needle, depth=0.10):
    rng = random.Random(seed)
    n_words = int(approx_tokens / 1.3)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    at = int(n_words * depth)
    ins = f". IMPORTANT FACT: the secret passphrase is {needle} . "
    body = " ".join(words[:at]) + ins + " ".join(words[at:])
    return (
        "Read the following notes carefully.\n\n" + body +
        "\n\nQuestion: What is the secret passphrase mentioned in the notes?\n"
        "Answer: The secret passphrase is"
    )


def one(idx, port, model, approx, timeout):
    needle = f"orbital-falcon-{idx}{idx}{idx}{idx}"
    prompt = build(approx, 100 + idx, needle)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps({"model": model, "prompt": prompt, "max_tokens": 24,
                    "temperature": 0}).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        text = d["choices"][0]["text"]
        n_prompt = d.get("usage", {}).get("prompt_tokens", 0)
        others = [f"orbital-falcon-{j}{j}{j}{j}" for j in range(9) if j != idx]
        bled = [o for o in others if o in text]
        return {"idx": idx, "prompt_tokens": n_prompt,
                "elapsed": round(time.perf_counter() - t0, 1),
                "verdict": "PASS" if (needle in text and not bled)
                           else ("CROSS-TALK" if bled else "MISS"),
                "text": text.strip()[:60]}
    except Exception as e:
        return {"idx": idx, "prompt_tokens": 0,
                "elapsed": round(time.perf_counter() - t0, 1),
                "verdict": "ERROR", "text": f"{type(e).__name__}: {str(e)[:70]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    print(f"firing {a.n} concurrent prompts of ~{a.tokens} target tokens each")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=a.n) as ex:
        res = list(ex.map(
            lambda i: one(i, a.port, a.model, a.tokens, a.timeout), range(a.n)))
    for r in sorted(res, key=lambda x: x["idx"]):
        print(f"  req{r['idx']}  {r['prompt_tokens']:>8} tok  "
              f"{r['elapsed']:>6}s  {r['verdict']:<10} {r['text']}")
    print(f"wall {time.perf_counter() - t0:.1f}s  "
          f"-> {sum(1 for r in res if r['verdict'] == 'PASS')}/{a.n} PASS")


if __name__ == "__main__":
    main()
