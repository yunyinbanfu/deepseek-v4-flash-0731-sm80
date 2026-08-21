#!/usr/bin/env python3
"""Context-ceiling probe: walk a ladder of prompt sizes and report the exact
real-token boundary between PASS and worker-death.

Unlike bench_needle.py this is parameterised (port/model/sizes) and it
distinguishes the three outcomes that matter:
  PASS   - needle recovered
  WRONG  - answered, needle missing (attention degraded, not a crash)
  DEAD   - request failed AND the server stopped answering /v1/models
           (i.e. the worker was killed -- the Xid 31 signature)

Usage: bench_ceiling.py --port 8098 --model deepseek-v4-flash 150000 158000 ...
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

NEEDLE = "quantum-lighthouse-4471"
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
    needle = f". IMPORTANT FACT: the secret passphrase is {NEEDLE} . "
    body = " ".join(words[:at]) + needle + " ".join(words[at:])
    return (
        "Read the following notes carefully.\n\n" + body +
        "\n\nQuestion: What is the secret passphrase mentioned in the notes?\n"
        "Answer: The secret passphrase is"
    )


def alive(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10)
        return True
    except Exception:
        return False


def probe(port, model, approx, seed, timeout):
    prompt = build(approx, seed)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps({
            "model": model, "prompt": prompt, "max_tokens": 24,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    out, n_prompt = [], 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    n_prompt = d["usage"].get("prompt_tokens", n_prompt)
                for ch in d.get("choices", []):
                    txt = ch.get("text", "")
                    if txt:
                        if t_first is None:
                            t_first = time.perf_counter()
                        out.append(txt)
    except Exception as e:
        return {"target": approx, "prompt_tokens": n_prompt, "ttft": None,
                "verdict": "DEAD" if not alive(port) else "ERROR",
                "detail": f"{type(e).__name__}: {str(e)[:160]}"}

    text = "".join(out)
    ttft = (t_first - t0) if t_first else None
    verdict = "PASS" if NEEDLE in text else "WRONG"
    return {"target": approx, "prompt_tokens": n_prompt, "ttft": ttft,
            "verdict": verdict, "detail": text.strip()[:80],
            "prefill_tps": (n_prompt / ttft) if (ttft and n_prompt) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--stop-on-dead", action="store_true")
    ap.add_argument("sizes", nargs="+", type=int)
    a = ap.parse_args()

    print(f"{'target':>9} {'real tok':>9} {'TTFT s':>8} {'prefill t/s':>12}  verdict")
    results = []
    for s in a.sizes:
        r = probe(a.port, a.model, s, a.seed, a.timeout)
        results.append(r)
        ttft_s = ("%.1f" % r["ttft"]) if r["ttft"] else "-"
        tps_s = ("%.0f" % r["prefill_tps"]) if r.get("prefill_tps") else "-"
        print("%9d %9d %8s %12s  %s  %s" % (
            r["target"], r["prompt_tokens"], ttft_s, tps_s,
            r["verdict"], r["detail"][:60]), flush=True)
        if r["verdict"] == "DEAD" and a.stop_on_dead:
            print("server is down -- stopping ladder", flush=True)
            break
    print("\nJSON:", json.dumps(results))


if __name__ == "__main__":
    main()
