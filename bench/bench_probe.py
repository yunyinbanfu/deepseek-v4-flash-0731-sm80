#!/usr/bin/env python3
"""Dump deterministic (temp 0) completions so two configs can be diffed.

Speculative decoding is supposed to be output-identical to non-speculative
greedy decoding. Any divergence is either a real verifier bug or floating-point
reassociation from the different batch shapes the target sees; either way we
want to see WHICH prompts diverge and where, not a summary statistic.

Usage: bench_probe.py <outfile>
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8098/v1/completions"
MODEL = "dsv4s"

PROMPTS = [
    "Question: I have 17 sheep and all but 9 run away. How many sheep do I have left?\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one paragraph why the sky appears blue.",
    "List the first ten prime numbers:",
    "Translate to French: 'The weather is beautiful today.'\nFrench:",
]


def main():
    out_path = sys.argv[1]
    lines = []
    for i, p in enumerate(PROMPTS):
        req = urllib.request.Request(
            URL,
            json.dumps(
                {"model": MODEL, "prompt": p, "max_tokens": 96, "temperature": 0}
            ).encode(),
            {"Content-Type": "application/json"},
        )
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        lines.append(f"### PROMPT {i}\n{r['choices'][0]['text']!r}\n")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {len(PROMPTS)} completions to {out_path}")


if __name__ == "__main__":
    main()
