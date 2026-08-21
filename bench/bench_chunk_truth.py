"""Which decode number is real: 24.5 t/s or 88 t/s?

Two harnesses disagree 3.6x at the same depth on the same server at the same
moment. One counts SSE CHUNKS, the other trusts usage.completion_tokens. A chunk
can carry several tokens, so chunk-counting understates -- but completion_tokens
could in principle be inflated by speculative drafts under DSpark.

Ground truth neither can fudge: TOKENIZE THE GENERATED TEXT and compare. If
tokenize(text) == completion_tokens, the server is honest and the chunk-counting
harness is undercounting. Measured at several depths, because the chunk/token
ratio need not be constant -- and if it is not, the published decode-vs-context
CURVE is distorted in shape, not just offset.
"""
import argparse, json, random, time, urllib.request, zlib

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()


def count_tokens(port, model, text):
    b = json.dumps({"model": model, "prompt": text}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=b,
                               headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=300).read())
    return d.get("count") or len(d.get("tokens", []))


def probe(port, model, ctx_target, max_tokens, salt, timeout=1800):
    rng = random.Random(zlib.crc32(f"{salt}|{ctx_target}".encode()))
    # 1.65 tok/word, matching the original harness so depths are directly
    # comparable. (A repeating 4-word pattern measures 1.5 -- random selection
    # from the 26-word list does not. Using 1.5 overflows max_model_len.)
    body_words = [rng.choice(WORDS) for _ in range(max(1, int(ctx_target / 1.65)))]
    prompt = "Notes:\n" + " ".join(body_words) + "\n\nSummarise.\nAnswer:"

    req_body = json.dumps({"model": model, "prompt": prompt,
                           "max_tokens": max_tokens, "temperature": 0,
                           "ignore_eos": True, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=req_body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0          # SSE events carrying text  (what the OLD harness counted)
    pieces = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                usage = d["usage"]
            for ch in (d.get("choices") or []):
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    pieces.append(ch["text"])
    total = time.time() - t0
    text = "".join(pieces)
    gen_s = total - ttft if ttft else None
    ctok = usage.get("completion_tokens")
    real = count_tokens(port, model, text)          # <-- ground truth
    return {
        "ctx": usage.get("prompt_tokens"), "ttft": round(ttft, 2) if ttft else None,
        "gen_seconds": round(gen_s, 2) if gen_s else None,
        "chunks": chunks, "completion_tokens": ctok, "tokenized_text": real,
        "chars": len(text),
        "tokens_per_chunk": round(real / chunks, 2) if chunks else None,
        "decode_BY_CHUNKS": round((chunks - 1) / gen_s, 1) if gen_s else None,
        "decode_BY_USAGE": round((ctok - 1) / gen_s, 1) if (gen_s and ctok) else None,
        "decode_BY_TOKENIZED": round((real - 1) / gen_s, 1) if gen_s else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--model", default="dsv4s")
    ap.add_argument("--depths", default="2000,30000,120000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--salt", default="t1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for d in [int(x) for x in a.depths.split(",")]:
        r = probe(a.port, a.model, d, a.max_tokens, a.salt)
        rows.append(r)
        print(f"ctx={r['ctx']:>7}  chunks={r['chunks']:>4}  usage_tok={r['completion_tokens']:>4}  "
              f"tokenized={r['tokenized_text']:>4}  tok/chunk={r['tokens_per_chunk']:>5}  "
              f"| decode  by_chunks={r['decode_BY_CHUNKS']:>6}  "
              f"by_usage={r['decode_BY_USAGE']:>6}  by_tokenized={r['decode_BY_TOKENIZED']:>6}",
              flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
