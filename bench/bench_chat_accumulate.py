"""Accumulating-conversation probe: does a REAL chat rot as context grows?

Every prior long-context result on this stack is a SINGLE one-shot prefill of a
giant prompt, scored by a SINGLE needle. That tests the prefill path (where our
row-chunk patch lives) and nothing else. This harness tests the path a human
actually drives:

  turn -> small new message -> prefix cache reuses all prior KV -> decode at depth

which exercises (a) DECODE at depth, (b) prefix-cache reuse across turns, and
(c) generation coherence over many samples instead of one needle.

THREE OUTPUTS
  1. CORRECTNESS   canary facts planted at known depths, recalled on a schedule
  2. COHERENCE     mechanical degeneration metrics per turn (no judge model):
                   repeated-n-gram counts, distinct-n, top-token share, and the
                   mean token logprob (a repetition collapse shows up as the
                   model becoming near-certain BEFORE the text visibly breaks)
  3. SPEED SHAPE   per-turn TTFT / decode / cached-vs-new prompt tokens, to see
                   whether chat decays like the one-shot curve or stays fast

Results stream to JSONL as they happen -- a crash at turn 180 must not cost the
first 179 turns.
"""
import argparse, json, os, random, re, time, urllib.request
from collections import Counter

# ---------------------------------------------------------------- canaries
# Distinctive, verifiable, and shaped like something that could plausibly sit in
# a technical document -- so it is not trivially salient to the model.
CANARY_TMPL = [
    ("ZULU-{n}",     "the internal tracking code for the {topic} unit is {v}"),
    ("{n} volts",    "the bench supply for the {topic} unit is set to {v}"),
    ("build {n}",    "the {topic} unit regression was first seen in {v}"),
    ("{n} ms",       "the {topic} unit was measured at {v} end to end"),
    ("rack {n}",     "the {topic} unit is installed in {v}"),
]
# ---- TOPICS MUST NOT COLLIDE WITH THE CORPUS VOCABULARY ----
# The smoke test used words like "indexer"/"allocator". The corpus is our own
# engineering docs, which are SATURATED with those terms, so the model answered
# from the document instead of from the planted note -- scoring a MISS that was
# really my question being ambiguous. Invented proper nouns cannot appear in any
# source file, so a miss is now unambiguously a retrieval failure.
TOPICS = ["Kestrel", "Marlowe", "Vandergriff", "Ashcombe", "Pellworth",
          "Thornbury", "Quillfeather", "Braxton", "Ravensworth", "Lindquist",
          "Ottoline", "Deverell", "Fenwick", "Harrowgate", "Silverbourne"]


def make_canary(rng, idx):
    tmpl, sent = CANARY_TMPL[idx % len(CANARY_TMPL)]
    n = rng.randint(1000, 9999)
    value = tmpl.format(n=n)
    topic = TOPICS[idx % len(TOPICS)]
    return {
        "id": idx,
        "value": value,
        "topic": topic,
        "sentence": sent.format(v=value, topic=topic),
        # Points explicitly at the parenthetical note, so this measures whether
        # the model can still REACH that depth -- not whether it can guess which
        # of several plausible referents I meant.
        "question": (f"Earlier in this conversation I asked you to remember a note "
                     f"about the {topic} unit. What was the specific value in that "
                     f"note? Answer with the value only."),
    }


# ------------------------------------------------------------ degeneration
def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def degeneration(text):
    """Mechanical repetition metrics. Catches the NOW.NOW.NOW collapse mode."""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    out = {"chars": len(text), "tokens_approx": len(toks)}
    if len(toks) < 8:
        out.update({"distinct4": None, "max_rep_8gram": 0, "top_token_share": None,
                    "degenerate": False})
        return out
    c4 = Counter(ngrams(toks, 4))
    c8 = Counter(ngrams(toks, 8))
    top_tok = Counter(toks).most_common(1)[0][1]
    out["distinct4"] = round(len(c4) / max(1, len(toks) - 3), 4)
    out["max_rep_8gram"] = c8.most_common(1)[0][1] if c8 else 0
    out["top_token_share"] = round(top_tok / len(toks), 4)
    # Any 8-token span repeated 5+ times, or one token owning a third of the
    # output, is degenerate by inspection -- this is the screenshot failure.
    out["degenerate"] = bool(out["max_rep_8gram"] >= 5 or out["top_token_share"] >= 0.33)
    return out


# ------------------------------------------------------------------ client
def chat_stream(port, model, messages, max_tokens, timeout, want_logprobs=True,
                think_effort=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    # Per-request thinking, so BOTH arms of an A/B can run against ONE container
    # with nothing else differing. Top-level `reasoning_effort` is the ambiguous
    # path (it changes behaviour but /tokenize cannot see it); chat_template_kwargs
    # is the documented one, and `thinking` alone is not enough -- the effort key
    # must be present too.
    if think_effort:
        body["chat_template_kwargs"] = {"thinking": True,
                                       "reasoning_effort": think_effort}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})

    t0 = time.time()
    ttft = None
    pieces, think, lps = [], [], []
    usage = {}
    ntok = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in (d.get("choices") or []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                # DeepSeek-V4 is a REASONING model: <think> tokens stream as
                # reasoning_content, NOT delta.content. Counting only content
                # undercounted generation ~2.7x and made decode read 30 t/s
                # against a real ~88. It also means a model looping INSIDE the
                # think block scores clean unless the reasoning text is fed to
                # the degeneration metrics -- so capture both.
                # ★ On DeepSeek-V4-0731 the streamed field is `reasoning`, NOT
                # `reasoning_content` (per thomaslwang, vllm#50576). Accept both:
                # reading only the latter silently zeroes every think block and
                # makes a thinking run look identical to a non-thinking one.
                rc = delta.get("reasoning") or delta.get("reasoning_content")
                if rc:
                    if ttft is None:
                        ttft = time.time() - t0
                    think.append(rc)
                    ntok += 1
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    pieces.append(piece)
                    ntok += 1
                lp = ch.get("logprobs")
                if lp and lp.get("content"):
                    for e in lp["content"]:
                        if e.get("logprob") is not None:
                            lps.append(e["logprob"])
    total = time.time() - t0
    text = "".join(pieces)
    reasoning = "".join(think)
    # Prefer the SERVER's completion_tokens -- it is the ground truth for how many
    # tokens were actually generated, including any the stream does not surface.
    ctok = usage.get("completion_tokens") or ntok
    decode_tps = ((ctok - 1) / (total - ttft)) if (ttft and ctok > 1 and total > ttft) else None
    row = {
        "text": text, "reasoning": reasoning,
        "ttft": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "gen_tokens_streamed": ntok, "reasoning_chars": len(reasoning),
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    det = usage.get("prompt_tokens_details") or {}
    row["cached_tokens"] = det.get("cached_tokens")
    if lps:
        row["mean_logprob"] = round(sum(lps) / len(lps), 4)
        # Fraction of tokens the model was essentially certain about. A collapse
        # into repetition drives this toward 1.0 before the text looks broken.
        row["frac_near_certain"] = round(sum(1 for x in lps if x > -0.05) / len(lps), 4)
    return row


# -------------------------------------------------------------------- corpus
def load_corpus(path):
    """Corpus dir -> list of (content_type, text). Type comes from extension."""
    kinds = {".md": "techdoc", ".txt": "prose", ".py": "code",
             ".js": "code", ".ts": "code", ".sh": "code", ".json": "structured"}
    docs = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in kinds:
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    t = f.read()
            except Exception:
                continue
            if len(t.strip()) > 500:
                docs.append((kinds[ext], fn, t))
    return docs


def chunk_docs(docs, target_chars, rng):
    """Interleave content types so depth is not confounded with content type."""
    by_kind = {}
    for kind, fn, text in docs:
        by_kind.setdefault(kind, []).append((fn, text))
    streams = []
    for kind, items in by_kind.items():
        buf = []
        for fn, text in items:
            for i in range(0, len(text), target_chars):
                piece = text[i:i + target_chars]
                if len(piece) > 200:
                    buf.append((kind, fn, piece))
        rng.shuffle(buf)
        streams.append(buf)
    out, i = [], 0
    while any(streams):
        s = streams[i % len(streams)]
        if s:
            out.append(s.pop())
        i += 1
        streams = [x for x in streams if x]
        if not streams:
            break
    return out


# Answers are deliberately asked as a PARAGRAPH, not "two sentences". The smoke
# test produced 19-46 token replies, over which (n-1)/(total-ttft) measures
# startup noise rather than decode -- it read 29 t/s against a known 88.8. A
# ~150-250 token answer is still realistic chat and gives a real decode window.
PROMPTS = {
    "techdoc": "Here is a section of an engineering document. Summarise its main "
               "point in a full paragraph, and note anything that looks like a "
               "risk or an open question.\n\n---\n{body}\n---",
    "code":    "Here is a source file excerpt. In a full paragraph, explain what it "
               "does and call out anything that looks fragile.\n\n---\n{body}\n---",
    "prose":   "Here is a passage. Summarise it in a full paragraph and note what "
               "it implies.\n\n---\n{body}\n---",
    "structured": "Here is a structured data excerpt. In a full paragraph, describe "
                  "what it records and what stands out.\n\n---\n{body}\n---",
}


def scrape_cache(port, timeout=15):
    """vLLM prefix-cache counters. usage.prompt_tokens_details.cached_tokens comes
    back null on this build, so the per-turn cached/new split is derived from the
    deltas of these counters instead -- that is what separates 'chat stays fast
    because the cache works' from 'chat stays fast for some other reason'."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
            body = r.read().decode(errors="ignore")
    except Exception:
        return {}
    out = {}
    for key, name in (("queries", "vllm:prefix_cache_queries_total"),
                      ("hits", "vllm:prefix_cache_hits_total")):
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*}}\s+([0-9.eE+-]+)$", body, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--model", default="dsv4s")
    ap.add_argument("--label", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--budget-tokens", type=int, default=118000,
                    help="stop accumulating when prompt_tokens exceeds this")
    ap.add_argument("--chunk-chars", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--canary-every", type=int, default=4, help="plant every N turns")
    ap.add_argument("--recall-every", type=int, default=8, help="recall battery every N turns")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip warmup so turn 1 measures the COLD server")
    ap.add_argument("--final-probe-efforts", default=None,
                    help="comma list of reasoning_effort values. After reaching the "
                         "budget, re-probe EVERY canary once per value against the "
                         "SAME conversation -- a paired comparison where context, "
                         "canaries and history are identical and only effort differs.")
    ap.add_argument("--think", default=None,
                    help="reasoning_effort (none|low|high|max); omit for thinking OFF")
    ap.add_argument("--grounding", action="store_true",
                    help="prepend the abstention/grounding system prompt")
    a = ap.parse_args()

    # Tests the DISPOSITION the AA-Omniscience benchmark scores: when the model
    # cannot retrieve a fact, does it abstain or substitute a plausible one?
    # Measured 8/05 at short context: grounding fixed fabrication 10/10 and was
    # FASTER (abstentions are short). This checks it under retrieval failure.
    GROUNDING = (
        "Answer only from information actually present in this conversation. "
        "If you cannot find the specific detail you were asked for, say so "
        "explicitly -- reply that you have no record of it. Do NOT substitute a "
        "similar-looking value from a different item, and do NOT guess. An "
        "honest 'I cannot find that' is always better than a plausible answer."
    )

    rng = random.Random(a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    jsonl = os.path.join(a.outdir, f"accum_{a.label}.jsonl")
    fh = open(jsonl, "w", buffering=1)

    docs = load_corpus(a.corpus)
    if not docs:
        raise SystemExit(f"no usable files under {a.corpus}")
    chunks = chunk_docs(docs, a.chunk_chars, rng)
    print(f"=== {a.label}: corpus {len(docs)} files -> {len(chunks)} chunks ===", flush=True)

    if not a.no_warmup:
        try:
            chat_stream(a.port, a.model, [{"role": "user", "content": "hi"}], 8, 120,
                        think_effort=a.think)
            print("warmup done (discarded)", flush=True)
        except Exception as e:
            print(f"warmup failed: {e}", flush=True)

    messages = [{"role": "system", "content": GROUNDING}] if a.grounding else []
    canaries = []
    turn = 0
    degen_turn = None
    print(f"grounding prompt: {'ON' if a.grounding else 'OFF'}  |  "
          f"thinking: {a.think or 'OFF'}", flush=True)

    for kind, fn, body in chunks:
        turn += 1
        user = PROMPTS.get(kind, PROMPTS["prose"]).format(body=body)

        # Plant a canary INSIDE the user turn, mid-body, on a schedule.
        planted = None
        if turn % a.canary_every == 0:
            c = make_canary(rng, len(canaries))
            mid = len(user) // 2
            user = (user[:mid] + f"\n\n(Note for later: {c['sentence']}.)\n\n" + user[mid:])
            canaries.append(dict(c, turn=turn))
            planted = c["value"]

        messages.append({"role": "user", "content": user})
        cache_before = scrape_cache(a.port)
        try:
            r = chat_stream(a.port, a.model, messages, a.max_tokens, a.timeout,
                            think_effort=a.think)
        except Exception as e:
            rec = {"turn": turn, "kind": "ERROR", "err": f"{type(e).__name__}: {str(e)[:200]}"}
            fh.write(json.dumps(rec) + "\n")
            print(f"[{turn:03d}] ERROR {rec['err']}", flush=True)
            messages.pop()
            break

        messages.append({"role": "assistant", "content": r["text"]})
        # Score BOTH streams: visible answer, and the answer+reasoning together.
        # A loop that starts inside <think> is invisible to the content-only view.
        deg = degeneration(r["text"])
        deg_all = degeneration(r["reasoning"] + "\n" + r["text"])
        cache_after = scrape_cache(a.port)
        cq = cache_after.get("queries", 0) - cache_before.get("queries", 0)
        chh = cache_after.get("hits", 0) - cache_before.get("hits", 0)
        rec = {"turn": turn, "kind": "chat", "content_type": kind, "src": fn,
               "planted": planted,
               **{k: v for k, v in r.items() if k not in ("text", "reasoning")},
               "cache_queries_delta": cq, "cache_hits_delta": chh,
               "cache_new_delta": cq - chh,
               "degen": deg, "degen_with_reasoning": deg_all,
               "text_head": r["text"][:160],
               "reasoning_head": r["reasoning"][:160]}
        fh.write(json.dumps(rec) + "\n")

        pt = r.get("prompt_tokens")
        cached = int(chh) if chh else r.get("cached_tokens")
        flag = deg["degenerate"] or deg_all["degenerate"]
        print(f"[{turn:03d}] ctx={pt} cached={cached} TTFT={r['ttft']}s "
              f"dec={r['decode_tps']} t/s gen={r.get('completion_tokens')} "
              f"think={r['reasoning_chars']}c "
              f"d4={deg_all['distinct4']} rep8={deg_all['max_rep_8gram']} "
              f"mlp={r.get('mean_logprob')}"
              + ("  <== DEGENERATE" if flag else ""), flush=True)

        if flag and degen_turn is None:
            degen_turn = turn
            print(f"  !! first degenerate output at turn {turn}, ctx={pt}", flush=True)

        # ---- recall battery: ask about canaries at several depths ----
        if canaries and turn % a.recall_every == 0:
            idxs = sorted({0, len(canaries) // 4, len(canaries) // 2,
                           (3 * len(canaries)) // 4, len(canaries) - 1})
            for ci in idxs:
                c = canaries[ci]
                probe = messages + [{"role": "user", "content": c["question"]}]
                try:
                    # ★ Reasoning tokens are billed from the SAME allowance, so with
                    # thinking on a fixed 300 truncates mid-think and returns EMPTY
                    # content -- every probe would score a false MISS. Scale the
                    # probe budget with --max-tokens.
                    rr = chat_stream(a.port, a.model, probe,
                                     max(300, a.max_tokens), a.timeout,
                                     think_effort=a.think)
                except Exception as e:
                    fh.write(json.dumps({"turn": turn, "kind": "recall_error",
                                         "canary_turn": c["turn"],
                                         "err": str(e)[:200]}) + "\n")
                    continue
                # Score on the NUMERIC CORE as well as the full string. Asked for
                # "the value only", the model returns "3820" for "build 3820" --
                # correct recall that a literal match scores as a MISS.
                ans = rr["text"].lower()
                core = re.sub(r"[^0-9]", "", c["value"])
                hit = (c["value"].lower() in ans) or (bool(core) and core in ans)
                in_reasoning = bool(core) and core in rr["reasoning"].lower()
                depth_frac = c["turn"] / max(1, turn)
                fh.write(json.dumps({
                    "turn": turn, "kind": "recall", "canary_id": c["id"],
                    "canary_turn": c["turn"], "depth_frac": round(depth_frac, 3),
                    "ctx": rr.get("prompt_tokens"), "expected": c["value"],
                    "hit": hit, "hit_in_reasoning_only": in_reasoning and not hit,
                    "answer": rr["text"][:200],
                    "degen": degeneration(rr["reasoning"] + "\n" + rr["text"]),
                    "mean_logprob": rr.get("mean_logprob")}) + "\n")
                print(f"    recall t{c['turn']:03d} ({'HIT ' if hit else 'MISS'}) "
                      f"want={c['value']!r} got={rr['text'][:70]!r}", flush=True)

        if pt and pt >= a.budget_tokens:
            print(f"\nreached budget {a.budget_tokens} at turn {turn} (ctx={pt}) - stopping",
                  flush=True)
            break

    # ---- paired final probe: same context, one pass per effort ----
    if a.final_probe_efforts and canaries:
        efforts = [e.strip() for e in a.final_probe_efforts.split(",") if e.strip()]
        print(f"\n=== FINAL PAIRED PROBE: {len(canaries)} canaries x {efforts} ===", flush=True)
        for eff in efforts:
            hits = 0
            for c in canaries:
                probe = messages + [{"role": "user", "content": c["question"]}]
                try:
                    rr = chat_stream(a.port, a.model, probe,
                                     max(300, a.max_tokens), a.timeout,
                                     think_effort=eff)
                except Exception as e:
                    fh.write(json.dumps({"kind": "final_probe_error", "effort": eff,
                                         "canary_turn": c["turn"],
                                         "err": str(e)[:200]}) + "\n")
                    continue
                ans = rr["text"].lower()
                core = re.sub(r"[^0-9]", "", c["value"])
                hit = (c["value"].lower() in ans) or (bool(core) and core in ans)
                hits += bool(hit)
                fh.write(json.dumps({
                    "kind": "final_probe", "effort": eff, "canary_id": c["id"],
                    "canary_turn": c["turn"], "ctx": rr.get("prompt_tokens"),
                    "expected": c["value"], "hit": hit,
                    "answer": rr["text"][:300],
                    "reasoning_chars": rr.get("reasoning_chars"),
                    "completion_tokens": rr.get("completion_tokens"),
                    "degen": degeneration(rr["text"])}) + "\n")
            print(f"  effort={eff:5s}  {hits}/{len(canaries)} = "
                  f"{100*hits/len(canaries):.1f}%", flush=True)

    fh.close()
    summary = {"label": a.label, "turns": turn, "canaries": len(canaries),
               "first_degenerate_turn": degen_turn, "jsonl": jsonl}
    with open(os.path.join(a.outdir, f"accum_{a.label}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {jsonl}", flush=True)
    print(f"SUMMARY {a.label}: turns={turn} canaries={len(canaries)} "
          f"first_degenerate={degen_turn}", flush=True)


if __name__ == "__main__":
    main()
