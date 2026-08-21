"""Turn an accum_*.jsonl into the three answers the run was built to give.

  1. SPEED SHAPE  per-turn TTFT / decode / cache split, bucketed by depth, so the
                  accumulated curve can be laid next to the published ONE-SHOT
                  curve (prefill 4,687->1,904 t/s, decode 88.7->35.6).
  2. COHERENCE    degeneration metrics vs depth; flags the first degenerate turn.
  3. CORRECTNESS  canary recall vs depth, split by how far back the fact was.
"""
import argparse, json, statistics as st
from collections import defaultdict


def bucket(ctx, width=20000):
    if ctx is None:
        return None
    return (ctx // width) * width


def fmt(v, n=1):
    return "-" if v is None else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--width", type=int, default=20000)
    a = ap.parse_args()

    chats, recalls, errors = [], [], []
    for line in open(a.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        k = d.get("kind")
        if k == "chat":
            chats.append(d)
        elif k == "recall":
            recalls.append(d)
        elif k in ("ERROR", "recall_error"):
            errors.append(d)

    print(f"=== {a.jsonl} ===")
    print(f"turns={len(chats)}  recall probes={len(recalls)}  errors={len(errors)}")
    if chats:
        print(f"depth reached: {max(c.get('prompt_tokens') or 0 for c in chats):,} tokens")

    # ---------------------------------------------------------- 1. SPEED
    print("\n--- 1. SPEED SHAPE (accumulated chat) ---")
    print(f"{'ctx bucket':>14} {'turns':>5} {'TTFT s':>8} {'decode t/s':>11} "
          f"{'new tok/turn':>13} {'cache hit%':>11}")
    b = defaultdict(list)
    for c in chats:
        b[bucket(c.get("prompt_tokens"), a.width)].append(c)
    for key in sorted(x for x in b if x is not None):
        rows = b[key]
        tt = [r["ttft"] for r in rows if r.get("ttft")]
        dc = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        nw = [r["cache_new_delta"] for r in rows if r.get("cache_new_delta") is not None]
        cq = sum(r.get("cache_queries_delta") or 0 for r in rows)
        chh = sum(r.get("cache_hits_delta") or 0 for r in rows)
        print(f"{key:>10,}+ {len(rows):>5} {fmt(st.median(tt) if tt else None,2):>8} "
              f"{fmt(st.median(dc) if dc else None):>11} "
              f"{fmt(st.median(nw) if nw else None,0):>13} "
              f"{fmt(100*chh/cq if cq else None):>11}")

    # ------------------------------------------------------ 2. COHERENCE
    print("\n--- 2. COHERENCE (degeneration vs depth) ---")
    print(f"{'ctx bucket':>14} {'turns':>5} {'distinct4':>10} {'max rep8':>9} "
          f"{'top tok%':>9} {'mean logprob':>13} {'degen':>6}")
    for key in sorted(x for x in b if x is not None):
        rows = b[key]
        def med(path, sub=None):
            vals = []
            for r in rows:
                v = r.get("degen_with_reasoning") or r.get("degen") or {}
                x = v.get(path) if sub is None else v.get(path)
                if x is not None:
                    vals.append(x)
            return st.median(vals) if vals else None
        mlp = [r["mean_logprob"] for r in rows if r.get("mean_logprob") is not None]
        ndeg = sum(1 for r in rows
                   if (r.get("degen_with_reasoning") or {}).get("degenerate")
                   or (r.get("degen") or {}).get("degenerate"))
        print(f"{key:>10,}+ {len(rows):>5} {fmt(med('distinct4'),3):>10} "
              f"{fmt(med('max_rep_8gram'),0):>9} "
              f"{fmt(100*(med('top_token_share') or 0)):>9} "
              f"{fmt(st.median(mlp) if mlp else None,3):>13} {ndeg:>6}")

    first = next((r["turn"] for r in chats
                  if (r.get("degen_with_reasoning") or {}).get("degenerate")
                  or (r.get("degen") or {}).get("degenerate")), None)
    print(f"\nfirst degenerate turn: {first if first else 'NONE - no collapse observed'}")

    # ----------------------------------------------------- 3. CORRECTNESS
    print("\n--- 3. CORRECTNESS (canary recall vs depth) ---")
    if not recalls:
        print("no recall probes")
    else:
        hits = sum(1 for r in recalls if r["hit"])
        print(f"OVERALL: {hits}/{len(recalls)} = {100*hits/len(recalls):.1f}% recall")
        print(f"\n{'ctx bucket':>14} {'probes':>7} {'hits':>5} {'recall%':>8}")
        rb = defaultdict(list)
        for r in recalls:
            rb[bucket(r.get("ctx"), a.width)].append(r)
        for key in sorted(x for x in rb if x is not None):
            rows = rb[key]
            h = sum(1 for r in rows if r["hit"])
            print(f"{key:>10,}+ {len(rows):>7} {h:>5} {100*h/len(rows):>7.1f}%")

        # How far back the fact was, as a fraction of the conversation.
        print(f"\n{'fact age':>14} {'probes':>7} {'hits':>5} {'recall%':>8}")
        ab = defaultdict(list)
        for r in recalls:
            df = r.get("depth_frac")
            if df is None:
                continue
            band = "oldest 25%" if df <= .25 else "25-50%" if df <= .5 else \
                   "50-75%" if df <= .75 else "newest 25%"
            ab[band].append(r)
        for band in ("oldest 25%", "25-50%", "50-75%", "newest 25%"):
            rows = ab.get(band)
            if not rows:
                continue
            h = sum(1 for r in rows if r["hit"])
            print(f"{band:>14} {len(rows):>7} {h:>5} {100*h/len(rows):>7.1f}%")

        misses = [r for r in recalls if not r["hit"]]
        if misses:
            print(f"\n--- {len(misses)} MISSES (fluent-and-wrong is the dangerous mode) ---")
            for m in misses[:15]:
                print(f"  ctx={m.get('ctx')} planted_turn={m['canary_turn']} "
                      f"want={m['expected']!r} got={m['answer'][:80]!r}")

    if errors:
        print(f"\n--- {len(errors)} ERRORS ---")
        for e in errors[:5]:
            print(f"  turn={e.get('turn')} {e.get('err','')[:120]}")


if __name__ == "__main__":
    main()
