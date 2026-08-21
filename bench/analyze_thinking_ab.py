"""Thinking ON (reasoning_effort=max) vs the thinking-OFF baseline, matched depths.

Guards against three ways this comparison could lie:
  1. FALSE HITS  - with thinking on, answers are long and may hedge then answer, or
                   abstain while mentioning a number in passing. Any answer that
                   contains abstention language is printed in full for inspection.
  2. EMPTY CONTENT - max effort + a finite max_tokens can truncate mid-think and
                   return an empty `content`. Those must not be scored as answers.
  3. DEGENERATION not comparable - the baseline had NO reasoning to score, so only
                   the CONTENT-ONLY metric can be compared like-for-like.
"""
import json
import re
import sys
from collections import defaultdict

ABST = re.compile(r"no record|cannot find|can.t find|don.t have|not find|"
                  r"unable to find|no note|not mentioned|don.t see", re.I)
BASE = "results/accum_P2_CHUNK64.jsonl"
THINK = "results/accum_P3_THINK_MAX.jsonl"
CAP = 500_000          # baseline goes to 1M; compare only where both have data


def load(p):
    chat, rec = [], []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("kind") == "chat":
            chat.append(d)
        elif d.get("kind") == "recall":
            rec.append(d)
    return chat, rec


def bucket(c):
    return (c // 150_000) * 150_000


def recall_table(rec, label):
    b = defaultdict(lambda: [0, 0])
    for r in rec:
        c = r.get("ctx")
        if c is None or c > CAP:
            continue
        k = bucket(c)
        b[k][1] += 1
        if r["hit"]:
            b[k][0] += 1
    tot_h = sum(v[0] for v in b.values())
    tot_n = sum(v[1] for v in b.values())
    print("%-22s %s" % (label, "  ".join(
        "%s+: %d/%d=%.0f%%" % ("{:,}".format(k), v[0], v[1], 100 * v[0] / v[1])
        for k, v in sorted(b.items()))))
    return tot_h, tot_n


def main():
    bc, br = load(BASE)
    tc, tr = load(THINK)

    print("=== 1. RECALL, matched depths (<= %s tokens) ===" % "{:,}".format(CAP))
    bh, bn = recall_table(br, "thinking OFF (base)")
    th, tn = recall_table(tr, "thinking ON  (max) ")
    print()
    print("  OVERALL  OFF %d/%d = %.1f%%   |   ON %d/%d = %.1f%%"
          % (bh, bn, 100 * bh / bn, th, tn, 100 * th / tn))

    print("\n=== 2. FALSE-HIT AUDIT: answers scored HIT that contain abstention language ===")
    sus = [r for r in tr if r["hit"] and ABST.search(r["answer"])]
    if not sus:
        print("  none")
    for r in sus:
        print("  ctx=%s want=%r" % ("{:,}".format(r["ctx"]), r["expected"]))
        print("     FULL: %r" % r["answer"][:400])
    print("  -> %d suspicious of %d hits" % (len(sus), th))

    print("\n=== 3. EMPTY CONTENT (truncated mid-think) ===")
    for name, chat in (("OFF", bc), ("ON ", tc)):
        empty = [d for d in chat if not (d.get("text_head") or "").strip()]
        hit_cap = [d for d in chat if d.get("finish_reason") == "length"]
        print("  %s: %d/%d turns with empty content, %d hit max_tokens"
              % (name, len(empty), len(chat), len(hit_cap)))

    print("\n=== 4. DEGENERATION, content-only (the only like-for-like metric) ===")
    for name, chat in (("OFF", bc), ("ON ", tc)):
        sub = [d for d in chat if (d.get("prompt_tokens") or 0) <= CAP]
        deg = [d for d in sub if (d.get("degen") or {}).get("degenerate")]
        rep = [d for d in sub if ((d.get("degen") or {}).get("max_rep_8gram") or 0) >= 2]
        d4 = [(d.get("degen") or {}).get("distinct4") for d in sub]
        d4 = [x for x in d4 if x is not None]
        print("  %s: turns=%-4d degenerate=%-3d  repeated-8gram=%-4d (%.1f%%)  median distinct4=%.3f"
              % (name, len(sub), len(deg), len(rep), 100 * len(rep) / max(1, len(sub)),
                 sorted(d4)[len(d4) // 2] if d4 else 0))
    print("\n  (for reference, reasoning-INCLUSIVE on the thinking run:)")
    sub = [d for d in tc if (d.get("prompt_tokens") or 0) <= CAP]
    degall = [d for d in sub if (d.get("degen_with_reasoning") or {}).get("degenerate")]
    print("    degenerate=%d/%d -- reasoning text is inherently repetitive, so this"
          " is NOT comparable to the baseline" % (len(degall), len(sub)))

    print("\n=== 5. COST ===")
    for name, chat in (("OFF", bc), ("ON ", tc)):
        sub = [d for d in chat if (d.get("prompt_tokens") or 0) <= CAP]
        gen = [d.get("completion_tokens") or 0 for d in sub]
        th_c = [d.get("reasoning_chars") or 0 for d in sub]
        dec = [d.get("decode_tps") for d in sub if d.get("decode_tps")]
        print("  %s: mean gen tokens/turn=%.0f  mean reasoning chars=%.0f  median decode=%.1f t/s"
              % (name, sum(gen) / len(gen), sum(th_c) / len(th_c),
                 sorted(dec)[len(dec) // 2]))


if __name__ == "__main__":
    main()
