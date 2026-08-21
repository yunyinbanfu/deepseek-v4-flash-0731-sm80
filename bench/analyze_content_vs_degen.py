"""Is degeneration driven by SOURCE CONTENT or by DEPTH?

Both degenerate outputs seen so far were `techdoc` summaries of our own handoff
files (structurally very repetitive: tables, repeated headers, enumerations) --
at 88,137 and 866,282 tokens, i.e. wildly different depths. That suggests content,
not depth.

Counting the degenerate flag is hopeless: 2 events in ~900 turns. Instead compare
the DISTRIBUTION of the repetition metrics by content type, which uses every turn.

Confound check: if techdoc chunks happen to land at greater depths, a depth effect
would masquerade as a content effect -- so report depth per content type too, and
re-compare within a matched depth band.
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict


def med(v):
    # Short generations leave distinct4 unset; drop them rather than crashing.
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fmt(v, n=3):
    return "-" if v is None else ("%." + str(n) + "f") % v


def collect(paths):
    rows = []
    for p in paths:
        run = os.path.basename(p).replace("accum_", "").replace(".jsonl", "")
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") != "chat":
                continue
            g = d.get("degen_with_reasoning") or d.get("degen") or {}
            if g.get("max_rep_8gram") is None:
                continue
            rows.append({
                "run": run,
                "ct": d.get("content_type"),
                "ctx": d.get("prompt_tokens") or 0,
                "rep8": g.get("max_rep_8gram"),
                "d4": g.get("distinct4"),
                "top": g.get("top_token_share"),
                "deg": bool(g.get("degenerate")),
                "len_stop": d.get("finish_reason") == "length",
                "gen": d.get("completion_tokens") or 0,
            })
    return rows


def table(rows, title):
    print("\n=== %s  (n=%d) ===" % (title, len(rows)))
    print("%-11s %6s %9s %9s %9s %10s %9s %8s" %
          ("content", "turns", "rep8 mean", "rep8 max", "distinct4",
           "med ctx", "hit-len%", "degen"))
    by = defaultdict(list)
    for r in rows:
        by[r["ct"]].append(r)
    for ct in sorted(by, key=lambda k: -len(by[k])):
        v = by[ct]
        print("%-11s %6d %9s %9d %9s %10s %8s%% %8d" % (
            ct, len(v),
            fmt(mean([x["rep8"] for x in v]), 2),
            max(x["rep8"] for x in v),
            fmt(med([x["d4"] for x in v] or [None])),
            "{:,}".format(int(med([x["ctx"] for x in v]) or 0)),
            fmt(100 * sum(1 for x in v if x["len_stop"]) / len(v), 1),
            sum(1 for x in v if x["deg"]),
        ))


def main():
    base = "results"
    paths = sorted(glob.glob(os.path.join(base, "accum_*.jsonl")))
    paths = [p for p in paths if "SMOKE" not in p]
    print("runs:", ", ".join(os.path.basename(p) for p in paths))
    rows = collect(paths)

    table(rows, "ALL TURNS, by content type")

    # Confound control: same comparison inside one depth band.
    band = [r for r in rows if 50_000 <= r["ctx"] <= 150_000]
    table(band, "DEPTH-MATCHED 50k-150k only")

    # And the reverse view: does depth move the metrics within one content type?
    print("\n=== DEPTH effect WITHIN techdoc (content held fixed) ===")
    td = [r for r in rows if r["ct"] == "techdoc"]
    print("%-14s %6s %9s %9s %9s" % ("ctx band", "turns", "rep8 mean", "distinct4", "hit-len%"))
    for lo in (0, 150_000, 300_000, 500_000, 750_000):
        hi = {0: 150_000, 150_000: 300_000, 300_000: 500_000,
              500_000: 750_000, 750_000: 10**9}[lo]
        v = [r for r in td if lo <= r["ctx"] < hi]
        if not v:
            continue
        print("%-14s %6d %9s %9s %8s%%" % (
            "%s-%s" % ("{:,}".format(lo), "{:,}".format(hi) if hi < 10**9 else "max"),
            len(v),
            fmt(mean([x["rep8"] for x in v]), 2),
            fmt(med([x["d4"] for x in v])),
            fmt(100 * sum(1 for x in v if x["len_stop"]) / len(v), 1)))

    print("\n=== the degenerate turns themselves ===")
    for r in rows:
        if r["deg"]:
            print("  run=%-12s ctx=%-9s type=%-8s rep8=%d d4=%s gen=%d len_stop=%s"
                  % (r["run"], "{:,}".format(r["ctx"]), r["ct"], r["rep8"],
                     fmt(r["d4"]), r["gen"], r["len_stop"]))


if __name__ == "__main__":
    main()
