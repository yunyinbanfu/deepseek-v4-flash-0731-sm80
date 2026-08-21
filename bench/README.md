# Benchmark harnesses

Every number in [RESULTS.md](../RESULTS.md) came from these. They talk to
`http://127.0.0.1:8098/v1/completions` with `--served-model-name dsv4s`; edit `URL` / `MODEL`
at the top of each if yours differ.

Start the server with **`--no-enable-prefix-caching`** for any of these, or repeated prompts
skip prefill and the numbers are fiction.

**One exception:** `bench_chat_accumulate.py` requires prefix caching to be **ON** — reusing prior
KV across turns is the thing it exists to test, and it is the path where the accumulated-chat
ceiling lives. Run it against a normally-configured server, not a benchmark one.

| script | measures | notes |
|---|---|---|
| `bench_decode3.py` | end-to-end decode over 3 content types | The headline number. Speculative decoding's benefit is content-dependent, so one prompt is not a measurement. |
| `bench_concurrency.py` | aggregate decode and prefill vs concurrent requests | Threads, unique prompt per request. |
| `bench_prefill.py` | prefill vs prompt length | `max_tokens=1`, rate against the server's own `usage.prompt_tokens`. |
| `bench_longctx.py` | prefill at 4k → 100k+ | Shows PP's prefill climbing with context and TP's staying flat. |
| `bench_decode_stream.py` | decode vs context, by streaming | Timestamps first and last token — no subtraction. Requests `include_usage` and `ignore_eos`; see pitfalls below. |
| `bench_needle.py` | **correctness** at long context | Buries a passphrase at 10% depth and asks for it back. Tests that the sparse indexer really selects the right blocks — not merely that the run completes. |
| `bench_probe.py` | deterministic completions to a file | Diff two runs for greedy-equivalence and self-determinism checks. |
| `bench_ceiling.py` | **the context ceiling**, precisely | Walks a ladder of prompt sizes and separates **PASS / WRONG / DEAD** — DEAD meaning the server stopped answering `/v1/models`, i.e. the worker was killed, which is the Xid-31 signature. Takes `--port` / `--model` rather than editing the file. ⚠️ Its size arguments are **~1.30× the real token count** — read the `real tok` column, which comes from the server's own `usage.prompt_tokens`. |
| `bench_conc_needle.py` | **correctness under concurrency** at long context | Fires N long needle prompts simultaneously, each with its **own** passphrase, and checks every reply contains its own and no other. A prefill chunk can hold tokens from several requests, so this is what catches row/offset bleed across requests — single-request tests never exercise it. |
| `bench_decode_ctx.py` | decode vs context by subtraction | **Superseded** by `bench_decode_stream.py` — kept because the failure is instructive. |
| `bench_chat_accumulate.py` | ★ **a real multi-turn conversation** — the path every other harness here skips | Many small turns with **prefix caching ON**, growing to 1M. Plants canary facts at known turns and re-queries them on a schedule, so it scores retrieval *and* coherence, not just "did it finish". This is what found the ~725k accumulated-chat ceiling that one-shot needles miss. Needs a corpus directory (`--corpus`); `--grounding` adds an abstention system prompt. ⚠️ The one harness here that must run **with** prefix caching. |
| `analyze_accum.py` | turns a `bench_chat_accumulate` JSONL into the three answers | Speed shape (TTFT / decode / cache split by depth), coherence (repetition metrics by depth), correctness (canary recall by depth and by how far back the fact was). |
| `bench_chunk_truth.py` | ★ **whether your token rate is real** | In a *single* request, counts SSE chunks **and** tokenizes the generated text, then compares both to `usage.completion_tokens`. Run this once against any new streaming harness before trusting it — it is how the 3.3–4.6× chunk-counting error was proven rather than argued. |
| `analyze_content_vs_degen.py` | whether degeneration tracks content or depth | Pools every turn across runs, groups repetition metrics by content type **and** by depth band, and reports the depth-matched comparison. Written to test a hypothesis that it then **falsified** — kept as the template for checking a co-occurrence against its base rate. |

## Pitfalls these encode

Each of these produced a wrong number before it was caught. If you write your own harness,
these are the traps:

1. **Never count SSE chunks to get a token rate.** Under speculative decoding one chunk
   carries several tokens (roughly the acceptance length). Counting chunks reported
   24 tok/s where the truth was 79.5. Use `stream_options: {include_usage: true}` and the
   server's `completion_tokens`.
2. **Pass `ignore_eos`** when comparing two configs. Speculative output diverges and hits EOS
   early, so you end up comparing a 50-token generation against a 192-token one — and short
   generations are dominated by ramp-up.
3. **Discard the first request after boot.** It carries Triton JIT compilation and reads
   roughly 4× low. A cold 514 tok/s reading was really 1,966.
4. **Do not compute decode rate by subtracting two calls** (`max_tokens=1` vs `max_tokens=N`).
   That assumes both prefills cost the same; at 50k context they differ by seconds, which
   produced 135 tok/s sitting between neighbours of 43.8 and 38.6.
5. **A best steady-state window is not a benchmark.** Taking the fastest 10-second window
   from the engine's own log gave 3.6×; the honest end-to-end figure over mixed content was
   1.9×.
6. **Assert what is actually running** before trusting a sweep:
   `docker inspect NAME --format '{{join .Args " "}}'`.
