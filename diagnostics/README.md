# Historical neighbor-loop replay

The result JSON contains 304 per-question measurements, input SHA-256 hashes and
199/304-question means. It is a fixed-current-memory counterfactual, not a rerun
of the paper's historical models. Context tokens use cl100k_base, not DeepSeek billing.

## Check the published arithmetic (fresh clone; standard library only)

```bash
python repro/replay_history.py --verify-json diagnostics/history-replay.json
```

This recalculates all aggregate means. It does not independently reconstruct prompts.

## Full replay (requires original local inputs)

```bash
python -m pip install tiktoken==0.14.0
python repro/replay_history.py --repo . --log PATH_TO_ORIGINAL_MAIN_LOG --cache-dir PATH_TO_ORIGINAL_CACHE --out diagnostics/history-replay.json
```

Use the original 304-question main log and memory_cache_sample_0.pkl / sample_1.pkl.
They are ignored and are NOT supplied by a fresh clone. Their original SHA-256 hashes
are in the result JSON. Regenerating memory with a new LLM run will not reproduce these
inputs. The script requires full upstream Git history (85787ea, 3542c98, 5b345e6).
Only load trusted pickle files; the inert note classes do not make pickle safe.

The script reconstructs every current context and asserts that the December loop matches
the logged context, then replays all three loop bodies on the same direct retrievals.
It makes no model calls. On first use tiktoken may download its public dictionary;
subsequent runs use .cache/history-replay-tokenizer. For offline use pass
--tokenizer-file PATH_TO_cl100k_base.tiktoken (checksum verified by tiktoken).
The dictionary is a cache asset and is deliberately not committed.

Current/earliest context tokens are 9414.30/4380.75 = 2.149 on conv-26.
Historical counters append before checking their threshold; old block counts are
not k+6 or 2k+1. No answer quality or historical-paper token ratio is inferred.
