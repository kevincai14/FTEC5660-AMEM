# LoCoMo subset with A-Mem (authors' code), DeepSeek as the LLM

Replication study for FTEC5660 (reproducibility assignment).
Paper: *A-Mem: Agentic Memory for LLM Agents* (Xu et al., NeurIPS 2025, arXiv:2502.12110).
Code: `WujiangXu/A-mem` @ `f9a173d` (the repository the authors say is "specifically designed
to reproduce the results presented in our paper").

## What is being reproduced

The paper's **Table 1**, LoCoMo split, per-category **F1 (%)** and **BLEU-1 (%)** for A-Mem
(gpt-4o-mini row, used as the reference point):

| category | F1 | BLEU-1 |
|---|---|---|
| Multi-Hop | 27.02 | 20.09 |
| Temporal | 45.85 | 36.67 |
| Open-Domain | 12.14 | 12.00 |
| Single-Hop | 44.65 | 37.06 |
| Adversarial | 50.03 | 49.47 |

Two claims are tested, in this order:

1. **The scores themselves** on a fixed subset of LoCoMo, with the authors' pipeline and a
   different LLM (DeepSeek). Not expected to match: the paper's number is gpt-4o-mini's.
2. **The cost claim** — A-Mem answers a question while reading `1,200–2,500` tokens of memory,
   against ~`16,900` for full-context baselines. This one is largely model-independent, so it
   is the claim a model swap can still speak to.

## One modification

`retrieve_k` (the number of memory notes retrieved per question, upstream default 10), swept
over `{5, 10, 20}` on the 199 questions of conv-26. Memories are built once and cached, so the
arms differ *only* in how many notes they read — the same notes, links and evolutions, verified
by the cache being loaded rather than rebuilt. Reported: F1, prompt tokens per question and
$/question, paired per question, with a bootstrap CI on the difference.

The knob is linear but five times larger than its name suggests: `find_related_memories_raw`
appends up to `k` linked neighbours *for each* of the `k` retrieved notes, and each note carries
~4.1 links, so context grows as `5.13·k` blocks — `k=10` sends 51.4 blocks (32.1 distinct).
`repro/blocks.py` recounts this from a run log.

## Deviations from the paper (all recorded in each run's `manifest.json`)

| # | Deviation | Why | Consequence |
|---|---|---|---|
| 1 | LLM is `deepseek-flash`, not `gpt-4o-mini` | cost; assignment allows a model swap if recorded | scores are not directly comparable to Table 1 |
| 2 | 2 of 10 conversations (304 of 1986 questions) | budget | subset, per-category `n` reported |
| 3 | thinking mode **disabled** | DeepSeek enables it by default; reasoning tokens are billed as output, they consume the harness's `max_tokens=1000` (observed truncated answers), and thinking mode *ignores* `temperature`, which the harness sets per category | temperature now behaves as the code intends |
| 4 | BERTScore stubbed to 0 | appendix metric; needs roberta-large (1.4 GB) on CPU per question | F1/BLEU-1 unaffected, which is what Table 1 reports |
| 5 | category 5 scored **both** ways | upstream's `--legacy_c5_reference` documents that the paper scored adversarial questions against the *adversarial answer* | both numbers reported from the same predictions, no extra cost |
| 6 | `--legacy_c5_reference` left off at run time | corrected convention is the default in the current code | harmless: both conventions recomputed in `repro/score.py` |

## Result, in four lines

| | measured | paper (gpt-4o-mini) |
|---|---|---|
| F1 overall (k=10, 304 q) | 43.70 (CI 39.12–48.28) | — (Table 1 has no average) |
| Multi-Hop F1 | 29.85 | 27.02 |
| Temporal F1 | 33.70 | 45.85 |
| prompt tokens / question | **10,700** | **2,520** (claim: 85–93% below 16,910) |
| cost, all three arms | **¥6.24** (balance delta) | — |

The efficiency claim does not reproduce: `find_related_memories_raw` expands up to `k` neighbours
for each of the `k` retrieved notes, so a question reads 51.4 note blocks (37.0% duplicate text),
not 10. Sweeping `k` over {5, 10, 20} moves F1 by −1.90 / +2.50 points, both CIs containing zero,
while tokens move 4×.

## Environment

Python 3.12 (torch has no 3.14 wheel), `.venv`, CPU only. Pinned in
`requirements-repro.lock.txt`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-repro.lock.txt      # Linux/macOS
uv pip install --python .venv/Scripts/python.exe -r requirements-repro.lock.txt  # Windows
python repro/fetch_nltk.py        # tokenizer + wordnet into .nltk_data/, and verifies them
```

Two host fixes, applied by `repro/net.py` before any model import:

- HuggingFace goes through `hf-mirror.com` (direct huggingface.co returns 502 here) unless
  `HF_ENDPOINT` is already set;
- NLTK `punkt`/`punkt_tab`/`wordnet` must exist, or `nltk.word_tokenize` raises inside
  `calculate_bleu_scores`, which `utils.py` swallows into **BLEU = 0.0** — a silent zero
  rather than an error. That is why `fetch_nltk.py` exits non-zero when they are missing
  instead of letting the run continue.

## Running

```bash
cp .env.example .env           # add DEEPSEEK_API_KEY
python repro/smoke.py --turns 12 --questions 6 --budget-usd 0.10   # unit cost probe
python repro/run.py --dry-run --samples 2                          # plan, no API calls
python repro/run.py --samples 2 --retrieve-k 10 --tag main --budget-usd 1.60   # ~60 min
python repro/score.py runs/main_k10_s2_deepseek-flash_seed42 --paper
```

`--dry-run` is the cheapest end-to-end check: it loads the dataset, parses the conversations
and prints the question counts per category (43/63/13/114/71 = 304) without spending anything.

You do not have to spend anything: the three arms are committed, so every number quoted here
can be recomputed from the stored predictions, with no API calls —

```bash
python repro/score.py runs/main_k10_s2_deepseek-flash_seed42 \
                       runs/sweep5_k5_s1_deepseek-flash_seed42 \
                       runs/sweep20_k20_s1_deepseek-flash_seed42 \
                       --only-sample 0 --markdown runs/tables/table-sweep.md \
                       --tokens-md runs/tables/table-sweep-tokens.md
```

## Run-to-run variance

The harness does not seed the LLM: `--seed` only permutes the order of the two options shown for
category-5 questions (`test_advanced_robust.py`). Every run is therefore a fresh sample at
temperature 0.7, and "how far does F1 move if I run this again" is a real question the bootstrap
CIs do not answer. Four evaluations of the reported configuration are committed — `runs/main_k10_s2_*`
and `runs/repB_k10_s1_*` — and `repro/variance.py` reports mean, sd, range, and the within-run CI
width beside them:

```bash
python repro/variance.py runs/repB_k10_s1_deepseek-flash_seed4* \
                         runs/main_k10_s2_deepseek-flash_seed42 --only-sample 0
```

Result: overall F1 41.31 ± 1.13 (sd 1.13, range 2.61) against a within-run CI width of 11.1, and
9,572 ± 16 prompt tokens per question. F1 differences of two or three points are inside that band;
the token figures are not.

`runs/rep_k10_s1_*` are three further repeats made with `--refresh-index`, which is **not** a
"force rebuild" switch: it re-embeds neighbours as soon as evolution rewrites them, instead of
every 100 evolutions as in the paper. That is a different configuration, and it measures
differently — F1 44.82 ± 1.34 at 14,243 ± 51 tokens per question. Do not pool the two groups.

`repro/sweep.sh` is the paid half: it re-runs the two `k` arms against the cached memories
(~¥6 for both, off-peak) and then scores them. Re-running the scripts is only necessary if you
doubt the stored artefacts, which is the point of committing them.

`run.py` refuses to run if the dataset SHA-256 changes, writes a manifest pinning the upstream
commit, the prompt files' hashes, model, temperature, `k`, seed and package versions, and stops
the moment spend crosses `--budget-usd`.

## Layout

```
repro/net.py        proxy/HF-mirror/NLTK setup, applied first
repro/fetch_nltk.py one-shot NLTK resource fetch, non-zero exit if incomplete
repro/pricing.py    DeepSeek price table, peak vs off-peak
repro/telemetry.py  per-call token+latency accounting, per-question deltas, budget stop
repro/run.py        one arm: manifest + predictions + telemetry
repro/smoke.py      unit-cost probe
repro/score.py      F1/BLEU under both c5 conventions, bootstrap CI, paired comparison
repro/blocks.py     counts note blocks per prompt (the 5.13·k measurement)
repro/variance.py   mean / sd across repeated runs of one configuration
```

Upstream files are unmodified except `memory_layer_robust.py` (+3 lines: `base_url` on the
OpenAI controller, forwarded from the factory) so the harness can talk to any OpenAI-compatible
endpoint.
