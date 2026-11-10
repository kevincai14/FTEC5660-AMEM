#!/usr/bin/env python
"""Mean and spread across repeated runs of the same configuration.

The point is the noise floor. `score.py` puts a bootstrap CI on the mean, which answers "how
sure are we about this arm's F1 given these questions" -- but not "if I re-ran the whole thing
tomorrow, how far would F1 move". Those are different questions, and a difference between two
arms only means something if it is larger than the second one.

What a repeat is here: the same memory graph, every answer re-sampled. Nothing in the harness
seeds the LLM -- `--seed` only permutes the order of the two options shown for category-5
questions (`test_advanced_robust.py`, `random.Random(f"{seed}:...")`) -- so repeats differ by
both decoding and category-5 option order. Memory construction is held fixed; this does not measure full-pipeline variability.

The graph is deliberately held fixed, because `--refresh-index` is *not* a "force rebuild" switch:
it re-embeds neighbour notes the moment evolution rewrites them, where the paper's behaviour
refreshes embeddings only every 100 evolutions. Turning it on changes what is retrieved, so it
changes the configuration. Measured, on these same 199 questions: F1 44.8 vs 41.3 and 14.2k vs
9.6k prompt tokens per question. Repeats must not mix the two.

    python repro/variance.py runs/repB_k10_s1_* runs/main_k10_s2_* --only-sample 0
    python repro/variance.py runs/repB_k10_s1_* --only-sample 0 --markdown runs/tables/variance.md

Reports, per category: the mean across runs, the *between-run* sd, min/max, and -- for contrast --
the first repeat's within-run CI width. If the between-run sd is comparable to the CI, the honest
conclusion is that the arm's number is quoted to more precision than the experiment earns.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repro.net import configure_network, quiet_redundant_downloads  # noqa: E402

configure_network()
quiet_redundant_downloads()

from repro.score import CATEGORY_NAMES, bootstrap_ci, load_run, report, write_text, validate_runs  # noqa: E402

ORDER = ["overall"] + list(CATEGORY_NAMES.values())


def spread(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories, one per repeat, same configuration")
    ap.add_argument("--only-sample", default=None,
                    help="restrict to one conversation (e.g. 0), for pairing with narrower arms")
    ap.add_argument("--out", default=None, help="machine-readable repeat summary")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--tokens-md", default=None)
    ap.add_argument("--runs-md", default=None, help="one row per repeat, so every run shows")
    args = ap.parse_args()

    only = None if args.only_sample in (None, "all") else {int(args.only_sample)}
    loaded = [load_run(Path(d), only_sample=only) for d in args.runs]
    validate_runs(loaded, repeat=True)
    reps = [report(r) for r in loaded]

    def category_values(key: str) -> list[float]:
        """Per-question F1 for one category, from the first repeat -- only for its within-run CI."""
        return [s["f1_corrected"] for s in loaded[0]["scored"]
                if key == "overall" or CATEGORY_NAMES.get(s["category"]) == key]
    if len(reps) < 2:
        raise SystemExit("need at least two runs to measure spread")

    seeds = [r["arm"].rsplit("seed", 1)[-1] for r in reps]
    print(f"{len(reps)} repeats, seeds {', '.join(seeds)}, "
          f"{reps[0]['questions']} questions each\n")

    header = f"{'category':<13}{'n':>5}{'mean':>9}{'sd':>7}{'min':>8}{'max':>8}{'range':>8}{'CI width':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for key in ORDER:
        per_run = [r["metrics"]["f1_corrected"].get(key) for r in reps]
        per_run = [p for p in per_run if p]
        if not per_run:
            continue
        s = spread([p["mean"] for p in per_run])
        ci_lo, ci_hi = bootstrap_ci(category_values(key))
        rows.append({"key": key, "n": per_run[0]["n"], **s, "ci_width": ci_hi - ci_lo})
        print(f"{key:<13}{per_run[0]['n']:>5}{s['mean']:>9.2f}{s['sd']:>7.2f}"
              f"{s['min']:>8.2f}{s['max']:>8.2f}{s['max'] - s['min']:>8.2f}{ci_hi - ci_lo:>10.2f}")

    overall = rows[0]
    print(f"\nnoise floor: overall F1 moves {overall['sd']:.2f}pp (sd) between runs of the *same* "
          f"configuration, range {overall['max'] - overall['min']:.2f}pp")
    tok = [r["tokens_per_question"]["mean_prompt"] for r in reps]
    tok_sd = statistics.stdev(tok) if len(tok) > 1 else 0.0
    print(f"same for prompt tokens/question: {statistics.mean(tok):,.0f} ± {tok_sd:,.0f}")

    print("\nper repeat:")
    for d, r in zip(args.runs, reps):
        m = r["metrics"]["f1_corrected"]["overall"]
        lo, hi = m.get("ci95", (float("nan"), float("nan")))
        print(f"  {Path(d).name:<44} F1 {m['mean']:6.2f} [{lo:.2f}, {hi:.2f}]"
              f"  {r['tokens_per_question']['mean_prompt']:>9,.0f} tok/q")

    if args.out:
        import json
        write_text(args.out, json.dumps({"runs": [r["arm"] for r in reps],
            "repeat_scope": "fixed memory; decoding plus category-5 option order",
            "categories": rows, "prompt_tokens": spread(tok),
            "ci_scope": "question bootstrap on first listed repeat only",
            "memory_cache_hash_verified": False}, indent=2))

    if args.runs_md:
        lines = ["| repeat | overall F1 | its own 95% CI | prompt tok/q | modelled $/q |",
                 "|---|---|---|---|---|"]
        for d, r in zip(args.runs, reps):
            m = r["metrics"]["f1_corrected"]["overall"]
            lo, hi = m.get("ci95", (float("nan"), float("nan")))
            lines.append(f"| `{Path(d).name}` | {m['mean']:.2f} | [{lo:.2f}, {hi:.2f}] | "
                         f"{r['tokens_per_question']['mean_prompt']:,.0f} | "
                         f"{(r['cost_per_question'] or 0):.5f} |")
        write_text(args.runs_md, "\n".join(lines) + "\n")
        print(f"wrote {args.runs_md}")

    if args.markdown:
        lines = ["| category | n | mean F1 | sd | min | max | range | within-run CI width |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['key']} | {r['n']} | {r['mean']:.2f} | {r['sd']:.2f} | "
                         f"{r['min']:.2f} | {r['max']:.2f} | {r['max'] - r['min']:.2f} | "
                         f"{r['ci_width']:.2f} |")
        write_text(args.markdown, "\n".join(lines) + "\n")
        print(f"\nwrote {args.markdown}")

    if args.tokens_md:
        lines = ["| run | questions | prompt tok/q | completion tok/q | modelled $/q | elapsed |",
                 "|---|---|---|---|---|---|"]
        for d, r in zip(args.runs, reps):
            u = r.get("usage") or {}
            pt = r["tokens_per_question"]["mean_prompt"]
            ct = r["tokens_per_question"].get("mean_completion", 0)
            lines.append(f"| {Path(d).name} | {r['questions']} | {pt:,.0f} | {ct:,.0f} | "
                         f"{(r['cost_per_question'] or 0):.5f} | {(u.get('latency_s') or 0) / 60:.0f} min |")
        write_text(args.tokens_md, "\n".join(lines) + "\n")
        print(f"wrote {args.tokens_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
