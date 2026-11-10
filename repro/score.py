#!/usr/bin/env python
"""Recompute the paper's metrics from stored predictions, and compare arms.

Why recompute: the harness resolves category 5 (adversarial) against one
convention at run time, and the paper's own code scored that category against
the *adversarial* answer -- the repository's `--legacy_c5_reference` flag says so
explicitly. Both are recoverable from the stored predictions, so this script
reports:

  corrected  c5 reference = "Not mentioned in the conversation"  (the right answer)
  legacy     c5 reference = the adversarial answer               (what the paper ran)

and `decline_rate` = share of c5 questions the model correctly refused.

  python repro/score.py runs/pilot_k10_s1_deepseek-flash_seed42
  python repro/score.py --compare runs/a runs/b
"""
import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repro.net import configure_network, quiet_redundant_downloads  # noqa: E402

configure_network()
QUIET_NLTK = quiet_redundant_downloads()   # upstream utils.py re-downloads at import; see net.py

from utils import calculate_bleu_scores, calculate_rouge_scores, simple_tokenize  # noqa: E402

NOT_MENTIONED = "Not mentioned in the conversation"

CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}


def write_text(path, text: str) -> None:
    """Write, creating the parent directory first.

    A fresh clone has no ``runs/tables/``, and the documented command writes its tables
    there, so `mkdir -p` has to happen here rather than in the README.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

# arXiv:2502.12110 Table 1, gpt-4o-mini row (A-Mem), F1 / BLEU-1 in percent
PAPER_AMEM_GPT4O_MINI = {
    "Multi-Hop": (27.02, 20.09), "Temporal": (45.85, 36.67), "Open-Domain": (12.14, 12.00),
    "Single-Hop": (44.65, 37.06), "Adversarial": (50.03, 49.47),
}


def token_f1(prediction: str, reference: str) -> float:
    """The harness's F1: set overlap of lowercased word tokens."""
    pred, ref = set(simple_tokenize(prediction)), set(simple_tokenize(reference))
    if not pred or not ref:
        return 0.0
    common = pred & ref
    if not common:
        return 0.0
    precision, recall = len(common) / len(pred), len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def declines(prediction: str, trap: str) -> int:
    text = str(prediction).lower()
    says_no = "not mentioned" in text or "no information available" in text
    trap = str(trap or "").lower().strip(" .")
    gives_trap = bool(trap) and not trap.startswith("not mentioned") and trap in text
    return int(says_no and not gives_trap)


def score_row(row: dict) -> dict:
    """All metrics for one question, under both category-5 conventions."""
    pred = row.get("prediction") or ""
    category = int(row.get("category") or 0)
    answer = str(row.get("reference") if row.get("reference") is not None else "")
    trap = str(row.get("adversarial_answer") if row.get("adversarial_answer") is not None else "")
    out = {"category": category, "prediction": pred}

    ref_corrected = NOT_MENTIONED if category == 5 else answer
    ref_legacy = trap if category == 5 else answer
    out["f1_corrected"] = token_f1(pred, ref_corrected)
    out["f1_legacy"] = token_f1(pred, ref_legacy)
    out["em_corrected"] = int(str(pred).strip().lower() == str(ref_corrected).strip().lower())
    out["bleu1_corrected"] = calculate_bleu_scores(pred, ref_corrected)["bleu1"]
    out["bleu1_legacy"] = calculate_bleu_scores(pred, ref_legacy)["bleu1"]
    out["rougeL"] = calculate_rouge_scores(pred, ref_corrected)["rougeL_f"]
    if category == 5:
        out["decline"] = declines(pred, trap)
    out["prompt_tokens"] = row.get("prompt_tokens")
    out["completion_tokens"] = row.get("completion_tokens")
    out["cost_usd"] = row.get("cost_usd")
    out["calls"] = row.get("calls")
    return out


def load_run(run_dir: Path, only_sample=None) -> dict:
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        raise SystemExit(f"{run_dir}: no predictions.jsonl (was the run interrupted?)")
    rows = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only_sample is not None:
        # a narrower arm can still be paired against a wider one: keep the shared conversations
        rows = [r for r in rows if r.get("sample_id") in only_sample]
    scored = [score_row(r) for r in rows]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "manifest": manifest, "rows": rows, "scored": scored}


def question_map(run):
    """Align on conversation and exact question; reject ambiguous duplicate keys."""
    result = {}
    for row, scored in zip(run["rows"], run["scored"]):
        key = (row["sample_id"], row["question"])
        if key in result:
            raise ValueError(f"duplicate question identity: {key}")
        result[key] = (row, scored)
    if not result:
        raise ValueError("no questions selected")
    return result


def validate_runs(runs, repeat=False):
    """Require comparable inputs; subset selection must be explicit."""
    base = runs[0]["manifest"]
    first = question_map(runs[0])
    for run in runs:
        m = run["manifest"]
        if m.get("status") != "ok":
            raise ValueError(f"incomplete run: {m['arm']}")
        for field in ("model", "base_url", "prompts_sha256"):
            if m.get(field) != base.get(field):
                raise ValueError(f"incompatible {field}: {m['arm']}")
        if m["dataset"]["sha256"] != base["dataset"]["sha256"]:
            raise ValueError("dataset hash differs")
        for field in ("memory_scope", "refresh_index", "temperature_c5", "embedding",
                      "evo_threshold", "max_tokens_per_call") + (("retrieve_k",) if repeat else ()):
            if m["params"].get(field) != base["params"].get(field):
                raise ValueError(f"incompatible parameter {field}: {m['arm']}")
        other = question_map(run)
        if first.keys() != other.keys():
            raise ValueError("question sets differ; select the common samples with --only-sample")
        for key in first:
            for field in ("category", "reference", "adversarial_answer"):
                if first[key][0].get(field) != other[key][0].get(field):
                    raise ValueError(f"reference/category differs: {key}")


def paired_comparison(a, b):
    left, right = question_map(a), question_map(b)
    pairs = [(left[k][1], right[k][1]) for k in left]
    diffs = [y["f1_corrected"] - x["f1_corrected"] for x, y in pairs]
    tokens_a = statistics.mean(x["prompt_tokens"] for x, _ in pairs)
    tokens_b = statistics.mean(y["prompt_tokens"] for _, y in pairs)
    return {"baseline": a["manifest"]["arm"], "variant": b["manifest"]["arm"],
            "n": len(pairs), "mean_delta_f1_points": statistics.mean(diffs) * 100,
            "ci95_delta_f1_points": bootstrap_ci(diffs),
            "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "mean_prompt_tokens_baseline": tokens_a, "mean_prompt_tokens_variant": tokens_b,
            "prompt_token_change_fraction": tokens_b / tokens_a - 1 if tokens_a else None,
            "uncertainty": "paired question bootstrap; excludes run-to-run and memory-build variation",
            "pairing": "sample_id + exact question text"}


def aggregate(scored: list, key: str) -> dict:
    by_cat = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s.get(key, 0.0) or 0.0)
    overall = [s.get(key, 0.0) or 0.0 for s in scored]
    result = {"overall": {"mean": statistics.mean(overall) * 100 if overall else 0.0, "n": len(overall)}}
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        result[CATEGORY_NAMES.get(cat, str(cat))] = {
            "mean": statistics.mean(vals) * 100,
            "std": statistics.stdev(vals) * 100 if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return result


def bootstrap_ci(values: list, n_boot: int = 2000, seed: int = 7) -> tuple:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(statistics.mean(rng.choices(values, k=len(values))))
    means.sort()
    return means[int(0.025 * n_boot)] * 100, means[int(0.975 * n_boot)] * 100


def report(run: dict) -> dict:
    scored, manifest = run["scored"], run["manifest"]
    out = {"arm": manifest["arm"], "questions": len(scored), "metrics": {}}
    for key in ("f1_corrected", "f1_legacy", "bleu1_corrected", "bleu1_legacy", "em_corrected"):
        out["metrics"][key] = aggregate(scored, key)
    f1 = [s["f1_corrected"] for s in scored]
    out["metrics"]["f1_corrected"]["overall"]["ci95"] = bootstrap_ci(f1)
    c5 = [s for s in scored if s["category"] == 5]
    out["decline_rate"] = round(100 * statistics.mean([s["decline"] for s in c5]), 2) if c5 else None
    out["usage"] = manifest.get("usage", {})
    out["peak"] = manifest.get("peak_start")
    row_cost = [s["cost_usd"] for s in scored if s.get("cost_usd") is not None]
    out["cost_per_question"] = round(statistics.mean(row_cost), 6) if row_cost else None
    n = len(scored)
    if n:
        pt = [s["prompt_tokens"] or 0 for s in scored]
        ct = [s["completion_tokens"] or 0 for s in scored]
        out["tokens_per_question"] = {"mean_prompt": round(statistics.mean(pt), 1),
                                      "median_prompt": statistics.median(pt),
                                      "mean_completion": round(statistics.mean(ct), 1),
                                      "mean_calls": round(statistics.mean([s["calls"] or 0 for s in scored]), 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--compare", action="store_true", help="compare every subsequent run against the first; validate question identities")
    ap.add_argument("--paper", action="store_true", help="show the paper's gpt-4o-mini A-Mem row next to ours")
    ap.add_argument("--markdown", default=None, help="write the per-category table as markdown")
    ap.add_argument("--tokens-md", default=None, help="write the token/cost table as markdown")
    ap.add_argument("--only-sample", default=None,
                    help="comma-separated sample_id values to keep, so arms over different subsets can be paired")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keep = None
    if args.only_sample is not None:
        keep = {int(x) for x in args.only_sample.split(",")}
    loaded = [load_run(Path(r) if Path(r).is_absolute() else ROOT / r, keep) for r in args.runs]
    reports = [report(r) for r in loaded]

    for run, rep in zip(loaded, reports):
        m = rep["metrics"]
        print(f"\n=== {rep['arm']}  ({rep['questions']} questions) ===")
        header = f"{'category':13s} {'n':>4s} {'F1(corr)':>9s} {'F1(legacy)':>11s} {'BLEU1':>7s}"
        print(header)
        for cat in [None] + list(CATEGORY_NAMES.values()):
            key = "overall" if cat is None else cat
            row = m["f1_corrected"].get(key)
            if not row:
                continue
            legacy = m["f1_legacy"].get(key, {}).get("mean", 0.0)
            bleu = m["bleu1_corrected"].get(key, {}).get("mean", 0.0)
            label = "OVERALL" if cat is None else cat
            print(f"{label:13s} {row['n']:4d} {row['mean']:8.2f}% {legacy:10.2f}% {bleu:6.2f}%")
            if args.paper and cat in PAPER_AMEM_GPT4O_MINI:
                pf1, pbleu = PAPER_AMEM_GPT4O_MINI[cat]
                print(f"{'':13s} {'':4s} paper {pf1:5.2f}%  (Δ {row['mean'] - pf1:+6.2f})   bleu {pbleu:5.2f}% (Δ {bleu - pbleu:+6.2f})")
        ci = m["f1_corrected"]["overall"]["ci95"]
        print(f"F1 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]   decline_rate(c5): {rep['decline_rate']}%  "
              f"prompt tokens/question: {rep['tokens_per_question']['mean_prompt']:.0f}")
        print(f"cost: ${rep['usage'].get('cost_usd', 0):.4f}  calls: {rep['usage'].get('calls', 0)}")

    comparisons = []
    if args.compare:
        if len(loaded) < 2:
            raise ValueError("--compare requires at least two runs")
        validate_runs(loaded)
        comparisons = [paired_comparison(loaded[0], b) for b in loaded[1:]]
        for c in comparisons:
            lo, hi = c["ci95_delta_f1_points"]
            print(f"Paired {c['baseline']} -> {c['variant']}: n={c['n']}, "
                  f"delta F1={c['mean_delta_f1_points']:+.2f}, CI [{lo:+.2f}, {hi:+.2f}]")

    payload = {"arms": [{"dir": str(r["dir"].relative_to(ROOT)), **rep}
                        for r, rep in zip(loaded, reports)], "comparisons": comparisons}
    if args.out:
        out_path = Path(args.out)
    elif len(loaded) > 1 or keep:
        # writing into the first run's directory would replace that arm's own score.json with
        # a multi-arm payload, or with a subset of its own questions -- a committed artefact
        # silently overwritten by a read-only command
        out_path = ROOT / "runs" / "tables" / "score.json"
    else:
        out_path = loaded[0]["dir"] / "score.json"
    write_text(out_path, json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")

    if args.markdown:
        lines = ["| arm | category | n | F1 (corrected) | F1 (legacy c5) | BLEU-1 | paper F1 (gpt-4o-mini) |",
                 "|---|---|---|---|---|---|---|"]
        for rep in reports:
            m = rep["metrics"]
            for key in ["overall"] + list(CATEGORY_NAMES.values()):
                row = m["f1_corrected"].get(key)
                if not row:
                    continue
                label = "**overall**" if key == "overall" else key
                paper = PAPER_AMEM_GPT4O_MINI.get(key, (None, None))[0]
                lines.append(
                    f"| {rep['arm']} | {label} | {row['n']} | {row['mean']:.2f} | "
                    f"{m['f1_legacy'].get(key, {}).get('mean', 0):.2f} | "
                    f"{m['bleu1_corrected'].get(key, {}).get('mean', 0):.2f} | "
                    f"{'-' if paper is None else f'{paper:.2f}'} |")
        write_text(args.markdown, "\n".join(lines) + "\n")
        print(f"wrote {args.markdown}")

    if args.tokens_md:
        lines = ["| arm | questions | prompt tok/q | completion tok/q | total tok/q | modelled $/q | modelled total $ | elapsed | peak pricing |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for rep in reports:
            u = rep["usage"]
            n = rep["questions"]
            pt = rep["tokens_per_question"]["mean_prompt"]
            ct = rep["tokens_per_question"].get("mean_completion", 0)
            lines.append(
                f"| {rep['arm']} | {n} | {pt:.0f} | {ct:.0f} | {pt + ct:.0f} | "
                f"{(rep['cost_per_question'] or 0):.5f} | {u.get('cost_usd', 0):.3f} | "
                f"{u.get('latency_s', 0) / 60:.0f} min | {'yes' if rep.get('peak') else 'no'} |")
        lines += ["", "Modelled at the published per-token prices. The account balance moved less than this",
                  "model predicts (see README-REPRO.md): these are estimates, not verified bills or guaranteed upper bounds. Whole-run totals include memory construction and all conversations; per-question columns use only the selected questions."]
        write_text(args.tokens_md, "\n".join(lines) + "\n")
        print(f"wrote {args.tokens_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
