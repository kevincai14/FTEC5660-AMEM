#!/usr/bin/env python
"""Cheap unit-cost probe: build a few memories, ask a few questions, measure.

Answers one question before committing money to an arm: how much does one turn of
memory construction and one answered question cost, and does the pipeline produce
parseable notes and non-empty answers on this stack (Windows, Py3.12, DeepSeek)?

  python repro/smoke.py --turns 20 --questions 10 --budget-usd 0.15
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repro.net import configure_network, quiet_redundant_downloads, ensure_nltk  # noqa: E402

NET = configure_network()
quiet_redundant_downloads()   # upstream utils.py re-downloads at import; see net.py

from repro import pricing, telemetry  # noqa: E402
from repro.run import DATASET, load_env  # noqa: E402


def pick_questions(sample, n: int):
    """Round-robin over categories so a small probe still touches all five."""
    buckets = {}
    for qa in sample.qa:
        buckets.setdefault(int(qa.category), []).append(qa)
    picked, i = [], 0
    while len(picked) < n and any(buckets.values()):
        for cat in sorted(buckets):
            if buckets[cat] and len(picked) < n:
                picked.append(buckets[cat].pop(0))
        i += 1
        if i > 500:
            break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--budget-usd", type=float, default=0.15)
    args = ap.parse_args()

    env = load_env()
    out_dir = ROOT / "runs" / f"smoke_{args.turns}t_{args.questions}q"
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry.open_jsonl(out_dir / "telemetry_calls.jsonl")
    telemetry.LEDGER.budget_usd = args.budget_usd
    telemetry.patch_harness()

    from load_dataset import load_locomo_dataset
    from utils import score_prediction
    from llm_text_parsers import parse_plain_text_answer
    import test_advanced_robust as tar
    import random

    sample = load_locomo_dataset(str(DATASET))[0]
    agent = tar.RobustAdvancedMemAgent(args.model, "openai", args.retrieve_k, 0.5)

    print(f"model={args.model} base={env['base_url']} peak={pricing.is_peak(datetime.now(timezone.utc))}")
    turns = []
    for _, session in list(sample.conversation.sessions.items()):
        for turn in session.turns:
            if len(turns) >= args.turns:
                break
            turns.append((turn, session.date_time))
        if len(turns) >= args.turns:
            break

    t0 = time.perf_counter()
    before = telemetry.LEDGER.snapshot()
    for turn, date in turns:
        agent.add_memory(f"Speaker {turn.speaker}says : {turn.text}", time=date)
    after = telemetry.LEDGER.snapshot()
    mem_s = time.perf_counter() - t0
    n_notes = len(agent.memory_system.memories)
    kw = sum(len(n.keywords) for n in agent.memory_system.memories.values()) / max(1, n_notes)
    links = sum(len(n.links) for n in agent.memory_system.memories.values())
    print(f"memory: {n_notes} notes from {len(turns)} turns in {mem_s:.0f}s, "
          f"{after[0] - before[0]} calls, ${after[5] - before[5]:.4f}, "
          f"avg keywords/note {kw:.1f}, links {links}")

    picked = pick_questions(sample, args.questions)
    rows = []
    for qa in picked:
        b = telemetry.LEDGER.snapshot()
        pred, prompt, context = agent.answer_question(qa.question, qa.category, qa.final_answer,
                                                      random.Random(42))
        a = telemetry.LEDGER.snapshot()
        pred = parse_plain_text_answer(pred)
        ref, metrics = score_prediction(pred, qa)
        rows.append({"category": qa.category, "question": qa.question, "prediction": pred,
                     "reference": ref, "adversarial_answer": qa.adversarial_answer,
                     "f1": metrics["f1"], "bleu1": metrics["bleu1"],
                     "adv_correct": metrics.get("adversarial_correct"),
                     "prompt_tokens": a[1] - b[1], "cost_usd": round(a[5] - b[5], 6),
                     "context_chars": len(context)})
        print(f"  cat{qa.category} f1={metrics['f1']:.3f} ctx={len(context):5d}ch "
              f"in={a[1] - b[1]:5d}tok ${a[5] - b[5]:.4f} | {qa.question[:60]} -> {pred[:50]}")

    telemetry.close_jsonl()
    usage = telemetry.LEDGER.summary()
    n_turn_calls = after[0] - before[0]
    q_cost = sum(r["cost_usd"] for r in rows)
    per_turn = (after[5] - before[5]) / max(1, len(turns))
    per_q = q_cost / max(1, len(rows))
    qa_prompt = sum(r["prompt_tokens"] for r in rows) / max(1, len(rows))

    summary = {
        "model": args.model, "turns": len(turns), "questions": len(rows),
        "calls_per_turn": round(n_turn_calls / max(1, len(turns)), 2),
        "usd_per_turn": round(per_turn, 6),
        "usd_per_question": round(per_q, 6),
        "prompt_tokens_per_question": round(qa_prompt, 1),
        "seconds_per_turn": round(mem_s / max(1, len(turns)), 2),
        "usage": usage,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({**r, "q_index": i}, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps({
        "arm": out_dir.name, "status": "smoke", "model": args.model, "retrieve_k": args.retrieve_k,
        "dataset": {"path": "data/locomo10.json", "samples": 1, "turns": len(turns)},
        "usage": usage,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n--- unit costs ---")
    print(f"per turn     ${per_turn:.5f}  ({n_turn_calls / max(1, len(turns)):.2f} calls)")
    print(f"per question ${per_q:.5f}  ({qa_prompt:.0f} prompt tokens, k={args.retrieve_k})")
    full = 5872 * per_turn + 1986 * per_q
    print(f"extrapolated to all 10 conversations (5872 turns, 1986 qa): ${full:.2f} / CNY {full * pricing.USD_TO_CNY:.0f}")
    print(f"sample 0 only (419 turns, 199 qa): ${419 * per_turn + 199 * per_q:.2f}")
    print(f"total: {usage['calls']} calls, ${usage['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
