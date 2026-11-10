#!/usr/bin/env python
"""Run one arm of the A-Mem reproduction on LoCoMo.

An "arm" is one configuration (model, retrieve_k, memory scope, seed). The script
wraps the authors' `evaluate_dataset` and adds what a reproduction needs: a
manifest that pins every input, a hard budget stop, and per-question token
accounting.

  python repro/run.py --dry-run                       # plan + estimate, no API calls
  python repro/run.py --samples 1 --retrieve-k 10 --tag pilot --budget-usd 1.5
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
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

DATASET = ROOT / "data" / "locomo10.json"
EXPECTED_DATASET_SHA = "553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365"
PROMPT_FILES = ["llm_text_parsers.py", "memory_layer_robust.py", "test_advanced_robust.py", "utils.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def load_env():
    """Read .env without touching the process environment of the caller."""
    env_file = ROOT / ".env"
    values = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    key = values.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("no DEEPSEEK_API_KEY in .env or environment")
    base = values.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    # the OpenAI SDK reads these two, and the harness builds a plain OpenAI client
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = base
    return {"base_url": base, "key_present": True}


def dataset_path(args) -> Path:
    return (ROOT / args.dataset).resolve()


def plan(args) -> dict:
    data = json.loads(dataset_path(args).read_text(encoding="utf-8"))
    samples = data[: max(1, min(args.samples, len(data)))]
    qa = sum(len(s["qa"]) for s in samples)
    turns = sum(len(v) for s in samples for k, v in s["conversation"].items()
                if k.startswith("session_") and isinstance(v, list))
    from collections import Counter
    cats = Counter(q.get("category") for s in samples for q in s["qa"])
    return {"samples": len(samples), "qa": qa, "turns": turns,
            "categories": dict(sorted(cats.items()))}


def estimate(pl: dict, per_turn_usd: float, per_question_usd: float) -> dict:
    low = pl["turns"] * per_turn_usd + pl["qa"] * per_question_usd
    return {"usd": round(low, 3), "cny": round(low * pricing.USD_TO_CNY, 2),
            "note": "x1.0 if run off-peak, x2.0 if run in 01-04/06-10 UTC on a weekday"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--dataset", default="data/locomo10.json",
                    help="relative to the repo root; a smaller file is only for wiring tests")
    ap.add_argument("--samples", type=int, default=1, help="first N LoCoMo conversations (upstream --ratio is N/10)")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature-c5", type=float, default=0.5)
    ap.add_argument("--memory-scope", choices=["conversation", "pooled"], default="conversation")
    ap.add_argument("--refresh-index", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate-per-turn-usd", type=float, default=None,
                    help="from a previous run's summary.json, for the cost estimate")
    ap.add_argument("--estimate-per-question-usd", type=float, default=None)
    args = ap.parse_args()

    stamped = datetime.now(timezone.utc).astimezone()
    env = load_env() if not args.dry_run else {"base_url": "https://api.deepseek.com", "key_present": False}
    pl = plan(args)
    pl["peak_now"] = pricing.is_peak(stamped)
    arm = f"{args.tag}_k{args.retrieve_k}_s{args.samples}_{args.model}_seed{args.seed}"
    run_dir = ROOT / "runs" / arm

    print(f"arm      {arm}")
    print(f"model    {args.model} @ {env['base_url']}")
    print(f"samples  {pl['samples']} conversations, {pl['qa']} questions, {pl['turns']} turns "
          f"{pl['categories']}")
    print(f"prices   {'PEAK' if pl['peak_now'] else 'off-peak'} right now "
          f"({stamped.isoformat(timespec='minutes')})")
    if args.estimate_per_turn_usd is not None:
        print(f"estimate {estimate(pl, args.estimate_per_turn_usd, args.estimate_per_question_usd or 0)}")
    if args.dry_run:
        print(f"would write {run_dir.relative_to(ROOT)}")
        return 0

    data_file = dataset_path(args)
    if data_file.name == "locomo10.json" and sha256(data_file) != EXPECTED_DATASET_SHA:
        raise SystemExit("dataset hash changed: refusing to compare arms across datasets")

    telemetry.open_jsonl(run_dir / "telemetry_calls.jsonl")
    telemetry.LEDGER.budget_usd = args.budget_usd
    mlr, tar = telemetry.patch_harness()
    nltk_state = ensure_nltk()
    if not all(nltk_state.values()):
        print(f"warning: missing NLTK resources {nltk_state} — BLEU/METEOR would read 0", file=sys.stderr)

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_results.json"
    started = time.perf_counter()
    status, error = "ok", None
    try:
        results = tar.evaluate_dataset(
            str(data_file), args.model, str(raw_path), pl["samples"] / 10.0, "openai",
            args.temperature_c5, args.retrieve_k, "http://localhost", 30000,
            False, args.seed, args.refresh_index, args.memory_scope,
        )
    except telemetry.BudgetExceeded as e:
        status, error = "budget_exceeded", str(e)
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        results = None
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
        results = None
    except Exception as e:  # keep the telemetry that was paid for
        status, error = "error", f"{type(e).__name__}: {e}"
        print(f"\nFAILED: {error}", file=sys.stderr)
        results = None
    finally:
        telemetry.close_jsonl()
        elapsed = round(time.perf_counter() - started, 1)

    usage = telemetry.LEDGER.summary()
    manifest = {
        "arm": arm,
        "status": status,
        "error": error,
        "started": stamped.isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "peak_start": pricing.is_peak(stamped),
        "model": args.model,
        "base_url": env["base_url"],
        "backend": "openai-compatible (upstream RobustOpenAIController)",
        "dataset": {"path": str(data_file.relative_to(ROOT)) if data_file.is_relative_to(ROOT) else str(data_file),
                    "sha256": sha256(data_file), "samples": pl["samples"],
                    "qa": pl["qa"], "turns": pl["turns"], "categories": pl["categories"]},
        "params": {"retrieve_k": args.retrieve_k, "seed": args.seed, "temperature_c5": args.temperature_c5,
                   "memory_scope": args.memory_scope, "refresh_index": args.refresh_index,
                   "evo_threshold": 100, "max_tokens_per_call": 1000, "embedding": "all-MiniLM-L6-v2"},
        "prompts_sha256": {f: sha256(ROOT / f) for f in PROMPT_FILES},
        "upstream": {"base_commit": git("rev-list", "--max-parents=0", "HEAD"),
                     "base_subject": git("log", "--max-parents=0", "--format=%s"),
                     "head": git("log", "-1", "--format=%H"),
                     "head_subject": git("log", "-1", "--format=%s"),
                     "dirty": bool(git("status", "--porcelain"))},
        "environment": {"python": sys.version.split()[0],
                        "packages": {p: (md.version(p) if _has(p) else None)
                                     for p in ("torch", "transformers", "sentence-transformers", "openai")},
                        "network": NET, "nltk": nltk_state,
                        "bert_score": "disabled (stubbed to 0; appendix metric, roberta-large skipped)"},
        "usage": usage,
        "budget_usd": args.budget_usd,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if results:
        rows = results["individual_results"]
        per_q = {(r["q_index"]): r for r in telemetry.LEDGER.per_question}
        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({**r, **per_q.get(i, {})}, ensure_ascii=False) + "\n")
        print(f"\nwrote {raw_path.relative_to(ROOT)}, predictions.jsonl, manifest.json")

    print("\nusage:", json.dumps({k: usage[k] for k in ("calls", "prompt_tokens", "completion_tokens",
                                                      "cache_hit_tokens", "cost_usd", "cost_cny", "latency_s")}))
    print("stages:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'prompt'}
                                 for k, v in usage["by_stage"].items()}))
    print(f"elapsed {elapsed}s  status={status}")
    return 0 if status == "ok" else 3


def _has(pkg: str) -> bool:
    try:
        md.version(pkg)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
