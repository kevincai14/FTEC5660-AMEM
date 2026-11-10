"""Token/cost telemetry for a harness whose LLM calls we do not control.

The authors' evaluation loop constructs its own LLM controllers deep inside
`RobustAgenticMemorySystem`, so instead of editing that code we wrap the one
place every call goes through: the OpenAI client's `chat.completions.create`.
`patch_harness()` installs two hooks:

  1. on `RobustLLMController.__init__`, instrument the client just created;
  2. on `RobustAdvancedMemAgent.answer_question`, measure the token delta of one
     question, so "tokens per question" (paper Table 1's Token Length) is known
     per question rather than per run.

Every call is appended to a JSONL file. `BudgetExceeded` is raised *after* a call
that crosses the budget, so a run can never silently overspend by more than one
call.
"""
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import pricing


class BudgetExceeded(BaseException):
    """BaseException on purpose: the harness retries on `Exception`, and a budget stop
    must not be retried into three more paid calls."""


class Ledger:
    """Cumulative usage for one process, plus the per-question deltas."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.latency_s = 0.0
        self.cost_usd = 0.0
        self.current_stage = "setup"
        self.model = None
        self.per_question = []
        self.jsonl = None
        self.budget_usd: Optional[float] = None
        self.by_stage = {}

    # -- stage tagging -----------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        previous = self.current_stage
        self.current_stage = name
        try:
            yield
        finally:
            self.current_stage = previous

    def snapshot(self):
        with self.lock:
            return (self.calls, self.prompt_tokens, self.completion_tokens,
                    self.cache_hit_tokens, self.cache_miss_tokens, self.cost_usd)

    # -- recording ---------------------------------------------------------
    def record(self, prompt: int, completion: int, hit: int, miss: int,
               latency: float, model: str, when: datetime):
        cost = pricing.cost_usd(model, hit, miss, completion, when)
        with self.lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cache_hit_tokens += hit
            self.cache_miss_tokens += miss
            self.latency_s += latency
            self.cost_usd += cost
            bucket = self.by_stage.setdefault(self.current_stage, {"calls": 0, "prompt": 0, "completion": 0, "cost_usd": 0.0})
            bucket["calls"] += 1
            bucket["prompt"] += prompt
            bucket["completion"] += completion
            bucket["cost_usd"] += cost
            if self.jsonl is not None:
                self.jsonl.write(json.dumps({
                    "ts": when.isoformat(), "stage": self.current_stage, "model": model,
                    "prompt_tokens": prompt, "completion_tokens": completion,
                    "cache_hit_tokens": hit, "cache_miss_tokens": miss,
                    "latency_s": round(latency, 3), "cost_usd": round(cost, 8),
                }) + "\n")
                self.jsonl.flush()
            over = self.budget_usd is not None and self.cost_usd > self.budget_usd
        if over:
            raise BudgetExceeded(
                f"spent ${self.cost_usd:.4f} (> ${self.budget_usd:.4f} budget) after {self.calls} calls"
            )

    def summary(self) -> dict:
        with self.lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cache_hit_tokens": self.cache_hit_tokens,
                "cache_miss_tokens": self.cache_miss_tokens,
                "latency_s": round(self.latency_s, 1),
                "cost_usd": round(self.cost_usd, 6),
                "cost_cny": round(self.cost_usd * pricing.USD_TO_CNY, 4),
                "by_stage": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.by_stage.items()},
            }


LEDGER = Ledger()

# Injected into every chat.completions call. DeepSeek enables thinking mode by default
# (effort "high"), which on this harness has three consequences: reasoning tokens are
# billed as output, they compete with the paper's max_tokens=1000 cap (we observed
# truncated answers), and thinking mode *ignores* temperature, so the per-category
# temperatures in test_advanced_robust would silently do nothing. Disabling it makes the
# harness behave as written; the choice is recorded in every manifest.
LLM_EXTRA_BODY = {"thinking": {"type": "disabled"}}


def open_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.jsonl = open(path, "w", encoding="utf-8")
    return LEDGER.jsonl


def close_jsonl():
    if LEDGER.jsonl is not None:
        LEDGER.jsonl.close()
        LEDGER.jsonl = None


def _instrument_controller(controller):
    client = getattr(controller, "client", None)
    if client is None or getattr(client, "_amem_instrumented", False):
        return
    completions = client.chat.completions
    create = completions.create

    def wrapped(*args, **kwargs):
        if LLM_EXTRA_BODY:
            body = dict(kwargs.get("extra_body") or {})
            body.update(LLM_EXTRA_BODY)
            kwargs["extra_body"] = body
        started = time.perf_counter()
        response = create(*args, **kwargs)
        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        # DeepSeek splits prompt tokens into cached/uncached; OpenAI merely reports a
        # total, so all of it counts as a miss there (an over-estimate, never under).
        hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(usage, "prompt_cache_miss_tokens", prompt - hit) or 0)
        model = getattr(response, "model", None) or getattr(controller, "model", "unknown")
        LEDGER.model = model
        LEDGER.record(prompt, completion, hit, miss, latency, model, datetime.now(timezone.utc))
        return response

    completions.create = wrapped
    client._amem_instrumented = True


def patch_harness(disable_bert: bool = True):
    """Install the hooks on the authors' classes (no upstream edits involved).

    `disable_bert` swaps out `utils.calculate_bert_scores` for a zero-returning stub.
    BERTScore loads roberta-large (~1.4 GB) and runs it on CPU for every question;
    it is an appendix metric, while F1 and BLEU-1 are what the paper's Table 1
    reports, so the run keeps the metrics it needs and drops the expensive one.
    The substitution is recorded in the manifest.
    """
    import memory_layer_robust as mlr
    import test_advanced_robust as tar
    import utils as metrics_utils

    if disable_bert:
        metrics_utils.calculate_bert_scores = lambda *a, **k: {
            "bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    original_init = mlr.RobustLLMController.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _instrument_controller(self.llm)

    mlr.RobustLLMController.__init__ = patched_init

    # RobustAdvancedMemAgent built its controllers through the factory, which was
    # already replaced above; controllers created before this point are unreachable,
    # so patch first, then import the agent.

    original_answer = tar.RobustAdvancedMemAgent.answer_question
    counter = {"n": 0}

    def patched_answer(self, question, category, answer, rng=None):
        index = counter["n"]
        counter["n"] += 1
        before = LEDGER.snapshot()
        try:
            with LEDGER.stage("qa"):
                return original_answer(self, question, category, answer, rng)
        finally:
            after = LEDGER.snapshot()
            LEDGER.per_question.append({
                "q_index": index, "category": category,
                "calls": after[0] - before[0],
                "prompt_tokens": after[1] - before[1],
                "completion_tokens": after[2] - before[2],
                "cache_hit_tokens": after[3] - before[3],
                "cost_usd": round(after[5] - before[5], 8),
            })

    tar.RobustAdvancedMemAgent.answer_question = patched_answer

    # memory construction happens outside answer_question: tag it
    original_add = mlr.RobustAgenticMemorySystem.add_note

    def patched_add(self, *args, **kwargs):
        with LEDGER.stage("memory"):
            return original_add(self, *args, **kwargs)

    mlr.RobustAgenticMemorySystem.add_note = patched_add
    return mlr, tar
