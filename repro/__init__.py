"""Reproducibility harness for A-Mem (NeurIPS 2025, arXiv:2502.12110) on LoCoMo.

Nothing here reimplements the method: `run.py` calls the authors' own
`test_advanced_robust.evaluate_dataset`, which builds memories with
`RobustAgenticMemorySystem` and answers questions through the paper's prompts.
This package only adds the things a reproduction needs and the upstream code
does not have:

  pricing.py    -- a cost model for the LLM actually used (DeepSeek), with peak/off-peak
  telemetry.py  -- per-call token/latency accounting, per-question deltas, budget hard stop
  run.py        -- runs one "arm" (one configuration) and writes a manifest + predictions
  score.py      -- recomputes F1/BLEU from stored predictions under both category-5 conventions
  check_env.py  -- offline self-check: versions, dataset hash, endpoint, cost estimate
"""
