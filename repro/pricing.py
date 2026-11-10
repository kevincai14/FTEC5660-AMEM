"""DeepSeek price model, used only to report what a run cost.

Prices are USD per 1M tokens, from https://api-docs.deepseek.com/quick_start/pricing
(read 2026-11-10). Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri; the
published page excludes Chinese public holidays from "peak", which this model
ignores (so a holiday run is over-estimated, never under-estimated).
"""
from datetime import datetime, timezone
from typing import Dict

# model -> (cache_hit, cache_miss, output) for off-peak and peak
PRICES: Dict[str, Dict[str, Dict[str, float]]] = {
    "deepseek-flash": {
        "off_peak": {"hit": 0.003, "miss": 0.15, "out": 0.60},
        "peak": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    },
    "deepseek-v4-pro": {
        "off_peak": {"hit": 0.022, "miss": 0.66, "out": 1.98},
        "peak": {"hit": 0.044, "miss": 1.32, "out": 3.96},
    },
}

USD_TO_CNY = 7.15  # reporting only; the balance is in CNY

# KNOWN LIMITATION, measured 2026-11-10: three arms of 3834 calls / 9.99M prompt tokens
# modelled to $2.28 (CNY 16.3) moved the account balance by only CNY 6.24. The implied
# blended rate (USD 0.087 per 1M prompt tokens) corresponds to ~40% prefix-cache hits,
# while the API's per-call `prompt_cache_hit_tokens` reported 15.3%. So this model is an
# upper bound, not a bill; the balance delta is quoted as the real cost.


def is_peak(when: datetime) -> bool:
    """True inside the two daily peak windows (UTC), Monday-Friday."""
    when = when.astimezone(timezone.utc)
    if when.weekday() >= 5:  # Sat/Sun are off-peak all day
        return False
    return 1 <= when.hour < 4 or 6 <= when.hour < 10


def rates(model: str, when: datetime) -> Dict[str, float]:
    table = PRICES.get(model)
    if table is None:
        raise KeyError(f"no price for model {model!r}; known: {sorted(PRICES)}")
    return table["peak" if is_peak(when) else "off_peak"]


def cost_usd(model: str, hit: int, miss: int, out: int, when: datetime) -> float:
    r = rates(model, when)
    return (hit * r["hit"] + miss * r["miss"] + out * r["out"]) / 1_000_000.0
