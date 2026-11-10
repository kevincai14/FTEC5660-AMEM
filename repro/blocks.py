"""Count what one question actually sends to the LLM.

The efficiency argument rests on 51.4 note blocks per question, only 32.1 of them distinct. That
number decides whether A-Mem's "2,520 tokens" is reachable, so it should be recountable by
the reader rather than taken on trust. Two modes:

    python repro/blocks.py --prompt path/to/one_prompt.txt      # a saved prompt
    python repro/blocks.py --log logs/eval_..._conversation.log # every prompt in a run

Run logs are large (27 MB for the 304-question arm) and are not committed, so each arm stores the
counting for every question in `runs/<arm>/blocks.json`; those rows are what the means printed
below are computed from, and `--json` regenerates the file.

A *block* is one `talk start time:` segment, which `find_related_memories_raw` emits per
note: the retrieved note itself plus one per expanded link. Two blocks are the same note if
their text is identical, which is what makes the duplicate share meaningful.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")
SPLIT = "talk start time:"
ANSWER_TAIL = re.compile(r"\n\s*Question: .*$", re.S)


def blocks_in(prompt: str) -> list[str]:
    """Split one prompt into note blocks, dropping the question that follows them."""
    body = ANSWER_TAIL.sub("", prompt)
    parts = [p.strip() for p in body.split(SPLIT)[1:]]
    return [p for p in parts if "memory content:" in p]


def note_key(block: str) -> str:
    """Two blocks are the same note when the conversation text matches.

    Keying on the whole block instead would count a note twice if its context or keywords
    were rewritten between the two retrievals; keying on the content is the conservative
    choice — it gives the *smaller* duplicate share of the two.
    """
    after = block.split("memory content:", 1)
    if len(after) < 2:
        return block
    return after[1].split("memory context:", 1)[0].strip()


def stats(prompts: list[str]) -> dict:
    rows: list[dict] = []
    for prompt in prompts:
        blocks = blocks_in(prompt)
        if not blocks:
            continue
        distinct = len({note_key(b) for b in blocks})
        rows.append(
            {
                "blocks": len(blocks),
                "distinct": distinct,
                "duplicate_share": (len(blocks) - distinct) / len(blocks),
                "chars": len(prompt),
                "chars_per_block": len(prompt) / len(blocks),
            }
        )
    if not rows:
        return {}
    return {
        "per_question": rows,
        "questions": len(rows),
        "blocks": statistics.mean(r["blocks"] for r in rows),
        "distinct": statistics.mean(r["distinct"] for r in rows),
        "duplicate_share": statistics.mean(r["duplicate_share"] for r in rows),
        "chars": statistics.mean(r["chars"] for r in rows),
        "chars_median": statistics.median(r["chars"] for r in rows),
        "chars_per_block": statistics.mean(r["chars_per_block"] for r in rows),
        "blocks_median": statistics.median(r["blocks"] for r in rows),
        "blocks_range": (min(r["blocks"] for r in rows), max(r["blocks"] for r in rows)),
    }


def from_log(path: Path) -> list[str]:
    """Every `User Prompt:` block in a run log, in order."""
    prompts, current = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if current is not None:
            if STAMP.match(line) or "User Prompt:" in line:
                prompts.append("\n".join(current))
                current = None
            else:
                current.append(line)
                continue
        if "User Prompt:" in line:
            current = [line.split("User Prompt:", 1)[1]]
    if current:
        prompts.append("\n".join(current))
    return prompts


def report(kind: str, s: dict) -> None:
    if not s:
        sys.exit(f"no note blocks found in {kind}")
    print(f"{kind}: {s['questions']} prompt(s)")
    print(f"  blocks/question   {s['blocks']:.1f}  (median {s['blocks_median']:.0f}, "
          f"range {s['blocks_range'][0]}–{s['blocks_range'][1]})")
    print(f"  distinct notes    {s['distinct']:.1f}")
    print(f"  duplicate share   {s['duplicate_share'] * 100:.1f}%")
    print(f"  characters        {s['chars']:,.0f}  (median {s['chars_median']:,.0f}, "
          f"{s['chars_per_block']:.0f} per block)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=Path, help="a saved prompt: one User Prompt block from a run log")
    g.add_argument("--log", type=Path, help="a run log; every User Prompt in it is counted")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the per-question counts, so the mean can be re-derived later")
    args = ap.parse_args()

    if args.prompt:
        text = args.prompt.read_text(encoding="utf-8")
        source, s = str(args.prompt), stats([text])
    else:
        source, s = str(args.log), stats(from_log(args.log))

    report(source, s)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": Path(source).name, **{k: v for k, v in s.items() if k != "per_question"},
                   "per_question": s["per_question"]}
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
