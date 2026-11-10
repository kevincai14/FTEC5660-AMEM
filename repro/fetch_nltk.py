"""Fetch the NLTK resources the scorer needs, into the repo-local `.nltk_data/`.

Why this exists as a script rather than a line in the README: the failure mode is silent.
`calculate_bleu_scores` in `utils.py` wraps tokenisation in `except Exception: score = 0.0`,
so a missing `punkt_tab` does not error -- it reports BLEU-1 as 0.0 for every question and
looks like a scientific finding. Running one command that *verifies* the resources are
present is the difference between a bug and a measurement.

    python repro/fetch_nltk.py

Writes to `.nltk_data/` at the repo root (gitignored: 174 MB of corpora), which
`repro/net.py` then puts on `NLTK_DATA` automatically. If the download fails behind a
proxy that NLTK refuses to use, the message says so instead of pretending to succeed.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repro.net import configure_network, ensure_nltk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".nltk_data"
PACKAGES = ("punkt", "punkt_tab", "wordnet", "omw-1.4")


def main() -> int:
    applied = configure_network()
    print(f"proxy state : {applied.get('HTTPS_PROXY', 'untouched (AMEM_KEEP_PROXY=1)')}")
    LOCAL.mkdir(exist_ok=True)
    os.environ["NLTK_DATA"] = str(LOCAL)

    import nltk

    for package in PACKAGES:
        print(f"fetching    : {package}", end="  ", flush=True)
        try:
            nltk.data.find(f"tokenizers/{package}")
            print("already present")
            continue
        except LookupError:
            pass
        try:
            nltk.data.find(f"corpora/{package}")
            print("already present")
            continue
        except LookupError:
            pass
        ok = nltk.download(package, download_dir=str(LOCAL), quiet=True)
        print("ok" if ok else "FAILED (offline?)")

    found = ensure_nltk()
    print()
    for package, ok in found.items():
        print(f"  {package:<10} {'present' if ok else 'MISSING'}")
    if not all(found.values()):
        print(
            "\nSome resources could not be fetched. In order of preference:\n"
            "\n"
            "  1. If you are behind a proxy, NLTK 3.10 refuses proxied downloads by default\n"
            "     ('Security Violation [pathsec.urlopen]'). Re-run with\n"
            "         NLTK_ALLOW_PROXIED_URLOPEN=1 AMEM_KEEP_PROXY=1 python repro/fetch_nltk.py\n"
            "  2. Otherwise fetch the four zips through a working client and unzip them into\n"
            "     .nltk_data/ (each zip unpacks to tokenizers/ or corpora/):\n"
            "         https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip\n"
            "         https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip\n"
            "         https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip\n"
            "         https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip\n"
            "  3. Or point NLTK_DATA at a directory that already has them.\n"
            "\n"
            "Do not skip this and run anyway: `utils.calculate_bleu_scores` swallows the\n"
            "LookupError, so BLEU-1 reads 0.00 for every question instead of failing.",
            file=sys.stderr,
        )
        return 1

    print(f"\nok -- {LOCAL} ({sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file()) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
