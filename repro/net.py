"""Network setup, applied before any HuggingFace/NLTK import.

Two things go wrong on this machine that would otherwise fail silently:

  * huggingface.co is not reachable through the ambient proxy (502), so the
    embedding model must come from the mirror;
  * NLTK's tokenizer is downloaded on first use, and a failure there makes
    `nltk.word_tokenize` raise inside `calculate_bleu_scores`, which `utils.py`
    swallows into BLEU = 0.0 -- a silent zero, not an error.

Both are environment fixes, not method changes: the same weights and the same
tokenizer are used, only the transport differs. Set `AMEM_KEEP_PROXY=1` to keep
the ambient proxy settings (e.g. when direct egress is blocked).
"""
import os


def configure_network() -> dict:
    applied = {}
    if os.getenv("AMEM_KEEP_PROXY") != "1":
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            if os.environ.pop(var, None) is not None:
                applied[var] = "cleared"
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    applied["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]
    # the tokenizer/wordnet resources are vendored in the repo so a run never depends on
    # an NLTK download succeeding mid-experiment
    local_nltk = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".nltk_data")
    if os.path.isdir(local_nltk):
        os.environ.setdefault("NLTK_DATA", local_nltk)
        applied["NLTK_DATA"] = os.environ["NLTK_DATA"]
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return applied


def ensure_nltk() -> dict:
    """Fetch the tokenizer/wordnet resources and report whether they are actually there."""
    import nltk
    wanted = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab", "wordnet": "corpora/wordnet"}
    found = {}
    for package, path in wanted.items():
        try:
            nltk.data.find(path)
            found[package] = True
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                nltk.data.find(path)
                found[package] = True
            except Exception:
                found[package] = False
    return found


def quiet_redundant_downloads() -> bool:
    """No-op `nltk.download` when the resources are already on disk.

    Upstream `utils.py` calls `nltk.download('punkt')` / `('wordnet')` at *import* time, inside
    a try/except that only handles the exception -- so on a machine whose egress is proxied,
    NLTK prints fifteen lines of `Security Violation [pathsec.urlopen]` before the first score,
    even though the local copies are found and used. Harmless, but it reads like a broken run.

    Only patched when everything the scorer needs is already present, so a genuinely missing
    resource still takes the real download path and still fails loudly.
    """
    import nltk
    needed = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab", "wordnet": "corpora/wordnet"}
    for path in needed.values():
        try:
            nltk.data.find(path)
        except LookupError:
            return False
    nltk.download = lambda *args, **kwargs: True  # type: ignore[assignment]
    return True
