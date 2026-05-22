from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def word_freq() -> dict[str, int]:
    """Lowercased word -> total count across Brown + Reuters + Webtext.

    Lazy + cached: first call downloads NLTK data and builds the FreqDist
    (a few seconds); subsequent calls are free.
    """
    import nltk
    from itertools import chain

    for c in ("brown", "reuters", "webtext"):
        nltk.download(c, quiet=True)

    from nltk.corpus import brown, reuters, webtext
    from nltk.probability import FreqDist

    fdist = FreqDist(
        w.lower()
        for w in chain(brown.words(), reuters.words(), webtext.words())
        if w.isalpha()
    )
    return dict(fdist)
