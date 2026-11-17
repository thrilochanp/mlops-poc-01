"""Feature extraction shared by train.py and evaluate.py / serving.

Kept as a thin wrapper so the exact same vectorizer object that was fit
during training is what scores requests at inference time (loaded back out
of the persisted sklearn Pipeline — see train.py, which bundles the
vectorizer and classifier into a single Pipeline before logging to MLflow,
so there is no train/serve skew from re-fitting features differently).
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
