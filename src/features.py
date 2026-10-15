"""Text cleaning + feature engineering shared by training and the Flask app.

Kept in its own module so the pickled pipeline can be loaded anywhere.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
HTML_RE = re.compile(r"&\w+;")

# keep a few negations/short words out of the stop list: they carry signal in tweets
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never"}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/HTML entities/punctuation/digits, keep hashtag words."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)          # '#flood' -> 'flood'
    text = NON_ALPHA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, remove_stop: bool = True):
    """Whitespace tokenisation on cleaned text (drops 1-char tokens and stop-words)."""
    toks = [t for t in clean_text(text).split() if len(t) > 1]
    if remove_stop:
        toks = [t for t in toks if t not in STOP_WORDS]
    return toks


def preprocess(text: str) -> str:
    """Cleaned, stop-word-free string used as TF-IDF input."""
    return " ".join(tokenize(text))


class TextPrep(BaseEstimator, TransformerMixin):
    """DataFrame(text, keyword) -> Series of model-ready strings.

    The tweet's `keyword` (if present) is appended as an extra token so the model can use it.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        kw = X["keyword"].fillna("").astype(str).str.replace("%20", " ", regex=False)
        out = X["text"].astype(str).map(preprocess)
        return (out + " " + kw.str.lower()).str.strip().values


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted numeric features: length, hashtags, mentions, urls, caps ratio, etc."""

    names_ = [
        "n_chars", "n_words", "avg_word_len", "n_hashtags", "n_mentions",
        "has_url", "n_exclaim", "n_question", "caps_ratio", "has_keyword",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = X["text"].astype(str)
        words = t.str.split()
        n_chars = t.str.len()
        feats = pd.DataFrame({
            "n_chars": np.log1p(n_chars),
            "n_words": np.log1p(words.str.len()),
            "avg_word_len": words.map(lambda w: np.mean([len(x) for x in w]) if w else 0) / 10,
            "n_hashtags": t.str.count(r"#\w+"),
            "n_mentions": t.str.count(r"@\w+"),
            "has_url": t.str.contains(URL_RE).astype(int),
            "n_exclaim": t.str.count("!"),
            "n_question": t.str.count(r"\?"),
            "caps_ratio": t.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
            "has_keyword": X["keyword"].notna().astype(int) if "keyword" in X else 0,
        })
        return feats.values.astype(float)
