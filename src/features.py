import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_RE = re.compile(r"[a-z0-9]+")
FEATURE_NAMES = [
    "word_tfidf_cosine",
    "char_tfidf_cosine",
    "query_token_coverage",
    "token_jaccard",
    "exact_query_in_title",
    "brand_in_query",
    "color_in_query",
    "numeric_model_token_overlap",
    "query_length",
    "title_length",
    "length_ratio",
    "missing_brand",
    "missing_color",
]
TFIDF_FEATURE_NAMES = ["word_tfidf_cosine", "char_tfidf_cosine"]
HANDCRAFTED_FEATURE_NAMES = FEATURE_NAMES[2:]
FEATURE_SETS = {
    "tfidf_2": TFIDF_FEATURE_NAMES,
    "all_13": FEATURE_NAMES,
}


def normalize_text(value):
    return " ".join(TOKEN_RE.findall(str(value).lower()))


def tokenize(value):
    return TOKEN_RE.findall(str(value).lower())


def fit_feature_bundle(train_frame, tfidf_config):
    unique_queries = train_frame["query"].fillna("").astype(str).drop_duplicates()
    unique_titles = train_frame["product_title"].fillna("").astype(str).drop_duplicates()
    fit_corpus = pd.concat([unique_queries, unique_titles], ignore_index=True).drop_duplicates().tolist()
    if not fit_corpus:
        raise ValueError("Cannot fit TF-IDF on an empty training corpus")

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=tuple(tfidf_config["word_ngram_range"]),
        min_df=int(tfidf_config["min_df"]),
        max_features=int(tfidf_config["word_max_features"]),
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=tuple(tfidf_config["char_ngram_range"]),
        min_df=int(tfidf_config["min_df"]),
        max_features=int(tfidf_config["char_max_features"]),
        dtype=np.float32,
    )
    word_vectorizer.fit(fit_corpus)
    char_vectorizer.fit(fit_corpus)
    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "feature_names": FEATURE_NAMES,
        "fit_corpus_size": len(fit_corpus),
        "unique_training_queries": int(unique_queries.nunique()),
        "unique_training_titles": int(unique_titles.nunique()),
    }


def _rowwise_tfidf_cosine(vectorizer, queries, titles):
    query_values = pd.Series(queries, dtype=str).fillna("").to_numpy()
    title_values = pd.Series(titles, dtype=str).fillna("").to_numpy()
    unique_queries, query_codes = np.unique(query_values, return_inverse=True)
    unique_titles, title_codes = np.unique(title_values, return_inverse=True)
    query_matrix = vectorizer.transform(unique_queries)[query_codes]
    title_matrix = vectorizer.transform(unique_titles)[title_codes]
    return np.asarray(query_matrix.multiply(title_matrix).sum(axis=1)).ravel().astype(np.float32)


def transform_features(frame, bundle, feature_names=None):
    requested = list(FEATURE_NAMES if feature_names is None else feature_names)
    unknown = set(requested) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown feature name(s): {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one feature must be requested")

    queries = frame["query"].fillna("").astype(str)
    titles = frame["product_title"].fillna("").astype(str)
    values = {}
    if "word_tfidf_cosine" in requested:
        values["word_tfidf_cosine"] = _rowwise_tfidf_cosine(
            bundle["word_vectorizer"], queries, titles
        )
    if "char_tfidf_cosine" in requested:
        values["char_tfidf_cosine"] = _rowwise_tfidf_cosine(
            bundle["char_vectorizer"], queries, titles
        )

    if set(requested) & set(HANDCRAFTED_FEATURE_NAMES):
        brands = (
            frame.get("product_brand", pd.Series("", index=frame.index))
            .fillna("")
            .astype(str)
        )
        colors = (
            frame.get("product_color", pd.Series("", index=frame.index))
            .fillna("")
            .astype(str)
        )
        rows = []
        for query, title, brand, color in zip(queries, titles, brands, colors):
            query_tokens = tokenize(query)
            title_tokens = tokenize(title)
            query_set = set(query_tokens)
            title_set = set(title_tokens)
            union = query_set | title_set
            query_model_tokens = {
                token for token in query_set if any(char.isdigit() for char in token)
            }
            query_norm = " ".join(query_tokens)
            title_norm = " ".join(title_tokens)
            brand_norm = normalize_text(brand)
            color_norm = normalize_text(color)
            rows.append(
                (
                    len(query_set & title_set) / len(query_set) if query_set else 0.0,
                    len(query_set & title_set) / len(union) if union else 0.0,
                    float(bool(query_norm) and f" {query_norm} " in f" {title_norm} "),
                    float(bool(brand_norm) and f" {brand_norm} " in f" {query_norm} "),
                    float(bool(color_norm) and f" {color_norm} " in f" {query_norm} "),
                    (
                        len(query_model_tokens & title_set) / len(query_model_tokens)
                        if query_model_tokens
                        else 0.0
                    ),
                    len(query_tokens),
                    len(title_tokens),
                    len(title_tokens) / max(len(query_tokens), 1),
                    float(not brand.strip()),
                    float(not color.strip()),
                )
            )
        handcrafted = np.asarray(rows, dtype=np.float32).reshape(
            len(rows), len(HANDCRAFTED_FEATURE_NAMES)
        )
        for index, name in enumerate(HANDCRAFTED_FEATURE_NAMES):
            values[name] = handcrafted[:, index]

    return pd.DataFrame({name: values[name] for name in requested}, index=frame.index)


def combined_lexical_score(features, word_weight):
    return (
        float(word_weight) * features["word_tfidf_cosine"].to_numpy()
        + (1.0 - float(word_weight)) * features["char_tfidf_cosine"].to_numpy()
    )
