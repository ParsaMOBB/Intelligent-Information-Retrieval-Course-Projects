# retrieval_models.py
from collections import defaultdict
import math

def bm25_score(query, inverted_index, k=1.5, b=0.75):
    """
    Compute BM25 scores for a given query against all documents in the inverted index.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - k: BM25 k parameter (default=1.5)
    - b: BM25 b parameter (default=0.75)

    Returns:
    - scores: Dictionary mapping doc_id to BM25 score
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  # Incorporate term weight
    return scores

def idf_overlap_score(query, inverted_index):
    """
    Method 1: Sum of IDF values for overlapping terms (no normalization).
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, _ in postings:
            scores[doc_id] += query_weight * idf
    return scores


def saturated_tf_score(query, inverted_index, k=1.5):
    """
    Method 2: Saturated TF weighting without IDF.
    """
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            score = (freq * (k + 1)) / (freq + k)
            scores[doc_id] += query_weight * score
    return scores


def bm11_score(query, inverted_index, k=1.5):
    """
    Method 3: BM11 variant (doc length normalization without b parameter).
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            denominator = k * (doc_len / avg_doc_len) + freq
            score = idf * (freq * (k + 1)) / denominator
            scores[doc_id] += query_weight * score
    return scores


def boolean_overlap_score(query, inverted_index):
    """
    Method 4: Boolean overlap (1 if the document contains the term).
    """
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, _ in postings:
            scores[doc_id] += query_weight
    return scores


def rare_term_emphasis_score(query, inverted_index, k=1.5, b=0.75):
    """
    Method 5: Emphasize rare terms by using IDF^2 and a softer TF normalization.
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = (idf ** 2) * freq / denominator
            scores[doc_id] += query_weight * score
    return scores


def bm25_plus_score(query, inverted_index, k=1.5, b=0.75, delta=0.5):
    """
    Method 6: BM25+ variant with additive delta floor.
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            norm = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            base = (freq * (k + 1)) / norm
            score = idf * (base + delta)
            scores[doc_id] += query_weight * score
    return scores


def pivoted_length_normalization(query, inverted_index, s=0.2):
    """
    Pivoted Length Normalization with nested log TF component.
    """
    scores = defaultdict(float)
    avg_doc_len = inverted_index.avg_doc_len or 1.0
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        if not postings:
            continue
        df = inverted_index.term_doc_freq.get(term, 0) or 1
        N = inverted_index.doc_count or 1
        idf = math.log((N + 1) / df)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            if freq <= 0:
                continue
            tf_weight = math.log(1 + math.log(1 + freq))
            length_norm = 1.0 / (1 - s + s * (inverted_index.doc_lengths[doc_id] / avg_doc_len))
            scores[doc_id] += query_weight * tf_weight * idf * length_norm
    return scores


def pivoted_length_normalization_simple(query, inverted_index, s=0.2):
    """
    Pivoted LN without the nested log TF term.
    """
    scores = defaultdict(float)
    avg_doc_len = inverted_index.avg_doc_len or 1.0
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        if not postings:
            continue
        df = inverted_index.term_doc_freq.get(term, 0) or 1
        N = inverted_index.doc_count or 1
        idf = math.log((N + 1) / df)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            if freq <= 0:
                continue
            tf_weight = math.log(1 + freq)
            length_norm = 1.0 / (1 - s + s * (inverted_index.doc_lengths[doc_id] / avg_doc_len))
            scores[doc_id] += query_weight * tf_weight * idf * length_norm
    return scores
