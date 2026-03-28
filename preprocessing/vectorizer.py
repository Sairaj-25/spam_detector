# preprocessing/vectorizer.py

import numpy as np
import math
from collections import Counter
from typing import List, Dict

class TFIDFVectorizer:
    """
    TF-IDF from scratch — no sklearn.

    INTUITION:
    - "free" appears in 8000/10000 spam emails → low IDF (not informative)
    - "viagra" appears in 500/10000 emails → high IDF (very informative)
    - TF-IDF = how often word appears in THIS email × how rare it is globally
    
    FORMULA:
    TF(word, doc)  = count(word in doc) / total_words_in_doc
    IDF(word)      = log((1 + N) / (1 + df(word))) + 1
    TF-IDF         = TF × IDF   (then L2 normalize the whole vector)
    """

    def __init__(self,
                 max_features: int   = 10000,
                 min_df:       int   = 2,
                 max_df:       float = 0.95):

        self.max_features = max_features
        self.min_df       = min_df
        self.max_df       = max_df

        # Learned during fit() — saved with pickle
        self.vocabulary_:    Dict[str, int]   = {}
        self.idf_:           Dict[str, float] = {}
        self.feature_names_: List[str]        = []
        self._is_fitted = False

    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        """Learn vocabulary + IDF from training docs ONLY."""
        print(f"  Fitting TF-IDF on {len(documents):,} documents...")

        n_docs = len(documents)

        # Count: how many documents contain each word?
        doc_freq = Counter()
        for doc in documents:
            unique_words = set(doc.split())
            for word in unique_words:
                doc_freq[word] += 1

        # Filter by min/max document frequency
        min_count = self.min_df
        max_count = int(self.max_df * n_docs)

        filtered = {
            word: count
            for word, count in doc_freq.items()
            if min_count <= count <= max_count
        }
        print(f"  Words after df filtering: {len(filtered):,}")

        # Sort by frequency, take top max_features
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_features]

        # Build vocabulary: word → index
        self.vocabulary_    = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.feature_names_ = [word for word, _ in sorted_words]

        # Compute IDF for each vocabulary word
        for word in self.vocabulary_:
            df = doc_freq[word]
            self.idf_[word] = math.log((1 + n_docs) / (1 + df)) + 1.0

        self._is_fitted = True
        print(f"  Vocabulary size: {len(self.vocabulary_):,}")
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Convert cleaned documents to TF-IDF matrix."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        n_docs     = len(documents)
        n_features = len(self.vocabulary_)
        matrix     = np.zeros((n_docs, n_features), dtype=np.float32)

        for doc_idx, doc in enumerate(documents):
            words = doc.split()
            if not words:
                continue

            word_counts = Counter(words)
            total_words = len(words)

            for word, count in word_counts.items():
                if word not in self.vocabulary_:
                    continue  # Unknown word — skip (CRITICAL for inference)

                word_idx = self.vocabulary_[word]
                tf  = (1 + math.log(count)) / total_words  # Sublinear TF
                idf = self.idf_[word]
                matrix[doc_idx, word_idx] = tf * idf

        # L2 normalize each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def transform_single(self, document: str) -> np.ndarray:
        """Transform one document. Used during inference."""
        return self.transform([document])[0]