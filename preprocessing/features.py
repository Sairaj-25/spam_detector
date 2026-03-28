# preprocessing/features.py

import re
import numpy as np
from typing import List

class MetadataFeatureExtractor:
    """
    Extracts 10 hand-crafted features from RAW email text.
    
    WHY RAW text? Because cleaning destroys these signals:
    - Cleaning removes "!!!" → we lose exclamation count
    - Cleaning removes URLs → we lose link count
    So we extract metadata BEFORE cleaning.
    """

    URGENT_KEYWORDS = [
        "urgent", "immediately", "act now", "limited time",
        "expires", "deadline", "last chance", "final notice",
        "respond now", "important notice", "don't delay",
        "winner", "won", "free", "prize", "claim", "guaranteed",
        "congratulations", "selected", "lucky"
    ]

    def extract(self, raw_text: str) -> np.ndarray:
        """Returns a 10-dimensional feature vector."""
        if not raw_text or not isinstance(raw_text, str):
            return np.zeros(10, dtype=np.float32)

        features = [
            self._count_links(raw_text),           # f1
            self._has_urgent_keywords(raw_text),   # f2
            self._special_char_ratio(raw_text),    # f3
            self._uppercase_ratio(raw_text),       # f4
            self._exclamation_count(raw_text),     # f5
            self._dollar_sign_count(raw_text),     # f6
            self._html_tag_count(raw_text),        # f7
            self._word_count(raw_text),            # f8
            self._avg_word_length(raw_text),       # f9
            self._number_ratio(raw_text),          # f10
        ]
        return np.array(features, dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        return [
            "link_count", "has_urgent_keyword", "special_char_ratio",
            "uppercase_ratio", "exclamation_count", "dollar_sign_count",
            "html_tag_count", "word_count", "avg_word_length", "number_ratio"
        ]

    @property
    def num_features(self) -> int:
        return 10

    def _count_links(self, text: str) -> float:
        return float(len(re.findall(r'http[s]?://', text)))

    def _has_urgent_keywords(self, text: str) -> float:
        text_lower = text.lower()
        return float(any(kw in text_lower for kw in self.URGENT_KEYWORDS))

    def _special_char_ratio(self, text: str) -> float:
        if not text: return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return round(special / max(len(text), 1), 4)

    def _uppercase_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters: return 0.0
        return round(sum(1 for c in letters if c.isupper()) / len(letters), 4)

    def _exclamation_count(self, text: str) -> float:
        return float(text.count('!'))

    def _dollar_sign_count(self, text: str) -> float:
        return float(text.count('$'))

    def _html_tag_count(self, text: str) -> float:
        return float(len(re.findall(r'<[^>]+>', text)))

    def _word_count(self, text: str) -> float:
        return float(len(text.split()))

    def _avg_word_length(self, text: str) -> float:
        words = text.split()
        if not words: return 0.0
        return round(sum(len(w) for w in words) / len(words), 4)

    def _number_ratio(self, text: str) -> float:
        if not text: return 0.0
        digits = sum(1 for c in text if c.isdigit())
        return round(digits / max(len(text), 1), 4)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = MetadataFeatureExtractor()

    spam_email = """
    CONGRATULATIONS!!! You've WON $1,000,000!!!
    Click NOW: https://claim-prize.com https://win-money.net
    <html><b>ACT IMMEDIATELY</b></html>
    Limited time offer expires TODAY!!!
    """

    ham_email = """
    Hi John, please find attached the meeting notes from yesterday.
    We discussed the Q3 roadmap and agreed on the timeline.
    Let me know if you have questions.
    Thanks, Sarah
    """

    spam_features = extractor.extract(spam_email)
    ham_features  = extractor.extract(ham_email)

    print(f"{'Feature':<25} {'SPAM':>10} {'HAM':>10}")
    print("-" * 47)
    for name, sv, hv in zip(extractor.feature_names, spam_features, ham_features):
        print(f"{name:<25} {sv:>10.4f} {hv:>10.4f}")