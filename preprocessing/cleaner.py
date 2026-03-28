# preprocessing/cleaner.py

import re
import html

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "as", "if", "then", "because", "while", "also", "about", "up",
    "out", "there", "here", "any", "been", "get", "got", "use", "used",
}

class EmailCleaner:
    """
    Cleans raw email text step by step.
    Each step removes noise so the model sees signal, not garbage.
    """

    def __init__(self,
                 remove_stopwords: bool = True,
                 min_word_length:  int  = 2):
        self.remove_stopwords = remove_stopwords
        self.min_word_length  = min_word_length

    def clean(self, text: str) -> str:
        """Master cleaning pipeline — returns cleaned string."""
        if not text or not isinstance(text, str):
            return ""

        text = self._decode_html_entities(text)   # &amp; → &
        text = self._remove_html_tags(text)        # <b>hi</b> → hi
        text = self._remove_urls(text)             # https://... → URL
        text = self._remove_email_addresses(text)  # x@y.com → EMAIL
        text = self._lowercase(text)               # HELLO → hello
        text = self._remove_punctuation(text)      # hello! → hello
        text = self._remove_extra_whitespace(text) # "a  b" → "a b"

        if self.remove_stopwords:
            text = self._remove_stopwords_fn(text)

        text = self._remove_short_words(text)      # "a b cat" → "cat"

        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        return html.unescape(text)

    def _remove_html_tags(self, text: str) -> str:
        # Remove <script> and <style> blocks entirely
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ',
                      text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        return text

    def _remove_urls(self, text: str) -> str:
        # Replace URLs with the token "URL" (keeps the signal)
        text = re.sub(r'http\S+|www\.\S+', ' url ', text)
        return text

    def _remove_email_addresses(self, text: str) -> str:
        return re.sub(r'\S+@\S+', ' email ', text)

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_punctuation(self, text: str) -> str:
        # Keep only alphanumeric and spaces
        return re.sub(r'[^\w\s]', ' ', text)

    def _remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text)

    def _remove_stopwords_fn(self, text: str) -> str:
        words = text.split()
        return ' '.join(w for w in words if w not in STOPWORDS)

    def _remove_short_words(self, text: str) -> str:
        words = text.split()
        return ' '.join(w for w in words if len(w) >= self.min_word_length)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = EmailCleaner()

    sample = """
    Subject: FREE iPhone!!!

    <html><body>
    <b>Congratulations!</b> You've WON a <i>FREE</i> iPhone!!!
    Click here: https://totally-legit.com/claim?id=123
    Contact us: winner@spam.com
    &amp; don't miss this LIMITED TIME offer!!!
    </body></html>
    """

    cleaned = cleaner.clean(sample)
    print("=== BEFORE ===")
    print(sample[:300])
    print("\n=== AFTER ===")
    print(cleaned)