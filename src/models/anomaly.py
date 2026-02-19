from difflib import SequenceMatcher

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.iso = None

    def fit_embeddings(self, embeddings):
        """Train an isolation forest on numeric embeddings."""
        if IsolationForest is None:
            raise ImportError("scikit-learn not available for anomaly detection")
        self.iso = IsolationForest(contamination=0.05, random_state=42)
        self.iso.fit(embeddings)
        return self.iso

    def predict_embedding(self, emb):
        if self.iso is None:
            return False
        score = self.iso.decision_function([emb])[0]
        return score < 0  # negative score = anomaly

    def is_duplicate(self, text, other_text):
        if not text or not other_text:
            return False
        return SequenceMatcher(None, text, other_text).ratio() > 0.9

    def short_review(self, text, min_words=3):
        if not text:
            return True
        return len(text.split()) < min_words

    def rating_mismatch(self, rating, sentiment_label):
        try:
            r = float(rating)
        except Exception:
            return False
        if r >= 4 and sentiment_label == "negative":
            return True
        if r <= 2 and sentiment_label == "positive":
            return True
        return False
