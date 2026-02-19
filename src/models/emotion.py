from ..config import EMOTION_KEYWORDS


# optional ML dependency
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

class EmotionDetector:
    def __init__(self, keywords=None, use_ml=False):
        self.keywords = keywords or EMOTION_KEYWORDS
        self.use_ml = use_ml
        self.model = None

    def detect_emotion(self, text: str):
        if self.use_ml and self.model is not None:
            pred = self.model.predict([text])[0]
            prob = None
            try:
                prob = max(self.model.predict_proba([text])[0])
            except Exception:
                pass
            return pred, prob or 0.0

        # fallback rule-based
        text_low = (text or "").lower()
        scores = {}
        for emo, kws in self.keywords.items():
            cnt = sum(1 for k in kws if k in text_low)
            if cnt:
                scores[emo] = cnt
        if not scores:
            return "neutral", 0.0
        best = max(scores.items(), key=lambda x: x[1])
        # simple confidence normalization
        total = sum(scores.values())
        return best[0], best[1] / total

    def train_ml(self, texts, labels):
        """Train a simple classifier given text and emotion label arrays."""
        clf = LogisticRegression(max_iter=1000)
        clf.fit(texts, labels)
        self.model = clf
        return clf

    def save(self, filename="emotion_model.joblib"):
        import joblib, os
        from ..config import ARTIFACTS_DIR
        path = os.path.join(ARTIFACTS_DIR, filename)
        if self.model is not None:
            joblib.dump(self.model, path)
            return path
        return None

    def load(self, filename="emotion_model.joblib"):
        import joblib, os
        from ..config import ARTIFACTS_DIR
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.use_ml = True
            return True
        return False
