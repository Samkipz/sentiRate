from sklearn.linear_model import LogisticRegression
import joblib
import os
from ..config import ARTIFACTS_DIR

class SentimentClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)
        return preds, probs

    def save(self, filename="sentiment_model.joblib"):
        path = os.path.join(ARTIFACTS_DIR, filename)
        joblib.dump(self.model, path)
        return path

    def load(self, filename="sentiment_model.joblib"):
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            return True
        return False
