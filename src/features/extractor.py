from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from ..config import TFIDF_MAX_FEATURES, ARTIFACTS_DIR

class FeatureExtractor:
    def __init__(self, max_features=TFIDF_MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, filename="tfidf_vectorizer.joblib"):
        path = os.path.join(ARTIFACTS_DIR, filename)
        joblib.dump(self.vectorizer, path)
        return path

    def load(self, filename="tfidf_vectorizer.joblib"):
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            self.vectorizer = joblib.load(path)
            return True
        return False
