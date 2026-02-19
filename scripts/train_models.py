import os
import pandas as pd
from src.data.parser import ReviewParser
from src.preprocessing.cleaner import TextCleaner
from src.features.extractor import FeatureExtractor
from src.models.sentiment import SentimentClassifier
from src.config import ARTIFACTS_DIR

def _fake_labels(texts):
    # naive labeling for demo: presence of good words -> positive
    labels = []
    for t in texts:
        tl = (t or "").lower()
        if any(w in tl for w in ["love","great","good","excellent","like","amazing","wooow"]):
            labels.append(1)
        else:
            labels.append(0)
    return labels

def train_on_demo(csv_path):
    parser = ReviewParser()
    df = parser.parse_csv(csv_path)
    cleaner = TextCleaner()
    df["clean_text"] = df["review_text"].apply(lambda t: cleaner.clean(t))
    feats = FeatureExtractor()
    X = feats.fit_transform(df["clean_text"].fillna(""))
    y = _fake_labels(df["clean_text"].fillna(""))
    model = SentimentClassifier()
    model.train(X, y)
    feats.save()
    model.save()
    print("Saved artifacts to", ARTIFACTS_DIR)

if __name__ == "__main__":
    demo = os.path.join(os.path.dirname(__file__), "..", "sample_data", "reviews_demo.csv")
    demo = os.path.abspath(demo)
    if os.path.exists(demo):
        train_on_demo(demo)
    else:
        print("Demo CSV not found at", demo)
