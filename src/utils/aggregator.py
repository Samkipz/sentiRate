import pandas as pd

class ResultAggregator:
    def __init__(self):
        pass

    def aggregate(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        metrics = {}
        if "sentiment" in df.columns:
            metrics["positive_count"] = int((df["sentiment"] == "positive").sum())
            metrics["negative_count"] = int((df["sentiment"] == "negative").sum())
        if "star_rating" in df.columns:
            try:
                df["star_rating_num"] = pd.to_numeric(df["star_rating"], errors="coerce")
                metrics["avg_rating"] = float(df["star_rating_num"].mean())
            except Exception:
                metrics["avg_rating"] = None
        if "emotion" in df.columns:
            metrics["emotion_distribution"] = df["emotion"].value_counts().to_dict()
        return df, metrics
