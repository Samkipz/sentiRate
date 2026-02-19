import streamlit as st
import pandas as pd
import os
import altair as alt
import plotly.express as px
from src.data.parser import ReviewParser
from src.preprocessing.cleaner import TextCleaner
from src.features.extractor import FeatureExtractor
from src.models.sentiment import SentimentClassifier
from src.models.emotion import EmotionDetector
from src.models.anomaly import AnomalyDetector
from src.utils.aggregator import ResultAggregator
from src.config import ARTIFACTS_DIR

st.set_page_config(layout="wide", page_title="Review Sentiment Dashboard")

@st.cache_resource
def load_models():
    feats = FeatureExtractor()
    model = SentimentClassifier()
    loaded_feat = feats.load()
    loaded_model = model.load()
    if not (loaded_feat and loaded_model):
        # Auto-train on demo data if artifacts missing
        st.info("Training models on demo data (first time only)...")
        from src.preprocessing.cleaner import TextCleaner
        from src.data.parser import ReviewParser
        demo_path = os.path.join(os.path.dirname(__file__), "sample_data", "reviews_demo.csv")
        if os.path.exists(demo_path):
            parser = ReviewParser()
            df_demo = parser.parse_csv(demo_path)
            cleaner = TextCleaner()
            df_demo["clean_text"] = df_demo["review_text"].fillna("").apply(lambda t: cleaner.clean(t))
            # Simple labeling: presence of positive words
            y_demo = [1 if any(w in str(t).lower() for w in ["love","great","good","excellent","like","amazing","wooow","best","excellent","satisfied","recommend"]) else 0 for t in df_demo["clean_text"].fillna("")]
            X = feats.fit_transform(df_demo["clean_text"].fillna(""))
            model.train(X, y_demo)
            feats.save()
            model.save()
            st.success("Models trained successfully!")
            return feats, model, True
        return feats, model, False
    return feats, model, loaded_feat and loaded_model

def main():
    st.sidebar.title("Inputs")
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    paste = st.sidebar.text_area("Or paste reviews (optional)")
    analysis_type = st.sidebar.radio("Analysis mode", ["Basic", "Advanced"], index=0)
    search = st.sidebar.text_input("Filter text (reviewer/date/product)")
    run_demo_train = st.sidebar.button("Train demo artifacts")

    if run_demo_train:
        st.info("Run `python scripts/train_models.py` in terminal to train demo artifacts.")

    parser = ReviewParser()
    if upload is not None:
        df = parser.parse_csv(upload)
    elif paste:
        df = parser.parse_pasted_text(paste)
    else:
        st.info("Upload a CSV or paste reviews. You can also use sample_data/reviews_demo.csv")
        return

    # apply simple search filter
    if search:
        mask = (
            df['reviewer_name'].astype(str).str.contains(search, case=False, na=False)
            | df['date'].astype(str).str.contains(search, case=False, na=False)
            | df['product_tag'].astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    st.header("Parsed Reviews")
    # Format display: convert star_rating to numeric, clean up display
    display_df = df.copy()
    if 'star_rating' in display_df.columns:
        display_df['star_rating'] = pd.to_numeric(display_df['star_rating'], errors='coerce')
        display_df['star_rating'] = display_df['star_rating'].apply(lambda x: int(x) if pd.notna(x) else '-')
    # highlight non-English reviews
    if 'detected_language' in display_df.columns:
        st.dataframe(display_df.style.apply(lambda row: ['background-color: #fff3cd' if row['detected_language'] and row['detected_language']!='en' else '' for _ in row], axis=1), use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

    cleaner = TextCleaner()
    df["clean_text"] = df["review_text"].fillna("").apply(lambda t: cleaner.clean(t))

    feats, model, ready = load_models()
    if ready:
        X = feats.transform(df["clean_text"].fillna(""))
        preds, probs = model.predict(X)
        df["sentiment"] = ["positive" if p==1 else "negative" for p in preds]
        df["sentiment_prob"] = [max(p) for p in probs]
    else:
        st.warning("Model artifacts not found. Run training script to create demo models.")

    emo = EmotionDetector()
    df["emotion"], df["emotion_confidence"] = zip(*df["clean_text"].apply(lambda t: emo.detect_emotion(t)))

    anomaly = AnomalyDetector()
    flags = []
    # use enumerate to avoid non-integer index values
    for idx, row in enumerate(df.itertuples(index=False)):
        reasons = []
        text_val = getattr(row, 'review_text', '')
        if anomaly.short_review(text_val):
            reasons.append("short_review")
        # check duplicates against prior rows
        for prev in range(0, idx):
            prev_text = df.iloc[prev].get("review_text", "")
            if anomaly.is_duplicate(text_val, prev_text):
                reasons.append("duplicate")
                break
        if "sentiment" in df.columns and anomaly.rating_mismatch(getattr(row, 'star_rating', ""), getattr(row, 'sentiment', "")):
            reasons.append("rating_mismatch")
        flags.append(";".join(reasons) if reasons else "")
    df["suspicious_flags"] = flags

    agg = ResultAggregator()
    df, metrics = agg.aggregate(df)

    st.header("Metrics")
    st.write(metrics)


    # charts for basic and advanced modes
    if "sentiment" in df.columns:
        st.header("Sentiment Proportions")
        chart = alt.Chart(df).mark_bar().encode(
            x='sentiment',
            y='count()',
            color='sentiment'
        )
        st.altair_chart(chart, use_container_width=True)
    if analysis_type == "Advanced":
        if "emotion" in df.columns:
            st.header("Emotion Distribution")
            pie = px.pie(df, names='emotion', title='Emotion breakdown')
            st.plotly_chart(pie, use_container_width=True)
        if "star_rating" in df.columns:
            st.header("Star Ratings Histogram")
            hist = px.histogram(df, x='star_rating', nbins=5, title='Star rating distribution')
            st.plotly_chart(hist, use_container_width=True)
        st.header("Suspicious Reviews")
        st.dataframe(df[df["suspicious_flags"]!=""][["reviewer_name","review_text","suspicious_flags"]])

    st.sidebar.download_button("Download enriched CSV", df.to_csv(index=False).encode("utf-8"), file_name="enriched_reviews.csv")

if __name__ == "__main__":
    main()
