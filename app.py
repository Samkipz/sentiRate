# shim for removed stdlib module `imghdr` (Python 3.13+).
# Streamlit still does `import imghdr` when handling images, so we
# preload our own copy from the repo root before pulling in Streamlit.
import sys, importlib.util, os

shim_path = os.path.join(os.path.dirname(__file__), "imghdr.py")
if "imghdr" not in sys.modules and os.path.exists(shim_path):
    spec = importlib.util.spec_from_file_location("imghdr", shim_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    sys.modules["imghdr"] = module

import streamlit as st
import pandas as pd
import os
import re
try:
    import altair as alt
except ImportError:
    alt = None
import plotly.express as px
from src.data.parser import ReviewParser
from src.preprocessing.cleaner import TextCleaner
from src.features.extractor import FeatureExtractor
from src.models.sentiment import SentimentClassifier
from src.models.emotion import EmotionDetector
from src.models.anomaly import AnomalyDetector
from src.utils.aggregator import ResultAggregator
from src.config import ARTIFACTS_DIR

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Try to ensure VADER lexicon exists, but don't fail if download fails
_vader_available = False
try:
    nltk.data.find('vader_lexicon')
    _vader_available = True
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
        _vader_available = True
    except:
        _vader_available = False

st.set_page_config(layout="wide", page_title="Review Sentiment Dashboard")

@st.cache_resource
def get_vader():
    if _vader_available:
        try:
            return SentimentIntensityAnalyzer()
        except:
            return None
    return None


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
            try:
                parser = ReviewParser()
                df_demo = parser.parse_csv(demo_path)
                cleaner = TextCleaner()
                df_demo["clean_text"] = df_demo["review_text"].fillna("").apply(lambda t: cleaner.clean(t))
                # Simple labeling: presence of positive/negative words
                pos_words = set(["love", "great", "good", "excellent", "like", "amazing", "wooow", "best", "satisfied", 
                                "recommend", "fantastic", "wonderful", "awesome", "perfect", "brilliant", "happy",
                                "beautiful", "outstanding", "superb", "exceptional", "quality", "value"])
                neg_words = set(["hate", "terrible", "awful", "bad", "poor", "worst", "useless", "waste", "disappointed",
                                "angry", "horrible", "disgusting", "disappointing", "issue", "problem", 
                                "broken", "defective", "cheap", "inferior"])
                y_demo = [1 if (any(w in str(t).lower() for w in pos_words) and 
                               not any(w in str(t).lower() for w in neg_words)) else 0 
                         for t in df_demo["clean_text"].fillna("")]
                X = feats.fit_transform(df_demo["clean_text"].fillna(""))
                model.train(X, y_demo)
                feats.save()
                model.save()
                st.success("Models trained successfully! Refresh the page to see results.")
                return feats, model, True
            except Exception as e:
                st.error(f"Training failed: {e}")
                return feats, model, False
        return feats, model, False
    return feats, model, loaded_feat and loaded_model

def generate_recommendations(df, metrics):
    """Return a list of recommendation strings based on review DataFrame and computed metrics.
    Logic tries to be balanced by reporting positive signals as well as areas for improvement.
    """
    recs = []
    total = len(df)
    if total == 0:
        return recs

    pos = metrics.get('positive_count', 0)
    neg = metrics.get('negative_count', 0)
    pos_pct = pos / total if total else 0
    neg_pct = neg / total if total else 0

    # basic sentiment summary
    if neg_pct > 0.5:
        recs.append("More than half of reviews are negative. Investigate frequent complaints and engage with your customers.")
    elif pos_pct > 0.5:
        recs.append("Majority of reviews are positive. Consider amplifying the messages that resonate with customers.")
    else:
        recs.append("Mixed feedback received; dive deeper into common themes on both sides.")

    # compute distinctive terms by comparing negative vs positive
    from collections import Counter
    neg_texts = df[df.get('sentiment')=='negative']['clean_text'].astype(str)
    pos_texts = df[df.get('sentiment')=='positive']['clean_text'].astype(str)
    # simple word lists to ignore (common words and keywords)
    ignore = set(["the","and","with","this","that","product","review","buy","order","item"])
    # positive keywords to avoid flagging
    pos_keywords = set(["love","great","good","excellent","amazing","best","satisfied","recommend","happy","nzuri","furaha","tamu","poa"])

    def word_counts(texts):
        ctr = Counter()
        for t in texts:
            for w in re.findall(r"\w+", t.lower()):
                if len(w) > 3 and w not in ignore:
                    ctr[w] += 1
        return ctr

    neg_ctr = word_counts(neg_texts)
    pos_ctr = word_counts(pos_texts)

    # compute words strongly associated with negative reviews
    distinctive = []
    for w, cnt in neg_ctr.most_common():
        if w in pos_keywords or w in ignore:
            continue
        if cnt >= 2 and cnt > pos_ctr.get(w, 0):
            distinctive.append(w)
        if len(distinctive) >= 5:
            break
    if distinctive:
        recs.append("Distinctive negative terms: " + ", ".join(distinctive))

    # add positive highlights when available
    if pos_pct >= 0.3:
        top_pos = []
        for w, cnt in pos_ctr.most_common():
            if w in pos_keywords or w in ignore:
                continue
            if cnt >= 2:
                top_pos.append(w)
            if len(top_pos) >= 5:
                break
        if top_pos:
            recs.append("Positive highlights: " + ", ".join(top_pos))

    return recs


def main():
    # Initialize session state for data persistence across reruns
    if "parsed_df" not in st.session_state:
        st.session_state.parsed_df = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None  # track whether it came from upload or paste
    
    st.sidebar.title("Inputs")
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    paste = st.sidebar.text_area("Or paste reviews (optional)")
    analysis_type = st.sidebar.radio("Analysis mode", ["Basic", "Advanced"], index=0)
    search = st.sidebar.text_input("Filter text (reviewer/date/product)")
    run_demo_train = st.sidebar.button("Train demo artifacts")

    if run_demo_train:
        st.info("Run `python scripts/train_models.py` in terminal to train demo artifacts.")

    parser = ReviewParser()
    
    # Parse new uploads/pastes and cache the DataFrame
    if upload is not None and st.session_state.data_source != "upload":
        try:
            st.session_state.parsed_df = parser.parse_csv(upload)
            st.session_state.data_source = "upload"
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
            st.session_state.parsed_df = None
            return
    elif paste and paste != st.session_state.get("last_paste", ""):
        try:
            st.session_state.parsed_df = parser.parse_pasted_text(paste)
            st.session_state.data_source = "paste"
            st.session_state.last_paste = paste
            st.success("Reviews pasted successfully!")
        except Exception as e:
            st.error(f"Failed to parse pasted text: {e}")
            st.session_state.parsed_df = None
            return
    
    # Use cached dataframe
    if st.session_state.parsed_df is not None:
        df = st.session_state.parsed_df.copy()
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

    # convert and filter by date range if available
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            min_d = df['date'].min()
            max_d = df['date'].max()
            date_range = st.sidebar.date_input("Date range", [min_d, max_d])
            if len(date_range) == 2:
                df = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
        except Exception:
            pass

    # preprocessing for display
    display_df = df.copy()
    if 'star_rating' in display_df.columns:
        display_df['star_rating'] = pd.to_numeric(display_df['star_rating'], errors='coerce')
        display_df['star_rating'] = display_df['star_rating'].apply(lambda x: int(x) if pd.notna(x) else '-')

    # highlight non-English reviews when showing raw data later
    def highlight_non_en(row):
        return ['background-color: #fff3cd' if row.get('detected_language') and row.get('detected_language')!='en' else '' for _ in row]

    cleaner = TextCleaner()
    df["clean_text"] = df["review_text"].fillna("").apply(lambda t: cleaner.clean(t))

    # Use VADER if available, fallback to rule-based
    vader = get_vader()
    
    # Rule-based fallback keywords
    pos_words_en = set(["love", "great", "good", "excellent", "like", "amazing", "best", "satisfied", 
                        "recommend", "fantastic", "wonderful", "awesome", "perfect", "brilliant", "happy",
                        "beautiful", "outstanding", "superb", "exceptional", "quality", "value", "worth"])
    neg_words_en = set(["hate", "terrible", "awful", "bad", "poor", "worst", "useless", "waste", "disappointed",
                        "angry", "horrible", "disgusting", "disappointing", "issue", "problem", 
                        "broken", "defective", "cheap", "inferior"])
    pos_words_sw = set(["nzuri", "ajabu", "furaha", "bora", "safi", "jalifu", "tamu", "karibu", "asante"])
    neg_words_sw = set(["mbaya", "hapana", "duni", "kasoro", "tatizo", "kufa", "haiwezi", "dhaifu", "maadhimisho"])
    
    sentiments = []
    probs = []
    
    for idx, text in enumerate(df["clean_text"].fillna("")):
        text_str = str(text).strip()
        lang = df.iloc[idx].get("detected_language", "en") if idx < len(df) else "en"
        
        # Try VADER first if available
        if vader:
            scores = vader.polarity_scores(text_str)
            compound = scores['compound']
            vader_confidence = abs(compound)
            
            # Use VADER if confident
            if compound >= 0.05:
                sentiments.append("positive")
                probs.append(scores['pos'] + 0.2)
                continue
            elif compound <= -0.05:
                sentiments.append("negative")
                probs.append(scores['neg'] + 0.2)
                continue
            # Otherwise fall through to fallback
        
        # Fallback rule-based approach
        pos_set = pos_words_sw if lang in ["sw", "unknown"] else pos_words_en
        neg_set = neg_words_sw if lang in ["sw", "unknown"] else neg_words_en
        
        pos_count = sum(1 for w in pos_set if w in text_str.lower())
        neg_count = sum(1 for w in neg_set if w in text_str.lower())
        
        if pos_count > neg_count:
            sentiments.append("positive")
            probs.append(0.7 + min(pos_count * 0.1, 0.25))
        elif neg_count > pos_count:
            sentiments.append("negative")
            probs.append(0.7 + min(neg_count * 0.1, 0.25))
        else:
            # neutral - use star rating
            rating = df.iloc[idx].get("star_rating", None) if idx < len(df) else None
            try:
                rating_num = float(rating) if rating else 0
                if rating_num >= 4:
                    sentiments.append("positive")
                    probs.append(0.6)
                else:
                    sentiments.append("negative")
                    probs.append(0.6)
            except:
                sentiments.append("negative")
                probs.append(0.5)
    
    df["sentiment"] = sentiments
    df["sentiment_prob"] = probs

    emo = EmotionDetector()
    df["emotion"], df["emotion_confidence"] = zip(*df["clean_text"].apply(lambda t: emo.detect_emotion(t)))

    anomaly = AnomalyDetector()
    flags = []
    seen_texts = set()
    # use enumerate to avoid non-integer index values
    for idx, row in enumerate(df.itertuples(index=False)):
        reasons = []
        text_val = getattr(row, 'review_text', '')
        if anomaly.short_review(text_val):
            reasons.append("short_review")
        # check duplicates using a set (O(1) lookup instead of O(n))
        text_hash = text_val.lower().strip()
        if text_hash in seen_texts:
            reasons.append("duplicate")
        seen_texts.add(text_hash)
        if "sentiment" in df.columns and anomaly.rating_mismatch(getattr(row, 'star_rating', ""), getattr(row, 'sentiment', "")):
            reasons.append("rating_mismatch")
        flags.append(";".join(reasons) if reasons else "")
    df["suspicious_flags"] = flags

    agg = ResultAggregator()
    df, metrics = agg.aggregate(df)

    # convert date column to datetime if present
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass

    # summary metrics and tabs
    st.header("Dashboard")
    tab_overview, tab_charts, tab_flags, tab_data = st.tabs(["Overview", "Charts", "Flagged Reviews", "Raw Data"])

    with tab_overview:
        st.subheader("Key metrics")
        cols = st.columns(4)
        cols[0].metric("Total reviews", len(df))
        cols[1].metric("Positive", metrics.get('positive_count', 0))
        cols[2].metric("Negative", metrics.get('negative_count', 0))
        cols[3].metric("Avg rating", f"{metrics.get('avg_rating', '-'):.2f}" if metrics.get('avg_rating') is not None else "-")

        if analysis_type == "Advanced" and 'emotion' in df.columns:
            st.subheader("Emotion breakdown")
            if alt:
                df_chart = df.copy()
                for c in df_chart.select_dtypes(include=["string"]).columns:
                    df_chart[c] = df_chart[c].astype("object")
                emo_bar = alt.Chart(df_chart).mark_bar().encode(
                    x='emotion',
                    y='count()',
                    color='emotion'
                )
                st.altair_chart(emo_bar, use_container_width=True)
            else:
                # fallback to plotly bar
                fig = px.bar(df, x='emotion', title='Emotion breakdown')
                st.plotly_chart(fig, use_container_width=True)

    with tab_charts:
        if "sentiment" in df.columns:
            st.subheader("Sentiment proportions")
            if alt:
                # Altair/vega sometimes chokes on pyarrow's LargeUtf8 string
                # dtype, which can appear when pandas is using the Arrow
                # extension.  Convert any "string" columns back to plain
                # Python objects before handing the data to Altair.
                df_chart = df.copy()
                for c in df_chart.select_dtypes(include=["string"]).columns:
                    df_chart[c] = df_chart[c].astype("object")

                chart = alt.Chart(df_chart).mark_bar().encode(
                    x='sentiment',
                    y='count()',
                    color='sentiment'
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                fig = px.bar(df, x='sentiment', title='Sentiment proportions')
                st.plotly_chart(fig, use_container_width=True)

            if 'date' in df.columns:
                st.subheader("Sentiment over time")
                if alt:
                    df_chart = df.copy()
                    for c in df_chart.select_dtypes(include=["string"]).columns:
                        df_chart[c] = df_chart[c].astype("object")
                    time_chart = alt.Chart(df_chart).mark_line().encode(
                        x='yearmonth(date):T',
                        y='count()',
                        color='sentiment'
                    )
                    st.altair_chart(time_chart, use_container_width=True)
                else:
                    fig2 = px.line(df, x='date', y=df.index, color='sentiment', title='Sentiment over time')
                    st.plotly_chart(fig2, use_container_width=True)

        if analysis_type == "Advanced":
            if 'emotion' in df.columns and 'date' in df.columns:
                st.subheader("Emotion trends")
                if alt:
                    df_chart = df.copy()
                    for c in df_chart.select_dtypes(include=["string"]).columns:
                        df_chart[c] = df_chart[c].astype("object")
                    emo_time = alt.Chart(df_chart).mark_line().encode(
                        x='yearmonth(date):T',
                        y='count()',
                        color='emotion'
                    )
                    st.altair_chart(emo_time, use_container_width=True)
                else:
                    fig3 = px.line(df, x='date', y=df.index, color='emotion', title='Emotion trends')
                    st.plotly_chart(fig3, use_container_width=True)

            if "star_rating" in df.columns:
                st.subheader("Rating distribution")
                hist = px.histogram(df, x='star_rating', nbins=5, title='Star rating distribution')
                st.plotly_chart(hist, use_container_width=True)

            # recommendation insights based on simple heuristics
            st.subheader("Recommendations")
            recs = generate_recommendations(df, metrics)
            if recs:
                for r in recs:
                    st.write("- " + r)
            else:
                st.write("No strong recommendations based on the current data.")

    with tab_flags:
        st.subheader("Suspicious reviews")
        flags_df = df[df["suspicious_flags"]!=""][["reviewer_name","review_text","suspicious_flags"]].copy()
        for c in flags_df.select_dtypes(include=["string"]).columns:
            flags_df[c] = flags_df[c].astype("object")
        st.dataframe(flags_df, use_container_width=True)

    with tab_data:
        st.subheader("Parsed data")
        # explicitly select and reorder columns to avoid alignment issues
        cols_to_show = ["reviewer_name", "date", "product_tag", "star_rating", "review_text", "detected_language"]
        # only keep columns that exist
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        show_df = display_df[cols_to_show].copy()
        # reset index to ensure clean display
        show_df = show_df.reset_index(drop=True)
        # aggressively convert all Arrow string types to plain Python objects
        for col in show_df.columns:
            try:
                if 'string' in str(show_df[col].dtype).lower():
                    show_df[col] = show_df[col].astype('object')
            except:
                pass
        if 'detected_language' in show_df.columns:
            st.dataframe(show_df.style.apply(highlight_non_en, axis=1), use_container_width=True)
        else:
            st.dataframe(show_df, use_container_width=True)

    # sidebar download remains
    st.sidebar.download_button("Download enriched CSV", df.to_csv(index=False).encode("utf-8"), file_name="enriched_reviews.csv")

if __name__ == "__main__":
    main()
