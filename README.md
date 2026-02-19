# Review Sentiment Dashboard

This repository contains a lightweight pipeline for parsing, analyzing, and visualizing customer reviews. It's designed to run on modest hardware and can be deployed locally or via Streamlit Cloud.

## 🚀 Features
- **Flexible input**: upload CSV, paste raw text, or plug in a future scraper.
- **Parsing**: extracts reviewer names, dates, ratings, product tags, and auto-detects language.
- **Cleaning/Preprocessing**: removes HTML, emojis, punctuation, and stopwords (English, Swahili, Sheng). Supports tokenization and optional user stopword files.
- **Feature extraction**:
  - TF‑IDF vectors (fast CPU)
  - Optional sentence-transformer embeddings
- **Models**:
  - Logistic Regression classifier (sentiment)
  - Optional small LSTM model (built with Keras)
  - Rule-based or ML emotion detector
  - Anomaly detector (duplicate, short review, rating mismatch, optional IsolationForest)
- **Aggregation & metrics**: compute sentiment ratios, average ratings, emotion distribution.
- **Streamlit dashboard**: tables, bar/pie charts, histogram, filtering, download enriched CSV, basic/advanced mode.

## 🏗️ Architecture Overview
1. **Data layer** (CSV / pasted text)
2. **Cleaning & parsing** with `src/data/parser.py`
3. **Preprocessing**: `TextCleaner`, stopword management, `TokenizerHelper` for sequences
4. **Feature extraction**: `FeatureExtractor` (TF‑IDF) and `EmbeddingExtractor`
5. **Model layer**: sentiment, emotion, anomalies, optionally LSTM
6. **Postprocessing**: `ResultAggregator` computes metrics
7. **Dashboard**: `app.py` builds UI, charts with Altair/Plotly, download button

## 📁 Project Structure
```
app.py
src/
  config.py           # settings & artifact path
  data/parser.py      # CSV/paste parser
  preprocessing/
    cleaner.py        # text cleaner
    tokenizer.py      # sequence helper (optional deps)
  features/
    extractor.py      # TF‑IDF vectorizer
    embeddings.py     # sentence-transformer wrapper
  models/
    sentiment.py      # logistic regression
    lstm.py           # small LSTM (optional, requires Keras)
    emotion.py        # rule/ML emotion detection
    anomaly.py        # duplicate/short/rating mismatch + IsolationForest
  utils/aggregator.py # summarise results
sample_data/           # demo CSV
scripts/train_models.py
tests/                 # unit tests
```

## 📦 Installation

```sh
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Some features (LSTM, embeddings) rely on heavy packages that may not install in all environments. The code is defensive and will still run without them, but you should install `tensorflow` and `sentence-transformers` for full functionality.

## 🧩 Running the Dashboard

```sh
streamlit run app.py
```

Upload `sample_data/reviews_demo.csv` or paste text to see parsing, sentiment, emotion, and anomalies. Use the sidebar to toggle between **Basic** and **Advanced** analysis.

## 🧪 Testing

The repo includes pytest tests covering parser, cleaner, models, and utilities. Tests gracefully skip when optional dependencies are missing.

```sh
.venv\Scripts\python -m pytest tests
```

## 📝 Tips for Demo
- Create a CSV with 20–50 reviews in English, Swahili, or Sheng.
- Include missing star ratings and gibberish to showcase parsing robustness.
- Use `python scripts/train_models.py` to generate sample TF‑IDF and sentiment artifacts.

## ⚙️ Future Enhancements
- Add live web scraping or API connectors.
- Integrate multilingual embeddings for richer analysis.
- Predict star-rating from review text.
- Add time-series trends, deploy on Vercel/Streamlit Cloud.

---

The architecture at the top of this README retains the original detailed plan that guided development. Modify it as your project evolves.
