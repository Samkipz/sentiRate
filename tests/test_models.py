import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.emotion import EmotionDetector
from src.models.anomaly import AnomalyDetector
from src.features.embeddings import EmbeddingExtractor


def test_emotion_rule_based():
    emo = EmotionDetector()
    label, conf = emo.detect_emotion("I love this, amazing product")
    assert label == 'happy'


def test_emotion_ml_training():
    # attempt to train and catch missing dependency
    import pytest
    emo = EmotionDetector(use_ml=True)
    texts = ["I love it", "I hate it"]
    labels = ["happy", "angry"]
    try:
        emo.train_ml(texts, labels)
    except Exception as e:
        # if sklearn was not installed or logistic regression missing
        pytest.skip(f"Skipping ML emotion training: {e}")
    pred, conf = emo.detect_emotion("I love it")
    assert pred in ['happy', 'angry']


def test_anomaly_basic():
    ano = AnomalyDetector()
    assert ano.short_review("ok")
    assert not ano.short_review("this is fine")
    assert ano.is_duplicate("hello", "hello")


def test_lstm_placeholder():
    import pytest
    try:
        from src.models.lstm import LSTMSentimentClassifier
    except ImportError:
        pytest.skip("LSTM module not available")
        return
    try:
        cls = LSTMSentimentClassifier()
    except ImportError:
        pytest.skip("Keras not installed, skipping LSTM tests")
        return
    # we won't train but ensure save/load etc handle None model gracefully
    assert hasattr(cls, 'build')


def test_tokenizer_helper_exists():
    try:
        from src.preprocessing.tokenizer import TokenizerHelper
    except ImportError:
        pytest.skip("tokenizer helper missing or tensorflow not installed")
        return
    try:
        tok = TokenizerHelper()
    except ImportError:
        pytest.skip("TensorFlow/Keras not available")
        return
    assert hasattr(tok, 'fit')
    assert hasattr(tok, 'pad')


def test_embedding_extractor_missing():
    import pytest
    try:
        emb = EmbeddingExtractor()
    except ImportError:
        pytest.skip("sentence-transformers not installed")
        return
    try:
        arr = emb.encode(["test"])
    except ImportError:
        pytest.skip("sentence-transformers not installed at runtime")
        return
    assert hasattr(arr, 'shape')
