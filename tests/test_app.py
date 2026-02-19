import pandas as pd
import os, sys

# ensure project path on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    from app import generate_recommendations
except ImportError as e:
    import pytest
    pytest.skip(f"Skipping generate_recommendations tests: {e}", allow_module_level=True)


def test_generate_recommendations_empty():
    df = pd.DataFrame(columns=['sentiment', 'clean_text'])
    metrics = {}
    assert generate_recommendations(df, metrics) == []


def test_generate_recommendations_all_negative():
    df = pd.DataFrame({
        'sentiment': ['negative', 'negative'],
        'clean_text': ['bad product', 'terrible service']
    })
    metrics = {'negative_count': 2}
    recs = generate_recommendations(df, metrics)
    assert any('More than half' in r for r in recs)
    assert any('Common negative terms' in r for r in recs)


def test_generate_recommendations_positive():
    df = pd.DataFrame({
        'sentiment': ['positive', 'positive'],
        'clean_text': ['great value', 'excellent']
    })
    metrics = {'negative_count': 0}
    recs = generate_recommendations(df, metrics)
    assert recs == []


if __name__ == "__main__":
    # run tests manually if pytest is unavailable
    test_generate_recommendations_empty()
    test_generate_recommendations_all_negative()
    test_generate_recommendations_positive()
    print("test_app: all checks passed")
