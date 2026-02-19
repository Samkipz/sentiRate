import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.data.parser import ReviewParser
except ImportError:
    ReviewParser = None


import pytest


def test_parse_demo_csv():
    if ReviewParser is None:
        pytest.skip("ReviewParser not available (missing dependencies)")
    demo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'reviews_demo.csv'))
    parser = ReviewParser()
    df = parser.parse_csv(demo)
    assert 'review_text' in df.columns
    assert len(df) >= 1



def test_parse_pasted_text_basic():
    if ReviewParser is None:
        pytest.skip("ReviewParser not available")
    raw = """Alice\n2026-01-10 | white | 5 stars\nI love this product!"""
    parser = ReviewParser()
    df = parser.parse_pasted_text(raw)
    assert df.iloc[0]['reviewer_name'] == 'Alice'
    assert '2026-01-10' in df.iloc[0]['date']
    assert df.iloc[0]['product_tag'].strip() == 'white'
    assert df.iloc[0]['star_rating'] == 5.0
    assert 'love this product' in df.iloc[0]['review_text']



def test_parse_web_scraped_stub():
    if ReviewParser is None:
        pytest.skip("ReviewParser not available")
    parser = ReviewParser()
    df = parser.parse_web_scraped('<html></html>')
    assert df.empty
    expected_cols = {"reviewer_name", "date", "product_tag", "star_rating", "review_text", "detected_language"}
    assert set(df.columns) == expected_cols
