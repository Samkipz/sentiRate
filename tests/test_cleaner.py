import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.cleaner import TextCleaner


def test_cleaner_basic():
    cleaner = TextCleaner()
    out = cleaner.clean("Hello, THIS is a Test!!!")
    assert 'hello' in out


def test_cleaner_stopwords_from_file(tmp_path):
    # create temporary stopword file
    sw = tmp_path / "extras.txt"
    sw.write_text("foo\nbar\n")
    cleaner = TextCleaner(stopword_files=[str(sw)])
    # words 'foo' and 'bar' should be removed
    txt = "foo something bar baz"
    cleaned = cleaner.clean(txt, remove_punct=False, remove_stop=True)
    assert 'foo' not in cleaned and 'bar' not in cleaned

