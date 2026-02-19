import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from ..config import SWAHILI_STOPWORDS, SHENG_STOPWORDS

try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("punkt")
    nltk.download("stopwords")

class TextCleaner:
    def __init__(self, extra_stopwords=None, stopword_files=None):
        # base english stopwords from nltk
        self.stopwords = set(stopwords.words("english"))
        # include swahili/sheng from config (possibly loaded from files)
        self.stopwords.update(SWAHILI_STOPWORDS)
        self.stopwords.update(SHENG_STOPWORDS)
        # allow passing additional words or files
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)
        if stopword_files:
            # stopword_files can be list of paths
            for f in stopword_files:
                try:
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            w = line.strip()
                            if w and not w.startswith("#"):
                                self.stopwords.add(w)
                except FileNotFoundError:
                    pass

    def lowercase(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text)

    def remove_stopwords(self, text: str) -> str:
        # simple word split to avoid heavy tokenizer dependencies
        tokens = re.findall(r"\w+", text)
        filtered = [t for t in tokens if t.lower() not in self.stopwords]
        return " ".join(filtered)

    def clean(self, text: str, remove_punct=True, remove_stop=True) -> str:
        t = self.lowercase(text)
        if remove_punct:
            t = self.remove_punctuation(t)
        if remove_stop:
            t = self.remove_stopwords(t)
        return t
