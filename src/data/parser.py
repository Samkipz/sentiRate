import re
try:
    from langdetect import detect
except ImportError:
    def detect(text):
        return "unknown"
import pandas as pd
import numpy as np
from datetime import datetime
from ..config import EMOTION_KEYWORDS

def clean_raw_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove html tags
    text = re.sub(r"<[^>]+>", " ", text)
    # remove emojis (basic)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def auto_detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

class ReviewParser:
    def __init__(self):
        pass

    def parse_web_scraped(self, html: str) -> pd.DataFrame:
        """Stub for future scraping support. Returns empty dataframe with expected columns."""
        cols = ["reviewer_name", "date", "product_tag", "star_rating", "review_text", "detected_language"]
        return pd.DataFrame(columns=cols)

    def parse_csv(self, path: str) -> pd.DataFrame:
        # Read CSV without assuming an index column
        df = pd.read_csv(path, index_col=False, on_bad_lines='skip')
        # drop any unnamed index columns that may have been created
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        # ensure expected columns exist
        for col in ["review_text", "reviewer_name", "date", "product_tag", "star_rating"]:
            if col not in df.columns:
                df[col] = np.nan
        df["review_text"] = df["review_text"].fillna("").apply(clean_raw_text)
        df["detected_language"] = df["review_text"].apply(auto_detect_language)
        # Return only expected columns in correct order
        return df[["reviewer_name", "date", "product_tag", "star_rating", "review_text", "detected_language"]]

    def parse_pasted_text(self, raw: str) -> pd.DataFrame:
        # handle table-like paste first (header present)
        if "\n" in raw and ("\t" in raw or "," in raw):
            first_line = raw.strip().splitlines()[0].lower()
            if any(col in first_line for col in ["review_text", "reviewer_name", "star_rating"]):
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(raw), sep=None, engine="python", index_col=False, on_bad_lines='skip')
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
                    for col in ["review_text", "reviewer_name", "date", "product_tag", "star_rating"]:
                        if col not in df.columns:
                            df[col] = np.nan
                    df["review_text"] = df["review_text"].fillna("").apply(clean_raw_text)
                    df["detected_language"] = df["review_text"].apply(auto_detect_language)
                    return df[["reviewer_name", "date", "product_tag", "star_rating", "review_text", "detected_language"]]
                except Exception:
                    pass
        # if not table, split by lines but treat each line as a record if possible
        date_pattern = re.compile(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if lines and all(date_pattern.search(l) for l in lines):
            # simple delimited lines, assume order: reviewer,date,product_tag,star_rating,review_text,lang?
            rows = []
            for l in lines:
                parts = re.split(r"\t|,", l)
                # pad to 6 elements
                parts += [""] * (6 - len(parts))
                reviewer, date, product, rating, text, lang = parts[:6]
                try:
                    rating = float(rating)
                except:
                    rating = np.nan
                text = clean_raw_text(text)
                lang = lang or auto_detect_language(text or reviewer)
                rows.append({"reviewer_name": reviewer, "date": date, "product_tag": product, "star_rating": rating, "review_text": text, "detected_language": lang})
            return pd.DataFrame(rows)
        # fallback: split blank lines into review blocks
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        rows = []
        # simple regex patterns for date, rating, product
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ \d,\.\-/]+)", re.IGNORECASE)
        rating_pattern = re.compile(r"([1-5])\s?(?:stars?|/5)?", re.IGNORECASE)
        product_pattern = re.compile(r"\|\s*([^\(\n]+)")

        for b in blocks:
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            reviewer = lines[0] if lines else ""
            date = ""
            rating = np.nan
            product = ""
            # combine remaining lines for searching
            rest = " ".join(lines[1:]) if len(lines) > 1 else ""
            # look for date in any line
            m = date_pattern.search(b)
            if m:
                date = m.group(1)
            # look for rating
            m = rating_pattern.search(b)
            if m:
                try:
                    rating = float(m.group(1))
                except Exception:
                    rating = np.nan
            # look for a product tag after pipe or parentheses
            m = product_pattern.search(b)
            if m:
                product = m.group(1).strip()
            text = rest
            text = clean_raw_text(text)
            lang = auto_detect_language(text or reviewer)
            rows.append({"reviewer_name": reviewer, "date": date, "product_tag": product, "star_rating": rating, "review_text": text, "detected_language": lang})
        return pd.DataFrame(rows)
