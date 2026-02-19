import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "..", "models", "artifacts")
ARTIFACTS_DIR = os.path.abspath(ARTIFACTS_DIR)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 500

# Minimal Swahili & Sheng stopwords examples (extend as needed)
# These sets are populated by reading optional files under src/stopwords
def _load_stopwords(filename: str):
    path = os.path.join(ROOT_DIR, filename)
    words = set()
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    words.add(w)
    except FileNotFoundError:
        pass
    return words

SWAHILI_STOPWORDS = _load_stopwords('stopwords/swahili.txt') or {"na", "ya", "kwa", "ni", "si", "siya", "mimi", "sisi"}
SHENG_STOPWORDS = _load_stopwords('stopwords/sheng.txt') or {"mambo", "sasa", "msee"}

EMOTION_KEYWORDS = {
    "happy": ["love", "great", "excellent", "happy", "amazing", "wooow", "like"],
    "angry": ["hate", "terrible", "awful", "disappointed", "angry"],
    "excited": ["excited", "wow", "omg", "amazed"],
    "disappointed": ["disappointed", "bad", "poor", "waste"],
}
