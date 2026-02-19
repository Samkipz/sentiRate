import os
import joblib

from ..config import ARTIFACTS_DIR

# tensorflow imports are optional; we delay to constructor

class TokenizerHelper:
    def __init__(self, num_words=10000, oov_token='<OOV>'):
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            raise ImportError("TensorFlow/Keras is required for TokenizerHelper")
        self.Tokenizer = Tokenizer
        self.pad_sequences = pad_sequences
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.num_words = num_words
        self.oov_token = oov_token

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def pad(self, sequences, maxlen=None, padding='post', truncating='post'):
        return self.pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)

    def save(self, filename='tokenizer.joblib'):
        path = os.path.join(ARTIFACTS_DIR, filename)
        joblib.dump(self.tokenizer, path)
        return path

    def load(self, filename='tokenizer.joblib'):
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            self.tokenizer = joblib.load(path)
            return True
        return False
