try:
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
except ImportError:
    # keras not installed; LSTM functionality will not be available
    Sequential = None

import os
import joblib

from ..config import ARTIFACTS_DIR

class LSTMSentimentClassifier:
    def __init__(self, vocab_size=10000, embedding_dim=64, maxlen=100):
        if Sequential is None:
            raise ImportError("Keras is required for LSTM model")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.model = None
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

    def build(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.maxlen))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, texts, labels, epochs=5, batch_size=32):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.maxlen, padding='post')
        self.model = self.build()
        self.model.fit(padded, labels, epochs=epochs, batch_size=batch_size)

    def predict(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.maxlen, padding='post')
        probs = self.model.predict(padded)
        preds = ['positive' if p>0.5 else 'negative' for p in probs.flatten()]
        return preds, probs

    def save(self, base_name='lstm_sentiment'):
        if self.model:
            model_path = os.path.join(ARTIFACTS_DIR, f"{base_name}.h5")
            tokenizer_path = os.path.join(ARTIFACTS_DIR, f"{base_name}_tokenizer.joblib")
            self.model.save(model_path)
            joblib.dump(self.tokenizer, tokenizer_path)
            return model_path, tokenizer_path
        return None

    def load(self, base_name='lstm_sentiment'):
        model_path = os.path.join(ARTIFACTS_DIR, f"{base_name}.h5")
        tokenizer_path = os.path.join(ARTIFACTS_DIR, f"{base_name}_tokenizer.joblib")
        loaded = False
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            from keras.models import load_model
            self.model = load_model(model_path)
            self.tokenizer = joblib.load(tokenizer_path)
            loaded = True
        return loaded
