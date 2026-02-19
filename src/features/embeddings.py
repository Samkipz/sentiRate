"""Provides optional sentence-transformer embeddings for multilingual text.

This module wraps the sentence_transformers library but falls back gracefully
if the dependency is missing (e.g. during demo or offline development).
"""

import os

class EmbeddingExtractor:
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            # model unavailable; user must install sentence-transformers
            self.model = None

    def encode(self, texts, normalize: bool = True):
        if self.model is None:
            raise ImportError("sentence-transformers not installed")
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=normalize)

    def save(self, filename="embeddings.npy", embeddings=None):
        import numpy as np
        from ..config import ARTIFACTS_DIR
        path = os.path.join(ARTIFACTS_DIR, filename)
        if embeddings is not None:
            np.save(path, embeddings)
        return path

    def load(self, filename="embeddings.npy"):
        import numpy as np
        from ..config import ARTIFACTS_DIR
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            return np.load(path)
        return None
