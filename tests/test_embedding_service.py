import numpy as np
import pytest
from app.embedding_service import EmbeddingService

class DummyModel:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(384)
        return [np.ones(384) for _ in texts]

def test_embedding_service_monkeypatch(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "_load_model", lambda self: DummyModel())
    service = EmbeddingService()
    emb = service.embed_text("hello world")
    assert isinstance(emb, (list, np.ndarray))
    assert len(emb) == 384 