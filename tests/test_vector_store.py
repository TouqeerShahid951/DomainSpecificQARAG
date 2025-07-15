import numpy as np
from app.vector_store import VectorStore

class DummyEmbeddingService:
    def embed_text(self, text):
        return np.ones(384)

def test_vector_store_add_and_search(tmp_path):
    embedding_service = DummyEmbeddingService()
    store = VectorStore(embedding_service)
    docs = [
        {"text": "The quick brown fox.", "metadata": {"filename": "a.txt", "chunk_id": 0}},
        {"text": "Jumps over the lazy dog.", "metadata": {"filename": "a.txt", "chunk_id": 1}},
    ]
    store.add_documents(docs)
    results = store.search("fox", top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert "text" in results[0] 