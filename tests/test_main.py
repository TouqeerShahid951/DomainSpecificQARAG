import os
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"

def test_upload_txt(monkeypatch):
    # Mock RAGService.upload_document to avoid real processing
    def mock_upload_document(self, file_path):
        return {
            "filename": "test.txt",
            "status": "success",
            "message": "Mocked upload",
            "chunks_processed": 1,
            "file_size": 10,
            "processing_time": 0.01
        }
    from app import rag_service
    monkeypatch.setattr(rag_service.RAGService, "upload_document", mock_upload_document)

    file_content = b"This is a test."
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test.txt" 