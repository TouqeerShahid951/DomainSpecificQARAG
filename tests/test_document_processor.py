import os
import tempfile
from app.document_processor import DocumentProcessor

def test_chunking_simple_text():
    processor = DocumentProcessor()
    text = "This is a test document. " * 30  # long enough for multiple chunks
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as f:
        f.write(text)
        temp_path = f.name
    try:
        chunks = processor.process_document(temp_path)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert len(chunk["text"]) > 0
    finally:
        os.remove(temp_path) 