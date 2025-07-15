import os
import fitz  # PyMuPDF
from docx import Document
import tiktoken
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings

class DocumentProcessor:
    """Handles document processing and text chunking."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on its extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into chunks and return with metadata."""
        if not text.strip():
            raise ValueError("Empty text content")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": i,
                    "chunk_index": i,
                    "source": filename,
                    "chunk_size": self._count_tokens(chunk)
                }
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document and return chunked text with metadata."""
        filename = os.path.basename(file_path)
        
        # Extract text from document
        text = self.extract_text(file_path)
        
        # Chunk the text
        chunks = self.chunk_text(text, filename)
        
        return chunks 