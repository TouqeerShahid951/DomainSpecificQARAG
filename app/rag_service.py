import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.document_processor import DocumentProcessor
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.llm_service import LLMService
from app.config import settings

class RAGService:
    """Main RAG service that orchestrates all components."""
    
    def __init__(self):
        """Initialize all RAG components."""
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(self.embedding_service)
        self.llm_service = LLMService()
        
        # Create necessary directories
        settings.create_directories()
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload and process a document."""
        try:
            start_time = time.time()
            
            # Get file info
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # Process document
            chunks = self.document_processor.process_document(file_path)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            processing_time = time.time() - start_time
            
            return {
                "filename": filename,
                "status": "success",
                "message": f"Document processed successfully in {processing_time:.2f}s",
                "chunks_processed": len(chunks),
                "file_size": file_size,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "filename": os.path.basename(file_path) if file_path else "unknown",
                "status": "error",
                "message": str(e),
                "chunks_processed": 0,
                "file_size": 0,
                "processing_time": 0
            }
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question and get an answer using RAG."""
        try:
            start_time = time.time()
            
            # Search for relevant documents
            search_results = self.vector_store.search(question, top_k)
            
            if not search_results:
                return {
                    "question": question,
                    "answer": "I don't have any relevant documents to answer your question. Please upload some documents first.",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Generate answer using LLM
            answer = self.llm_service.generate_answer(question, search_results)
            
            # Calculate confidence based on search scores
            avg_confidence = sum(result["score"] for result in search_results) / len(search_results)
            
            # Format sources
            sources = []
            for result in search_results:
                source = {
                    "filename": result["metadata"]["filename"],
                    "chunk_id": result["metadata"]["chunk_id"],
                    "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                    "score": result["score"]
                }
                sources.append(source)
            
            processing_time = time.time() - start_time
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": avg_confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of uploaded documents."""
        try:
            filenames = self.vector_store.list_filenames()
            documents = []
            
            for filename in filenames:
                # Get document chunks
                chunks = self.vector_store.get_documents_by_filename(filename)
                
                # Calculate total size (approximate)
                total_size = sum(len(chunk["text"]) for chunk in chunks)
                
                document_info = {
                    "filename": filename,
                    "upload_date": datetime.now(),  # This would need to be stored in metadata
                    "file_size": total_size,
                    "chunks_count": len(chunks),
                    "file_type": os.path.splitext(filename)[1].lower()
                }
                documents.append(document_info)
            
            return documents
            
        except Exception as e:
            print(f"Error getting documents: {str(e)}")
            return []
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete a document from the vector store."""
        try:
            self.vector_store.delete_documents_by_filename(filename)
            
            return {
                "filename": filename,
                "status": "success",
                "message": f"Document '{filename}' deleted successfully"
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "message": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and component information."""
        return {
            "embedding_service": {
                "model": settings.EMBEDDING_MODEL,
                "device": settings.EMBEDDING_DEVICE,
                "status": "loaded"
            },
            "vector_store": {
                "type": "ChromaDB",
                "collection": settings.CHROMA_COLLECTION_NAME,
                "document_count": self.vector_store.get_document_count(),
                "status": "ready"
            },
            "llm_service": {
                "model_path": settings.LLM_MODEL_PATH,
                "model_type": settings.LLM_MODEL_TYPE,
                "status": "loaded" if self.llm_service.is_available() else "not_loaded"
            },
            "document_processor": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "status": "ready"
            }
        } 