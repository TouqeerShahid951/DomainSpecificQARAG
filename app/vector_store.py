import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import os
from app.config import settings
from app.embedding_service import EmbeddingService

class VectorStore:
    """Vector store service using ChromaDB."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"ChromaDB initialized with collection: {settings.CHROMA_COLLECTION_NAME}")
            
        except Exception as e:
            raise Exception(f"Error initializing ChromaDB: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        try:
            if not documents:
                return
            
            # Extract texts and metadata
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            ids = [f"{doc['metadata']['filename']}_{doc['metadata']['chunk_id']}" for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_service.generate_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_single_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i]  # Convert distance to similarity score
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error searching vector store: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
            return 0
    
    def get_documents_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """Get all documents for a specific filename."""
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    result = {
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error getting documents by filename: {str(e)}")
    
    def delete_documents_by_filename(self, filename: str) -> None:
        """Delete all documents for a specific filename."""
        try:
            # Get document IDs for the filename
            results = self.collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            if results["metadatas"]:
                ids_to_delete = [f"{filename}_{metadata['chunk_id']}" for metadata in results["metadatas"]]
                self.collection.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} documents for filename: {filename}")
            
        except Exception as e:
            raise Exception(f"Error deleting documents by filename: {str(e)}")
    
    def list_filenames(self) -> List[str]:
        """Get list of all unique filenames in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            filenames = set()
            
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    filenames.add(metadata["filename"])
            
            return list(filenames)
            
        except Exception as e:
            print(f"Error listing filenames: {str(e)}")
            return [] 