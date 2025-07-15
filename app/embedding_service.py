import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from app.config import settings
import os

class EmbeddingService:
    """Service for generating embeddings using SentenceTransformers."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Embedding model loaded successfully")
            
        except Exception as e:
            raise Exception(f"Error loading embedding model: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            return embeddings_list
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.model is None:
            raise Exception("Model not loaded")
        return self.model.get_sentence_embedding_dimension() 