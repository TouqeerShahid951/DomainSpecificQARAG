from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str
    status: str
    message: str
    chunks_processed: int
    file_size: int

class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class QuestionResponse(BaseModel):
    """Response model for question answers."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float

class DocumentInfo(BaseModel):
    """Model for document information."""
    filename: str
    upload_date: datetime
    file_size: int
    chunks_count: int
    file_type: str

class DocumentsListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentInfo]
    total_count: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str = "1.0.0" 