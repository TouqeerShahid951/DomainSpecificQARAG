import os
import shutil
from datetime import datetime
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.models import (
    DocumentUploadResponse, QuestionRequest, QuestionResponse,
    DocumentsListResponse, DocumentInfo, HealthResponse
)
from app.rag_service import RAGService
from app.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="RAG Q&A Chatbot API",
    description="A Retrieval-Augmented Generation based Q&A chatbot for domain-specific documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = None

def get_rag_service():
    """Dependency to get RAG service instance."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_service
    try:
        rag_service = RAGService()
        print("RAG service initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG service: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Q&A Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "ask": "POST /ask",
            "documents": "GET /documents",
            "health": "GET /health",
            "status": "GET /status"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/status")
async def get_status(rag: RAGService = Depends(get_rag_service)):
    """Get system status and component information."""
    try:
        return rag.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service)
):
    """Upload and process a document."""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Validate file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Save file temporarily
        temp_file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        result = rag.upload_document(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return DocumentUploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """Ask a question and get an answer using RAG."""
    try:
        result = rag.ask_question(request.question, request.top_k)
        
        if "error" in result["answer"].lower():
            raise HTTPException(status_code=500, detail=result["answer"])
        
        return QuestionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentsListResponse)
async def get_documents(rag: RAGService = Depends(get_rag_service)):
    """Get list of uploaded documents."""
    try:
        documents = rag.get_documents()
        
        # Convert to DocumentInfo objects
        document_infos = []
        for doc in documents:
            document_info = DocumentInfo(
                filename=doc["filename"],
                upload_date=doc["upload_date"],
                file_size=doc["file_size"],
                chunks_count=doc["chunks_count"],
                file_type=doc["file_type"]
            )
            document_infos.append(document_info)
        
        return DocumentsListResponse(
            documents=document_infos,
            total_count=len(document_infos)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    rag: RAGService = Depends(get_rag_service)
):
    """Delete a document from the vector store."""
    try:
        result = rag.delete_document(filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return {"message": result["message"]}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 