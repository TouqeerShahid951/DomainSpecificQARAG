#!/usr/bin/env python3
"""
Startup script for the RAG Q&A Chatbot
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("🤖 Starting RAG Q&A Chatbot...")
    print(f"📡 API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"🌐 Frontend will be available at: http://localhost:8501")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    ) 