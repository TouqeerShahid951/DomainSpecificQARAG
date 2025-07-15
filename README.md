# Domain-Specific RAG Q&A Chatbot

## Project Description

This is a Retrieval-Augmented Generation (RAG) based Q&A chatbot system designed for domain-specific document analysis and question answering. The system allows users to upload various document formats (PDF, DOCX, TXT) and ask questions about their content, receiving AI-generated answers backed by relevant document sources.

### Key Features

- **Document Upload & Processing**: Supports multiple file formats with automatic text extraction and chunking
- **Semantic Search**: Uses advanced embedding models to find relevant document chunks for questions
- **Local LLM Inference**: Runs Mistral 7B model locally for privacy and offline operation
- **Source Attribution**: Provides confidence scores and source references for all answers
- **Web Interface**: User-friendly Streamlit frontend with real-time chat interface
- **RESTful API**: FastAPI backend for easy integration and extensibility
- **Vector Database**: ChromaDB for efficient document storage and retrieval

### How It Works

1. **Document Processing**: Uploaded documents are processed, chunked, and converted to embeddings
2. **Vector Storage**: Document chunks are stored in ChromaDB with their embeddings for fast retrieval
3. **Question Processing**: User questions are converted to embeddings and matched against stored documents
4. **Answer Generation**: The LLM generates contextual answers using retrieved relevant chunks
5. **Source Attribution**: The system provides confidence scores and source references for transparency

## Tools and Technologies

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs with automatic OpenAPI documentation
- **Uvicorn**: ASGI server for running the FastAPI application

### Machine Learning & AI
- **llama-cpp-python**: Python bindings for running GGUF models locally
- **Mistral 7B Instruct**: Large language model for answer generation (GGUF format)
- **Sentence Transformers**: For generating document and query embeddings
- **all-MiniLM-L6-v2**: Lightweight embedding model for semantic search

### Vector Database & Storage
- **ChromaDB**: Open-source embedding database for storing and retrieving document vectors
- **Chunking**: Intelligent document segmentation with configurable overlap

### Document Processing
- **PyMuPDF (fitz)**: PDF text extraction and processing
- **python-docx**: Microsoft Word document processing
- **Text Processing**: Custom chunking and text cleaning utilities

### Frontend
- **Streamlit**: Interactive web application framework for the chat interface
- **Real-time Chat**: Dynamic conversation interface with message history
- **File Upload**: Drag-and-drop document upload with progress indicators
- **Source Visualization**: Expandable source references with confidence scores

### Development & Configuration
- **Python 3.8+**: Core programming language
- **Environment Variables**: Configuration management via .env files
- **Type Hints**: Full type annotation for better code quality
- **Error Handling**: Comprehensive exception handling and user feedback

### Data Flow Architecture
```
Document Upload → Text Extraction → Chunking → Embedding → Vector Store
                                                           ↓
User Question → Embedding → Semantic Search → Context Retrieval → LLM → Answer
```

### System Requirements
- Python 3.8 or higher
- Sufficient RAM for LLM model loading (8GB+ recommended)
- Local storage for document embeddings and vector database
- Internet connection for initial model downloads

This system provides a complete, self-contained solution for document-based question answering with full control over data privacy and processing capabilities. 