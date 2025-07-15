#!/usr/bin/env python3
"""
Test script to verify RAG chatbot setup
"""

import sys
import os
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    packages = [
        "fastapi",
        "uvicorn",
        "chromadb",
        "transformers",
        "torch",
        "sentence_transformers",
        "fitz",  # PyMuPDF
        "docx",
        "llama_cpp",
        "streamlit",
        "numpy",
        "tiktoken",
        "langchain",
        "dotenv"
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from app.config import settings
        print("✅ Configuration loaded successfully")
        print(f"   - LLM Model: {settings.LLM_MODEL_PATH}")
        print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   - ChromaDB Path: {settings.CHROMA_PERSIST_DIRECTORY}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_directories():
    """Test if required directories exist or can be created."""
    print("\n📁 Testing directories...")
    
    from app.config import settings
    
    directories = [
        settings.UPLOAD_DIR,
        settings.CHROMA_PERSIST_DIRECTORY,
        os.path.dirname(settings.LLM_MODEL_PATH)
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}")
        except Exception as e:
            print(f"❌ {directory}: {e}")

def test_model_file():
    """Test if LLM model file exists."""
    print("\n🧠 Testing LLM model...")
    
    from app.config import settings
    
    if os.path.exists(settings.LLM_MODEL_PATH):
        print(f"✅ Model file found: {settings.LLM_MODEL_PATH}")
        file_size = os.path.getsize(settings.LLM_MODEL_PATH)
        print(f"   Size: {file_size / (1024**3):.2f} GB")
    else:
        print(f"❌ Model file not found: {settings.LLM_MODEL_PATH}")
        print("   Please download a GGUF model and update the LLM_MODEL_PATH in your .env file")

def main():
    """Run all tests."""
    print("🧪 RAG Chatbot Setup Test")
    print("=" * 40)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test configuration
    config_ok = test_config()
    
    # Test directories
    test_directories()
    
    # Test model file
    test_model_file()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} package(s) failed to import:")
        for pkg in failed_imports:
            print(f"   - {pkg}")
        print("\n💡 Run: pip install -r requirements.txt")
    else:
        print("✅ All packages imported successfully")
    
    if config_ok:
        print("✅ Configuration loaded successfully")
    else:
        print("❌ Configuration issues detected")
    
    print("\n🚀 Next steps:")
    print("1. Download a GGUF model file if not already done")
    print("2. Run: python run.py (for backend)")
    print("3. Run: streamlit run frontend/streamlit_app.py (for frontend)")

if __name__ == "__main__":
    main() 