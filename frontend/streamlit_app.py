import streamlit as st
import requests
import json
import time
from datetime import datetime
import os

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="RAG Q&A Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG Q&A Chatbot")
    st.markdown("Upload your documents and ask questions!")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    try:
                        files = {"file": uploaded_file}
                        response = requests.post(f"{API_BASE_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.info(f"📊 Processed {result['chunks_processed']} chunks")
                        else:
                            st.error(f"❌ Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Document list
        st.subheader("📋 Uploaded Documents")
        try:
            response = requests.get(f"{API_BASE_URL}/documents")
            if response.status_code == 200:
                documents = response.json()["documents"]
                if documents:
                    for doc in documents:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📄 {doc['filename']}")
                            st.caption(f"Chunks: {doc['chunks_count']} | Size: {doc['file_size']} chars")
                        with col2:
                            if st.button("🗑️", key=f"del_{doc['filename']}"):
                                delete_response = requests.delete(f"{API_BASE_URL}/documents/{doc['filename']}")
                                if delete_response.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.error("Delete failed")
                else:
                    st.info("No documents uploaded yet")
            else:
                st.error("Failed to load documents")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
        
        st.divider()
        
        # System status
        st.subheader("⚙️ System Status")
        try:
            response = requests.get(f"{API_BASE_URL}/status")
            if response.status_code == 200:
                status = response.json()
                
                # Embedding service
                emb_status = status.get("embedding_service", {})
                st.write(f"🔤 Embeddings: {emb_status.get('status', 'Unknown')}")
                st.caption(f"Model: {emb_status.get('model', 'Unknown')}")
                
                # Vector store
                vs_status = status.get("vector_store", {})
                st.write(f"🗄️ Vector Store: {vs_status.get('status', 'Unknown')}")
                st.caption(f"Documents: {vs_status.get('document_count', 0)}")
                
                # LLM service
                llm_status = status.get("llm_service", {})
                st.write(f"🧠 LLM: {llm_status.get('status', 'Unknown')}")
                st.caption(f"Model: {llm_status.get('model_type', 'Unknown')}")
            else:
                st.error("Failed to get system status")
        except Exception as e:
            st.error(f"Error getting status: {str(e)}")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}** (Score: {source['score']:.3f})")
                        st.write(f"File: {source['filename']}")
                        st.write(f"Text: {source['text']}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt, "top_k": 5}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.markdown(result["answer"])
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "confidence": result["confidence"],
                            "processing_time": result["processing_time"]
                        })
                        
                        # Show confidence and processing time
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{result['confidence']:.3f}")
                        with col2:
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        
                        # Show sources
                        if result["sources"]:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(result["sources"], 1):
                                    st.write(f"**Source {i}** (Score: {source['score']:.3f})")
                                    st.write(f"File: {source['filename']}")
                                    st.write(f"Text: {source['text']}")
                                    st.divider()
                    else:
                        error_msg = f"❌ Error: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.session_state.messages and st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 