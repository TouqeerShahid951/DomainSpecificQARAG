import os
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from app.config import settings

class LLMService:
    """Service for local LLM inference using llama-cpp-python."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the local LLM model."""
        try:
            model_path = settings.LLM_MODEL_PATH
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading LLM model: {model_path}")
            
            # Initialize Llama model
            self.model = Llama(
                model_path=model_path,
                n_ctx=settings.LLM_MAX_TOKENS,
                n_threads=os.cpu_count(),
                n_gpu_layers=0,  # Set to higher value if GPU is available
                verbose=False
            )
            
            print("LLM model loaded successfully")
            
        except Exception as e:
            raise Exception(f"Error loading LLM model: {str(e)}")
    
    def _create_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM with context and question."""
        # Format context
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"Document {i} (from {doc['metadata']['filename']}):\n{doc['text']}\n\n"
        
        # Create prompt based on model type
        if settings.LLM_MODEL_TYPE.lower() == "mistral":
            prompt = f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find the answer in the provided documents."

Context:
{context_text}

Question: {question}

Answer: [/INST]"""
        else:
            # Generic prompt format
            prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Generate an answer using the local LLM."""
        try:
            if not self.model:
                raise Exception("LLM model not loaded")
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                stop=["</s>", "[INST]", "Question:", "Context:"],
                echo=False
            )
            
            # Extract answer
            answer = response['choices'][0]['text'].strip()
            
            # Clean up the answer
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            return answer
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": settings.LLM_MODEL_PATH,
            "model_type": settings.LLM_MODEL_TYPE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE
        } 