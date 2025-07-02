# src/rag_app/rag_engine.py - Optimized for Speed with Dual Model Support and SystemSettings Integration

import os
import pickle
import faiss
import logging
import torch
import requests
import json
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from django.conf import settings
from .models import SystemSettings
import threading
import time
import sys

# Configure logger
logger = logging.getLogger(__name__)

class MauritaniaRAG:
    def __init__(self):
        self.config = settings.RAG_CONFIG
        self.embedding_model = None
        self.llm = None  # Local GGUF model instance
        self.index = None
        self.metadatas = []
        self.documents = []
        self.initialized_core = False # Tracks core components (embeddings, index)
        self.initialized_local_llm = False # Tracks local LLM loading
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self.gemini_api_key = getattr(settings, 'GEMINI_API_KEY', None)
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        # Pre-compiled context template for speed
        self.context_template = """أجب إجابة تفصيلية على السؤال بناءً على السياق التالي:

{context}

السؤال: {question}
الإجابة التفصيلية:"""

        logger.info(f"Using device for embeddings: {self.device}" )
        self.initialize_core_components() # Initialize non-LLM parts first

    def initialize_core_components(self):
        """Initialize embedding model and vector store."""
        if self.initialized_core:
            return
        with self._lock:
            if self.initialized_core:
                return
            try:
                self._load_embedding_model()
                self._load_vector_store()
                self.initialized_core = True
                logger.info("RAG core components (Embeddings, FAISS) initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize RAG core components: {e}")
                raise

    def ensure_local_llm_loaded(self):
        """Load the GGUF model if it hasn't been loaded yet."""
        if self.initialized_local_llm:
            return
        with self._lock:
            if self.initialized_local_llm:
                return
            try:
                self._load_gguf_model()
                self.initialized_local_llm = True
                logger.info("Local GGUF LLM initialized.")
            except Exception as e:
                logger.error(f"Failed to load GGUF model on demand: {e}")
                # Don't raise here, allow fallback or API usage
                self.llm = None # Ensure llm is None if loading failed

    def _load_embedding_model(self):
        """Load embedding model with speed optimizations."""
        emb_model_name = self.config["EMBEDDING_MODEL_NAME"]
        logger.info(f"Loading embedding model: {emb_model_name}...")
        # Use smaller, faster model for speed
        if "MiniLM" in emb_model_name:
            self.embedding_model = SentenceTransformer(emb_model_name, device=self.device)
        else:
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
        self.embedding_model.eval()
        if hasattr(self.embedding_model, '_modules'):
            for module in self.embedding_model._modules.values():
                if hasattr(module, 'eval'):
                    module.eval()
        logger.info("Embedding model loaded.")

    def _load_gguf_model(self):
        """Load GGUF model with maximum speed optimizations."""
        model_path = self.config.get('GGUF_MODEL_PATH', 'models/SILMA-Kashif-2B-Instruct-v1.0.i1-IQ3_XS.gguf')
        if not os.path.exists(model_path):
            logger.warning(f"GGUF model not found at {model_path}. Local model will be unavailable.")
            self.llm = None
            return # Don't raise, just log and disable local model

        logger.info(f"Loading GGUF model: {model_path}...")
        n_gpu_layers = -1 if torch.cuda.is_available() else 0
        n_ctx = min(self.config.get('CONTEXT_LENGTH', 1024), 1024)
        n_threads = os.cpu_count() or 4
        
        # Get model settings from SystemSettings
        model_settings = self.get_model_settings('local')
        temperature = model_settings.get('temperature', 0.7)
        max_tokens = model_settings.get('max_tokens', 1000)
        top_k = model_settings.get('top_k', 40)
        
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=512,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                seed=-1,
                f16_kv=True,
                logits_all=False,
                vocab_only=False,
                embedding=False,
                chat_format="chatml"
            )
            logger.info(f"GGUF model loaded with {n_gpu_layers} GPU layers, {n_threads} threads")
        except Exception as e:
            logger.error(f"Failed to load GGUF model with optimized settings: {e}. Trying minimal settings.")
            try:
                self.llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=0,
                    n_ctx=512,
                    n_threads=2,
                    verbose=False,
                    use_mmap=True,
                    chat_format="chatml"
                )
                logger.info("GGUF model loaded with minimal settings.")
            except Exception as e_minimal:
                 logger.error(f"Failed to load GGUF model even with minimal settings: {e_minimal}")
                 self.llm = None # Ensure llm is None if loading failed

    def _load_vector_store(self):
        """Load FAISS index and metadata with optimizations."""
        index_path = self.config["VECTOR_INDEX_PATH"]
        metadata_path = self.config["METADATA_PATH"]
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("FAISS index or metadata not found. Run indexing script first.")
        self.index = faiss.read_index(str(index_path))
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(32, self.index.nprobe or 32)
        with open(metadata_path, "rb") as f:
            self.metadatas = pickle.load(f)
        self.documents = [meta.get('text', '') for meta in self.metadatas]
        logger.info(f"Loaded {self.index.ntotal} vectors and {len(self.metadatas)} metadata entries.")

    def get_model_settings(self, model_type):
        """
        Get model settings based on model type from SystemSettings
        """
        try:
            # Try to get from database
            model_params_setting = SystemSettings.objects.filter(key='model_params').first()
            if model_params_setting:
                all_params = json.loads(model_params_setting.value)
                if model_type in all_params:
                    return all_params.get(model_type, {})
        except Exception as e:
            logger.error(f"Error getting model settings: {str(e)}")
        
        # Default values if database retrieval fails
        if model_type == 'gemini':
            return {
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_k': 20
            }
        else:  # local model
            return {
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_k': 40
            }

    def get_rag_settings(self):
        """
        Get RAG settings from SystemSettings
        """
        try:
            # Try to get from database
            rag_settings_obj = SystemSettings.objects.filter(key='rag_settings').first()
            if rag_settings_obj:
                return json.loads(rag_settings_obj.value)
        except Exception as e:
            logger.error(f"Error getting RAG settings: {str(e)}")
        
        # Default values if database retrieval fails
        return {
            'num_docs': 3,
            'similarity_threshold': 0.5
        }

    def search_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Optimized document search."""
        if not self.initialized_core:
            self.initialize_core_components()
        if not self.initialized_core: # Check again after attempt
             logger.error("Core components not initialized, cannot search.")
             return []

        # Get RAG settings for top_k if not provided
        rag_settings = self.get_rag_settings()
        if top_k is None:
            top_k = rag_settings.get('num_docs', 3)
            
        query = query[:200] if len(query) > 200 else query
        try:
            with torch.no_grad():
                query_vector = self.embedding_model.encode([query], convert_to_tensor=False)
            distances, indices = self.index.search(query_vector, top_k)
            results = []
            
            # Apply similarity threshold from settings
            similarity_threshold = rag_settings.get('similarity_threshold', 0.5)
            
            for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if 0 <= idx < len(self.metadatas):
                    # Skip results below threshold
                    if score < similarity_threshold:
                        continue
                        
                    metadata_entry = self.metadatas[idx]
                    if isinstance(metadata_entry, dict) and 'text' in metadata_entry:
                        content = metadata_entry.get('text', '').strip()[:500]
                        results.append({
                            "rank": i + 1,
                            "content": content,
                            "source": metadata_entry.get("source", "Unknown"),
                            "similarity_score": float(score),
                            "chunk_id": int(idx)
                        })
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _generate_answer_local(self, question: str, context_str: str) -> Dict:
        """Generate answer using the local GGUF model."""
        self.ensure_local_llm_loaded() # Load LLM if not already loaded
        if not self.llm:
            logger.error("Local LLM not available or failed to load.")
            return {"answer": "عذراً، النموذج المحلي غير متاح حالياً.", "confidence": 0.0, "model_type": "local_unavailable"}

        # Get model settings from SystemSettings
        model_settings = self.get_model_settings('local')
        temperature = model_settings.get('temperature', 0.7)
        max_tokens = model_settings.get('max_tokens', 1000)
        top_k = model_settings.get('top_k', 40)

        prompt = self.context_template.format(context=context_str, question=question)
        try:
            start_time = time.time()
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.7,
                top_k=top_k,
                repeat_penalty=1.05,
                stop=["السؤال:", "\n\n", "السياق:", "---"],
                echo=False,
                stream=False
            )
            generation_time = time.time() - start_time

            if response and 'choices' in response and response['choices']:
                answer = response['choices'][0]['text'].strip()
                if answer.startswith("الإجابة"):
                    answer = answer.split(":", 1)[-1].strip()
                answer = " ".join(line.strip() for line in answer.split('\n') if line.strip())
                if not answer:
                    answer = "لم أتمكن من إنتاج إجابة واضحة."
            else:
                answer = "حدث خطأ في الإجابة."

            confidence = min(0.9, len(answer) / 100.0) if answer else 0.1
            return {
                "answer": answer,
                "confidence": confidence,
                "generation_time": generation_time,
                "model_type": "local_gguf"
            }
        except Exception as e:
            logger.error(f"Local GGUF generation error: {e}")
            return {"answer": "عذراً، حدث خطأ في الإجابة باستخدام النموذج المحلي.", "confidence": 0.0, "model_type": "local_error"}

    def _generate_answer_gemini(self, question: str, context_str: str) -> Dict:
        """Generate answer using the Gemini API."""
        if not self.gemini_api_key:
            logger.error("Gemini API key not configured.")
            return {"answer": "عذراً، خدمة Gemini API غير مهيأة.", "confidence": 0.0, "model_type": "gemini_error"}

        # Get model settings from SystemSettings
        model_settings = self.get_model_settings('gemini')
        temperature = model_settings.get('temperature', 0.7)
        max_tokens = model_settings.get('max_tokens', 1000)
        top_k = model_settings.get('top_k', 20)

        # Format the prompt for Gemini API
        prompt_text = f"أجب باللغة العربية عن هذا السياق:\nContext:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topK": top_k
            }
        }
        api_url_with_key = f"{self.gemini_api_url}?key={self.gemini_api_key}"

        try:
            start_time = time.time()
            response = requests.post(api_url_with_key, headers=headers, json=data, timeout=60) # Added timeout
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            generation_time = time.time() - start_time

            result = response.json()
            logger.info(f"Full Gemini API response: {json.dumps(result, indent=2, ensure_ascii=False)}") # ADD THIS LINE FOR DETAILED LOGGING
    
            answer = "لم يتم العثور على إجابة في استجابة Gemini." # Default if parsing fails
    
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                    answer_part = candidate['content']['parts'][0]
                    if 'text' in answer_part:
                        answer = answer_part['text'].strip()
                        logger.info(f"Extracted answer from Gemini: '{answer}'") # ADD/VERIFY THIS LOG
                    else:
                        logger.warning("Gemini response 'parts' does not contain 'text'.")
                elif 'error' in candidate: # Check for candidate-level errors
                    logger.error(f"Gemini API candidate error: {candidate['error']}")
                    answer = f"خطأ من Gemini: {candidate['error'].get('message', 'خطأ غير معروف')}"
                else:
                    logger.warning("Gemini response 'candidates' structure unexpected (no content/parts).")
            elif 'error' in result: # Check for top-level errors in the JSON response
                logger.error(f"Gemini API top-level error: {result['error']}")
                answer = f"خطأ من Gemini: {result['error'].get('message', 'خطأ غير معروف')}"
            else:
                logger.warning(f"Unexpected Gemini response format or empty candidates: {result}")
    
            confidence = 0.8 if answer and not answer.startswith("لم يتم") and not answer.startswith("خطأ من Gemini") else 0.1 # Adjust confidence logic
    
            return {
                "answer": answer,
                "confidence": confidence,
                "generation_time": generation_time,
                "model_type": "gemini_api" # This part seems to be working
            }
    
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request error: {e}")
            return {"answer": f"عذراً، حدث خطأ أثناء الاتصال بـ Gemini API: {e}", "confidence": 0.0, "model_type": "gemini_error"}
        except Exception as e:
            logger.error(f"Error processing Gemini response: {e}")
            return {"answer": "عذراً، حدث خطأ أثناء معالجة استجابة Gemini.", "confidence": 0.0, "model_type": "gemini_error"}

    # src/rag_app/rag_engine.py - FIXED Model Type Selection

    def ask_question(self, question: str, model_type: str = None) -> Dict:
        """Main RAG pipeline supporting model selection."""
        if not self.initialized_core:
            self.initialize_core_components()
            if not self.initialized_core:
                 return {"answer": "فشل تهيئة المكونات الأساسية.", "confidence": 0.0, "sources": [], "retrieved_documents": [], "total_time": 0}

        # FIXED: Only get from settings if model_type is explicitly None or empty
        if model_type is None or model_type == '':
            try:
                model_type_setting = SystemSettings.objects.filter(key='model_type').first()
                model_type = model_type_setting.value if model_type_setting else 'local'
                logger.info(f"No model_type provided, got from settings: {model_type}")
            except Exception as e:
                logger.error(f"Error getting model type from settings: {str(e)}")
                model_type = 'local'  # Default to local if error
        else:
            logger.info(f"Using provided model_type: {model_type}")

        logger.info(f"Processing user question with model_type='{model_type}'...")
        start_time = time.time()

        # Get RAG settings
        rag_settings = self.get_rag_settings()
        num_docs = rag_settings.get('num_docs', 3)

        relevant_docs = self.search_documents(question, top_k=num_docs)
        search_time = time.time() - start_time

        if not relevant_docs:
            return {
                "answer": "لم أجد معلومات ذات صلة.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_documents": [],
                "search_time": round(search_time, 2),
                "generation_time": 0,
                "total_time": round(search_time, 2),
                "model_used": "none"
            }

        # Use documents for context
        context_docs = relevant_docs
        context_chunks = [doc["content"] for doc in context_docs]

        generation_start = time.time()
        # Pass the selected model_type to generate_answer
        generation_result = self.generate_answer(question, context_chunks, model_type=model_type)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Extract sources for citation
        sources = []
        for doc in context_docs:
            if doc["source"] and doc["source"] not in sources:
                sources.append(doc["source"])

        return {
            "answer": generation_result.get("answer", "لا توجد إجابة."),
            "confidence": generation_result.get("confidence", 0.0),
            "sources": sources,
            "retrieved_documents": relevant_docs,
            "search_time": round(search_time, 2),
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
            "model_used": generation_result.get("model_type", "unknown")
        }

    def generate_answer(self, question: str, context_chunks: List[str], model_type: str = None) -> Dict:
        """Generate answer using the selected model type ('local' or 'gemini')."""
        if not self.initialized_core:
             return {"answer": "عذراً، الخدمة غير متاحة حالياً (core not ready).", "confidence": 0.0}

        if not context_chunks:
            return {"answer": "لم أجد معلومات ذات صلة.", "confidence": 0.0}

        # FIXED: Only get from settings if model_type is explicitly None or empty
        if model_type is None or model_type == '':
            try:
                model_type_setting = SystemSettings.objects.filter(key='model_type').first()
                model_type = model_type_setting.value if model_type_setting else 'local'
                logger.info(f"No model_type provided to generate_answer, got from settings: {model_type}")
            except Exception as e:
                logger.error(f"Error getting model type from settings: {str(e)}")
                model_type = 'local'  # Default to local if error
        else:
            logger.info(f"Using provided model_type in generate_answer: {model_type}")

        # Get RAG settings
        rag_settings = self.get_rag_settings()
            
        # Limit context size for speed/cost
        limited_context = []
        total_chars = 0
        # Adjust max_context based on model? Gemini might handle more.
        max_context = 1500 # Increased slightly

        for chunk in context_chunks:
            if total_chars + len(chunk) > max_context:
                break
            limited_context.append(chunk) # Use full chunk up to limit
            total_chars += len(chunk)

        context_str = "\n---\n".join(limited_context) # Use separator
        question = question[:200] if len(question) > 200 else question # Truncate question

        logger.info(f"Generating answer using model type: {model_type}")

        if model_type == 'gemini':
            return self._generate_answer_gemini(question, context_str)
        elif model_type == 'local':
            return self._generate_answer_local(question, context_str)
        else:
            logger.warning(f"Unknown model type requested: {model_type}. Defaulting to local.")
            return self._generate_answer_local(question, context_str)
# Initialize the RAG engine as a singleton
rag_engine_instance = MauritaniaRAG()

def rag_engine(query, model_type=None):
    """
    RAG engine function that processes queries and returns responses with sources.
    
    Args:
        query (str): The user's query
        model_type (str): The model to use ('local' or 'gemini'), if None, gets from settings
        
    Returns:
        tuple: (response, sources, confidence)
    """
    try:
        result = rag_engine_instance.ask_question(query, model_type)
        return result["answer"], result["sources"], result["confidence"]
    except Exception as e:
        logger.error(f"Error in RAG engine: {str(e)}")
        return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.", [], 0.0

# Initialize the RAG engine as a singleton
rag_engine_instance = MauritaniaRAG()