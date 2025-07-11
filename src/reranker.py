"""
Reranker module for RAG system.
Extends EmbeddingRetriever with cross-encoder reranking capabilities.
"""

import torch
from typing import List
from langchain_core.documents import Document
from embeddings import EmbeddingRetriever, MODEL_NAME, USE_GPU

# Reranking configurations
RERANKING_MODEL = "BAAI/bge-reranker-base"  # Fast and effective cross-encoder
USE_RERANKING = True
INITIAL_RETRIEVAL_K = 20  # Retrieve more docs for reranking
RERANK_TOP_K = 5  # Final number after reranking

class RerankingEmbeddingRetriever(EmbeddingRetriever):
    """
    Enhanced retriever with cross-encoder reranking capabilities.
    
    Implements a two-stage retrieval:
    1. Initial retrieval with embedding similarity (FAISS)
    2. Reranking with cross-encoder for better relevance
    """
    
    def __init__(
        self,
        texts: List[str],
        k: int = 3,
        model_name: str = MODEL_NAME,
        reranking_model: str = RERANKING_MODEL,
        use_reranking: bool = USE_RERANKING,
        initial_k: int = INITIAL_RETRIEVAL_K,
        use_gpu: bool = USE_GPU
    ):
        # Initialize parent class (embedding retriever)
        super().__init__(texts, k, model_name, use_gpu)
        
        # Reranking configurations
        self.use_reranking = use_reranking
        self.initial_k = initial_k
        self.reranking_model_name = reranking_model
        self.reranking_model = None
        
        # Initialize reranking model if enabled
        if self.use_reranking:
            self._load_reranking_model()
    
    def _load_reranking_model(self):
        """Load the cross-encoder reranking model."""
        try:
            from sentence_transformers import CrossEncoder
            print(f"🎯 Loading reranking model: {self.reranking_model_name}")
            self.reranking_model = CrossEncoder(
                self.reranking_model_name, 
                device=self.device
            )
            print(f"✅ Reranking model loaded successfully")
        except ImportError:
            print("❌ sentence-transformers not found. Install with: pip install sentence-transformers")
            self.use_reranking = False
        except Exception as e:
            print(f"❌ Failed to load reranking model: {e}")
            self.use_reranking = False
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and rerank relevant documents for a query.
        
        Process:
        1. Initial retrieval with embedding similarity
        2. Cross-encoder reranking for better relevance
        3. Return top-k reranked documents
        """
        
        if not self.use_reranking or self.reranking_model is None:
            # Fall back to original embedding-only retrieval
            return super().get_relevant_documents(query)
        
        # Step 1: Initial retrieval with higher k
        initial_k = max(self.initial_k, self.k)  # Ensure we retrieve at least k docs
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=self.device)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in FAISS index for initial candidates
        distances, indices = self.index.search(query_embedding, min(initial_k, len(self.texts)))
        
        # Get initial documents with metadata
        initial_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                initial_docs.append({
                    "text": self.texts[idx],
                    "index": int(idx),
                    "distance": float(distances[0][i]),
                    "initial_rank": i + 1
                })
        
        if len(initial_docs) <= self.k:
            # If we have few documents, return them directly
            return [
                Document(
                    page_content=doc["text"],
                    metadata={
                        "distance": doc["distance"],
                        "index": doc["index"],
                        "initial_rank": doc["initial_rank"]
                    }
                ) 
                for doc in initial_docs
            ]
        
        # Step 2: Rerank using cross-encoder
        try:
            # Prepare query-document pairs for reranking
            query_doc_pairs = [[query, doc["text"]] for doc in initial_docs]
            
            # Get reranking scores
            rerank_scores = self.reranking_model.predict(query_doc_pairs)
            
            # Combine scores with documents
            for i, doc in enumerate(initial_docs):
                doc["rerank_score"] = float(rerank_scores[i])
            
            # Sort by reranking score (higher is better)
            reranked_docs = sorted(initial_docs, key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top-k reranked documents
            top_k_docs = reranked_docs[:self.k]
            
            return [
                Document(
                    page_content=doc["text"],
                    metadata={
                        "rerank_score": doc["rerank_score"],
                        "distance": doc["distance"],
                        "index": doc["index"],
                        "initial_rank": doc["initial_rank"],
                        "final_rank": i + 1
                    }
                ) 
                for i, doc in enumerate(top_k_docs)
            ]
            
        except Exception as e:
            print(f"⚠️ Reranking failed, falling back to embedding similarity: {str(e)}")
            
            # Fall back to original similarity-based ranking
            return [
                Document(
                    page_content=doc["text"],
                    metadata={
                        "distance": doc["distance"],
                        "index": doc["index"],
                        "initial_rank": doc["initial_rank"],
                        "reranking_failed": True
                    }
                ) 
                for doc in initial_docs[:self.k]
            ]
    
    def set_reranking(self, use_reranking: bool):
        """Enable or disable reranking."""
        if use_reranking and self.reranking_model is None:
            # Try to load reranking model
            self._load_reranking_model()
        
        self.use_reranking = use_reranking and self.reranking_model is not None
    
    def set_initial_k(self, initial_k: int):
        """Set the number of documents to retrieve before reranking."""
        self.initial_k = max(initial_k, self.k)  # Ensure initial_k >= k
    
    def get_reranking_info(self) -> dict:
        """Get information about the reranking setup."""
        return {
            "use_reranking": self.use_reranking,
            "reranking_model": self.reranking_model_name,
            "initial_k": self.initial_k,
            "final_k": self.k,
            "model_loaded": self.reranking_model is not None
        } 